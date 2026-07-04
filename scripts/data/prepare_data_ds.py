#!/usr/bin/env python3
"""Generate synthetic target cache data for DeepSeek-V4 Flash training.

Generates random hidden states shaped according to the target model config.
Each sample contains complete hidden states for all captured target layers
(target_layer_ids = [40, 41, 42], i.e. 3 layers), plus the final last_hidden_state.

Usage::

    # Default: 8 samples, seq_len=2048
    python scripts/data/prepare_data_ds.py

    # Custom sample count and sequence length
    python scripts/data/prepare_data_ds.py --num-samples 64 --seq-len 4096

    # Custom output directory
    python scripts/data/prepare_data_ds.py --output-dir /path/to/cache
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Allow running from any directory — resolve project root relative to this script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepspec.data.target_cache_dataset import (  # noqa: E402
    LocalCacheWriteSummary,
    LocalTargetCacheWriter,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)

# ---------------------------------------------------------------------------
# Defaults sourced from config/dspark/dspark_deepseek_v4_flash.py and
# models/deepseek_v4_flash_hf_config/config.json
# ---------------------------------------------------------------------------
DEFAULT_TARGET_LAYER_IDS = [40, 41, 42]  # Ascend DSpark target layers
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_VOCAB_SIZE = 129280
DEFAULT_MAX_LENGTH = 4096
DEFAULT_TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic target cache for DeepSeek-V4 Flash training"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of synthetic samples to generate (default: 8)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length for each sample (default: 2048)",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/ds_target_cache",
        help="Output directory for the target cache (default: /workspace/ds_target_cache)",
    )
    parser.add_argument(
        "--config-path",
        default=os.path.join(_PROJECT_ROOT, "config", "dspark", "dspark_deepseek_v4_flash.py"),
        help="Path to the training config .py file",
    )
    parser.add_argument(
        "--model-config-path",
        default=os.path.join(
            _PROJECT_ROOT, "models", "deepseek_v4_flash_hf_config", "config.json"
        ),
        help="Path to the model config.json",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=64 * 1024**3,
        help="Maximum shard size in bytes (default: 64 GiB)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--min-loss-tokens",
        type=int,
        default=14,
        help="Recorded min_loss_tokens in manifest (default: 14)",
    )
    return parser.parse_args()


def _load_model_params(model_config_path: str) -> dict:
    """Load key model parameters from config.json."""
    with open(model_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "hidden_size": int(cfg["hidden_size"]),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "vocab_size": int(cfg["vocab_size"]),
    }


def _resolve_target_layer_ids(config_path: str) -> list[int]:
    """Extract target_layer_ids from the training config .py file.

    Falls back to the hard-coded default if the config cannot be parsed.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_cfg", config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load config from {config_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model_cfg = getattr(mod, "model", {})
        layer_ids = model_cfg.get("target_layer_ids")
        if layer_ids is not None:
            return [int(lid) for lid in layer_ids]
    except Exception:
        pass
    return list(DEFAULT_TARGET_LAYER_IDS)


def main() -> None:
    args = parse_args()

    # Resolve model parameters -------------------------------------------------
    model_params = _load_model_params(args.model_config_path)
    hidden_size = model_params["hidden_size"]
    vocab_size = model_params["vocab_size"]
    num_hidden_layers = model_params["num_hidden_layers"]

    # Resolve target layers ----------------------------------------------------
    target_layer_ids = _resolve_target_layer_ids(args.config_path)
    num_target_layers = len(target_layer_ids)

    output_dir = os.path.abspath(args.output_dir)
    seq_len = args.seq_len

    # Print summary ------------------------------------------------------------
    print("=== Synthetic Target Cache Generator ===")
    print(f"  hidden_size:        {hidden_size}")
    print(f"  num_hidden_layers:  {num_hidden_layers}")
    print(f"  vocab_size:         {vocab_size}")
    print(f"  target_layer_ids:   {target_layer_ids}")
    print(f"  num_target_layers:  {num_target_layers}")
    print(f"  seq_len:            {seq_len}")
    print(f"  num_samples:        {args.num_samples}")
    print(f"  output_dir:         {output_dir}")
    print(f"  target_hidden_states shape:  (seq_len, {num_target_layers * hidden_size})")
    print(f"  target_last_hidden_states:    (seq_len, {hidden_size})")
    print()

    # Reproducibility ----------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare output directory -------------------------------------------------
    prepare_target_cache_output_dir(output_dir)

    rank_dir = os.path.join(output_dir, "_tmp", "rank_0")
    os.makedirs(rank_dir, exist_ok=True)

    # Write samples ------------------------------------------------------------
    writer = LocalTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=args.max_shard_bytes,
    )

    for sample_id in range(args.num_samples):
        # -- input_ids: random token ids ---------------------------------------
        input_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.int32)

        # -- attention_mask: random valid-length region ------------------------
        actual_len = torch.randint(seq_len // 2, seq_len + 1, (1,)).item()
        attention_mask = torch.zeros(seq_len, dtype=torch.uint8)
        attention_mask[:actual_len] = 1

        # -- loss_mask: assistant tokens in the latter half --------------------
        loss_mask = torch.zeros(seq_len, dtype=torch.uint8)
        loss_start = actual_len // 2
        loss_mask[loss_start:actual_len] = 1

        # -- target_hidden_states: (seq_len, num_target_layers * hidden_size) --
        # Concatenation of per-layer hidden states along the last dimension.
        # Each of the 3 target layers produces a (seq_len, hidden_size) tensor;
        # they are concatenated into one (seq_len, 3*4096) = (seq_len, 12288).
        target_hidden_states = torch.randn(
            seq_len, num_target_layers * hidden_size, dtype=torch.bfloat16
        )

        # -- target_last_hidden_states: (seq_len, hidden_size) -----------------
        # The final output of the target model (last_hidden_state).
        target_last_hidden_states = torch.randn(
            seq_len, hidden_size, dtype=torch.bfloat16
        )

        writer.write_sample(
            sample_id=sample_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
        )

        if (sample_id + 1) % 100 == 0 or sample_id == args.num_samples - 1:
            print(f"  [{sample_id + 1:6d}/{args.num_samples}] samples written", flush=True)

    writer.close()
    print(f"  Wrote {writer.num_local_samples} samples across "
          f"{len(writer.local_shard_files)} local shard(s)")

    # -- Rank summary ----------------------------------------------------------
    summary = LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=0,
        source_sample_end=args.num_samples,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))

    # -- Global shard map & rename ---------------------------------------------
    summaries = [load_local_cache_write_summary(rank_dir)]
    shard_map, shards = build_global_target_cache_shard_map(summaries)

    local_summary = load_local_cache_write_summary(rank_dir)
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=local_summary,
        shard_map=shard_map,
    )

    # -- Global index ----------------------------------------------------------
    num_valid_samples = finalize_target_cache_index(
        output_dir=output_dir,
        summaries=summaries,
        shard_map=shard_map,
    )

    # -- Manifest --------------------------------------------------------------
    manifest = build_target_cache_manifest(
        num_samples=num_valid_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": DEFAULT_TARGET_MODEL_NAME,
            "source_jsonl_paths": ["synthetic_data"],
            "chat_template": "deepseek",
            "max_length": seq_len,
            "min_loss_tokens": int(args.min_loss_tokens),
            "project_name": "deepspec",
            "exp_name": "dspark_block5_deepseek_v4_flash_synthetic",
            "git_sha": "synthetic",
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)

    # -- Cleanup temp directory ------------------------------------------------
    cleanup_target_cache_tmp_dir(output_dir)

    # -- Done ------------------------------------------------------------------
    total_mb = sum(
        os.path.getsize(os.path.join(output_dir, s["file_name"])) for s in shards
    ) / (1024**2)
    print(f"\n=== Done ===")
    print(f"  Samples:    {num_valid_samples}")
    print(f"  Shards:     {len(shards)} ({total_mb:.1f} MB total)")
    print(f"  Output:     {output_dir}")
    for f in ["manifest.json", "samples.idx"] + [s["file_name"] for s in shards]:
        fpath = os.path.join(output_dir, f)
        if os.path.exists(fpath):
            print(f"    {f}  ({os.path.getsize(fpath) / (1024**2):.1f} MB)")


if __name__ == "__main__":
    main()
