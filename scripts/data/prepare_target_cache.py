import argparse
from dataclasses import dataclass
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from deepspec.data import ConversationCollator
from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    compute_local_sample_range,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.utils import (
    CustomJSONEncoder,
    get_git_diff,
    get_git_sha,
    init_dist,
    is_global_main_process,
    load_config,
    main_process_first,
    parse_opts_to_config,
    print_on_global_main,
    print_on_local_main,
    seed_all,
)

os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class TargetForwardResult:
    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


# ---- Model loading (NPU-aware, with optional device-map sharding) ----


def _resolve_device_map(*, model_type: str, cli_device_map: str | None, device: torch.device) -> str | dict | None:
    """Determine the ``device_map`` value for ``AutoModel.from_pretrained``.

    ================  ========================================================
    ``--device-map``  behaviour
    ================  ========================================================
    ``"auto"``        Use ``accelerate`` to shard layers across all visible
                      devices.  Best for large models (e.g. DeepSeek-V4).
    ``"single"``      Load the entire model onto *device* (no sharding).
                      Suitable when the model fits on a single NPU / GPU.
    ``None`` (omit)   Auto-detect: ``"auto"`` for ``deepseek_v4`` models,
                      ``"single"`` for everything else.
    ================  ========================================================
    """
    if cli_device_map == "auto":
        return "auto"
    if cli_device_map == "single":
        return {"": device}
    if cli_device_map is not None:
        raise ValueError(f"Unsupported --device-map value: {cli_device_map!r}")

    # Auto-detect: large / custom-attention models → auto shard.
    if model_type in ("deepseek_v4",):
        return "auto"
    return {"": device}


def _load_target_model(*, model_name_or_path: str, dtype: torch.dtype, attn_impl: str, device_map):
    """Load the target model, placing parameters directly on the target device(s).

    When *device_map* is ``"auto"``, ``accelerate`` distributes decoder layers
    across all visible NPUs / GPUs so that a model larger than a single device
    can still run.  When *device_map* is a ``{"": device}`` dict the entire
    model stays on *device* (fast path for small models).
    """
    load_kwargs = dict(
        dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    print_on_local_main(
        f"Loading target model from {model_name_or_path!r} "
        f"(device_map={device_map!r}, dtype={dtype}, attn={attn_impl})..."
    )
    model = AutoModel.from_pretrained(model_name_or_path, **load_kwargs).eval()

    # Freeze the entire model — we only need forward passes.
    for p in model.parameters():
        p.requires_grad_(False)

    return model


def _empty_device_cache(device: torch.device):
    """Clear accelerator memory cache (NPU or CUDA)."""
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _get_target_backbone(target_model):
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model") and hasattr(target_model.model, "language_model"):
            return target_model.model.language_model
        assert False, "Gemma4 target model must expose a text language_model."
    # DeepSeek-V4 (and similar deepseek_v* models) expose the backbone
    # as ``.model`` (a DeepseekV4Model).
    if model_type in ("deepseek_v4",):
        return target_model.model
    return getattr(target_model, "model", target_model)


def _get_target_hidden_size(target_model) -> int:
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified"):
        return int(target_model.config.text_config.hidden_size)
    return int(target_model.config.hidden_size)


def _get_hook_tensor(output):
    """Extract a single (N, H) tensor from a layer output.

    Handles two special cases:

    * **Tuple output** (e.g. ``(hidden_states, residual)``): takes the first
      element.
    * **Hyper-connection output** (shape ``(N, hc_mult, H)``): mean-folds
      the ``hc_mult`` dimension into a single ``(N, H)`` tensor.  This is
      needed for DeepSeek-V4 and similar models that maintain multiple
      residual streams.
    """
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            tensor = first
        else:
            raise TypeError(f"Unsupported target hook output type: {type(output)!r}")
    else:
        raise TypeError(f"Unsupported target hook output type: {type(output)!r}")

    # Hyper-connection mean-folding: DeepSeek-V4 layers return (N, hc_mult, H).
    if tensor.ndim == 3 and tensor.shape[1] > 1:
        return tensor.mean(dim=1).detach()
    return tensor.detach()


def run_target_forward_with_hooks(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids,
):
    backbone = _get_target_backbone(target_model)
    layer_modules = backbone.layers
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    captured_hidden_states = {}
    handles = []

    # When device_map="auto" shards layers across devices, each hook captures
    # its tensor on the layer's host device.  We record the device of the
    # first captured tensor and move everything there before concatenation.
    _capture_device = [None]  # mutable container for the nested hook

    def capture_layer(layer_id: int):
        def hook(_module, _inputs, output):
            tensor = _get_hook_tensor(output)
            if _capture_device[0] is None:
                _capture_device[0] = tensor.device
            captured_hidden_states[layer_id] = tensor

        return hook

    try:
        if -1 in target_layer_ids:
            handles.append(
                backbone.embed_tokens.register_forward_hook(capture_layer(-1))
            )
        for layer_id in target_layer_ids:
            if layer_id < 0:
                continue
            handles.append(
                layer_modules[layer_id].register_forward_hook(capture_layer(layer_id))
            )

        with torch.no_grad():
            target_output = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            target_last_hidden_states = target_output.last_hidden_state.detach()
            # Move all captured tensors to a common device before concatenation.
            common_device = _capture_device[0] or target_last_hidden_states.device
            aligned = []
            for layer_id in target_layer_ids:
                t = captured_hidden_states[layer_id]
                if t.device != common_device:
                    t = t.to(device=common_device, non_blocking=True)
                aligned.append(t)
            target_hidden_states = torch.cat(aligned, dim=-1)
    finally:
        for handle in handles:
            handle.remove()
        captured_hidden_states.clear()

    return TargetForwardResult(
        target_hidden_states=target_hidden_states,
        target_last_hidden_states=target_last_hidden_states,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Training JSONL path. Repeat this argument to use multiple files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--model-config-path",
        default=None,
        help=(
            "Path to model config files (config.json, tokenizer.json, etc.). "
            "When set, AutoConfig and AutoTokenizer load from this path "
            "instead of target_model_name_or_path.  Useful when the config "
            "has been pre-converted (e.g. via cann-recipes-infer convert_config.py) "
            "but the original model dir is read-only.  Weights are still loaded "
            "from target_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--device-map",
        choices=["auto", "single"],
        default=None,
        help=(
            "Device placement strategy for the target model. "
            "'auto' uses accelerate to shard layers across all visible devices "
            "(best for large models like DeepSeek-V4). "
            "'single' loads the entire model onto one device. "
            "Default: auto-detect based on model type."
        ),
    )
    cli_args = parser.parse_args()
    config = parse_opts_to_config(cli_args.opts, load_config(cli_args.config))
    return cli_args, config


def _write_manifest(
    *,
    output_dir: str,
    config,
    train_data_paths,
    target_layer_ids,
    hidden_size: int,
    min_loss_tokens: int,
    shards,
):
    num_samples = sum(
        int(
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )["num_local_samples"]
        )
        for rank in range(dist.get_world_size())
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(config.model.target_model_name_or_path),
            "source_jsonl_paths": [str(path) for path in train_data_paths],
            "chat_template": str(config.data.chat_template),
            "max_length": int(config.data.max_length),
            "min_loss_tokens": int(min_loss_tokens),
            "project_name": (
                str(config.get("project_name"))
                if config.get("project_name") is not None
                else None
            ),
            "exp_name": (
                str(config.get("exp_name"))
                if config.get("exp_name") is not None
                else None
            ),
            "git_sha": str(get_git_sha()),
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)


def _print_prepare_progress(*, global_rank: int, processed_samples: int, total_samples: int):
    print(
        f"[prepare rank {global_rank}] {processed_samples}/{total_samples} samples",
        flush=True,
    )


def main():
    cli_args, config = parse_args()
    train_data_paths = list(cli_args.train_data_path)
    target_layer_ids = [int(layer_id) for layer_id in config.model.target_layer_ids]
    min_loss_tokens = int(cli_args.min_loss_tokens)
    seed_all(int(config.seed))
    device, global_rank, world_size = init_dist()
    output_dir = os.path.abspath(cli_args.output_dir)
    print_on_local_main(json.dumps(config, indent=4, cls=CustomJSONEncoder), flush=True)
    print_on_local_main(
        json.dumps(
            {
                "train_data_path": train_data_paths,
                "output_dir": output_dir,
                "target_layer_ids": target_layer_ids,
                "min_loss_tokens": min_loss_tokens,
                "max_shard_bytes": int(cli_args.max_shard_bytes),
                "local_batch_size": int(cli_args.local_batch_size),
                "num_workers": int(cli_args.num_workers),
            },
            indent=4,
        ),
        flush=True,
    )
    if global_rank == 0:
        prepare_target_cache_output_dir(output_dir)
    dist.barrier()

    rank_dir = os.path.join(output_dir, "_tmp", f"rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)

    with main_process_first():
        dataset = JsonLineDataset(data_paths=train_data_paths)

    local_start, local_end = compute_local_sample_range(
        num_samples=len(dataset),
        rank=global_rank,
        world_size=world_size,
    )
    local_total_samples = local_end - local_start

    local_subset = Subset(dataset, range(local_start, local_end))

    # Resolve paths: --model-config-path overrides where config/tokenizer are
    # loaded from (useful when config.json has been pre-converted).
    _weight_path = config.model.target_model_name_or_path
    _config_path = cli_args.model_config_path or _weight_path

    tokenizer = AutoTokenizer.from_pretrained(
        _config_path,
        trust_remote_code=True,
    )

    # Resolve model config (lightweight — no weights downloaded).
    _target_cfg = AutoConfig.from_pretrained(
        _config_path,
        trust_remote_code=True,
    )
    _model_type = str(_target_cfg.model_type)

    # DeepSeek-V4 and other MLA / custom-attention models require eager
    # attention; SDPA does not support their attention patterns.
    _attn = "eager" if _model_type in ("deepseek_v4",) else "sdpa"

    # Resolve device_map strategy and load directly onto NPU / GPU.
    _device_map = _resolve_device_map(
        model_type=_model_type,
        cli_device_map=cli_args.device_map,
        device=device,
    )
    if _device_map == "auto" and world_size > 1:
        print_on_local_main(
            "WARNING: device_map='auto' shards one model across all visible "
            "devices.  Running more than one process per node "
            f"(current world_size={world_size}) will cause each process to "
            "compete for the same devices, likely OOMing.  "
            "Re-launch with --nproc-per-node=1 to use layer-wise sharding, "
            "or use --device-map single for data-parallel mode."
        )
    target_model = _load_target_model(
        model_name_or_path=_weight_path,
        dtype=torch.bfloat16,
        attn_impl=_attn,
        device_map=_device_map,
    )
    target_hidden_size = _get_target_hidden_size(target_model)
    train_collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template=config.data.chat_template,
        max_length=config.data.max_length,
        min_loss_tokens=min_loss_tokens,
    )
    dataloader = DataLoader(
        local_subset,
        batch_size=int(cli_args.local_batch_size),
        collate_fn=train_collator,
        num_workers=int(cli_args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(cli_args.max_shard_bytes),
        max_queue_size=int(cli_args.local_batch_size) * 4,
    )

    processed_local_samples = 0
    last_progress_printed = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                processed_local_samples = min(
                    (batch_idx + 1) * int(cli_args.local_batch_size),
                    local_total_samples,
                )
                should_print_progress = (
                    processed_local_samples - last_progress_printed >= 100
                    or processed_local_samples == local_total_samples
                )
                if batch is None:
                    if should_print_progress:
                        _print_prepare_progress(
                            global_rank=global_rank,
                            processed_samples=processed_local_samples,
                            total_samples=local_total_samples,
                        )
                        last_progress_printed = processed_local_samples
                    continue
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                target_result = run_target_forward_with_hooks(
                    target_model=target_model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target_layer_ids=target_layer_ids,
                )
                seq_lens = batch["attention_mask"].sum(dim=1).tolist()
                for sample_idx_in_batch, seq_len in enumerate(seq_lens):
                    seq_len = int(seq_len)
                    writer.write_sample(
                        input_ids=batch["input_ids"][sample_idx_in_batch, :seq_len],
                        attention_mask=batch["attention_mask"][
                            sample_idx_in_batch, :seq_len
                        ],
                        loss_mask=batch["loss_mask"][sample_idx_in_batch, :seq_len],
                        target_hidden_states=target_result.target_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                        target_last_hidden_states=target_result.target_last_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                    )
                if should_print_progress:
                    _print_prepare_progress(
                        global_rank=global_rank,
                        processed_samples=processed_local_samples,
                        total_samples=local_total_samples,
                    )
                    last_progress_printed = processed_local_samples
    finally:
        writer.close()
    del target_model
    _empty_device_cache(device)
    dataset.close()
    summary = LocalCacheWriteSummary(
        global_rank=global_rank,
        source_sample_start=local_start,
        source_sample_end=local_end,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))
    dist.barrier()

    shard_map = None
    summaries = None
    if is_global_main_process():
        summaries = [
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )
            for rank in range(world_size)
        ]
        shard_map, shards = build_global_target_cache_shard_map(summaries)
    broadcast_payload = [shard_map]
    dist.broadcast_object_list(broadcast_payload, src=0)
    shard_map = broadcast_payload[0]
    local_summary = load_local_cache_write_summary(rank_dir)
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=local_summary,
        shard_map=shard_map,
    )
    dist.barrier()

    if is_global_main_process():
        assert summaries is not None
        num_valid_samples = finalize_target_cache_index(
            output_dir=output_dir,
            summaries=summaries,
            shard_map=shard_map,
        )
        _write_manifest(
            output_dir=output_dir,
            config=config,
            train_data_paths=train_data_paths,
            target_layer_ids=target_layer_ids,
            hidden_size=target_hidden_size,
            min_loss_tokens=min_loss_tokens,
            shards=shards,
        )
        cleanup_target_cache_tmp_dir(output_dir)
        print_on_global_main(
            f"Prepared target cache at {output_dir} with "
            f"{num_valid_samples}/{len(dataset)} valid samples."
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print(f"git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    main()
