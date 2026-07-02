#!/usr/bin/env python3
"""Minimal weight loader: extract embed_tokens and lm_head from DeepSeek-V4
safetensors shards without loading the full 275 GB model.

Usage::

    python scripts/data/init_draft_weights.py \\
        --draft-config config/dspark/dspark_deepseek_v4_flash.py \\
        --output /workspace/ds_draft_init_weights
"""

import argparse
import json
import os
import sys
import struct

import torch
from safetensors import safe_open

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepspec.modeling.dspark.deepseek_v4.config import build_draft_config
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.utils.config import ConfigNode, load_config
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and save minimal draft init weights from DeepSeek-V4"
    )
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--output", required=True,
                        help="Output directory for the init weights checkpoint")
    parser.add_argument("--target-model-dir",
                        default="/workspace/deepseek-v4-flash",
                        help="Directory containing DeepSeek-V4 safetensors shards")
    parser.add_argument("--target-model-id",
                        default="deepseek-ai/DeepSeek-V4-Flash",
                        help="HuggingFace model ID (for config loading)")
    parser.add_argument("--allow-missing-head", action="store_true",
                        help="If head.weight is missing, use random init for lm_head")
    return parser.parse_args()


def _find_safetensors_files(model_dir: str) -> dict:
    """Scan model_dir for safetensors files and return {weight_name: file_path}."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        # Map weight name → absolute path
        return {
            name: os.path.join(model_dir, shard)
            for name, shard in weight_map.items()
        }

    # No index: scan for .safetensors files
    import glob
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    result = {}
    for fpath in files:
        with safe_open(fpath, framework="pt") as sf:
            for key in sf.keys():
                result[key] = fpath
    return result


def _load_weight(weight_map: dict, weight_name: str) -> torch.Tensor | None:
    """Load a single weight tensor from safetensors shards."""
    if weight_name not in weight_map:
        return None
    shard_path = weight_map[weight_name]
    with safe_open(shard_path, framework="pt") as sf:
        return sf.get_tensor(weight_name)


def main():
    args = parse_args()

    # Resolve weight locations
    weight_map = _find_safetensors_files(args.target_model_dir)
    print(f"Found {len(weight_map)} weight entries across safetensors files")

    # ---- Load target weights ----
    # DeepSeek-V4 uses: embed.weight (token embedding), head.weight (lm_head)
    embed_weight = _load_weight(weight_map, "embed.weight")
    head_weight = _load_weight(weight_map, "head.weight")

    if embed_weight is not None:
        print(f"embed.weight: shape={tuple(embed_weight.shape)}, dtype={embed_weight.dtype}")
    else:
        print("ERROR: embed.weight not found!")
        sys.exit(1)

    if head_weight is not None:
        print(f"head.weight:  shape={tuple(head_weight.shape)}, dtype={head_weight.dtype}")
    elif args.allow_missing_head:
        print("head.weight not found — will use random init for lm_head")
    else:
        print("ERROR: head.weight not found! Download model-00045-of-00046.safetensors")
        print("  or pass --allow-missing-head to use random init")
        sys.exit(1)

    # ---- Build draft config & model ----
    target_config = AutoConfig.from_pretrained(
        args.target_model_id, trust_remote_code=True
    )

    # Parse model_args from the training config
    config = load_config(args.draft_config)
    model_args = config.model

    draft_config = build_draft_config(
        target_config=target_config,
        model_args=model_args,
    )
    draft_model = Qwen3DSparkModel(draft_config)
    print(f"Draft model: {sum(p.numel() for p in draft_model.parameters()):,} params total")

    # ---- Copy weights ----
    # draft_model.embed_tokens.weight shape: (129280, 4096)
    # embed.weight shape: (129280, 4096)
    assert embed_weight.shape == draft_model.embed_tokens.weight.shape, (
        f"Shape mismatch: embed.weight {tuple(embed_weight.shape)} vs "
        f"embed_tokens {tuple(draft_model.embed_tokens.weight.shape)}"
    )
    draft_model.embed_tokens.weight.data.copy_(embed_weight)
    print("✓ embed_tokens initialized from embed.weight")

    # draft_model.lm_head.weight shape: (129280, 4096)
    if head_weight is not None:
        # head.weight might be (vocab_size, hidden_size) = (129280, 4096)
        # lm_head.weight should be the same
        assert head_weight.shape == draft_model.lm_head.weight.shape, (
            f"Shape mismatch: head.weight {tuple(head_weight.shape)} vs "
            f"lm_head {tuple(draft_model.lm_head.weight.shape)}"
        )
        draft_model.lm_head.weight.data.copy_(head_weight)
        print("✓ lm_head initialized from head.weight")
    else:
        print("⚠ lm_head uses random init (head.weight not available)")

    # ---- Save ----
    os.makedirs(args.output, exist_ok=True)
    draft_model.save_pretrained(args.output)
    # Also save target config and tokenizer for downstream use
    target_config.save_pretrained(args.output)
    print(f"\nSaved draft init weights → {args.output}")
    print(f"Use with: --opts 'model.target_model_name_or_path={args.output}'")


if __name__ == "__main__":
    main()
