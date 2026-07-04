#!/usr/bin/env python3
"""Minimal single-NPU training smoke test for DSpark DeepSeek-V4.

Bypasses the full 275 GB target model download — only loads config and
tokenizer from the local HF config directory, then runs one forward +
backward pass on NPU to verify operator compatibility.
"""

import os
import sys

import torch
import torch.distributed as dist

# Resolve project root.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepspec.data.target_cache_dataset import CacheDataset
from deepspec.modeling.dspark.deepseek_v4.modeling import DeepSeekV4DSparkModel
from deepspec.modeling.dspark.deepseek_v4.config import build_draft_config, TRAIN_ATTN_IMPLEMENTATION
from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.utils.device import is_npu_available, device_type
from transformers import AutoConfig, AutoTokenizer


def main():
    # ---- Device & distributed setup ----
    if is_npu_available():
        device = torch.device("npu", 0)
        torch.npu.set_device(0)
        backend = "hccl"
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(0)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    # Loss computation uses dist.get_world_size() — init even for 1 device.
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=0,
            world_size=1,
        )

    dtype = torch.bfloat16
    device_str = device_type()
    print(f"Device: {device}  dtype: {dtype}")
    print(f"Attention implementation: {TRAIN_ATTN_IMPLEMENTATION}")

    # ---- Load config & tokenizer from local HF config dir ----
    hf_config_dir = os.path.join(
        _PROJECT_ROOT, "models", "deepseek_v4_flash_hf_config"
    )
    target_config = AutoConfig.from_pretrained(hf_config_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_config_dir, trust_remote_code=True)
    print(f"Target config loaded: model_type={target_config.model_type}, "
          f"hidden_size={target_config.hidden_size}, "
          f"vocab_size={target_config.vocab_size}")

    # ---- Build draft config & model (same as DeepSeekV4DSparkTrainer) ----
    # Mimic model_args from config/dspark/dspark_deepseek_v4_flash.py
    from deepspec.utils.config import ConfigNode
    model_args = ConfigNode(
        block_size=5,
        num_draft_layers=3,
        target_layer_ids=[40, 41, 42],
        mask_token_id=128799,
        num_anchors=512,
        markov_rank=256,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )

    draft_config = build_draft_config(
        target_config=target_config,
        model_args=model_args,
    )
    print(f"Draft config: num_hidden_layers={draft_config.num_hidden_layers}, "
          f"hidden_size={draft_config.hidden_size}, "
          f"vocab_size={draft_config.vocab_size}, "
          f"attn={draft_config._attn_implementation}")

    draft_model = DeepSeekV4DSparkModel(draft_config)
    draft_model = draft_model.to(device=device, dtype=dtype).train()

    # Count params
    total = sum(p.numel() for p in draft_model.parameters())
    trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  Trainable: {trainable:,}")

    # ---- Load target cache ----
    cache_dir = "/workspace/ds_target_cache"
    ds = CacheDataset(cache_dir)
    print(f"Cache: {len(ds)} samples, "
          f"target_layer_ids={ds.target_layer_ids}, "
          f"hidden_size={ds.hidden_size}")

    # ---- Get a batch ----
    sample = ds[0]
    batch = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "loss_mask": sample["loss_mask"].unsqueeze(0).to(device),
        "target_hidden_states": sample["target_hidden_states"].unsqueeze(0).to(device),
        "target_last_hidden_states": sample["target_last_hidden_states"].unsqueeze(0).to(device),
    }
    print(f"Batch: input_ids={batch['input_ids'].shape}, "
          f"target_hidden_states={batch['target_hidden_states'].shape}")

    # ---- Loss config (from dspark_deepseek_v4_flash.py) ----
    loss_config = {
        "ce_loss_alpha": 0.1,
        "l1_loss_alpha": 0.9,
        "confidence_head_alpha": 1.0,
        "loss_decay_gamma": 4.0,
    }

    # ---- Forward pass ----
    print("\n=== Running forward pass ===")
    output: DSparkForwardOutput = draft_model(
        input_ids=batch["input_ids"],
        target_hidden_states=batch["target_hidden_states"],
        loss_mask=batch["loss_mask"],
        target_last_hidden_states=batch["target_last_hidden_states"],
    )
    print(f"  draft_logits:     {output.draft_logits.shape}")
    print(f"  target_ids:       {output.target_ids.shape}")
    if output.confidence_pred is not None:
        print(f"  confidence_pred:  {output.confidence_pred.shape}")

    # ---- Loss ----
    loss = compute_dspark_loss(outputs=output, **loss_config)
    print(f"  Loss: {loss.item():.6f}")

    # ---- Backward pass ----
    print("\n=== Running backward pass ===")
    loss.backward()
    print("  Backward pass completed successfully!")

    # ---- Check gradients ----
    grad_norm = 0.0
    for name, p in draft_model.named_parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"  Gradient norm: {grad_norm:.4f}")

    # ---- Verify all grads are on NPU ----
    for name, p in draft_model.named_parameters():
        if p.grad is not None:
            assert p.grad.device.type == device_str, (
                f"Gradient of {name} is on {p.grad.device}, expected {device_str}"
            )

    print(f"\n=== All checks passed on {device_str.upper()}! ===")

    ds.close()


if __name__ == "__main__":
    main()
