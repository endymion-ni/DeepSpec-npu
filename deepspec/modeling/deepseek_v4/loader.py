"""Load DeepSeek-V4 ``inference/model.py`` Transformer with BF16 weights.

Reference: ``cann-recipes-infer/models/deepseek-v4/utils/convert_model.py``

Strategy (mirrors cann-recipes-infer):
1. Single-pass read of all safetensors into memory.
2. FP8 / FP4 → BF16 dequantisation using per-block ``weight * scale``.
3. Minimal name mapping: checkpoint uses ``attn.wq_a`` / ``ffn.experts.{e}.w1``
   while ``inference/model.py`` uses ``attention.wq_a`` / ``feed_forward.experts.*.gate_up_proj``.
4. ``sys.modules["kernel"]`` is patched with CPU fallbacks so the model runs
   without tilelang CUDA kernels.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FP8_BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Weight dequantisation  (adapted from cann-recipes-infer convert_model.py)
# ---------------------------------------------------------------------------

def weight_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantise FP8-e4m3 weight with per-block E8M0 scale → BF16.

    Mirrors ``cann-recipes-infer/utils/convert_model.py:weight_dequant``.

    Args:
        weight: (M, N)  float8_e4m3fn
        scale:  (ceil(M/block), ceil(N/block))  float8_e8m0fnu (power-of-2)
        block_size: typically 128

    Returns:
        (M, N)  bfloat16
    """
    M, N = weight.shape
    w = weight.float()
    s = scale.float()

    scale_m, scale_n = s.shape
    assert scale_m == (M + block_size - 1) // block_size, (
        f"scale rows mismatch: {scale_m} vs {(M + block_size - 1) // block_size}"
    )
    assert scale_n == (N + block_size - 1) // block_size, (
        f"scale cols mismatch: {scale_n} vs {(N + block_size - 1) // block_size}"
    )

    # Expand scale to full weight shape
    s_expanded = s.repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=1
    )[:M, :N]

    return (w * s_expanded).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Iterate safetensors
# ---------------------------------------------------------------------------

def _iter_safetensors(model_dir: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (key, tensor) from all ``*.safetensors`` in *model_dir*."""
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


# ---------------------------------------------------------------------------
# Name mapping: checkpoint native → inference/model.py attribute path
# ---------------------------------------------------------------------------

def _map_ckpt_to_model(ckpt_key: str) -> str | None:
    """Map a checkpoint tensor key to ``Transformer`` state_dict key.

    Checkpoint naming (native DeepSeek)::

        embed.weight
        layers.{i}.attn.wq_a.weight     / .scale
        layers.{i}.attn.wo_b.weight      / .scale
        layers.{i}.attn.wkv.weight       / .scale
        layers.{i}.attn.attn_sink
        layers.{i}.attn.q_norm.weight
        layers.{i}.attn.kv_norm.weight
        layers.{i}.attn.compressor.*
        layers.{i}.attn.indexer.*
        layers.{i}.ffn.gate.weight
        layers.{i}.ffn.experts.{e}.w{1,2,3}.weight  / .scale
        layers.{i}.ffn.shared_experts.w{1,2,3}.weight / .scale
        layers.{i}.attn_norm.weight
        layers.{i}.ffn_norm.weight
        layers.{i}.hc_attn_{fn,base,scale}
        layers.{i}.hc_ffn_{fn,base,scale}
        norm.weight
        head.weight
        hc_head_{fn,base,scale}

    ``inference/model.py`` naming::

        layers.{i}.attention.*   (instead of layers.{i}.attn.*)
        layers.{i}.feed_forward.* (instead of layers.{i}.ffn.*)
        layers.{i}.attn_norm.*   (same)
        layers.{i}.ffn_norm.*    (same)
        layers.{i}.attn_hc.*     (hc_attn_* → attn_hc.*)
        layers.{i}.ffn_hc.*      (hc_ffn_* → ffn_hc.*)
        embed.*  (same)
        head.*   (same)
        norm.*   (same)
        hc_head_* (same)
    """
    if ckpt_key.startswith("mtp."):
        return None  # skip MTP

    # --- Top-level ---
    if not ckpt_key.startswith("layers."):
        # embed, head, norm, hc_head_*
        return ckpt_key  # identities

    # --- Layer keys ---
    parts = ckpt_key.split(".")
    layer_idx = parts[1]  # keep as string
    rest = ".".join(parts[2:])

    # attn → attention
    if rest.startswith("attn."):
        rest = "attention." + rest[5:]
    # ffn → feed_forward
    elif rest.startswith("ffn."):
        rest = "feed_forward." + rest[4:]
    # hc_attn_* → attn_hc.*
    elif rest.startswith("hc_attn_"):
        param = rest[8:]  # fn / base / scale
        rest = f"attn_hc.{param}"
    # hc_ffn_* → ffn_hc.*
    elif rest.startswith("hc_ffn_"):
        param = rest[7:]  # fn / base / scale
        rest = f"ffn_hc.{param}"

    return f"layers.{layer_idx}.{rest}"


# ---------------------------------------------------------------------------
# Expert w1/w3 → gate_up_proj merging
# ---------------------------------------------------------------------------

def _merge_expert_weights(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merge per-expert ``w1`` + ``w3`` → ``gate_up_proj`` for all layers.

    The checkpoint stores expert weights as separate tensors::

        layers.{i}.ffn.experts.{e}.w1.weight  (intermediate, hidden)
        layers.{i}.ffn.experts.{e}.w3.weight  (intermediate, hidden)

    ``inference/model.py`` expects a fused stacked tensor::

        layers.{i}.feed_forward.experts.gate_up_proj:
            (n_experts, 2 * intermediate, hidden)

    Shared experts are merged similarly.
    """
    # Collect per-layer expert w1/w3
    expert_w1: dict[int, dict[int, torch.Tensor]] = {}
    expert_w3: dict[int, dict[int, torch.Tensor]] = {}
    shared_w1: dict[int, torch.Tensor] = {}
    shared_w3: dict[int, torch.Tensor] = {}
    n_experts: dict[int, int] = {}

    merged = {}

    for ckpt_key, tensor in raw.items():
        # Routed experts
        if ".ffn.experts." in ckpt_key and (
            ckpt_key.endswith(".w1.weight") or ckpt_key.endswith(".w3.weight")
        ):
            parts = ckpt_key.split(".")
            layer_idx = int(parts[1])
            exp_idx = int(parts[4])
            is_w3 = parts[5] == "w3"
            n_experts[layer_idx] = max(n_experts.get(layer_idx, 0), exp_idx + 1)
            t = _dequant_if_needed(ckpt_key, tensor, raw)
            if is_w3:
                expert_w3.setdefault(layer_idx, {})[exp_idx] = t
            else:
                expert_w1.setdefault(layer_idx, {})[exp_idx] = t
            continue

        # Shared experts
        if ".ffn.shared_experts." in ckpt_key and (
            ckpt_key.endswith(".w1.weight") or ckpt_key.endswith(".w3.weight")
        ):
            parts = ckpt_key.split(".")
            layer_idx = int(parts[1])
            is_w3 = parts[4] == "w3"
            t = _dequant_if_needed(ckpt_key, tensor, raw)
            if is_w3:
                shared_w3[layer_idx] = t
            else:
                shared_w1[layer_idx] = t
            continue

    # Merge routed experts
    for layer_idx in sorted(expert_w1.keys()):
        n_exp = n_experts[layer_idx]
        gate_up = torch.stack([
            torch.cat([expert_w1[layer_idx][e], expert_w3[layer_idx][e]], dim=0)
            for e in range(n_exp)
        ])
        merged[f"layers.{layer_idx}.feed_forward.experts.gate_up_proj"] = gate_up

    # Merge shared experts
    for layer_idx in sorted(shared_w1.keys()):
        gate_up = torch.cat([shared_w1[layer_idx], shared_w3[layer_idx]], dim=0)
        merged[f"layers.{layer_idx}.feed_forward.shared_experts.gate_up_proj"] = gate_up

    return merged


def _dequant_if_needed(
    ckpt_key: str, tensor: torch.Tensor, raw: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Dequantise FP8 → BF16 if the tensor is quantised."""
    if tensor.dtype == torch.float8_e4m3fn:
        scale_key = ckpt_key.replace(".weight", ".scale")
        scale = raw.get(scale_key)
        if scale is not None:
            return weight_dequant(tensor, scale)
    return tensor.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Main weight loading
# ---------------------------------------------------------------------------

def _load_weights(model: torch.nn.Module, model_dir: str) -> None:
    """Dequantise and load all weights from *model_dir* into *model*.

    Single-pass: read all shards once → process → ``load_state_dict``.
    """
    # 1. Read everything into memory
    raw: dict[str, torch.Tensor] = {}
    for ckpt_key, tensor in _iter_safetensors(model_dir):
        raw[ckpt_key] = tensor

    # 2. Merge expert weights (consumes expert w1/w3 keys)
    state_dict = _merge_expert_weights(raw)

    # 3. Process remaining keys
    for ckpt_key, tensor in raw.items():
        if ckpt_key.startswith("mtp."):
            continue
        if ckpt_key.endswith(".scale"):
            continue  # handled with weight

        model_key = _map_ckpt_to_model(ckpt_key)
        if model_key is None:
            continue

        # Skip expert w1/w3 — already merged
        if (
            ".ffn.experts." in ckpt_key
            or ".ffn.shared_experts." in ckpt_key
        ) and (ckpt_key.endswith(".w1.weight") or ckpt_key.endswith(".w3.weight")):
            continue
        # Map w2.weight to down_proj.weight
        if ".ffn.experts." in ckpt_key and ckpt_key.endswith(".w2.weight"):
            model_key = model_key.replace("feed_forward.experts.w2.weight",
                                          "feed_forward.experts.down_proj.weight")
        if ".ffn.shared_experts." in ckpt_key and ckpt_key.endswith(".w2.weight"):
            model_key = model_key.replace("feed_forward.shared_experts.w2.weight",
                                          "feed_forward.shared_experts.down_proj.weight")

        state_dict[model_key] = _dequant_if_needed(ckpt_key, tensor, raw)

    # 4. Load
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        n = min(5, len(missing))
        print(f"[loader] {len(missing)} missing keys (first {n}: {missing[:n]})")
    if unexpected:
        n = min(5, len(unexpected))
        print(f"[loader] {len(unexpected)} unexpected keys (first {n}: {unexpected[:n]})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class LoadedDeepSeekV4Model:
    model: torch.nn.Module
    config: dict
    model_args: object
    hc_mult: int
    n_layers: int
    hidden_size: int


def load_model(
    model_dir: str,
    *,
    device: torch.device | None = None,
) -> LoadedDeepSeekV4Model:
    """Load DeepSeek-V4 ``Transformer`` in BF16 inference mode.

    1. Patch ``sys.modules["kernel"]`` → CPU fallback.
    2. Parse ``ModelArgs`` from ``config.json``.
    3. Instantiate ``Transformer``.
    4. Dequantise and load all weights.
    5. Move to *device* (CPU if ``None``).
    """
    from deepspec.modeling.deepseek_v4 import kernel_cpu
    sys.modules["kernel"] = kernel_cpu

    # Add inference dir to path
    inference_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "models", "deepseek_v4_flash_hf_config", "inference")
    )
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    import model as ds_model

    # Config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(inference_dir), "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # ModelArgs (mirrors cann-recipes-infer parse)
    model_args = _build_model_args(config)

    print(f"[loader] Transformer(n_layers={model_args.n_layers}, "
          f"vocab={model_args.vocab_size}, dim={model_args.dim})")
    transformer = ds_model.Transformer(model_args)

    print(f"[loader] Loading + dequantising weights from {model_dir}")
    _load_weights(transformer, model_dir)

    if device is not None:
        transformer = transformer.to(device)

    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    return LoadedDeepSeekV4Model(
        model=transformer, config=config, model_args=model_args,
        hc_mult=model_args.hc_mult, n_layers=model_args.n_layers,
        hidden_size=model_args.dim,
    )


def _build_model_args(config: dict):
    """Build ``ModelArgs`` from config.json."""
    from models.deepseek_v4_flash_hf_config.inference.model import ModelArgs

    # Direct 1:1 mappings
    direct = {
        "vocab_size": "vocab_size",
        "dim": "dim",
        "n_layers": "n_layers",
        "n_routed_experts": "n_routed_experts",
        "n_shared_experts": "n_shared_experts",
        "n_activated_experts": "num_experts_per_tok",
        "score_func": "scoring_func",
        "route_scale": "routed_scaling_factor",
        "q_lora_rank": "q_lora_rank",
        "head_dim": "head_dim",
        "rope_head_dim": "qk_rope_head_dim",
        "norm_eps": "rms_norm_eps",
        "o_groups": "o_groups",
        "o_lora_rank": "o_lora_rank",
        "window_size": "sliding_window",
        "compress_rope_theta": "compress_rope_theta",
        "rope_theta": "rope_theta",
        "index_n_heads": "index_n_heads",
        "index_head_dim": "index_head_dim",
        "index_topk": "index_topk",
        "hc_mult": "hc_mult",
        "hc_sinkhorn_iters": "hc_sinkhorn_iters",
        "hc_eps": "hc_eps",
        "n_hash_layers": "n_hash_layers",
        "n_mtp_layers": "num_nextn_predict_layers",
    }
    kwargs = {}
    for ma, ck in direct.items():
        if ck in config:
            kwargs[ma] = config[ck]

    # Nested: rope_factor / beta_fast / beta_slow
    rp = config.get("rope_parameters", {}).get("main", {})
    for k in ("factor", "beta_fast", "beta_slow"):
        if k in rp:
            kwargs[f"rope_{k}"] = rp[k]

    kwargs.setdefault("max_seq_len", config.get("max_position_embeddings", 4096))
    kwargs.setdefault("dtype", "bf16")  # force BF16 path

    return ModelArgs(**kwargs)


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedHiddenStates:
    target_hidden_states: torch.Tensor       # (seq, n_layers * hidden)
    target_last_hidden_states: torch.Tensor  # (seq, hidden)


def extract_hidden_states(
    model_loaded: LoadedDeepSeekV4Model,
    input_ids: torch.Tensor,
    target_layer_ids: list[int],
) -> ExtractedHiddenStates:
    """Forward pass with hooks at *target_layer_ids*.

    Hooked layer output ``(N, hc_mult, hidden)`` is mean-folded → ``(N, hidden)``.
    The hc_head collapsed state (pre-norm) is the "last" hidden state.
    """
    transformer = model_loaded.model
    device = input_ids.device
    captured = {}
    handles = []

    def make_hook(lid: int):
        def hook(_m, _i, output):
            captured[lid] = output.mean(dim=1).detach()
        return hook

    for lid in target_layer_ids:
        handles.append(transformer.layers[lid].register_forward_hook(make_hook(lid)))

    try:
        with torch.no_grad():
            h = transformer.embed(input_ids)
            h = h.unsqueeze(2).repeat(1, 1, transformer.hc_mult, 1)
            for layer in transformer.layers:
                h = layer(h, 0, input_ids)

            # hc_head collapse (before final norm)
            fn = transformer.hc_head_fn.float()
            hc = (fn @ h.float().transpose(-1, -2)).transpose(-1, -2)
            hc = hc.squeeze(2) * transformer.hc_head_scale + transformer.hc_head_base
            last_hidden = hc.to(h.dtype)
    finally:
        for hdl in handles:
            hdl.remove()

    target_hidden = torch.cat(
        [captured[lid].to(device) for lid in target_layer_ids], dim=-1
    )
    return ExtractedHiddenStates(
        target_hidden_states=target_hidden.squeeze(0),
        target_last_hidden_states=last_hidden.squeeze(0),
    )
