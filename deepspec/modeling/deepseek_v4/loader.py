"""Load DeepSeek-V4 Transformer from ``inference/model.py`` with BF16 weights.

Patches ``sys.modules["kernel"]`` with a CPU fallback so that the model can
run without the tilelang CUDA kernels.  FP8 / FP4 weights are dequantised
to BF16 during loading, which means the ``linear()`` dispatcher inside
``inference/model.py`` always takes the plain ``F.linear`` path.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Constants (must match inference/model.py)
# ---------------------------------------------------------------------------
FP8_BLOCK_SIZE = 128
FP4_BLOCK_SIZE = 32

# FP4 lookup table (from inference/convert.py)
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Weight dequantisation helpers
# ---------------------------------------------------------------------------

def _dequant_fp8_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantise an FP8-e4m3 weight with per-128-block E8M0 scale → BF16.

    *weight*  : (out_features, in_features)   float8_e4m3fn
    *scale*   : (ceil(out/128), ceil(in/128)) float8_e8m0fnu  (power-of-2)
    """
    out_dim, in_dim = weight.shape
    b_out = (out_dim + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE
    b_in = (in_dim + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE

    # Pad scale to full block grid if needed
    s = scale.float()[:b_out, :b_in]

    # Expand scale to (out_dim, in_dim) via nearest-neighbour
    s = s.repeat_interleave(FP8_BLOCK_SIZE, dim=0)[:out_dim]
    s = s.repeat_interleave(FP8_BLOCK_SIZE, dim=1)[:in_dim]

    return (weight.float() * s).to(torch.bfloat16)


def _dequant_fp4_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantise an FP4-e2m1 weight with per-32-block E8M0 scale → BF16.

    *weight*  : (out_features, in_features//2)  float4_e2m1fn_x2  (packed)
    *scale*   : (out_features, ceil(in/32))     float8_e8m0fnu
    """
    out_dim = weight.shape[0]
    logical_in_dim = weight.shape[1] * 2  # packed: 2 FP4 values per byte

    # Unpack FP4: each uint8 contains two 4-bit values
    w_u8 = weight.view(torch.uint8)
    low = w_u8 & 0x0F
    high = (w_u8 >> 4) & 0x0F
    w_decomp = torch.stack(
        [FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1
    ).flatten(1)  # (out_dim, logical_in_dim)

    # Expand scale
    s = scale.float()
    s = s.repeat_interleave(FP4_BLOCK_SIZE, dim=1)[:, :logical_in_dim]

    return (w_decomp * s).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Name mapping: checkpoint key → model state_dict key
# ---------------------------------------------------------------------------

def _map_ckpt_key_to_model(ckpt_key: str) -> str | None:
    """Convert a checkpoint tensor name to a ``Transformer`` state_dict key.

    Returns ``None`` for keys that should be skipped (e.g. MTP weights).
    """
    # Skip MTP layers entirely
    if ckpt_key.startswith("mtp."):
        return None

    key = ckpt_key

    # Embed / head / norm (top-level)
    if key == "embed.weight":
        return "embed.weight"
    if key == "head.weight":
        return "head.weight"
    if key == "norm.weight":
        return "norm.weight"

    # Hyper-connection head parameters
    if key == "hc_head_fn":
        return "hc_head_fn"
    if key == "hc_head_base":
        return "hc_head_base"
    if key == "hc_head_scale":
        return "hc_head_scale"

    # Layer-specific keys: layers.{i}.{rest}
    if key.startswith("layers."):
        parts = key.split(".")
        layer_idx = int(parts[1])
        rest = ".".join(parts[2:])  # e.g. "attn.wq_a.weight"

        # --- Attention ---
        if rest.startswith("attn."):
            sub = rest[5:]  # strip "attn."
            # Linear weights: wq_a, wq_b, wo_a, wo_b, wkv
            for lin_name in ("wq_a", "wq_b", "wo_a", "wo_b", "wkv"):
                if sub == f"{lin_name}.weight":
                    return f"layers.{layer_idx}.attention.{lin_name}.weight"
                if sub == f"{lin_name}.scale":
                    return f"layers.{layer_idx}.attention.{lin_name}.scale"
            # Norms
            if sub == "q_norm.weight":
                return f"layers.{layer_idx}.attention.q_norm.weight"
            if sub == "kv_norm.weight":
                return f"layers.{layer_idx}.attention.kv_norm.weight"
            # attn_sink
            if sub == "attn_sink":
                return f"layers.{layer_idx}.attention.attn_sink"
            # Compressor / indexer weights (sparse layers)
            if sub.startswith("compressor."):
                return f"layers.{layer_idx}.attention.{sub}"
            if sub.startswith("indexer."):
                return f"layers.{layer_idx}.attention.{sub}"
            # CSA / HCA compressor weights
            if "weights_proj" in sub:
                return f"layers.{layer_idx}.attention.{sub}"
            # Unknown attention key — warn and skip
            print(f"[loader] WARNING: unknown attention key: {ckpt_key}")
            return None

        # --- FFN ---
        if rest.startswith("ffn."):
            sub = rest[4:]  # strip "ffn."

            # Router gate
            if sub == "gate.weight":
                return f"layers.{layer_idx}.feed_forward.gate.weight"

            # Shared experts (DeepseekV4MLP)
            if sub.startswith("shared_experts."):
                se_rest = sub[16:]  # strip "shared_experts."
                # w1 + w3 → gate_up_proj (need special handling)
                if se_rest in ("w1.weight", "w3.weight"):
                    return None  # handled by _merge_gate_up
                if se_rest in ("w1.scale", "w3.scale"):
                    return None  # handled by _merge_gate_up
                if se_rest == "w2.weight":
                    return f"layers.{layer_idx}.feed_forward.shared_experts.down_proj.weight"
                if se_rest == "w2.scale":
                    return f"layers.{layer_idx}.feed_forward.shared_experts.down_proj.scale"
                return None

            # Routed experts
            if sub.startswith("experts."):
                exp_parts = sub.split(".")
                exp_idx = int(exp_parts[1])
                exp_rest = ".".join(exp_parts[2:])  # w1.weight / w1.scale / etc.

                # w1 + w3 → gate_up_proj (stacked across experts)
                if exp_rest in ("w1.weight", "w3.weight"):
                    return None  # handled by _merge_gate_up
                if exp_rest in ("w1.scale", "w3.scale"):
                    return None  # handled by _merge_gate_up
                if exp_rest == "w2.weight":
                    return f"layers.{layer_idx}.feed_forward.experts.down_proj.weight"
                if exp_rest == "w2.scale":
                    return f"layers.{layer_idx}.feed_forward.experts.down_proj.scale"
                return None

            return None

        # --- Hyper-connections ---
        if rest in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale"):
            param = rest[3:]  # "attn_fn" → "fn", "attn_base" → "base", "attn_scale" → "scale"
            hc_attr = {"attn_fn": "fn", "attn_base": "base", "attn_scale": "scale"}[param]
            return f"layers.{layer_idx}.attn_hc.{hc_attr}"
        if rest in ("hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            param = {"hc_ffn_fn": "fn", "hc_ffn_base": "base", "hc_ffn_scale": "scale"}[rest]
            return f"layers.{layer_idx}.ffn_hc.{param}"

        # --- Norms ---
        if rest == "attn_norm.weight":
            return f"layers.{layer_idx}.attn_norm.weight"
        if rest == "ffn_norm.weight":
            return f"layers.{layer_idx}.ffn_norm.weight"

        # Unknown layer key
        print(f"[loader] WARNING: unknown layer key: {ckpt_key}")
        return None

    # Unknown top-level key
    print(f"[loader] WARNING: unknown key: {ckpt_key}")
    return None


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

@dataclass
class LoadedDeepSeekV4Model:
    """Container for a loaded Transformer and its metadata."""

    model: torch.nn.Module    # the Transformer instance
    config: dict              # raw config.json
    model_args: object        # ModelArgs dataclass
    hc_mult: int
    n_layers: int
    hidden_size: int


def _parse_model_args(config: dict) -> object:
    """Build a ``ModelArgs`` instance from the raw config dict."""
    from models.deepseek_v4_flash_hf_config.inference.model import ModelArgs

    # Map config.json keys to ModelArgs fields
    field_map = {
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
        "rope_factor": ("rope_parameters", "main", "factor"),
        "beta_fast": ("rope_parameters", "main", "beta_fast"),
        "beta_slow": ("rope_parameters", "main", "beta_slow"),
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
    for model_arg_name, cfg_key in field_map.items():
        if isinstance(cfg_key, tuple):
            # Nested access
            val = config
            for k in cfg_key:
                val = val.get(k, {}) if isinstance(val, dict) else getattr(val, k, None)
        else:
            val = config.get(cfg_key)
        if val is not None:
            kwargs[model_arg_name] = val

    # Hard-code inference mode
    kwargs.setdefault("max_seq_len", config.get("max_position_embeddings", 4096))
    kwargs.setdefault("dtype", "bf16")  # Force BF16 to avoid FP8 kernels

    return ModelArgs(**kwargs)


def _iter_safetensors(model_dir: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (key, tensor) pairs from all safetensors shards in *model_dir*."""
    import glob

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _load_weights_into_model(model: torch.nn.Module, model_dir: str) -> None:
    """Load and dequantise weights from *model_dir* into *model*.

    Single-pass over all safetensors shards:
    1. Collect every (ckpt_key, tensor) pair.
    2. Dequantise FP8 / FP4 → BF16, pairing weights with their scales.
    3. Merge per-expert w1 + w3 → stacked gate_up_proj.
    4. ``model.load_state_dict(...)``.
    """
    raw: dict[str, torch.Tensor] = {}
    for ckpt_key, tensor in _iter_safetensors(model_dir):
        raw[ckpt_key] = tensor

    state_dict: dict[str, torch.Tensor] = {}
    expert_w1: dict[int, dict[int, torch.Tensor]] = {}
    expert_w3: dict[int, dict[int, torch.Tensor]] = {}
    shared_w1: dict[int, torch.Tensor] = {}
    shared_w3: dict[int, torch.Tensor] = {}
    n_experts: dict[int, int] = {}

    for ckpt_key, tensor in raw.items():
        if ckpt_key.startswith("mtp."):
            continue  # skip MTP

        # ---- expert w1 / w3 (defer to merge pass) ----
        if ".ffn.experts." in ckpt_key and (".w1." in ckpt_key or ".w3." in ckpt_key):
            parts = ckpt_key.split(".")
            layer_idx = int(parts[1])
            exp_idx = int(parts[4])
            w_idx = parts[5]  # "w1" or "w3"
            n_experts[layer_idx] = max(n_experts.get(layer_idx, 0), exp_idx + 1)

            scale_key = ckpt_key.replace(".weight", ".scale")
            scale = raw.get(scale_key)
            t = _dequant_fp8_weight(tensor, scale) if scale is not None and tensor.dtype == torch.float8_e4m3fn else tensor.to(torch.bfloat16)

            if w_idx == "w1":
                expert_w1.setdefault(layer_idx, {})[exp_idx] = t
            else:
                expert_w3.setdefault(layer_idx, {})[exp_idx] = t
            continue

        if ".ffn.shared_experts." in ckpt_key and (".w1." in ckpt_key or ".w3." in ckpt_key):
            parts = ckpt_key.split(".")
            layer_idx = int(parts[1])
            w_idx = parts[4]
            scale_key = ckpt_key.replace(".weight", ".scale")
            scale = raw.get(scale_key)
            t = _dequant_fp8_weight(tensor, scale) if scale is not None and tensor.dtype == torch.float8_e4m3fn else tensor.to(torch.bfloat16)
            if w_idx == "w1":
                shared_w1[layer_idx] = t
            else:
                shared_w3[layer_idx] = t
            continue

        # ---- scale tensors (handled with weight) ----
        if tensor.dtype == torch.float8_e8m0fnu:
            continue

        # ---- map & dequantise ----
        model_key = _map_ckpt_key_to_model(ckpt_key)
        if model_key is None:
            continue

        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = ckpt_key.replace(".weight", ".scale")
            scale = raw.get(scale_key)
            state_dict[model_key] = _dequant_fp8_weight(tensor, scale) if scale is not None else tensor.to(torch.bfloat16)
        elif tensor.dtype == torch.float4_e2m1fn_x2:
            scale_key = ckpt_key.replace(".weight", ".scale")
            scale = raw.get(scale_key)
            state_dict[model_key] = _dequant_fp4_weight(tensor, scale) if scale is not None else tensor.to(torch.bfloat16)
        else:
            state_dict[model_key] = tensor.to(torch.bfloat16)

    # ---- Merge routed experts: w1 + w3 → gate_up_proj ----
    for layer_idx in sorted(expert_w1.keys()):
        n_exp = n_experts[layer_idx]
        w1_list = [expert_w1[layer_idx][e] for e in range(n_exp)]
        w3_list = [expert_w3[layer_idx][e] for e in range(n_exp)]
        gate_up = torch.stack([torch.cat([a, b], dim=0) for a, b in zip(w1_list, w3_list)])
        state_dict[f"layers.{layer_idx}.feed_forward.experts.gate_up_proj"] = gate_up

    for layer_idx in sorted(shared_w1.keys()):
        gate_up = torch.cat([shared_w1[layer_idx], shared_w3[layer_idx]], dim=0)
        state_dict[f"layers.{layer_idx}.feed_forward.shared_experts.gate_up_proj"] = gate_up

    missing, unexpected = model.load_state_dict(state_dict, strict=False)


def load_model(
    model_dir: str,
    *,
    device: torch.device | None = None,
) -> LoadedDeepSeekV4Model:
    """Load the DeepSeek-V4 Transformer from *model_dir* in BF16 inference mode.

    1. Patches ``sys.modules["kernel"]`` → CPU fallback.
    2. Instantiates ``Transformer`` with ``ModelArgs`` from ``config.json``.
    3. Loads and dequantises all weights.
    4. Moves the model to *device* (CPU if ``None``).
    """
    # Patch kernel BEFORE importing model
    from deepspec.modeling.deepseek_v4 import kernel_cpu

    sys.modules["kernel"] = kernel_cpu

    # Add inference dir to path so model.py can be imported
    inference_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "models", "deepseek_v4_flash_hf_config", "inference",
    )
    inference_dir = os.path.abspath(inference_dir)
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)

    # Now import the model module
    import model as ds_model  # noqa: E402  (inference/model.py)

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        # Try the repo-local copy
        config_path = os.path.join(
            os.path.dirname(inference_dir), "config.json"
        )
    with open(config_path) as f:
        config = json.load(f)

    # Build ModelArgs
    model_args = _parse_model_args(config)

    # Instantiate
    print(f"[loader] Instantiating Transformer (dtype=bf16, n_layers={model_args.n_layers})")
    transformer = ds_model.Transformer(model_args)

    # Load weights
    print(f"[loader] Loading weights from {model_dir}")
    _load_weights_into_model(transformer, model_dir)

    # Move to device
    if device is not None:
        transformer = transformer.to(device)

    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    return LoadedDeepSeekV4Model(
        model=transformer,
        config=config,
        model_args=model_args,
        hc_mult=model_args.hc_mult,
        n_layers=model_args.n_layers,
        hidden_size=model_args.dim,
    )


# ---------------------------------------------------------------------------
# Hidden-state extraction hooks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedHiddenStates:
    """Hidden states captured during one forward pass."""

    # (seq_len, num_target_layers * hidden_size) — concatenated mid-layer outputs
    target_hidden_states: torch.Tensor
    # (seq_len, hidden_size) — hc_head output (before norm)
    target_last_hidden_states: torch.Tensor


def extract_hidden_states(
    model_loaded: LoadedDeepSeekV4Model,
    input_ids: torch.Tensor,
    target_layer_ids: list[int],
) -> ExtractedHiddenStates:
    """Run one forward pass and capture hidden states at *target_layer_ids*.

    For each layer in *target_layer_ids*, the 4-stream hyper-connection output
    ``(seq, hc_mult, hidden)`` is mean-folded into ``(seq, hidden)``.
    The hc_head collapsed output (before final norm) serves as the "last"
    hidden state.
    """
    transformer = model_loaded.model
    device = input_ids.device
    captured = {}

    # --- Register forward hooks on target layers ---
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            # output is (N, hc_mult, hidden)
            captured[layer_id] = output.mean(dim=1).detach()  # mean-fold

        return hook

    for layer_id in target_layer_ids:
        if 0 <= layer_id < len(transformer.layers):
            h = transformer.layers[layer_id].register_forward_hook(
                make_hook(layer_id)
            )
            handles.append(h)
        else:
            raise ValueError(
                f"target_layer_id {layer_id} out of range "
                f"[0, {len(transformer.layers)})"
            )

    # --- Run forward ---
    try:
        with torch.no_grad():
            # The Transformer forward does:
            #   h = embed → expand hc_mult → layers → head(hc_head + norm) → logits
            # We need to intercept after layers but before head.
            # Since head() is called inside forward(), we can't easily hook
            # the intermediate. So we run the forward manually.

            h = transformer.embed(input_ids)
            h = h.unsqueeze(2).repeat(1, 1, transformer.hc_mult, 1)

            for layer in transformer.layers:
                h = layer(h, 0, input_ids)

            # h is now (B, S, hc_mult, hidden) — the output of all layers
            # Capture the hc_head collapsed output (before norm)
            # hc_head: hc_head_fn, hc_head_base, hc_head_scale
            hc_collapsed = transformer.hc_head_fn.float() @ h.float().transpose(-1, -2)
            hc_collapsed = hc_collapsed.transpose(-1, -2)  # (B, S, 1, hidden)
            hc_collapsed = hc_collapsed.squeeze(2)  # (B, S, hidden)
            hc_collapsed = hc_collapsed * transformer.hc_head_scale + transformer.hc_head_base

            last_hidden = hc_collapsed.to(h.dtype)

    finally:
        for hdl in handles:
            hdl.remove()

    # --- Concatenate captured mid-layer states ---
    target_hidden = torch.cat(
        [captured[lid].to(device) for lid in target_layer_ids],
        dim=-1,
    )  # (B, S, num_layers * hidden)

    return ExtractedHiddenStates(
        target_hidden_states=target_hidden.squeeze(0),
        target_last_hidden_states=last_hidden.squeeze(0),
    )
