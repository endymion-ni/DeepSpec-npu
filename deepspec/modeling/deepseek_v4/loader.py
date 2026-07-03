"""Load DeepSeek-V4 ``inference/model.py`` Transformer with native FP8 weights.

Patches ``sys.modules["kernel"]`` with CPU fallbacks so the model can run
without tilelang CUDA kernels.  Weights stay in FP8 / FP4 — **no
dequantisation** — so the model runs A8W8 (slower on CPU, fast on NPU once
kernels are ported).

Reference: ``cann-recipes-infer/models/deepseek-v4/utils/convert_model.py``
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterator

import torch
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Iterate safetensors
# ---------------------------------------------------------------------------

def _iter_safetensors(model_dir: str) -> Iterator[tuple[str, torch.Tensor]]:
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
    """Checkpoint native → ``Transformer`` state_dict key.

    Checkpoint:  layers.{i}.attn.wq_a.weight  / .scale
    Model:       layers.{i}.attention.wq_a.weight  / .scale
    """
    if ckpt_key.startswith("mtp."):
        return None
    if not ckpt_key.startswith("layers."):
        return ckpt_key

    parts = ckpt_key.split(".")
    layer_idx = parts[1]
    rest = ".".join(parts[2:])

    # attn → attention, ffn → feed_forward
    if rest.startswith("attn."):
        rest = "attention." + rest[5:]
    elif rest.startswith("ffn."):
        rest = "feed_forward." + rest[4:]
    # hc_attn_* → attn_hc.*
    elif rest.startswith("hc_attn_"):
        param = rest[8:]
        rest = f"attn_hc.{param}"
    elif rest.startswith("hc_ffn_"):
        param = rest[7:]
        rest = f"ffn_hc.{param}"

    return f"layers.{layer_idx}.{rest}"


# ---------------------------------------------------------------------------
# Expert w1/w3 → gate_up_proj  (FP8-native, no dequant)
# ---------------------------------------------------------------------------

def _merge_expert_weights(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merge per-expert w1 + w3 → stacked gate_up_proj.

    Checkpoint:  layers.{i}.ffn.experts.{e}.w1.weight  (FP8)
                 layers.{i}.ffn.experts.{e}.w3.weight  (FP8)
    Model:       layers.{i}.feed_forward.experts.gate_up_proj
                   (n_experts, 2*inter_dim, hidden)  FP8
    """
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
            (expert_w3 if is_w3 else expert_w1).setdefault(layer_idx, {})[exp_idx] = tensor
            continue

        # Shared experts
        if ".ffn.shared_experts." in ckpt_key and (
            ckpt_key.endswith(".w1.weight") or ckpt_key.endswith(".w3.weight")
        ):
            parts = ckpt_key.split(".")
            layer_idx = int(parts[1])
            is_w3 = parts[4] == "w3"
            (shared_w3 if is_w3 else shared_w1)[layer_idx] = tensor
            continue

    for layer_idx in sorted(expert_w1.keys()):
        n_exp = n_experts[layer_idx]
        gate_up = torch.stack([
            torch.cat([expert_w1[layer_idx][e], expert_w3[layer_idx][e]], dim=0)
            for e in range(n_exp)
        ])
        merged[f"layers.{layer_idx}.feed_forward.experts.gate_up_proj"] = gate_up

    for layer_idx in sorted(shared_w1.keys()):
        gate_up = torch.cat([shared_w1[layer_idx], shared_w3[layer_idx]], dim=0)
        merged[f"layers.{layer_idx}.feed_forward.shared_experts.gate_up_proj"] = gate_up

    return merged


# ---------------------------------------------------------------------------
# Weight loading (FP8-native)
# ---------------------------------------------------------------------------

def _load_weights(model: torch.nn.Module, model_dir: str, device: torch.device) -> None:
    raw: dict[str, torch.Tensor] = {}

    # Load to CPU (safetensors requirement), move to device immediately.
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key).to(device)

    # Merge experts
    state_dict = _merge_expert_weights(raw)

    # Map remaining keys
    for ckpt_key, tensor in raw.items():
        if ckpt_key.startswith("mtp."):
            continue
        model_key = _map_ckpt_to_model(ckpt_key)
        if model_key is None:
            continue
        if (".ffn.experts." in ckpt_key or ".ffn.shared_experts." in ckpt_key) and (
            ckpt_key.endswith(".w1.weight") or ckpt_key.endswith(".w3.weight")
        ):
            continue
        if ".ffn.experts." in ckpt_key and ckpt_key.endswith(".w2.weight"):
            model_key = model_key.replace(
                "feed_forward.experts.w2.weight", "feed_forward.experts.down_proj")
        if ".ffn.shared_experts." in ckpt_key and ckpt_key.endswith(".w2.weight"):
            model_key = model_key.replace(
                "feed_forward.shared_experts.w2.weight",
                "feed_forward.shared_experts.down_proj.weight")
        state_dict[model_key] = tensor

    del raw

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[loader] {len(missing)} missing (first 3: {missing[:3]})")
    if unexpected:
        print(f"[loader] {len(unexpected)} unexpected (first 3: {unexpected[:3]})")


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
    num_layers: int | None = None,
) -> LoadedDeepSeekV4Model:
    """Load DeepSeek-V4 Transformer with native FP8 weights.

    If *num_layers* is given, the model is cropped to the first N layers
    (e.g. ``num_layers=3`` creates layers 0-2 only).
    """

    from deepspec.modeling.deepseek_v4 import kernel_cpu
    sys.modules["kernel"] = kernel_cpu

    inference_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "models", "deepseek_v4_flash_hf_config", "inference")
    )
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    import model as ds_model

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(inference_dir), "config.json")
    with open(config_path) as f:
        config = json.load(f)

    if num_layers is not None and num_layers < config.get("num_hidden_layers", 999):
        print(f"[loader] Cropping model from {config['num_hidden_layers']} → {num_layers} layers")
        config["num_hidden_layers"] = num_layers
        # Also crop layer-dependent arrays
        for key in ("layer_types", "mlp_layer_types", "compress_ratios"):
            if key in config and isinstance(config[key], list):
                config[key] = config[key][:num_layers]

    model_args = _build_model_args(config)

    # Transformer.__init__ calls set_default_dtype(fp8). Save & restore.
    _prev_dtype = torch.get_default_dtype()
    transformer = ds_model.Transformer(model_args)
    torch.set_default_dtype(_prev_dtype)

    _load_weights(transformer, model_dir, device=device or torch.device("cpu"))

    if device is not None:
        # Non-weight buffers may still be on CPU — move them.
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
    from models.deepseek_v4_flash_hf_config.inference.model import ModelArgs

    direct = {
        "vocab_size": "vocab_size", "dim": "dim", "n_layers": "n_layers",
        "n_routed_experts": "n_routed_experts",
        "n_shared_experts": "n_shared_experts",
        "n_activated_experts": "num_experts_per_tok",
        "score_func": "scoring_func", "route_scale": "routed_scaling_factor",
        "q_lora_rank": "q_lora_rank", "head_dim": "head_dim",
        "rope_head_dim": "qk_rope_head_dim", "norm_eps": "rms_norm_eps",
        "o_groups": "o_groups", "o_lora_rank": "o_lora_rank",
        "window_size": "sliding_window",
        "compress_rope_theta": "compress_rope_theta",
        "rope_theta": "rope_theta",
        "index_n_heads": "index_n_heads", "index_head_dim": "index_head_dim",
        "index_topk": "index_topk",
        "hc_mult": "hc_mult", "hc_sinkhorn_iters": "hc_sinkhorn_iters",
        "hc_eps": "hc_eps",
        "n_hash_layers": "n_hash_layers",
        "n_mtp_layers": "num_nextn_predict_layers",
    }
    kwargs = {}
    for ma, ck in direct.items():
        if ck in config:
            kwargs[ma] = config[ck]

    rp = config.get("rope_parameters", {}).get("main", {})
    for k in ("factor", "beta_fast", "beta_slow"):
        if k in rp:
            kwargs[f"rope_{k}"] = rp[k]

    kwargs.setdefault("max_seq_len", config.get("max_position_embeddings", 4096))
    kwargs.setdefault("dtype", "fp8")
    kwargs.setdefault("scale_dtype", "fp8")

    return ModelArgs(**kwargs)


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractedHiddenStates:
    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


def extract_hidden_states(
    model_loaded: LoadedDeepSeekV4Model,
    input_ids: torch.Tensor,
    target_layer_ids: list[int],
) -> ExtractedHiddenStates:
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
