"""DeepSeek-V4 native DSpark draft model.

Reference: `DeepSeek-V4-Flash-DSpark <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark>`_

The official Ascend inference model embeds the DSpark speculative decoder as
``mtp.*`` stages inside the full 43-layer DeepSeek-V4 backbone.  That path uses
Shared-KV/MQA attention, MoE and hyper-connections, but requires the full
checkpoint and NPU custom kernels.

This module provides a **standalone** draft model that keeps the same DSpark
training contract while using PyTorch/SDPA-friendly approximations for the
kernel-heavy pieces.  It exposes the same ``forward()`` / ``_forward_backbone()``
interface as
:class:`~deepspec.modeling.dspark.qwen3.modeling.Qwen3DSparkModel` so it is
a drop-in replacement in the training pipeline.
"""

from typing import Optional

import torch
from torch import nn
from typing_extensions import Tuple, Unpack

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs

from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.utils.sampling import sample_tokens


# ---------------------------------------------------------------------------
# RoPE — adapted for DeepSeek-V4 (YaRN-compatible, no Qwen3-specific fields)
# ---------------------------------------------------------------------------

class DeepSeekV4RotaryEmbedding(nn.Module):
    """RoPE compatible with DeepSeek-V4 configs (no ``rope_parameters`` dict)."""

    def __init__(self, config):
        super().__init__()
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 1048576)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        dim = self.rope_head_dim
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.LongTensor, dtype: torch.dtype, device: torch.device):
        """Return ``(cos, sin)`` each of shape ``(total_len, rope_head_dim)``."""
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(position_ids.reshape(-1).float(), inv_freq)  # (total_len, half_dim)
        emb = torch.cat((freqs, freqs), dim=-1)  # (total_len, rope_head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to *x*, shape ``(*, seq_len, head_dim)``.
    *cos* / *sin* shape ``(seq_len, rope_dim)``.  The seq_len dim in *x*
    is the second-to-last non-rope dim (position -2 counting from the
    right, excluding the rope_dim split)."""
    rope_dim = cos.shape[-1]
    x_rope, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    # Insert singleton dims so cos broadcasts over everything except seq_len.
    if x.ndim == 4:
        cos = cos.view(1, -1, 1, rope_dim)
        sin = sin.view(1, -1, 1, rope_dim)
    elif x.ndim == 3:
        cos = cos.view(1, -1, rope_dim)
        sin = sin.view(1, -1, rope_dim)
    x1, x2 = x_rope.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return torch.cat([x_rope * cos + rotated * sin, x_pass], dim=-1)


# ---------------------------------------------------------------------------
# RMSNorm (same as DeepSeek-V4 / Qwen3)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------------------
# DSpark attention — matches official ``DSparkAttention`` in cann-recipes
# ---------------------------------------------------------------------------

class DSparkAttention(nn.Module):
    """Training-friendly counterpart of the official ``DSparkAttention``.

    Matches ``cann-recipes-infer_dspark/models/deepseek-v4/models/dspark_modeling.py``:

    * ``_project_dspark_q`` / ``_project_dspark_kv`` — low-rank MLA projections
      with RoPE applied inside (QK-normalisation, kv-norm).
    * ``attn_sink`` — learnable attention-sink bias (official uses it in
      sparse-attn weighted softmax).
    * Shared-KV/MQA: 1 KV head broadcast to ``num_heads``.
    * Dense ``F.scaled_dot_product_attention`` instead of NPU sparse-attn
      kernels (no FP8 quantisation during training).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.q_lora_rank = getattr(config, "q_lora_rank", self.hidden_size)
        self.o_lora_rank = getattr(config, "o_lora_rank", self.hidden_size)
        self.o_groups = getattr(config, "o_groups", 1)
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.qk_nope_head_dim = self.head_dim - self.rope_head_dim  # 448
        self.scaling = self.head_dim ** -0.5
        self.eps = getattr(config, "rms_norm_eps", 1e-6)

        # Low-rank Q projection: wq_a → q_norm → wq_b
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # KV projection: head_dim (nope, shared) + rope_head_dim (k-only).
        self.wkv = nn.Linear(self.hidden_size, self.head_dim + self.rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim + self.rope_head_dim, eps=self.eps)

        # Low-rank O projection: wo_a (grouped) → wo_b
        self.wo_a = nn.Linear(
            self.num_heads * self.head_dim, self.o_lora_rank, bias=False,
        )
        self.wo_b = nn.Linear(self.o_lora_rank, self.hidden_size, bias=False)

        # attn_sink — official ``attn_sink`` for sparse-attn weighted softmax.
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads))

    # ------------------------------------------------------------------
    # MLA projection helpers  (match _project_dspark_q / _project_dspark_kv)
    # ------------------------------------------------------------------

    def _project_dspark_q(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Low-rank Q with QK-normalisation + RoPE — matches official."""
        q = self.wq_a(x)
        q = self.q_norm(q)           # no FP8 quant during training
        q = self.wq_b(q)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        # Split nope / rope, apply RoPE to rope portion.
        q_nope, q_rope = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]
        q_rope = _apply_rotary(q_rope, cos, sin)
        return torch.cat([q_nope, q_rope], dim=-1)

    def _project_dspark_kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """KV projection with kv-norm + RoPE — matches official."""
        kv = self.wkv(x)
        kv = self.kv_norm(kv)        # no FP8 quant during training
        kv = kv.unflatten(-1, (1, self.head_dim + self.rope_head_dim))
        # Split: kv_nope(512) shared, kv_rope(64) k-only.
        kv_nope = kv[..., : self.head_dim]
        kv_rope = kv[..., self.head_dim :]
        kv_rope = _apply_rotary(kv_rope, cos, sin)
        return kv_nope, kv_rope

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Flatten HC multiplicity into batch dim.
        _is_hc = hidden_states.ndim == 4
        if _is_hc:
            _bsz, q_len, hc_mult, _ = hidden_states.shape
            hidden_states = hidden_states.transpose(1, 2).reshape(
                _bsz * hc_mult, q_len, -1,
            )
            target_hidden_states = target_hidden_states.unsqueeze(1).expand(
                -1, hc_mult, -1, -1,
            ).reshape(_bsz * hc_mult, target_hidden_states.shape[1], -1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(hc_mult, dim=0)
        else:
            hc_mult = 1

        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        kv_len = ctx_len + q_len

        cos, sin = position_embeddings  # each: (total_len, rope_dim)

        # Q from draft hidden states.
        q = self._project_dspark_q(hidden_states, cos[kv_len - q_len:], sin[kv_len - q_len:])

        # KV from target context + draft hidden states.
        kv_ctx_nope, kv_ctx_rope = self._project_dspark_kv(
            target_hidden_states, cos[:ctx_len], sin[:ctx_len],
        )
        kv_draft_nope, kv_draft_rope = self._project_dspark_kv(
            hidden_states, cos[ctx_len:], sin[ctx_len:],
        )
        kv_nope = torch.cat([kv_ctx_nope, kv_draft_nope], dim=1)
        kv_rope = torch.cat([kv_ctx_rope, kv_draft_rope], dim=1)

        # Assemble K, V for MQA.
        k_nope = kv_nope[..., : self.qk_nope_head_dim]
        k = torch.cat([k_nope, kv_rope], dim=-1)
        v = kv_nope

        # MQA broadcast: (bsz, kv_len, 1, dim) → (bsz, n_heads, kv_len, dim)
        q = q.transpose(1, 2)
        k = k.expand(-1, -1, self.num_heads, -1).transpose(1, 2)
        v = v.expand(-1, -1, self.num_heads, -1).transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # Dense attention (training fallback; NPU sparse-attn kernel in production).
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim,
        )

        # Low-rank O projection.
        o = self.wo_a(attn_output)
        o = self.wo_b(o)

        if _is_hc:
            o = o.reshape(_bsz, hc_mult, q_len, -1).transpose(1, 2)
        return o, None


# ---------------------------------------------------------------------------
# MoE — matches official ``DeepseekV3MoE`` in cann-recipes (training fallback)
# ---------------------------------------------------------------------------

class DSparkMoE(nn.Module):
    """Structurally matches ``DeepseekV3MoE`` / ``DeepseekV3SharedExpert``.

    Checkpoint keys (per layer):
        ``ffn.gate.weight``, ``ffn.gate.bias`` — routing gate
        ``ffn.experts.{eid}.w1/w2/w3.*`` — 256 routed experts (FP4)
        ``ffn.shared_experts.w1/w2/w3.*`` — shared expert

    During training the routed experts are replaced with a single dense
    SwiGLU FFN.  The gate is created but unused until top-k routing is
    enabled.  Shared expert is a separate dense branch (additive residual).
    """

    def __init__(self, config, prefix: str = ""):
        super().__init__()
        _ = prefix  # kept for canonical alignment with official loaders
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(
            config, "moe_intermediate_size", config.hidden_size * 4,
        )
        self.num_experts = getattr(config, "n_routed_experts", 256)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 6)
        self.n_shared_experts = getattr(config, "n_shared_experts", 1)

        # Routing gate (unused in dense fallback; kept for weight loading).
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Dense "routed expert" — single SwiGLU (training fallback for 256 experts).
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Shared expert.
        if self.n_shared_experts > 0:
            self.shared_gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.shared_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.shared_down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        else:
            self.shared_gate_proj = None
            self.shared_up_proj = None
            self.shared_down_proj = None

    def _dense_expert(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def _shared_expert(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_gate_proj is None:
            return 0.0
        gate = torch.nn.functional.silu(self.shared_gate_proj(x))
        up = self.shared_up_proj(x)
        return self.shared_down_proj(gate * up)

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool = False,
        cur_topk_list=None,
        input_ids=None,
        shared_expert_stream=None,
    ):
        """Official MoE forward signature.  Dense fallback ignores routing."""
        y = self._dense_expert(hidden_states)
        y = y + self._shared_expert(hidden_states)
        return y


# ---------------------------------------------------------------------------
# Hyper-Connection — OpKernel dispatch (fused AscendC / PyPTO when available)
# ---------------------------------------------------------------------------


class _OpKernel:
    """Dispatcher matching ``cann-recipes-infer_dspark`` OpKernel HC API.

    Production: fused AscendC / PyPTO kernels (``npu_hc_pre``, ``npu_hc_post``).
    Fallback: pure-PyTorch native path (this file).  The signatures deliberately
    mirror ``models/deepseek-v4/models/modules/op_impls/mhc.py`` so that
    swapping in the real fused kernels requires only changing the backend.
    """

    @staticmethod
    def hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
        """Pre-sublayer HC: fold ``hc_mult``→1 via learned mixing + Sinkhorn.

        *x*: (bsz, seq_len, hc_mult, hidden_size).
        *hc_fn*: (mix_hc, hc_mult * hidden_size) where ``mix_hc = (2+hc)*hc``.
        *hc_scale*: (3,).  *hc_base*: (mix_hc,).
        Returns ``(y, post, comb)`` — *y* shape (bsz, seq_len, hidden_size).
        """
        return hc_pre_native(
            x, hc_fn, hc_scale, hc_base,
            hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps,
        )

    @staticmethod
    def hc_post(x, residual, post, comb):
        """Post-sublayer HC: expand 1→``hc_mult``.

        *x*: (bsz, seq_len, hidden_size).  *residual*: (bsz, seq_len, hc_mult, hidden_size).
        *post*: (bsz, seq_len, hc_mult).  *comb*: (bsz, seq_len, hc_mult, hc_mult).
        Returns (bsz, seq_len, hc_mult, hidden_size).
        """
        return hc_post_native(x, residual, post, comb)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Pure-PyTorch Sinkhorn decomposition — matches tilelang kernel
    ``hc_split_sinkhorn`` in cann-recipes.

    Reference: ``ops/tilelang/ds_v4/hc_split_sinkhorn.py`` and
    ``models/deepseek-v4/models/modules/op_impls/mhc.py``.
    """
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult)

    comb = comb.softmax(-1) + eps
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb


def hc_pre_native(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """``hc_pre`` native fallback — matches ``hc_pre_native`` in mhc.py."""
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(x_flat, hc_fn.float()) * rsqrt

    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
    )
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb


def hc_post_native(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """``hc_post`` native fallback — matches ``hc_post_native`` in mhc.py."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2,
    )
    return y.type_as(x)


def init_hc_parameters(
    fn_weight: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    mix_hc: int,
):
    """Initialize HC parameters with ``mix_hc``-sized tensors.

    *fn_weight*: (mix_hc, hc_mult * hidden) — mixing projection.
    *base*: (mix_hc,) — per-dim bias.
    *scale*: (3,) — per-group scale.
    """
    nn.init.normal_(fn_weight, mean=0.0, std=0.02)
    nn.init.zeros_(base)
    nn.init.ones_(scale)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DeepSeekV4DSparkDecoderLayer(nn.Module):
    """A single decoder layer for the DSpark draft model.

    Architecture: Pre-norm with Shared-KV/MQA attention + SwiGLU MLP + hyper-connection
    mixing.  Target hidden states are projected through the same attention K/V
    path (shared weights for context and draft positions).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = getattr(config, "hc_mult", 1)
        self.hc_sinkhorn_iters = getattr(config, "hc_sinkhorn_iters", 20)
        self.hc_eps = getattr(config, "hc_eps", 1e-6)
        self.norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size

        self.attn = DSparkAttention(config, layer_idx=layer_idx)
        self.mlp = DSparkMoE(config, prefix=f"mtp.{layer_idx}.ffn")

        self.attn_norm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=self.norm_eps)

        # Hyper-connection weights matching official DSpark shapes.
        # mix_hc = pre(hc) + post(hc) + comb(hc*hc) = 4+4+16 = 24.
        if self.hc_mult > 1:
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))
            init_hc_parameters(self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale, mix_hc)
            init_hc_parameters(self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale, mix_hc)
        else:
            self.hc_attn_fn = self.hc_attn_base = self.hc_attn_scale = None
            self.hc_ffn_fn = self.hc_ffn_base = self.hc_ffn_scale = None

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, post, comb = _OpKernel.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps,
        )
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = _OpKernel.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = _OpKernel.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps,
        )
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _OpKernel.hc_post(hidden_states, residual, post, comb)
        return hidden_states


# ---------------------------------------------------------------------------
# DSpark Model
# ---------------------------------------------------------------------------

class DeepSeekV4DSparkModel(PreTrainedModel):
    """Standalone DSpark draft model for DeepSeek-V4 Flash.

    Follows the Ascend DeepSeek-V4-Flash DSpark layout while keeping the
    training path in PyTorch/SDPA instead of NPU custom inference kernels.

    Interface-compatible with :class:`Qwen3DSparkModel` so the existing
    trainer and evaluator can use it without changes.
    """

    _no_split_modules = ["DeepSeekV4DSparkDecoderLayer"]
    config_class = DeepseekV4Config
    _supports_sdpa = True
    _supports_flash_attention_2 = False
    _supports_flex_attention = False
    _is_stateful = False

    def __init__(self, config):
        super().__init__(config)

        # Validate required fields.
        for field in (
            "target_layer_ids", "mask_token_id", "num_anchors",
            "enable_confidence_head", "markov_rank",
        ):
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type")

        self.target_layer_ids = config.target_layer_ids
        self.hidden_size = config.hidden_size
        self.hc_mult = getattr(config, "hc_mult", 1)
        num_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList([
            DeepSeekV4DSparkDecoderLayer(config, layer_idx)
            for layer_idx in range(num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.rotary_emb = DeepSeekV4RotaryEmbedding(config)

        # Target hidden states fusion.
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)
        self.hc_eps = float(getattr(config, "hc_eps", 1e-6))

        # Official DSpark uses a learned HC head mixer before the final norm.
        if self.hc_mult > 1:
            hc_dim = self.hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))
            self.hc_head_scale = nn.Parameter(torch.ones(1))
            # hc_head uses (hc_mult, hc_dim) not (mix_hc, hc_dim) — sigmoid only.
            nn.init.normal_(self.hc_head_fn, mean=0.0, std=0.02)
            nn.init.zeros_(self.hc_head_base)
            nn.init.ones_(self.hc_head_scale)
        else:
            self.hc_head_fn = self.hc_head_base = self.hc_head_scale = None

        # Markov head.
        self.markov_head = build_markov_head(config)

        # Confidence head.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(
                getattr(config, "confidence_head_with_markov", False)
            )
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = config.hidden_size
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

        self.post_init()

    def save_pretrained(self, save_directory, **kwargs):
        """Save with original architecture name, stripping quantization_config."""
        cfg = self.config
        saved_arch = getattr(cfg, "architectures", None)
        had_quant = hasattr(cfg, "quantization_config")
        saved_quant = cfg.quantization_config if had_quant else None
        try:
            cfg.architectures = ["DeepseekV4ForCausalLM"]
            if had_quant:
                del cfg.quantization_config
            super().save_pretrained(save_directory, **kwargs)
        finally:
            if saved_arch is not None:
                cfg.architectures = saved_arch
            if had_quant:
                cfg.quantization_config = saved_quant

    # -- weight init helpers (same interface as Qwen3DSparkModel) ----------

    def initialize_embeddings_and_head(self, *, embed_tokens, lm_head, freeze=True):
        if isinstance(embed_tokens, torch.Tensor):
            embed_weight = embed_tokens
        else:
            embed_weight = embed_tokens.weight
        if isinstance(lm_head, torch.Tensor):
            head_weight = lm_head
        else:
            head_weight = lm_head.weight
        assert self.embed_tokens.weight.shape == embed_weight.shape
        assert self.lm_head.weight.shape == head_weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight.detach())
            self.lm_head.weight.copy_(head_weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    # -- forward pass ------------------------------------------------------

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def fold_hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fold HC streams — matches official ``DSparkDraftBlock.forward_head_hidden``."""
        if self.hc_mult <= 1:
            return hidden_states

        orig_dtype = hidden_states.dtype
        bsz, seq_len, hc_mult, _ = hidden_states.shape
        eps = getattr(self.config, "hc_eps", 1e-6)
        norm_eps = getattr(self.config, "rms_norm_eps", 1e-6)

        flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat, self.hc_head_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + eps
        y = torch.sum(pre.unsqueeze(-1) * flat.view(bsz, seq_len, hc_mult, -1), dim=2)
        return y.to(orig_dtype)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits, token_ids=prev_token_ids, hidden_states=hidden_states,
            )
        sampled = sample_tokens(step_logits.unsqueeze(1), temperature=temperature).squeeze(1)
        return sampled, step_logits

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        position_embeddings = self.rotary_emb(
            position_ids, dtype=hidden_states.dtype, device=hidden_states.device,
        )

        # Expand to hyper-connection multiplicity.
        if self.hc_mult > 1:
            hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Official DSpark folds HC streams with a learned head mixer before norm.
        if self.hc_mult > 1:
            hidden_states = self.fold_hc_head(hidden_states)

        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> DSparkForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        context_position_ids = torch.arange(
            seq_len, device=device, dtype=torch.long,
        ).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        dspark_attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
            attn_implementation=self.config._attn_implementation,
        )
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1, anchor_positions.size(1), -1, -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1, -1, -1, target_last_hidden_states.shape[-1],
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)

        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions,
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden_4d)
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        log_sampler_stats(
            seq_len=seq_len,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
        )

        confidence_pred = None

        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(
                    prev_token_ids
                ).to(dtype=output_hidden_4d.dtype)
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


__all__ = [
    "DSparkAttention",
    "DSparkMoE",
    "DeepSeekV4DSparkModel",
    "DeepSeekV4DSparkDecoderLayer",
    "DeepSeekV4MLAttention",
]

DeepSeekV4MLAttention = DSparkAttention
