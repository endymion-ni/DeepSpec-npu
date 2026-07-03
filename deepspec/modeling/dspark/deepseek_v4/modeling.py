"""DeepSeek-V4 native DSpark draft model.

Reference: `DeepSeek-V4-Flash-DSpark <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark>`_

The official model embeds the DSpark speculative decoder as an ``mtp.0`` layer
inside the full 43-layer DeepSeek-V4 backbone.  That approach shares the
MLA + MoE + hyper-connection architecture of the target model, but requires
the full ~275 GB checkpoint.

This module provides a **standalone** draft model that shares the same layer
design (MLA attention, optional MoE, hyper-connections) while keeping only a
small number of draft layers (``num_draft_layers``, default 1).  It exposes
the same ``forward()`` / ``_forward_backbone()`` interface as
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
# MLA (Multi-head Latent Attention) — DeepSeek-V4 native attention
# ---------------------------------------------------------------------------

class DeepSeekV4MLAttention(nn.Module):
    """MLA layer matching the official DeepSeek-V4 DSpark ``DSparkAttention``.

    Differences from the inference-only CUDA path:

    * Uses standard ``F.scaled_dot_product_attention`` instead of
      ``sparse_attn`` (tilelang kernel).
    * Operates in bf16 (no FP8 quantisation during training).
    * Supports both context and draft positions in a single forward call.
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
        self.v_head_dim = self.head_dim  # v uses all non-rope dims
        self.scaling = self.head_dim ** -0.5
        self.eps = getattr(config, "rms_norm_eps", 1e-6)

        # Low-rank Q projection: wq_a → q_norm → wq_b
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # KV projection: head_dim (nope, shared by k & v) + rope_head_dim (k only).
        # k = [kv_nope[:qk_nope_head_dim], kv_rope]  → 448 + 64 = 512
        # v = kv_nope[:head_dim]                       → 512
        self.wkv = nn.Linear(self.hidden_size, self.head_dim + self.rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim + self.rope_head_dim, eps=self.eps)

        # Low-rank O projection: wo_a → wo_b
        self.wo_a = nn.Linear(
            self.num_heads * self.head_dim, self.o_lora_rank, bias=False
        )
        self.wo_b = nn.Linear(self.o_lora_rank, self.hidden_size, bias=False)

    def _proj_q(self, x: torch.Tensor) -> torch.Tensor:
        """Low-rank Q with QK-normalisation (like the official code)."""
        q = self.wq_a(x)
        q = self.q_norm(q)
        q = self.wq_b(q)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        # QK normalisation (rl * rsq)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        return q

    def _proj_kv(self, x: torch.Tensor) -> torch.Tensor:
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        return kv.unflatten(-1, (1, self.head_dim + self.rope_head_dim))

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
        # Handle hyper-connection multiplicity: flatten hc_mult into batch dim.
        _is_hc = hidden_states.ndim == 4  # (bsz, seq, hc_mult, hidden)
        if _is_hc:
            bsz, q_len, hc_mult, _ = hidden_states.shape
            hidden_states = hidden_states.transpose(1, 2).reshape(
                bsz * hc_mult, q_len, -1,
            )
            target_hidden_states = target_hidden_states.unsqueeze(1).expand(
                -1, hc_mult, -1, -1,
            ).reshape(bsz * hc_mult, target_hidden_states.shape[1], -1)
            # Repeat attention mask for HC multiplicity.
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(hc_mult, dim=0)
        else:
            hc_mult = 1

        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]

        # Q from draft hidden states (low-rank MLA path).
        q = self._proj_q(hidden_states)  # (bsz, q_len, n_heads, head_dim)

        # K / V from target context + draft hidden states (concatenated).
        kv_ctx = self._proj_kv(target_hidden_states)  # (bsz, ctx_len, 1, hd+rd)
        kv_noise = self._proj_kv(hidden_states)        # (bsz, q_len,  1, hd+rd)
        kv = torch.cat([kv_ctx, kv_noise], dim=1)      # (bsz, ctx_len+q_len, 1, hd+rd)

        # Split KV (MLA convention):
        #   kv_nope (512) = shared k/v non-RoPE portion
        #   kv_rope (64)  = k-only RoPE portion
        #   → k = [kv_nope[:448], kv_rope] = 512  (for attention dot-product)
        #   → v = kv_nope[:512]            = 512
        kv_nope = kv[..., : self.head_dim]              # (bsz, kv_len, 1, 512)
        kv_rope = kv[..., self.head_dim :]               # (bsz, kv_len, 1, 64)
        k_nope = kv_nope[..., : self.qk_nope_head_dim]   # (bsz, kv_len, 1, 448)
        q_nope, q_rope = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim :]

        # Apply RoPE to the rope portions (cos/sin shape: (total_len, rope_dim)).
        cos, sin = position_embeddings
        q_rope = _apply_rotary(q_rope, cos[ctx_len:], sin[ctx_len:])
        kv_rope = _apply_rotary(kv_rope, cos, sin)

        # Assemble Q, K, V for attention.
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)     # (bsz, n_heads, q_len, 512)
        k = torch.cat([k_nope, kv_rope], dim=-1)                     # (bsz, kv_len, 1, 512)
        k = k.expand(-1, -1, self.num_heads, -1).transpose(1, 2)    # MQA broadcast
        v = kv_nope                                                   # (bsz, kv_len, 1, 512)
        v = v.expand(-1, -1, self.num_heads, -1).transpose(1, 2)    # MQA broadcast

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim
        )

        # Low-rank O projection.
        o = self.wo_a(attn_output)  # (bsz, q_len, o_lora_rank)
        o = self.wo_b(o)            # (bsz, q_len, hidden_size)

        # Un-flatten hyper-connection streams.
        if _is_hc:
            o = o.reshape(bsz // hc_mult, hc_mult, q_len, -1).transpose(1, 2)
        return o, None


# ---------------------------------------------------------------------------
# Dense MLP (stand-in for MoE during training)
# ---------------------------------------------------------------------------

class DeepSeekV4MLP(nn.Module):
    """Dense SwiGLU MLP — training-friendly replacement for the 256-expert MoE.

    During inference the official model uses ``MoEGMM`` (FP4 quantised MoE).
    For training we use a dense FFN; the MoE can be enabled later via a config
    flag.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(
            config, "moe_intermediate_size", config.hidden_size * 4
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# Hyper-Connection (simplified for training)
# ---------------------------------------------------------------------------

def hc_fuse(
    x: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    """Hyper-connection mixing across ``hc_mult`` residual streams.

    *x*: (bsz, seq_len, hc_mult, hidden_size)
    *fn_weight*: (hc_mult, hc_mult * hidden_size)
    *scale*: (1,) output scale
    *base*: (hc_mult,) per-stream bias
    """
    hc_mult, hidden = x.shape[2], x.shape[3]
    # W: (hc_mult, hc_mult * hidden) → (hc_mult, hc_mult, hidden)
    w = fn_weight.view(hc_mult, hc_mult, hidden)
    # Mix across streams: out[:,:,i,:] = sum_j(x[:,:,j,:] * W[i,j,:])
    out = torch.einsum("bsjh,ijh->bsih", x, w)
    out = out * scale + base.view(1, 1, hc_mult, 1)
    return out


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DeepSeekV4DSparkDecoderLayer(nn.Module):
    """A single decoder layer for the DSpark draft model.

    Architecture: Pre-norm with MLA attention + SwiGLU MLP + hyper-connection
    mixing.  Target hidden states are projected through the same attention K/V
    path (shared weights for context and draft positions).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = getattr(config, "hc_mult", 1)
        hc_dim = self.hc_mult * config.hidden_size

        self.attn = DeepSeekV4MLAttention(config, layer_idx=layer_idx)
        self.mlp = DeepSeekV4MLP(config)

        self.attn_norm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.ffn_norm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))

        # Hyper-connection weights (per-stream mixing after attn / ffn).
        if self.hc_mult > 1:
            self.hc_attn_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(self.hc_mult))
            self.hc_attn_scale = nn.Parameter(torch.ones(1))
            self.hc_ffn_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))
            self.hc_ffn_base = nn.Parameter(torch.empty(self.hc_mult))
            self.hc_ffn_scale = nn.Parameter(torch.ones(1))
        else:
            self.hc_attn_fn = self.hc_attn_base = self.hc_attn_scale = None
            self.hc_ffn_fn = self.hc_ffn_base = self.hc_ffn_scale = None

    def _hc_residual(self, x, residual, fn_w, scale, base):
        """Apply hyper-connection mixing and add to residual stream."""
        if self.hc_mult > 1:
            mixed = hc_fuse(x, fn_w, scale, base)
            return residual + mixed
        return residual + x

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
        hidden_states = self._hc_residual(
            hidden_states, residual,
            self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
        )

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self._hc_residual(
            hidden_states, residual,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# DSpark Model
# ---------------------------------------------------------------------------

class DeepSeekV4DSparkModel(PreTrainedModel):
    """Standalone DSpark draft model for DeepSeek-V4 Flash.

    Shares the MLA / MoE / hyper-connection architecture of the target model
    but uses only a small number of draft layers (default 1, matching the
    official ``mtp.0`` design).

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

        # Mean-fold hyper-connection streams (matching prepare_target_cache).
        if self.hc_mult > 1:
            hidden_states = hidden_states.mean(dim=2)

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
    "DeepSeekV4DSparkModel",
    "DeepSeekV4DSparkDecoderLayer",
    "DeepSeekV4MLAttention",
]
