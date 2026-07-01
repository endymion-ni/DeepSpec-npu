"""Draft config builder for DeepSeek-V4 Flash as target model.

DeepSeek-V4 uses MoE + MLA with hyper-connections (hc_mult=4), which is
fundamentally incompatible with DSpark's dense decoder layers. Following the
approach from DFlash-Ascend-experiments, the draft model layer shapes are
based on Qwen3-8B (a proven dense template), while DeepSeek's ``vocab_size``
(129280) and ``hidden_size`` (4096) are kept.

- vocab_size must stay DeepSeek's — the target cache stores DeepSeek token ids.
- hidden_size must equal the extracted hidden dim (4096; same in both).
- Everything else (num heads, kv heads, head_dim, intermediate_size, rope,
  max_position) is taken from Qwen3-8B.
"""

import copy

from transformers import AutoConfig

from deepspec.modeling.dspark.common import validate_target_layer_ids

TRAIN_ATTN_IMPLEMENTATION = "flex_attention"

# Qwen3-8B shapes used as the dense draft-layer template.
_QWEN3_8B_DRAFT_BASE = "Qwen/Qwen3-8B"

# Fields copied from Qwen3-8B (NOT vocab_size / hidden_size).
_COPY_ATTRS = [
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "hidden_act",
    "rms_norm_eps",
    "max_position_embeddings",
]


def _get_qwen3_8b_config():
    """Load (and cache) the Qwen3-8B config used as draft shape template."""
    if not hasattr(_get_qwen3_8b_config, "_cached"):
        _get_qwen3_8b_config._cached = AutoConfig.from_pretrained(
            _QWEN3_8B_DRAFT_BASE
        )
    return _get_qwen3_8b_config._cached


def build_draft_config(target_config, model_args):
    """Build a DSpark draft config from DeepSeek-V4 target + model_args.

    The returned config has:
    - DeepSeek-V4's vocab_size (129280) and hidden_size (4096)
    - Qwen3-8B's attention / FFN / rope shapes for dense transformer compatibility
    - All DSpark-specific fields from model_args
    """
    qwen_cfg = _get_qwen3_8b_config()
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)

    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    # Start from Qwen3-8B config (dense draft template), then override with
    # DeepSeek-V4 fields that MUST be kept.
    draft_config = copy.deepcopy(qwen_cfg)
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.vocab_size = int(target_config.vocab_size)
    draft_config.hidden_size = int(target_config.hidden_size)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.tie_word_embeddings = False
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.pad_token_id = target_config.pad_token_id
    draft_config.bos_token_id = target_config.bos_token_id
    draft_config.eos_token_id = target_config.eos_token_id

    # DSpark-specific fields.
    draft_config.block_size = int(model_args.block_size)
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)

    return draft_config


__all__ = [
    "build_draft_config",
    "TRAIN_ATTN_IMPLEMENTATION",
]
