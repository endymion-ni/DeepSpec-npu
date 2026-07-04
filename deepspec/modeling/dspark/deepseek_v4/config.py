"""Draft config builder for DeepSeek-V4 Flash as target model.

Builds a config that keeps DeepSeek-V4 Flash's native DSpark shapes:

    head_dim=512, num_attention_heads=64, num_key_value_heads=1 (MQA),
    q_lora_rank=1024, o_lora_rank=1024, o_groups=8,
    qk_rope_head_dim=64, moe_intermediate_size=2048, hc_mult=4.

Only the layer count is reduced to ``num_draft_layers``.  Compatible with
:class:`~deepspec.modeling.dspark.deepseek_v4.modeling.DeepSeekV4DSparkModel`.
"""

import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids
from deepspec.utils.device import is_npu_available

TRAIN_ATTN_IMPLEMENTATION = "sdpa" if is_npu_available() else "flex_attention"


def _parse_dspark_args(model_args):
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        int(model_args.get("num_target_layers", 999)),
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args

    return {
        "target_layer_ids": target_layer_ids,
        "enable_confidence_head": enable_confidence_head,
        "markov_rank": markov_rank,
    }


def _add_dspark_fields(draft_config, model_args, target_layer_ids, enable_confidence_head, markov_rank):
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

    # Official Ascend DSpark config aliases.
    draft_config.dspark_block_size = draft_config.block_size
    draft_config.dspark_noise_token_id = draft_config.mask_token_id
    draft_config.dspark_target_layer_ids = list(target_layer_ids)
    draft_config.dspark_markov_rank = markov_rank


def _truncate_per_layer_lists(draft_config, num_layers):
    for attr in ("layer_types", "mlp_layer_types"):
        val = getattr(draft_config, attr, None)
        if isinstance(val, (list, tuple)) and len(val) > num_layers:
            setattr(draft_config, attr, val[:num_layers])
    ratios = getattr(draft_config, "compress_ratios", None)
    if isinstance(ratios, (list, tuple)) and len(ratios) > num_layers:
        setattr(draft_config, "compress_ratios", ratios[:num_layers])


def build_draft_config(target_config, model_args):
    """Build a DeepSeek-V4 native DSpark draft config.

    Clones the target model's architecture (Shared-KV/MQA, MoE, HC) and reduces the
    layer count to ``num_draft_layers``.  Compatible with
    :class:`~deepspec.modeling.dspark.deepseek_v4.modeling.DeepSeekV4DSparkModel`.
    """
    num_draft_layers = int(model_args.num_draft_layers)
    kwargs = _parse_dspark_args(model_args)

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["DeepSeekV4DSparkModel"]
    draft_config.num_target_layers = int(target_config.num_hidden_layers)
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.dspark_num_layers = num_draft_layers
    draft_config.tie_word_embeddings = False
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION

    _truncate_per_layer_lists(draft_config, num_draft_layers)
    _add_dspark_fields(draft_config, model_args, **kwargs)
    return draft_config


__all__ = [
    "build_draft_config",
    "TRAIN_ATTN_IMPLEMENTATION",
]
