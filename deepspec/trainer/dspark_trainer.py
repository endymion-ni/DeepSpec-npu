import json
import os

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from deepspec.data import CacheCollator
from deepspec.modeling.dspark.deepseek_v4.config import (
    build_draft_config as build_deepseek_v4_draft_config,
)
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


def _load_target_weights_from_safetensors(
    *,
    weight_dir: str,
    index_path: str | None = None,
) -> dict[str, object]:
    """Load only embed_tokens and lm_head weights from DeepSeek-V4 safetensors.

    Avoids loading the full ~275 GB model by reading individual tensors
    directly from shard files.
    """
    # Load the weight index to map weight names → shard files.
    if index_path is None:
        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Weight index not found: {index_path}")
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})

    def _load_one(name: str):
        if name not in weight_map:
            raise FileNotFoundError(
                f"Weight '{name}' not found in index {index_path}. "
                f"Make sure the index covers all shards."
            )
        shard_file = weight_map[name]
        # The index may use relative paths (e.g. "model-00045-of-00046.safetensors").
        shard = os.path.join(weight_dir, shard_file)
        if not os.path.exists(shard):
            raise FileNotFoundError(
                f"Shard file not found: {shard} (for weight '{name}'). "
                f"Download the missing safetensors file."
            )
        with safe_open(shard, framework="pt") as sf:
            return sf.get_tensor(name)

    return {
        "embed_tokens": _load_one("embed.weight"),
        "lm_head": _load_one("head.weight"),
    }


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)


class DeepSeekV4DSparkTrainer(Qwen3DSparkTrainer):
    """DSpark trainer for DeepSeek-V4 Flash as target model.

    The draft model uses Qwen3-8B shapes for dense transformer compatibility
    while keeping DeepSeek-V4's vocab_size (129280) and hidden_size (4096).
    The embed_tokens and lm_head are copied from the DeepSeek-V4 target model.

    Unlike the base trainer, this class does **not** load the full ~275 GB
    target model.  Instead it reads only ``embed.weight`` and ``head.weight``
    directly from the safetensors shards via :func:`_load_target_weights_from_safetensors`.
    The weight directory defaults to the ``DEEPSPEC_DSV4_WEIGHT_DIR``
    environment variable, falling back to ``/workspace/deepseek-v4-flash``.
    """

    def build_models(self):
        model_args = self.args.model

        # Config & tokenizer — uses the local HF config directory (lightweight).
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
            trust_remote_code=True,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
            trust_remote_code=True,
        )

        # Build draft model.
        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        # Load only embed_tokens + lm_head from safetensors shards.
        weight_dir = os.environ.get(
            "DEEPSPEC_DSV4_WEIGHT_DIR",
            "/workspace/deepseek-v4-flash",
        )
        # Full index with all 69187 weight entries (weight_dir only has
        # a partial index covering the 5 downloaded shards).
        _default_index = "/workspace/DeepSpec-npu/models/deepseek_v4_flash_hf_config/model.safetensors.index.json"
        _full_index = os.environ.get("DEEPSPEC_DSV4_INDEX_PATH", _default_index)
        weights = _load_target_weights_from_safetensors(
            weight_dir=weight_dir,
            index_path=_full_index,
        )

        # Copy weights and freeze — bypass initialize_embeddings_and_head
        # because we pass raw tensors, not nn.Module wrappers.
        with torch.no_grad():
            draft_model.embed_tokens.weight.copy_(weights["embed_tokens"])
            draft_model.lm_head.weight.copy_(weights["lm_head"])
        draft_model.set_embedding_head_trainable(False)
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_deepseek_v4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)
