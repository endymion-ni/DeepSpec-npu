"""DSpark training config for DeepSeek-V4 Flash as target model.

Data preparation::

    python scripts/data/prepare_target_cache.py \\
        --config config/dspark/dspark_deepseek_v4_flash.py \\
        --train-data-path <train_data.jsonl> \\
        --output-dir <target_cache_dir> \\
        --local-batch-size 16

Training::

    bash scripts/train/train.sh  # after setting config_path / target_cache_dir

The draft model uses Qwen3-8B dense shapes (32 heads, 8 kv_heads, head_dim=128,
intermediate_size=12288, plain rope) for compatibility with DSpark's decoder
layers, while keeping DeepSeek-V4's vocab_size=129280 and hidden_size=4096.

Target layer mapping
--------------------
Following the reference DFlash-Ascend-experiments extraction pipeline,
``target_layer_ids = [2, 21, 40]`` captures three mid-layers whose outputs
(after mean-folding the 4 hyper-connection residual streams) form the
cross-attention context.  The model's final ``last_hidden_state`` serves as
the verifier target for L1 loss and confidence head supervision.
"""

import os

from deepspec.trainer import DeepSeekV4DSparkTrainer

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")

project_name = "deepspec"
exp_name = "dspark_block7_deepseek_v4_flash"
seed = 42

model = dict(
    # ---- Target model ----
    target_model_name_or_path="deepseek-ai/DeepSeek-V4-Flash",

    # ---- DSpark block drafting ----
    block_size=7,
    num_draft_layers=5,

    # ---- Target layers captured into the cache ----
    # Layers 2, 21, 40 are mid-decoder layers whose hyper-connection outputs
    # (4 residual streams) are mean-folded into a single (seq, 4096) vector.
    # The last_hidden_state (layer 43 hc_head + norm output) serves as the
    # verifier target for L1 loss and confidence head.
    target_layer_ids=[2, 21, 40],

    # ---- Mask token for noise embedding ----
    # DeepSeek V4 vocab: 0-127999 regular, 128000-129279 special.
    # 128000 = <｜place▁holder▁no▁0｜> is an unused special token, suitable as mask.
    mask_token_id=128000,
    num_anchors=512,

    # ---- Markov head ----
    markov_rank=256,
    markov_head_type="vanilla",

    # ---- Confidence head ----
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    # ---- Loss ----
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=DeepSeekV4DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="deepseek",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name_str = str(cfg["project_name"])
    exp_name_str = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR, project_name_str, exp_name_str
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR, project_name_str, exp_name_str
    )
    cfg["logging"] = logging_cfg
    return cfg
