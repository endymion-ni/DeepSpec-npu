#!/usr/bin/env bash
set -euo pipefail

# Single-card training launcher for DeepSeek-V4-Flash DSpark on NPU.
#
# Usage:
#   bash scripts/train/train_single.sh
#
# Override defaults via environment variables:
#   target_cache_dir=/path/to/cache bash scripts/train/train_single.sh
#   global_batch_size=256 bash scripts/train/train_single.sh
#   max_train_steps=100 bash scripts/train/train_single.sh

# ---- device ----
export DEEPSPEC_DEVICE="${DEEPSPEC_DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---- distributed (single process) ----
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ---- DeepSeek-V4 weight paths ----
# Directory containing safetensors shards (at least model-00001 and model-00045).
export DEEPSPEC_DSV4_WEIGHT_DIR="${DEEPSPEC_DSV4_WEIGHT_DIR:-/workspace/deepseek-v4-flash}"
# Full weight index (69187 entries across all 46 shards).
export DEEPSPEC_DSV4_INDEX_PATH="${DEEPSPEC_DSV4_INDEX_PATH:-/workspace/DeepSpec-npu/models/deepseek_v4_flash_hf_config/model.safetensors.index.json}"

# ---- config ----
config_path="${config_path:-config/dspark/dspark_deepseek_v4_flash.py}"
target_cache_dir="${target_cache_dir:-/workspace/ds_target_cache}"
global_batch_size="${global_batch_size:-512}"
max_train_steps="${max_train_steps:-}"
logging_steps="${logging_steps:-10}"
checkpointing_steps="${checkpointing_steps:-3000}"
checkpoint_dir="${checkpoint_dir:-}"

echo "=== DeepSeek-V4-Flash DSpark — Single NPU Training ==="
echo "  config:          ${config_path}"
echo "  target_cache:    ${target_cache_dir}"
echo "  weight_dir:      ${DEEPSPEC_DSV4_WEIGHT_DIR}"
echo "  global_batch:    ${global_batch_size}"
echo "  max_steps:       ${max_train_steps:-auto}"
echo "  device:          ${DEEPSPEC_DEVICE}"
echo "  visible_devices: ${ASCEND_RT_VISIBLE_DEVICES}"

cmd=(
    torchrun
    --nproc-per-node=1
    train.py
    --config "${config_path}"
    --opts "data.target_cache_path=${target_cache_dir}"
    --opts "train.global_batch_size=${global_batch_size}"
    --opts "logging.logging_steps=${logging_steps}"
    --opts "logging.checkpointing_steps=${checkpointing_steps}"
)

if [[ -n "${max_train_steps}" ]]; then
    cmd+=(--opts "train.max_train_steps=${max_train_steps}")
fi
if [[ -n "${checkpoint_dir}" ]]; then
    cmd+=(--opts "logging.checkpoint_dir=${checkpoint_dir}")
fi

"${cmd[@]}"
