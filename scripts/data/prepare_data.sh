#!/usr/bin/env bash
set -euo pipefail

# ---- model / paths ----
# Override these defaults from the environment to switch target models.
model_path=${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}
config_path=${CONFIG_PATH:-config/dspark/dspark_deepseek_v4_flash.py}
# Local config directory (config.json + tokenizer, pre-converted via convert_config.py).
# This avoids network access for config files; weights still come from model_path.
model_config_path=${MODEL_CONFIG_PATH:-models/deepseek_v4_flash_hf_config}

dataset_name=mlabonne/open-perfectblend
test_size=0.05
train_split_path=train_datasets/perfectblend_train.jsonl
eval_data_dir=eval_datasets

train_data_path=train_datasets/deepseek_v4/perfectblend_train_regen.jsonl
cache_dir=${TARGET_CACHE_DIR:-${HOME}/.cache/deepspec/deepseek_v4_flash_target_cache}

server_host=127.0.0.1
num_workers=8
start_port=30000
concurrency=32
temperature=0.7
top_p=0.8
top_k=20
min_p=0
max_tokens=4096

# ---- device selection ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}

# Count devices for torchrun (NPU or CUDA).
if python3 -c "import torch; torch.npu.is_available()" 2>/dev/null; then
    NPROCS=$(python3 -c "import torch; print(torch.npu.device_count())")
else
    NPROCS=$(python3 -c "import torch; print(torch.cuda.device_count())")
fi
NPROCS=${NPROCS:-1}

server_addresses=()
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    server_addresses+=("${server_host}:$((start_port + worker_id))")
done

echo "Step 1/3: downloading and splitting ${dataset_name}"
python scripts/data/download_and_split.py \
    --dataset-name "${dataset_name}" \
    --test-size "${test_size}" \
    --train-output-path "${train_split_path}" \
    --test-output-dir "${eval_data_dir}" \
    --skip-existing

mkdir -p "$(dirname "${train_data_path}")"

echo "Step 2/3: generating train data (${model_path}): ${train_data_path}"
echo "Start inference server first."
python scripts/data/generate_train_data.py \
    --model "${model_path}" \
    --server-address "${server_addresses[@]}" \
    --concurrency "${concurrency}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --top-k "${top_k}" \
    --min-p "${min_p}" \
    --max-tokens "${max_tokens}" \
    --disable-thinking \
    --resume \
    --input-file-path "${train_split_path}" \
    --output-file-path "${train_data_path}"

echo "Stop inference server before Step 3 if it is using the same devices."
echo "Step 3/3: preparing target cache: ${cache_dir}"
echo "  config path : ${model_config_path}"
echo "  weight path : ${model_path}"
torchrun \
    --nproc-per-node="${NPROCS}" \
    scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --model-config-path "${model_config_path}" \
    --opts "model.target_model_name_or_path=${model_path}" \
    --local-batch-size 16
