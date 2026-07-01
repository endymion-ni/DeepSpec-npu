#!/usr/bin/env bash
#
# Launch training with torchrun.
#
# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE / LOCAL_WORLD_SIZE
# automatically; train.py reads them via init_dist().  Use the standard
# torchrun CLI for single- or multi-node runs.
#
# Single-node example (all visible devices):
#   bash scripts/train/train.sh
#
# Override NPU / GPU visibility or number of processes:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train.sh
#   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train.sh
#
# Multi-node: set MASTER_ADDR / MASTER_PORT / NNODES / NODE_RANK in the
# environment before invoking this script.

# ---- device selection ----
# NPU (Ascend): ASCEND_RT_VISIBLE_DEVICES  (falls back to CUDA_VISIBLE_DEVICES)
# GPU (NVIDIA): CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}

# Count devices: NPU or CUDA.
if python3 -c "import torch; torch.npu.is_available()" 2>/dev/null; then
    NPROCS=$(python3 -c "import torch; print(torch.npu.device_count())")
else
    NPROCS=$(python3 -c "import torch; print(torch.cuda.device_count())")
fi
NPROCS=${NPROCS:-1}

# ---- config selection ----
# Available public configs:
## dflash
#   config/dflash/dflash_gemma4_12b.py
#   config/dflash/dflash_qwen3_4b.py
#   config/dflash/dflash_qwen3_8b.py
#   config/dflash/dflash_qwen3_14b.py
## dspark
#   config/dspark/dspark_gemma4_12b.py
#   config/dspark/dspark_qwen3_4b.py
#   config/dspark/dspark_qwen3_8b.py
#   config/dspark/dspark_qwen3_14b.py
#   config/dspark/dspark_deepseek_v4_flash.py
## eagle3
#   config/eagle3/eagle3_gemma4_12b.py
#   config/eagle3/eagle3_qwen3_4b.py
#   config/eagle3/eagle3_qwen3_8b.py
#   config/eagle3/eagle3_qwen3_14b.py

config_path=${config_path:-config/dspark/dspark_qwen3_4b.py}
target_cache_dir=${target_cache_dir:-${HOME}/.cache/deepspec/qwen3_4b_target_cache}

# --opts overrides any config field by dotted key path: --opts "<key.path>=<value>".
# Values are parsed as Python scalars (int/float/bool/str). Repeat the flag to set
# multiple fields, e.g.:
#   --opts "data.target_cache_path=${target_cache_dir}" \
#   --opts "train.lr=3e-4" \
#   --opts "train.local_batch_size=2"
#
# local_batch_size is the per-device micro-batch size. Raise it to better utilize
# devices with more memory (e.g. 4 or 8 on 80GB cards), or keep it at 1 if you
# hit OOM. Override it without editing the config via:
#   --opts "train.local_batch_size=4"

torchrun \
    --nproc-per-node="${NPROCS}" \
    train.py \
    --config "${config_path}" \
    --opts "data.target_cache_path=${target_cache_dir}"
