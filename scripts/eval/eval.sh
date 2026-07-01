#!/usr/bin/env bash
#
# Launch evaluation with torchrun.
#
# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE / LOCAL_WORLD_SIZE
# automatically; eval.py reads them via init_dist().

# ---- device selection ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}

# Count devices for torchrun (NPU or CUDA).
if python3 -c "import torch; torch.npu.is_available()" 2>/dev/null; then
    NPROCS=$(python3 -c "import torch; print(torch.npu.device_count())")
else
    NPROCS=$(python3 -c "import torch; print(torch.cuda.device_count())")
fi
NPROCS=${NPROCS:-1}

# Match this to the target model used by the draft checkpoint.
target_name_or_path=${TARGET_NAME_OR_PATH:-deepseek-ai/DeepSeek-V4-Flash}

# Training writes checkpoints under ~/checkpoints/<project_name>/<exp_name>/step_*.
# Use step_latest for the most recent checkpoint, or replace it with step_<N>.
draft_name_or_path=${DRAFT_NAME_OR_PATH:-${HOME}/checkpoints/deepspec/dspark_block7_deepseek_v4_flash/step_latest}

torchrun \
    --nproc-per-node="${NPROCS}" \
    eval.py \
    --target_name_or_path "${target_name_or_path}" \
    --draft_name_or_path "${draft_name_or_path}"
