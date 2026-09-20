# Data Preparation

This directory contains an example data preparation pipeline using `DeepSeek-V4-Flash` as the target model.

DeepSpec trains draft models against a target model. The data pipeline does three things:

1. download and split prompt data,
2. regenerate assistant answers with the target model,
3. precompute the target cache used by training.

The example below targets `deepseek-ai/DeepSeek-V4-Flash`, but the same pipeline applies to other models (e.g. Gemma). To switch targets, change the model name (`--model` / `model_path`) and adjust the sampling parameters (`--temperature`, `--top-p`, `--top-k` and `--min-p`) to match the recommended generation settings for that model. Output paths in the examples reference `qwen3_4b`; rename them as needed.

The wrapper script [prepare_data.sh](./prepare_data.sh) records the default settings. The individual Python scripts are also documented below for users who want to run each stage manually.

## Outputs

Default outputs:

```text
train_datasets/perfectblend_train.jsonl
train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
~/.cache/deepspec/qwen3_4b_target_cache
```

The example scripts assume a single machine with eight visible GPUs by default. For fewer GPUs, edit `num_workers` and `CUDA_VISIBLE_DEVICES` in the shell scripts.

## Step 1: Download And Split Data

The source dataset is `mlabonne/open-perfectblend`. The train split is written as JSONL, and the held-out user turns are written under `eval_datasets/`.

```bash
python scripts/data/download_and_split.py /
    --dataset-name mlabonne/open-perfectblend /
    --test-size 0.05 /
    --train-output-path train_datasets/perfectblend_train.jsonl /
    --test-output-dir eval_datasets /
    --skip-existing
```

This produces:

```text
train_datasets/perfectblend_train.jsonl
eval_datasets/perfectblend.jsonl
```

## Step 2: Regenerate Answers With VLLM-Ascend

基于[VLLM-Ascend-v0.26.0](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/index.html)拉起推理服务，用Target model重生成回答。

支持模型DeepSeek-V4-Flash，参考[VLLM-Ascend DeepSeek-V4-Flash](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/models/DeepSeek-V4-Flash.html)配置VLLM-Ascend环境，准备模型权重，拉起推理服务。

等vllm服务拉起完成后，发送请求：

```
python scripts/data/generate_train_data_vllm.py --input-file-path 'train_datasets/perfectblend_train.jsonl' --output-file-path 'output/perfectblend_train_regen.jsonl'
```

## Step 3: Prepare Target Cache

### 准备代码

拉取特征提取的代码仓

```
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
git checkout 6ee37ec6e5e709213de1d76c756f81c097970f9d
```

将特征提取的代码复制到cann-recipes-infer仓

```
cp -r scripts/data/dspark_data_preparation cann-recipes-infer/module/
```

### 准备环境和权重

参考
`cann-recipes-infer/models/deepseek_v4/README.md`配置环境，推荐直接拉取镜像。

参考
`cann-recipes-infer/models/deepseek_v4/README.md`下载模型权重，并且转换为Hybrid INT8-INT4权重，适用于Atlas A3 Pod系列。

### 离线特征提取

进入cann-recipes-infer仓库，修改配置文件`module/dspark_data_preparation/config/extract_deepseek_v4.yaml`，快速启动：

```
bash module/dspark_data_preparation/extract_features.sh /
    config/extract_deepseek_v4.yaml data.jsonl output "40 41 42" --mean-hc-mult
```

具体说明见`scripts/data/dspark_data_preparation/README.md`，配置文件的说明见`scripts/data/dspark_data_preparation/config/README.md`