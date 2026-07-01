# DeepSpec on NPU — 训练 DeepSeek V4 Flash 的 DSpark Draft Model

本文档记录在昇腾 NPU（Ascend 910B3）上使用 DeepSpec 框架，为 **DeepSeek V4 Flash** 目标模型训练 **DSpark** speculative-decoding draft model 的完整流程。

> **背景**：DeepSpec 原仓库仅提供 Qwen3 / Gemma4 的配置文件。本文档扩展了 DeepSpec，使其支持 DeepSeek V4 Flash 作为目标模型（verifier），并适配了 NPU 硬件环境。

---

## 1. 环境

### 1.1 基础环境

| 组件 | 版本 |
|------|------|
| Python | 3.12+ |
| torch | 2.7.1+ |
| torch_npu | 2.7.1+ |
| transformers | 5.8.0+ |
| NPU | Ascend 910B3 |

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

### 1.2 HuggingFace 镜像

如果 HuggingFace 直接连接失败，使用镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 2. 架构概述

### 2.1 DeepSeek V4 Flash 的关键参数

| 参数 | 值 |
|------|-----|
| model_type | `deepseek_v4` |
| vocab_size | 129,280 |
| hidden_size | 4,096 |
| num_hidden_layers | 43 |
| attention | MLA（Multi-head Latent Attention），head_dim=512, kv_heads=1 |
| FFN | MoE，256 routed experts + 1 shared expert |
| hyper-connections | `hc_mult=4`（4 条残差流） |
| rope | yarn |
| 权重大小 | W8A8 量化约 275GB |

### 2.2 DSpark Draft 模型设计

DeepSeek V4 使用 MoE + MLA + hyper-connections，其 config 与 DSpark 的 dense decoder 不兼容。因此 draft 模型的**层形状**基于 **Qwen3-8B**（经过验证的 dense 模板），只保留 DeepSeek V4 的两个关键参数：

```
保留 DeepSeek V4:
  - vocab_size  = 129,280     （输入 token id 与 embedding 维度）
  - hidden_size = 4,096       （必须等于抽取的 hidden state 维度）

取自 Qwen3-8B:
  - num_attention_heads  = 32
  - num_key_value_heads  = 8
  - head_dim             = 128
  - intermediate_size    = 12,288
  - hidden_act           = silu
  - rms_norm_eps         = 1e-06
  - rope                 = default (theta=1,000,000)
```

draft 模型为 5 层 dense decoder（`Qwen3DSparkModel`），仅训练 transformer layers + markov head + confidence head；`embed_tokens` 和 `lm_head` 从 target model 复制并冻结。

### 2.3 Hyper-Connection 处理

DeepSeek V4 每层输出形状为 `(N, hc_mult=4, hidden_size)`，即 4 条并行的残差流。DSpark 期望的输入是 `(N, hidden_size)`。

**处理方式**：在 hook 捕获层输出时，对 `hc_mult` 维度做 **mean 折叠**——取 4 条流的均值，压缩为单条 `(N, hidden_size)`：

```python
# deepspec/data/prepare_target_cache.py:_get_hook_tensor
if tensor.ndim == 3 and tensor.shape[1] > 1:
    return tensor.mean(dim=1).detach()  # (N, hc_mult, H) → (N, H)
```

这与参考项目 [DFlash-Ascend-experiments](../DFlash-Ascend-experiments-main/) 中 `dsv4_eagle3_aux.py` 的 `hidden_states.mean(dim=1)` 策略一致。

### 2.4 Target 层选择

参考 DFlash-Ascend-experiments 的提取配置：

```
DSV4_AUX_LAYERS="2 21 40 43"
```

| 层 | 用途 | DeepSpec 映射 |
|----|------|---------------|
| 2, 21, 40 | 中间层 → aux context（cross-attention 的 K/V 来源） | `target_layer_ids = [2, 21, 40]` |
| 43 | hc_head 塌缩输出（norm 前）→ verifier 目标 | `target_last_hidden_states`（模型最终输出，norm 后） |

在 DeepSpec 的 target cache 中：
- `target_hidden_states`：层 2/21/40 的输出 concat → `(seq, 3×4096)`
- `target_last_hidden_states`：模型最后输出 → `(seq, 4096)`

---

## 3. 工作流

三个阶段按顺序执行，每个阶段的输出作为下一阶段的输入：

```
① 数据准备 → ② Target Cache 构建 → ③ 训练 → ④ 评估
```

---

## 4. 数据准备

### 4.1 Step 1 — 下载和切分数据

```bash
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing
```

产物：

```
train_datasets/perfectblend_train.jsonl
eval_datasets/perfectblend.jsonl
```

### 4.2 Step 2 — 重新生成答案

用 DeepSeek V4 Flash 重新生成 assistant 回答（Non-Think 模式）。可以使用任何兼容 OpenAI API 的推理引擎（SGLang、vLLM-Ascend 等）。

**重要**：DeepSeek V4 使用 `encode_messages(thinking_mode="chat")` 进行 Non-Think 格式化，而非标准 Jinja chat template。如果推理引擎不支持 DeepSeek V4 的原生格式化，需要在客户端侧预处理。

**如果使用 vLLM-Ascend**（参考 [DFlash-Ascend-experiments](../DFlash-Ascend-experiments-main/)）：

```bash
# 终端 1：启动 vLLM 服务（TP8）
bash scripts/data/launch_sglang_server.sh  # 替换为你的服务启动脚本

# 终端 2：重新生成答案
python scripts/data/generate_train_data.py \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --server-address 127.0.0.1:30000 ... \
    --concurrency 32 \
    --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/deepseek_v4/perfectblend_train_regen.jsonl
```

产物：

```
train_datasets/deepseek_v4/perfectblend_train_regen.jsonl
```

### 4.3 Step 3 — 构建 Target Cache

这是**关键步骤**：用 DeepSeek V4 Flash 对每条数据做一次完整 forward，捕获指定层的 hidden states，写入二进制 target cache。训练时直接从缓存读取，无需重复跑 target model。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_deepseek_v4_flash.py \
    --train-data-path train_datasets/deepseek_v4/perfectblend_train_regen.jsonl \
    --output-dir ${HOME}/.cache/deepspec/deepseek_v4_flash_target_cache \
    --local-batch-size 16
```

> **存储警告**：Target cache 存储每条 token 的 hidden states。对于 DeepSeek V4 Flash（hidden_size=4096，3 层 aux），每 1B token 约需 `1B × 3 × 4096 × 2 bytes ≈ 24 TB`。确保 `--output-dir` 文件系统有足够空间。如果存储有限，可以减少训练数据量或减少 `target_layer_ids`。

产物：

```
~/.cache/deepspec/deepseek_v4_flash_target_cache/
├── manifest.json           # 元数据（层信息、hidden_size、shard 列表）
├── samples.idx             # 全局索引（每条样本的偏移量）
├── shard-00000.bin          # 二进制数据分片
├── shard-00001.bin
└── ...
```

> **注意**：DeepSeek V4 Flash 约 275GB（W8A8 量化版），如需加载完整权重做 target cache，可能需要多卡（TP）或使用 `device_map="auto"`。如果单卡无法加载，建议参考 [DFlash-Ascend-experiments](../DFlash-Ascend-experiments-main/) 中的 vLLM-Ascend forward-dump 方案——通过正常 serve + prefill 触发将 hidden states 落盘为 safetensors，然后再转换为 DeepSpec 的 target cache 格式。

---

## 5. 训练

### 5.1 配置说明

训练使用配置文件 [`config/dspark/dspark_deepseek_v4_flash.py`](config/dspark/dspark_deepseek_v4_flash.py)。

关键超参：

| 参数 | 值 | 说明 |
|------|-----|------|
| `target_model_name_or_path` | `deepseek-ai/DeepSeek-V4-Flash` | 目标模型 |
| `target_layer_ids` | `[2, 21, 40]` | 3 个中间层用于 cross-attention |
| `block_size` | 7 | DSpark 块大小（→ speculative_tokens=7） |
| `num_draft_layers` | 5 | draft 模型层数 |
| `num_anchors` | 512 | 每样本采样的 anchor 数 |
| `mask_token_id` | 128000 | 噪声 token（DeepSeek 的 `<｜place▁holder▁no▁0｜>`） |
| `lr` | 6e-4 | 学习率 |
| `global_batch_size` | 512 | 全局 batch size |
| `num_train_epochs` | 10 | 训练轮数 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `precision` | bf16 | 训练精度 |

Loss 权重：

| Loss | α | 含义 |
|------|-----|------|
| CE Loss | 0.1 | draft logits 与 ground truth token 的交叉熵（带指数衰减权重） |
| L1 Loss | 0.9 | draft 概率分布与 target 概率分布的 L1 距离 |
| Confidence Loss | 1.0 | confidence head 预测 token-level accept rate |

### 5.2 启动训练

```bash
bash scripts/train/train.sh
```

`train.sh` 默认使用 `config/dspark/dspark_qwen3_4b.py`，需修改其中的配置路径和 target cache 路径：

```bash
# 在 train.sh 中修改：
target_cache_dir=${target_cache_dir:-${HOME}/.cache/deepspec/deepseek_v4_flash_target_cache}

python train.py \
    --config config/dspark/dspark_deepseek_v4_flash.py \
    --opts "data.target_cache_path=${target_cache_dir}"
```

也可以通过 `--opts` 覆盖更多超参：

```bash
python train.py \
    --config config/dspark/dspark_deepseek_v4_flash.py \
    --opts "data.target_cache_path=${target_cache_dir}" \
    --opts "train.lr=3e-4" \
    --opts "train.local_batch_size=4"
```

硬件假设：单机 8 GPU/NPU。如果 GPU/NPU 数量更少，调整 `CUDA_VISIBLE_DEVICES`。

Checkpoint 写入 `~/checkpoints/deepspec/dspark_block7_deepseek_v4_flash/step_*`。

### 5.3 NPU 注意事项

- **Attention 实现**：训练时 draft model 使用 `flex_attention`（仅 draft 的 self/cross-attention）。DeepSeek V4 target model 在构建 cache 时使用 `eager` attention（MLA 不支持 SDPA/FA）。
- **torch.compile**：默认开启 `torch_compile=True`。如果在 NPU 上遇到 compile 问题，可通过 `--opts "train.torch_compile=False"` 关闭。
- **FSDP sharding**：默认 `no_shard`（单节点等价于 DDP）。多节点训练时可改为 `full_shard` 或 `hybrid_shard`。

---

## 6. 评估

```bash
python eval.py \
    --target_name_or_path deepseek-ai/DeepSeek-V4-Flash \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_block7_deepseek_v4_flash/step_latest \
    --max-new-tokens 2048 \
    --temperature 1.0
```

评估在 9 个 benchmark 上测试 speculative decoding 的 accept rate：
- 数学：gsm8k, math500, aime25
- 代码：humaneval, mbpp, livecodebench
- 对话：mt-bench, alpaca, arena-hard-v2

---

## 7. 新增文件清单

为支持 DeepSeek V4 Flash，在 DeepSpec 基础上新增/修改了以下文件：

| 文件 | 说明 |
|------|------|
| `config/dspark/dspark_deepseek_v4_flash.py` | DeepSeek V4 Flash 的 DSpark 训练配置 |
| `deepspec/modeling/dspark/deepseek_v4/__init__.py` | 包初始化 |
| `deepspec/modeling/dspark/deepseek_v4/config.py` | Draft 配置构建器（嫁接 Qwen3-8B 形状） |
| `deepspec/trainer/dspark_trainer.py` | 新增 `DeepSeekV4DSparkTrainer` |
| `deepspec/trainer/__init__.py` | 导出新 trainer |
| `deepspec/data/parser.py` | 注册 `deepseek` 聊天模板 + Jinja 模板 |
| `scripts/data/prepare_target_cache.py` | Hyper-connection mean-folding + eager attention |

---

## 8. 与 DFlash-Ascend-Experiments 的关系

[DFlash-Ascend-experiments](../DFlash-Ascend-experiments-main/) 是 DFlash draft model 在 NPU 上大规模训练的独立实验项目。本文档的方案参考了其以下设计：

- 层选择：`target_layer_ids = [2, 21, 40]`（DFlash-Ascend 使用 `2 21 40 43`，其中 43 映射为 DeepSpec 的 `last_hidden_state`）
- Hyper-connection 处理：mean-fold 4 条残差流
- Draft 形状嫁接：Qwen3-8B dense 模板 + DeepSeek V4 vocab_size/hidden_size

主要区别：本文档使用 **DSpark**（更复杂的 cross-attention + markov head + confidence head），而 DFlash-Ascend 使用 **DFlash**（由 vLLM speculators 框架训练）。

---

## 9. 许可证

DeepSpec 基于 [MIT License](LICENSE)。新增代码沿用同一许可证。包含来自第三方项目的改编代码，完整归属见 [NOTICE](NOTICE)。
