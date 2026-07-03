# DeepSpec NPU (Ascend) 适配说明

本文档记录在华为 Ascend NPU 上单卡训练 DeepSeek-V4 Flash DSpark 草稿模型所需的全部改动。

## 环境

- **硬件**: Ascend 950
- **软件**: torch_npu, HCCL
- **训练配置**: `config/dspark/dspark_deepseek_v4_flash.py`
- **官方模型**: [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)

## 架构概览

### 训练流程

```
train.py → DeepSeekV4DSparkTrainer
  ├── config: DeepSeek-V4 原生 MLA + MoE + HC (单层 ~1.28B params)
  ├── 权重: 仅加载 embed.weight + head.weight (2 个 safetensors shard, ~2GB)
  ├── 数据: target cache (precomputed hidden states)
  └── 单卡: 跳过 FSDP, torch.compile, 所有 dist.barrier
```

### 草稿模型: Qwen3 → DeepSeek-V4 替换

原始代码使用 Qwen3-8B 的 dense transformer 作为草稿模型模板（仅保留 DeepSeek-V4 的 vocab_size 和 hidden_size）。现已全部替换为 DeepSeek-V4 原生架构：

| 维度 | 旧 (Qwen3-8B 模板) | 新 (DeepSeek-V4 原生) |
|------|---------------------|------------------------|
| 模型类 | `Qwen3DSparkModel` | `DeepSeekV4DSparkModel` |
| 配置构建 | `build_draft_config` (Qwen3 base) | `build_draft_config` (clone target config) |
| Attention | Dense MHA: 32 heads, 8 KV, head_dim=128 | **MLA**: 64 heads, 1 KV(MQA), head_dim=512, q_lora_rank=1024, o_lora_rank=1024 |
| FFN | Dense SwiGLU (intermediate=12288) | Dense SwiGLU (intermediate=2048, 训练期替 MoE) |
| Position | Qwen3 RoPE (`rope_parameters`) | 自建 RoPE (无 Qwen3 依赖) |
| Residual | Standard residual | **Hyper-Connection** (hc_mult=4) |
| HF 序列化 | Qwen3PreTrainedModel | PreTrainedModel + DeepseekV4Config |

### 官方 DSpark 简化项

当前模型相对官方 `DeepSeek-V4-Flash-DSpark` 的 `mtp.0` 层做了以下简化，目的是先打通训练流程，后续逐步补全：

| 组件 | 官方 | 当前 | 影响 |
|------|------|------|------|
| FFN | 256-expert MoE + FP4 量化 | Dense SwiGLU MLP | 参数多 ~50 倍 (但无量化)，FLOPs ↑ |
| Attention | `sparse_attn` (tilelang CUDA): KV 压缩 + Indexer top-512 稀疏 | `F.scaled_dot_product_attention`: 全 dense | O(n²) vs O(n)，长序列显著变慢 |
| HC | HC Sinkhorn 迭代 (20 轮归一化) | 简化 `einsum` 线性混合 | Sinkhorn 归一化是 DSv4 特有机制 |
| KV Cache | Window + Compressor 管理 | 训练不用 KV cache | 无 |
| 量化 | FP8 K/V/Q 量化 | bf16 全精度 | 训练不需要量化 |
| 采样 | Gumbel-max 采样 | 训练不涉及 | 无 |

## 核心 NPU 适配改动

### 1. Attention 实现切换 (`deepspec/modeling/dspark/deepseek_v4/config.py`)

NPU 不支持 `torch.nn.attention.flex_attention`，改为 `sdpa`：

```python
TRAIN_ATTN_IMPLEMENTATION = "sdpa" if is_npu_available() else "flex_attention"
```

### 2. int64 索引适配 (`deepspec/modeling/dspark/common.py`)

Ascend NPU 的 `IndexPut` 算子要求 self、indices、values 三者 dtype 一致，且 `torch.where(int32, int64)` 在 NPU 上不遵循标准类型提升规则。

- 新增 `_arange()` 辅助函数：NPU 上返回 `torch.int32`
- `create_noise_embed()`：self+values 统一 `torch.long`，索引 int32
- `sample_anchor_positions()`：anchors 在 NPU 上 int32

### 3. torch.compile 跳过 (`deepspec/trainer/base_trainer.py`)

```python
if self.args.train.torch_compile and device_type() != "npu":
    self.model = torch.compile(self.model, dynamic=True)
```

### 4. 单卡跳过 FSDP (`deepspec/trainer/base_trainer.py`)

HCCL 不支持 world_size=1 的 `all_reduce`/`barrier`。单卡时：
- 跳过 FSDP wrap → 使用 `nullcontext()` 替代 `no_sync()`
- `FSDP.clip_grad_norm_()` → `torch.nn.utils.clip_grad_norm_()`
- 所有 `dist.barrier()` 加 `world_size > 1` 保护

### 5. 权重加载优化 (`deepspec/trainer/dspark_trainer.py`)

`DeepSeekV4DSparkTrainer.build_models()` 不加载完整 275 GB 模型，直接从 safetensors shard 读取 `embed.weight` + `head.weight`：

```python
weights = _load_target_weights_from_safetensors(weight_dir=..., index_path=...)
draft_model.embed_tokens.weight.copy_(weights["embed_tokens"])
draft_model.lm_head.weight.copy_(weights["lm_head"])
draft_model.set_embedding_head_trainable(False)
```

环境变量：
- `DEEPSPEC_DSV4_WEIGHT_DIR`：safetensors shard 目录（默认 `/workspace/deepseek-v4-flash`）
- `DEEPSPEC_DSV4_INDEX_PATH`：完整 weight index 路径

### 6. 分布式操作的 world_size 保护

| 文件 | 跳过的操作 |
|------|-----------|
| `deepspec/utils/metrics.py` | `dist.all_reduce`, `dist.all_gather_object` |
| `deepspec/trainer/ckpt_manager.py` | `dist.barrier`, FSDP state_dict 断言 |
| `deepspec/trainer/base_trainer.py` | `save_and_eval_checkpoint` / `_save_and_suspend` / `clean_up` 中的 `dist.barrier` |

## 新增文件

### 核心模型 (`deepspec/modeling/dspark/deepseek_v4/`)

| 文件 | 说明 |
|------|------|
| `modeling.py` | `DeepSeekV4DSparkModel` — MLA + HC + 自建 RoPE，继承 `PreTrainedModel`，支持 `save_pretrained` / `from_pretrained` |
| `config.py` | `build_draft_config` — 直接从 target config clone，只改层数和 DSpark 字段，保留全部 MLA/MoE/HC 原生维度 |

### 推理 eval (`deepspec/eval/`)

| 文件 | 改动 |
|------|------|
| `eval/dspark/evaluator.py` | 新增 `DeepSeekV4DSparkEvaluator` |
| `eval/dspark/draft_ops.py` | `DSparkModel` union 加入 `DeepSeekV4DSparkModel` |
| `eval/dspark/__init__.py` | 导出 `DeepSeekV4DSparkEvaluator` |
| `eval/__init__.py` | 导出 `DeepSeekV4DSparkEvaluator` |
| `eval.py` | `EVALUATORS` 注册 `"DeepSeekV4DSparkModel"` → `DeepSeekV4DSparkEvaluator` |

### 训练

| 文件 | 改动 |
|------|------|
| `trainer/dspark_trainer.py` | `DeepSeekV4DSparkTrainer` 使用 `DeepSeekV4DSparkModel` + 直接 safetensors 权重加载 |
| `trainer/base_trainer.py` | torch.compile/FSDP/barrier 单卡保护 |
| `config/dspark/dspark_deepseek_v4_flash.py` | 更新 docstring |

## 快速开始（单卡 NPU）

### 1. 准备 target cache

```bash
python scripts/data/prepare_data_ds.py \
    --num-samples 520 --seq-len 128 \
    --output-dir /workspace/ds_target_cache
```

### 2. 下载权重（仅需 2 个 shard, ~2GB）

| 文件 | 包含权重 | 大小 |
|------|---------|------|
| `model-00001-of-00046.safetensors` | `embed.weight` | ~1 GB |
| `model-00045-of-00046.safetensors` | `head.weight` | ~1 GB |

```bash
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('deepseek-ai/DeepSeek-V4-Flash', 'model-00045-of-00046.safetensors',
                cache_dir='/workspace/deepseek-v4-flash')
"
```

### 3. 启动训练

```bash
# 使用单卡启动脚本
bash scripts/train/train_single.sh

# 或手动指定参数
target_cache_dir=/workspace/ds_target_cache \
global_batch_size=512 \
max_train_steps=1000 \
bash scripts/train/train_single.sh
```

## 完整文件清单

### 核心适配文件

| 文件 | 改动 |
|------|------|
| `deepspec/modeling/dspark/deepseek_v4/config.py` | flex_attention → sdpa (NPU)；`build_draft_config` 改为 clone target config；截断 per-layer 列表 |
| `deepspec/modeling/dspark/deepseek_v4/modeling.py` | **新建** — `DeepSeekV4DSparkModel` (MLA + HC + DeepSeekV4RotaryEmbedding) |
| `deepspec/modeling/dspark/common.py` | `_arange()` int32 索引；`create_noise_embed` IndexPut dtype 修复；`sample_anchor_positions` int32 anchors |
| `deepspec/trainer/base_trainer.py` | torch.compile NPU 跳过；单卡 FSDP 跳过；单卡 dist.barrier 保护；grad_norm 无 FSDP fallback |
| `deepspec/trainer/dspark_trainer.py` | `DeepSeekV4DSparkTrainer.build_models()` 直接读 safetensors 权重；使用 `DeepSeekV4DSparkModel` |
| `deepspec/trainer/ckpt_manager.py` | 单卡 dist.barrier 保护；非 FSDP state_dict 支持 |
| `deepspec/utils/metrics.py` | 单卡 all_reduce/all_gather 跳过 |

### Eval 适配文件

| 文件 | 改动 |
|------|------|
| `deepspec/eval/dspark/evaluator.py` | 新增 `DeepSeekV4DSparkEvaluator` |
| `deepspec/eval/dspark/draft_ops.py` | `DSparkModel` union 加入 `DeepSeekV4DSparkModel` |
| `deepspec/eval/dspark/__init__.py` | 导出 `DeepSeekV4DSparkEvaluator` |
| `deepspec/eval/__init__.py` | 导出 `DeepSeekV4DSparkEvaluator` |
| `eval.py` | `EVALUATORS` 注册 |

### 工具脚本

| 文件 | 用途 |
|------|------|
| `scripts/data/prepare_data_ds.py` | 生成合成 target cache |
| `scripts/data/init_draft_weights.py` | 从 safetensors 提取 embed_tokens + lm_head |
| `scripts/data/test_npu_train.py` | 单卡 NPU 训练烟雾测试 |
| `scripts/train/train_single.sh` | 单卡训练启动脚本 |

## 已知限制

1. **FSDP 跳过**：当前单卡完全跳过 FSDP，多卡训练需验证 NPU HCCL 兼容性
2. **torch.compile**：NPU 上未启用，待 torch_npu 版本更新后测试
3. **FFN 简化**：使用 Dense SwiGLU 替代 256-expert MoE，需后续替换以对齐官方效果
4. **HC 简化**：使用线性 `einsum` 混合替代 HC Sinkhorn 20 轮迭代
5. **Attention**：使用 dense SDPA 替代 `sparse_attn`，长序列性能差
6. **`UserWarning: Cannot create tensor with interal format`**：`torch.full_like` 在 NPU 上的非关键告警
7. **权重加载**：仅加载 `embed_tokens` 和 `lm_head`，draft 层随机初始化，需完整真实 target cache 进行有效训练
