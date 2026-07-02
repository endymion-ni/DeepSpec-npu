# DeepSpec NPU (Ascend) 适配说明

本文档记录在华为 Ascend NPU 上单卡训练 DeepSeek-V4 Flash DSpark 草稿模型所需的改动。

## 环境

- **硬件**: Ascend 910B (64 GB HBM)
- **软件**: torch_npu, HCCL
- **训练配置**: `config/dspark/dspark_deepseek_v4_flash.py`

## 核心改动

### 1. Attention 实现切换 (`deepspec/modeling/dspark/deepseek_v4/config.py`)

NPU 不支持 `torch.nn.attention.flex_attention`，改为 `sdpa`：

```python
# 之前（硬编码）
TRAIN_ATTN_IMPLEMENTATION = "flex_attention"

# 之后（NPU 自适应）
from deepspec.utils.device import is_npu_available
TRAIN_ATTN_IMPLEMENTATION = "sdpa" if is_npu_available() else "flex_attention"
```

同时，从 Qwen3-8B 继承的 `layer_types` 等 per-layer 配置列表需要截断到 `num_draft_layers`（5），否则 config 校验报错。

### 2. int64 索引适配 (`deepspec/modeling/dspark/common.py`)

Ascend NPU 的 `IndexPut` 算子要求 `self`（被索引张量）、`indices`（索引）、`values`（赋值）三者 **dtype 一致**。且 `torch.where(int32, int64)` 在 NPU 上不遵循标准的类型提升规则。

**新增 `_arange()` 辅助函数**：在 NPU 上返回 `torch.int32`，其他平台返回 `torch.int64`。替换所有用于索引的 `torch.arange` 调用。

**`create_noise_embed()`**：将 `noise_ids`、`anchor_tokens`、`mask_tensor` 统一为 `torch.long` (int64)，索引保持 int32（来自 `_arange`）。

**`sample_anchor_positions()`**：返回的 `anchors` tensor 在 NPU 上使用 int32。

### 3. torch.compile 跳过 (`deepspec/trainer/base_trainer.py`)

NPU 上 `torch.compile` 支持不成熟，自动跳过：

```python
if self.args.train.torch_compile and device_type() != "npu":
    self.model = torch.compile(self.model, dynamic=True)
elif self.args.train.torch_compile:
    print("torch.compile is not yet supported on NPU — skipping compilation.")
```

### 4. 单卡跳过 FSDP (`deepspec/trainer/base_trainer.py`)

单进程下 FSDP 无实际意义，且 HCCL 不支持 world_size=1 的 `all_reduce`/`barrier`：

```python
if self.world_size > 1:
    self.model = self._wrap_with_fsdp(self.model)
else:
    print("Single-device — skipping FSDP wrap.")
```

关联影响：
- `no_sync()`：单卡时永远不需要，使用 `nullcontext()`
- `FSDP.clip_grad_norm_()` → `torch.nn.utils.clip_grad_norm_()`
- `dist.barrier()`：`world_size > 1` 时才调用

### 5. 权重加载优化 (`deepspec/trainer/dspark_trainer.py`)

`DeepSeekV4DSparkTrainer` 不再加载完整 275 GB 的 DeepSeek-V4 模型，而是直接从 safetensors shard 文件中读取 `embed.weight` 和 `head.weight`：

```python
class DeepSeekV4DSparkTrainer(Qwen3DSparkTrainer):
    def build_models(self):
        # 1. 从缓存加载 config + tokenizer（轻量）
        # 2. 构建 draft model (Qwen3DSparkModel)
        # 3. 从 safetensors 直接读取 embed.weight + head.weight
        # 4. 复制到 draft_model 并冻结
```

权重文件路径通过环境变量配置：
- `DEEPSPEC_DSV4_WEIGHT_DIR`：safetensors shard 文件目录（默认 `/workspace/deepseek-v4-flash`）
- `DEEPSPEC_DSV4_INDEX_PATH`：完整 weight index 路径

### 6. 分布式操作的 world_size 保护

以下文件中 `world_size <= 1` 时跳过分布式操作：

| 文件 | 跳过的操作 |
|------|-----------|
| `deepspec/utils/metrics.py` | `dist.all_reduce`, `dist.all_gather_object` |
| `deepspec/trainer/ckpt_manager.py` | `dist.barrier`, FSDP state_dict 断言 |

## 快速开始（单卡 NPU）

### 准备目标缓存

```bash
# 生成合成数据（或使用真实 target cache）
python scripts/data/prepare_data_ds.py \
    --num-samples 520 \
    --seq-len 128 \
    --output-dir /workspace/ds_target_cache
```

### 下载 DeepSeek-V4 权重

只需要两个 safetensors shard 文件：

| 文件 | 包含权重 | 大小 |
|------|---------|------|
| `model-00001-of-00046.safetensors` | `embed.weight` | ~1 GB |
| `model-00045-of-00046.safetensors` | `head.weight` | ~1 GB |

通过镜像站下载：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('deepseek-ai/DeepSeek-V4-Flash', 'model-00045-of-00046.safetensors',
                cache_dir='/workspace/deepseek-v4-flash')
"
```

### 启动训练

```bash
ASCEND_RT_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 train.py \
    --config config/dspark/dspark_deepseek_v4_flash.py \
    --opts "data.target_cache_path=/workspace/ds_target_cache" \
    --opts "train.global_batch_size=512"
```

注：`global_batch_size` 需 ≤ target cache 样本数。

## 修改文件清单

### 核心适配文件

| 文件 | 改动 |
|------|------|
| `deepspec/modeling/dspark/deepseek_v4/config.py` | flex_attention → sdpa (NPU)；截断 per-layer 列表 |
| `deepspec/modeling/dspark/common.py` | `_arange()` int32 索引；`create_noise_embed` IndexPut dtype 修复；`sample_anchor_positions` int32 anchors |
| `deepspec/trainer/base_trainer.py` | torch.compile NPU 跳过；单卡 FSDP 跳过；单卡 dist.barrier 保护；grad_norm 无 FSDP fallback |
| `deepspec/trainer/dspark_trainer.py` | `DeepSeekV4DSparkTrainer.build_models()` 直接读 safetensors 权重 |
| `deepspec/trainer/ckpt_manager.py` | 单卡 dist.barrier 保护；非 FSDP state_dict 支持 |
| `deepspec/utils/metrics.py` | 单卡 all_reduce/all_gather 跳过 |

### 工具脚本

| 文件 | 用途 |
|------|------|
| `scripts/data/prepare_data_ds.py` | 生成合成 target cache |
| `scripts/data/init_draft_weights.py` | 从 safetensors 提取 embed_tokens + lm_head |
| `scripts/data/test_npu_train.py` | 单卡 NPU 训练烟雾测试 |

## 已知限制

1. **FSDP 跳过**：当前单卡完全跳过 FSDP，多卡训练需验证 NPU HCCL 兼容性
2. **torch.compile**：NPU 上未启用，待 torch_npu 版本更新后测试
3. **`UserWarning: Cannot create tensor with interal format`**：`torch.full_like` 在 NPU 上的非关键告警，不影响功能
4. **权重加载**：仅加载 `embed_tokens` 和 `lm_head`，其余权重随机初始化，需完整的真实 target cache 进行有效训练
