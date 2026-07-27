# DeepSpec NPU (Ascend) 适配说明

本文档记录在华为 Ascend NPU 上单卡训练 DeepSeek-V4 Flash DSpark 草稿模型所需的全部改动。

## 环境

- **硬件**: Ascend 950
- **软件**: torch_npu, HCCL
- **训练配置**: `config/dspark/dspark_deepseek_v4_flash.py`
- **官方模型**: [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
- **参考实现**: `cann-recipes-infer_dspark/models/deepseek-v4/`

## 架构概览

### 训练流程

```
train.py → DeepSeekV4DSparkTrainer
  ├── config: DeepSeek-V4 Flash DSpark Shared-KV/MQA + MoE + HC
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
| Attention | Dense MHA: 32 heads, 8 KV, head_dim=128 | **DSparkAttention**: 64 heads, 1 KV (MQA), head_dim=512, q_lora_rank=1024, o_lora_rank=1024 |
| FFN | Dense SwiGLU (intermediate=12288) | **DSparkMoE**: Gate + 256 个 top-k routed experts + Shared Expert |
| Position | Qwen3 RoPE (`rope_parameters`) | 自建 RoPE (无 Qwen3 依赖) |
| Residual | Standard residual | **Hyper-Connection**: hc_mult=4, Sinkhorn 20 轮 |
| HF 序列化 | Qwen3PreTrainedModel | PreTrainedModel + DeepseekV4Config |

### cann-recipes-infer_dspark 对齐

Attention、MoE、HC 三大模块已按 `cann-recipes-infer_dspark/models/deepseek-v4/` 重构，结构完全对齐生产推理代码，便于后续融合算子替换：

#### Hyper-Connection — `_OpKernel` 调度

```
_OpKernel.hc_pre / hc_post    ← 融合算子入口（生产: AscendC/PyPTO）
  └── hc_pre_native / hc_post_native  ← 纯 PyTorch fallback
        └── hc_split_sinkhorn          ← Sinkhorn 20 轮分解
```

| 参数 | Shape | 说明 |
|------|-------|------|
| `hc_attn_fn`, `hc_ffn_fn` | `(24, 16384)` | `mix_hc = (2+hc)*hc`，pre(4) + post(4) + comb(16) |
| `hc_attn_base`, `hc_ffn_base` | `(24,)` | 对齐生产 |
| `hc_attn_scale`, `hc_ffn_scale` | `(3,)` | pre/post/comb 各独立 scale |
| `hc_head_fn` (模型级) | `(4, 16384)` | sigmoid 门控降维，不含 Sinkhorn |

参考文件: `cann-recipes-infer_dspark/models/deepseek-v4/models/modules/op_impls/mhc.py`

#### DSparkAttention

| 方法 | 官方对应 | 说明 |
|------|---------|------|
| `_project_dspark_q(x, cos, sin)` | `_project_dspark_q` | wq_a → q_norm → wq_b → QK-norm → RoPE |
| `_project_dspark_kv(x, cos, sin)` | `_project_dspark_kv` | wkv → kv_norm → RoPE, 返回 `(kv_nope, kv_rope)` |
| `attn_sink` | `attn_sink` | `nn.Parameter(zeros(n_heads))` |
| Attention 计算 | `F.scaled_dot_product_attention` | 训练期 dense SDPA (生产用 sparse_attn kernel) |

参考文件: `cann-recipes-infer_dspark/models/deepseek-v4/models/dspark_modeling.py` → `DSparkAttention`

#### DSparkMoE

| 参数 | 官方 checkpoint key |
|------|-------------------|
| `gate` (Linear + correction bias) | `ffn.gate.weight` / `ffn.gate.e_score_correction_bias` |
| `experts.{0..255}.w1/w2/w3` | `ffn.experts.{0..255}.w1/w2/w3` |
| `shared_experts.w1/w2/w3` | `ffn.shared_experts.w1/w2/w3` |
| forward 签名 | `forward(hidden_states, is_prefill, cur_topk_list, input_ids, shared_expert_stream)` |

训练实现与推理侧保持相同的 scoring、`noaux_tc`/greedy top-k、概率归一化和
`routed_scaling_factor` 语义，仅将量化 GMM 与 EP dispatch/combine 替换为可微的
PyTorch 稀疏专家调度。DeepSeek-V4-Flash 每个 token 激活 6/256 个 routed experts，
并叠加 1 个 shared expert。

> **训练资源约束**：三个 DSpark MoE 层约含 194 亿参数，单 BF16 权重约
> 36.2 GiB；当前 `BF16Optimizer` 的 FP32 master 参数与 AdamW 状态会把仅
> MoE 的常驻状态提高到约 253 GiB（尚未包含梯度和激活）。完整 256-expert
> 训练不能使用单卡或 `no_shard`，需要多卡参数/优化器分片，并建议进一步接入
> 与推理侧一致的 Expert Parallel dispatch/combine。

DeepSeek-V4 DSpark 默认配置现使用 `sharding_strategy="full_shard"`。FSDP
包装后，每个 rank 会记录 draft model 的本地参数视图：

```text
[rank 0] FSDP draft parameter view: strategy=full_shard, sharded=yes,
local_total_numel=.../... (...%), local_trainable_numel=.../...,
local_parameter_gib=.../...
FSDP draft parameter sharding summary: strategy=full_shard, world_size=...,
pre_wrap_total_numel=..., sum_rank_local_numel=..., replication_factor=...
```

`sharded=yes` 且 `replication_factor` 接近 1 表示各 rank 合计约为一份完整参数；
若复制系数接近 `world_size`，则参数仍然是全量复制。

参考文件: `cann-recipes-infer_dspark/models/deepseek-v4/models/modeling_deepseek.py` → `DeepseekV3MoE`

### 官方 DSpark 简化项

| 组件 | 官方 | 当前 | 影响 |
|------|------|------|------|
| FFN 专家 | 256-expert MoE + FP4 | 256-expert top-k MoE + Shared Expert（BF16） | 计算语义对齐；未使用推理量化/GMM/EP 融合 |
| Attention | `sparse_attn` (KV 压缩 + top-512) | `F.scaled_dot_product_attention` (dense) | O(n²) vs O(n)，长序列变慢 |
| HC 融合算子 | AscendC `npu_hc_pre` / PyPTO | `hc_pre_native` (纯 PyTorch) | 同计算逻辑，性能差距 |
| KV Cache | Window + Compressor | 训练不用 | 无 |
| 量化 | FP8 K/V/Q, FP4 experts | bf16 全精度 | 训练不需要 |

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
| `modeling.py` | `DSparkAttention` (MQA + attn_sink)、`DSparkMoE` (Gate + dense + shared)、`_OpKernel` HC dispatch、`DeepSeekV4DSparkModel` |
| `config.py` | `build_draft_config` — 直接从 target config clone，保留原生维度 |

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
| `deepspec/modeling/dspark/deepseek_v4/modeling.py` | **新建** — `DSparkAttention` (MQA + attn_sink)、`DSparkMoE` (Gate + dense + shared expert)、`_OpKernel` HC dispatch、`DeepSeekV4DSparkModel` |
| `deepspec/modeling/dspark/deepseek_v4/config.py` | flex_attention → sdpa (NPU)；`build_draft_config` clone target config |
| `deepspec/modeling/dspark/common.py` | `_arange()` int32 索引；`create_noise_embed` IndexPut dtype 修复；`sample_anchor_positions` int32 anchors |
| `deepspec/trainer/base_trainer.py` | torch.compile NPU 跳过；单卡 FSDP 跳过；单卡 dist.barrier 保护 |
| `deepspec/trainer/dspark_trainer.py` | `DeepSeekV4DSparkTrainer.build_models()` 直接读 safetensors 权重 |
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
3. **FFN 简化**：Gate + dense SwiGLU 替 256-expert MoE，需后续实现 top-k routing
4. **HC 融合算子**：纯 PyTorch native 路径，生产环境替换 `_OpKernel` 为 AscendC/PyPTO 实现
5. **Attention**：Dense SDPA 替 `sparse_attn`，长序列性能较差
6. **`UserWarning: Cannot create tensor with interal format`**：`torch.full_like` 在 NPU 上的非关键告警
7. **权重加载**：仅加载 `embed_tokens` 和 `lm_head`，draft 层随机初始化，需完整真实 target cache 进行有效训练
