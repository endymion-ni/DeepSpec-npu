"""CPU fallback implementations for DeepSeek-V4 inference/kernel.py.

These replace the tilelang CUDA kernels with pure-PyTorch (eager) equivalents.
When weights are dequantised to BF16 the ``act_quant`` / ``fp8_gemm`` / ``fp4_gemm``
paths are never hit; only ``sparse_attn`` and ``hc_split_sinkhorn`` are required.

Usage::

    import sys
    sys.modules["kernel"] = deepspec.modeling.deepseek_v4.kernel_cpu
"""

from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Activation quantisation (CPU — pure PyTorch)
# ---------------------------------------------------------------------------

# FP4 lookup table (matches inference/convert.py)
_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

_FP8_MAX = 448.0


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Per-block FP8 quantisation (CPU fallback).

    Returns ``(x_fp8, scale)`` where *x_fp8* has dtype ``float8_e4m3fn``
    and *scale* has dtype *scale_dtype* (or ``float8_e8m0fnu`` when
    *scale_fmt* is ``"ue8m0"``).
    """
    N = x.shape[-1]
    assert N % block_size == 0, f"last dim {N} not divisible by {block_size}"
    x_flat = x.reshape(-1, N)
    M = x_flat.shape[0]
    nb = N // block_size

    x_blocks = x_flat.view(M, nb, block_size)
    amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-4)  # (M, nb)
    scale = amax / _FP8_MAX

    if scale_fmt == "ue8m0":
        # Round to power-of-2, store as E8M0
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale.float())))
        scale_s = scale.to(torch.float8_e8m0fnu)
    else:
        scale_s = scale.to(scale_dtype)

    x_q = (x_blocks / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    x_q = x_q.to(torch.float8_e4m3fn).view(M, N)

    if inplace:
        x_dq = (x_q.float() * scale.unsqueeze(-1)).view(M, N).to(x.dtype)
        x.copy_(x_dq.reshape_as(x))
        return x

    return x_q.reshape(*x.shape[:-1], N), scale_s


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
):
    """Per-block FP4 quantisation (CPU fallback)."""
    N = x.shape[-1]
    assert N % block_size == 0
    x_flat = x.reshape(-1, N)
    M = x_flat.shape[0]
    nb = N // block_size

    x_blocks = x_flat.view(M, nb, block_size)
    amax = x_blocks.abs().amax(dim=-1).clamp(min=6.0 * (2 ** -126))
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0))).to(torch.float8_e8m0fnu)
    x_q = (x_blocks / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    x_q = x_q.to(torch.float4_e2m1fn_x2).view(M, N // 2)

    if inplace:
        x_dq = (x_q.float() * scale.unsqueeze(-1)).view(M, N).to(x.dtype)
        x.copy_(x_dq.reshape_as(x))
        return x

    return x_q.reshape(*x.shape[:-1], N // 2), scale


# ---------------------------------------------------------------------------
# GEMM (CPU — dequantise + torch.matmul)
# ---------------------------------------------------------------------------

def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """FP8×FP8 GEMM (CPU fallback): dequantise both → matmul → bf16.

    *a* : (M, K)  float8_e4m3fn
    *a_s* : (M, ceil(K/128))  scale (float32 or float8_e8m0fnu)
    *b* : (N, K)  float8_e4m3fn
    *b_s* : (ceil(N/128), ceil(K/128))  scale
    """
    M, K = a.shape
    N = b.shape[0]
    bs = 128

    # Dequantise A: (M, K)
    a_s2 = a_s.float().repeat_interleave(bs, dim=1)[:, :K]
    a_dq = a.float() * a_s2

    # Dequantise B: (N, K)
    b_s2 = b_s.float().repeat_interleave(bs, dim=0)[:N].repeat_interleave(bs, dim=1)[:, :K]
    b_dq = b.float() * b_s2

    return torch.mm(a_dq, b_dq.T).to(torch.get_default_dtype())


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """FP8×FP4 GEMM (CPU fallback): dequantise both → matmul → bf16.

    *a* : (M, K)  float8_e4m3fn
    *a_s* : (M, ceil(K/128))  scale
    *b* : (N, K//2)  float4_e2m1fn_x2  (packed, 2 FP4 per byte)
    *b_s* : (N, ceil(K/32))  scale (float8_e8m0fnu)
    """
    M, K = a.shape
    N = b.shape[0]
    K_logical = b.shape[1] * 2  # unpacked

    # Dequantise A
    a_s2 = a_s.float().repeat_interleave(128, dim=1)[:, :K]
    a_dq = a.float() * a_s2

    # Unpack FP4: each uint8 → two 4-bit values → lookup table
    w_u8 = b.view(torch.uint8)
    low = w_u8 & 0x0F
    high = (w_u8 >> 4) & 0x0F
    tbl = _FP4_TABLE.to(device=b.device)
    b_dq = torch.stack([tbl[low.long()], tbl[high.long()]], dim=-1)
    b_dq = b_dq.reshape(N, K_logical).float()

    # Dequantise B: (N, K_logical)
    b_s2 = b_s.float().repeat_interleave(32, dim=1)[:, :K_logical]
    b_dq = b_dq * b_s2

    # Truncate to match K if K_logical > K
    if K_logical > K:
        b_dq = b_dq[:, :K]

    return torch.mm(a_dq, b_dq.T).to(torch.get_default_dtype())


# ---------------------------------------------------------------------------
# Sparse attention (CPU fallback)
# ---------------------------------------------------------------------------

def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Index-gather + online-softmax sparse attention (pure PyTorch).

    Parameters
    ----------
    q :  (B, M, H, D)   query
    kv : (B, N, D)       key-value (shared)
    attn_sink : (H,)     per-head learnable sink bias
    topk_idxs : (B, M, K)  indices of top-K KV positions per query
    softmax_scale : float
    """
    B, M, H, D = q.shape
    K = topk_idxs.shape[-1]

    # Gather KV for each query position: (B, M, K, D)
    kv_gathered = kv[torch.arange(B, device=q.device).unsqueeze(-1).unsqueeze(-1),
                     topk_idxs]  # (B, M, K, D)

    # Einsum attention: Q @ K^T -> (B, M, H, K)
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), kv_gathered.float())
    scores = scores * softmax_scale

    # Add sink bias
    scores = scores + attn_sink[None, None, :, None].float()

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1).to(q.dtype)

    # Weighted sum: (B, M, H, D)
    out = torch.einsum("bmhk,bmkd->bmhd", attn_weights, kv_gathered)
    return out


# ---------------------------------------------------------------------------
# Hyper-Connection Sinkhorn (CPU fallback)
# ---------------------------------------------------------------------------

def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Sinkhorn normalisation for hyper-connection mixing (pure PyTorch).

    Parameters
    ----------
    mixes : (B, S, mix_hc)  where mix_hc = (2 + hc_mult) * hc_mult
    hc_scale : (3,)
    hc_base : (mix_hc,)
    hc_mult : int
    sinkhorn_iters : int
    eps : float

    Returns
    -------
    pre  : (B, S, hc_mult)
    post : (B, S, hc_mult)
    comb : (B, S, hc_mult, hc_mult)
    """
    B, S, _ = mixes.shape

    # pre: sigmoid of first hc_mult channels
    pre = torch.sigmoid(
        mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    ) + eps

    # post: 2 * sigmoid of next hc_mult channels
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    )

    # comb: last hc_mult * hc_mult channels → matrix
    comb_raw = mixes[..., 2 * hc_mult:]  # (B, S, hc_mult * hc_mult)
    comb_base = hc_base[2 * hc_mult:]    # (hc_mult * hc_mult,)
    comb = (comb_raw * hc_scale[2] + comb_base).view(B, S, hc_mult, hc_mult)

    # Softmax over last dim, then Sinkhorn
    comb = torch.softmax(comb, dim=-1) + eps

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb
