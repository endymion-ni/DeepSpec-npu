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
# Activation quantisation stubs (not called with BF16 weights)
# ---------------------------------------------------------------------------

def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Stub — BF16 path does not quantise activations."""
    raise RuntimeError("act_quant should not be called in BF16 inference mode")


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
):
    """Stub — BF16 path does not quantise activations."""
    raise RuntimeError("fp4_act_quant should not be called in BF16 inference mode")


# ---------------------------------------------------------------------------
# GEMM stubs (not called with BF16 weights)
# ---------------------------------------------------------------------------

def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
):
    """Stub — BF16 path uses F.linear."""
    raise RuntimeError("fp8_gemm should not be called in BF16 inference mode")


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
):
    """Stub — BF16 path uses F.linear."""
    raise RuntimeError("fp4_gemm should not be called in BF16 inference mode")


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
