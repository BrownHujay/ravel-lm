"""Triton local_gate for CUDA/ROCm: silu(causal_conv7(u)) * v.

Forward is a single fused Triton kernel (replaces the pad + grouped-conv +
chunk + silu chain the fallback path dispatches). Backward composes standard
GPU ops with the conv_grad staged once — the same math as the Metal backward
in ``metal/local_gate.metal``. A fully fused Triton backward is future work;
it must be parity-tested on-device before it can be trusted.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # Triton is only present on CUDA/ROCm installs.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _lg_fwd_kernel(
        uv_ptr, w_ptr, b_ptr, out_ptr, conv_ptr,
        T, D,
        BLOCK_D: tl.constexpr,
    ):
        pid_row = tl.program_id(0)  # one program per (batch * T) row
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        b = pid_row // T
        t = pid_row % T

        z = tl.load(b_ptr + offs_d, mask=mask_d, other=0.0)
        for k in tl.static_range(7):
            s = t + k - 6
            u = tl.load(
                uv_ptr + (b * T + s) * 2 * D + offs_d,
                mask=mask_d & (s >= 0), other=0.0,
            )
            w = tl.load(w_ptr + offs_d * 7 + k, mask=mask_d, other=0.0)
            z += u * w
        v = tl.load(uv_ptr + (b * T + t) * 2 * D + D + offs_d, mask=mask_d, other=0.0)
        tl.store(conv_ptr + pid_row * D + offs_d, z, mask=mask_d)
        out = (z / (1.0 + tl.exp(-z))) * v
        tl.store(out_ptr + pid_row * D + offs_d, out, mask=mask_d)


if _HAS_TRITON:

    @triton.jit
    def _lg_bwd_g_kernel(
        uv_ptr, conv_ptr, gy_ptr, g_ptr, guv_ptr,
        T, D,
        BLOCK_D: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        pos = pid_row * D + offs_d
        z = tl.load(conv_ptr + pos, mask=mask_d, other=0.0)
        gy = tl.load(gy_ptr + pos, mask=mask_d, other=0.0)
        v = tl.load(uv_ptr + pid_row * 2 * D + D + offs_d, mask=mask_d, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-z))
        g = gy * v * sig * (1.0 + z * (1.0 - sig))
        tl.store(g_ptr + pos, g, mask=mask_d)
        tl.store(guv_ptr + pid_row * 2 * D + D + offs_d, gy * z * sig, mask=mask_d)

    @triton.jit
    def _lg_bwd_u_kernel(
        uv_ptr, w_ptr, g_ptr, guv_ptr, dw_ptr, db_ptr,
        T, D,
        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_d = tl.program_id(1)
        b = tl.program_id(2)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_t = offs_t < T
        mask_d = offs_d < D
        m2 = mask_t[:, None] & mask_d[None, :]

        g_here = tl.load(g_ptr + (b * T + offs_t)[:, None] * D + offs_d[None, :], mask=m2, other=0.0)
        db_acc = tl.sum(g_here, axis=0)
        gin = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
        for k in tl.static_range(7):
            w = tl.load(w_ptr + offs_d * 7 + k, mask=mask_d, other=0.0)
            fut = offs_t + 6 - k
            g_fut = tl.load(
                g_ptr + (b * T + fut)[:, None] * D + offs_d[None, :],
                mask=(fut < T)[:, None] & mask_d[None, :], other=0.0,
            )
            gin += g_fut * w[None, :]
            src = offs_t + k - 6
            u = tl.load(
                uv_ptr + (b * T + src)[:, None] * 2 * D + offs_d[None, :],
                mask=(src >= 0)[:, None] & (src < T)[:, None] & mask_d[None, :], other=0.0,
            )
            tl.atomic_add(dw_ptr + offs_d * 7 + k, tl.sum(g_here * u, axis=0), mask=mask_d)
        tl.store(guv_ptr + (b * T + offs_t)[:, None] * 2 * D + offs_d[None, :], gin, mask=m2)
        tl.atomic_add(db_ptr + offs_d, db_acc, mask=mask_d)


def can_use_triton_local_gate(uv: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _HAS_TRITON
        and uv.is_cuda
        and uv.dtype == torch.float32
        and weight.ndim == 2
        and weight.shape[-1] == 7
        and uv.ndim == 3
        and uv.shape[-1] % 2 == 0
        and uv.shape[-1] // 2 == weight.shape[0]
    )


class _TritonLocalGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uv, weight, bias):
        B, T, DD = uv.shape
        D = DD // 2
        uv = uv.contiguous()
        out = uv.new_empty(B, T, D)
        conv = uv.new_empty(B, T, D)
        BLOCK_D = 64
        grid = (B * T, triton.cdiv(D, BLOCK_D))
        _lg_fwd_kernel[grid](
            uv, weight.contiguous(), bias.contiguous(), out, conv, T, D, BLOCK_D=BLOCK_D
        )
        ctx.save_for_backward(uv, weight, conv)
        return out

    @staticmethod
    def backward(ctx, gy):
        uv, weight, conv = ctx.saved_tensors
        B, T, DD = uv.shape
        D = DD // 2
        gy = gy.contiguous()
        guv = torch.empty_like(uv)
        g = torch.empty(B, T, D, device=uv.device, dtype=uv.dtype)
        dw = torch.zeros_like(weight)
        db = weight.new_zeros(D)
        BLOCK_D = 64
        _lg_bwd_g_kernel[(B * T, triton.cdiv(D, BLOCK_D))](
            uv, conv, gy, g, guv, T, D, BLOCK_D=BLOCK_D
        )
        BLOCK_T = 64
        _lg_bwd_u_kernel[(triton.cdiv(T, BLOCK_T), triton.cdiv(D, BLOCK_D), B)](
            uv, weight.contiguous(), g, guv, dw, db, T, D,
            BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        )
        return guv, dw, db


def triton_local_gate(uv: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """silu(causal_depthwise_conv7(u)) * v with a fused Triton forward."""
    return _TritonLocalGate.apply(uv, weight, bias)
