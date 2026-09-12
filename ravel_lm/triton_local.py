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
        z = conv
        sig = torch.sigmoid(z)
        v = uv[..., D:]
        # conv_grad staged once, reused by all three gradient paths.
        g = gy * v * sig * (1.0 + z * (1.0 - sig))
        gv = gy * z * sig
        future = F.pad(g, (0, 0, 0, 6))
        gu = sum(future[:, 6 - k : 6 - k + T] * weight[:, k] for k in range(7))
        windows = F.pad(uv[..., :D], (0, 0, 6, 0)).unfold(1, 7, 1)
        dw = (windows * g.unsqueeze(-1)).sum((0, 1))
        db = g.sum((0, 1))
        return torch.cat([gu, gv], dim=-1), dw, db


def triton_local_gate(uv: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """silu(causal_depthwise_conv7(u)) * v with a fused Triton forward."""
    return _TritonLocalGate.apply(uv, weight, bias)
