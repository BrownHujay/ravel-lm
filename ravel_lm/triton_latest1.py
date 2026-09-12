"""RDNA4-friendly Triton latest-1 lookup: sort in torch, one fused kernel after.

The June kernels in ``triton_memory.py`` are sweep-based and measured 2-3x
slower than the plain torch path on RX 9070 XT / Windows. This module uses the
design that won on MPS: ``torch.sort`` on composite int32 keys (microseconds on
this card), then a single kernel that binary-searches, validates the stream,
and gathers the payload row; backward is a single atomic scatter-add. Values
and mask are bit-identical to the torch path (pure gather); the backward's
float atomics are the same nondeterminism class as ``index_add_``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

Tensor = torch.Tensor


if _HAS_TRITON:

    @triton.jit
    def _search_gather_kernel(
        sorted_keys_ptr, order_ptr, read_base_ptr, payload_ptr,
        out_ptr, src_ptr, mask_ptr,
        N, D, Tplus1, T, C,
        BLOCK_D: tl.constexpr,
    ):
        gid = tl.program_id(0)  # one program per read element (b, t, c)
        t = (gid // C) % T
        base = tl.load(read_base_ptr + gid)
        key = base * Tplus1 + t - 1
        lo = 0
        hi = N
        while lo < hi:
            mid = (lo + hi) // 2
            k = tl.load(sorted_keys_ptr + mid)
            take = k <= key
            lo = tl.where(take, mid + 1, lo)
            hi = tl.where(take, hi, mid)
        idx = lo - 1
        cand = tl.load(sorted_keys_ptr + tl.maximum(idx, 0))
        valid = (idx >= 0) & (cand // Tplus1 == base)
        source = tl.where(valid, tl.load(order_ptr + tl.maximum(idx, 0)), -1)
        tl.store(src_ptr + gid, source)
        tl.store(mask_ptr + gid, valid.to(tl.int8))
        offs = tl.arange(0, BLOCK_D)
        dmask = offs < D
        row = tl.load(payload_ptr + tl.maximum(source, 0) * D + offs,
                      mask=dmask & valid, other=0.0)
        tl.store(out_ptr + gid * D + offs, row, mask=dmask)

    @triton.jit
    def _scatter_grad_kernel(
        src_ptr, grad_out_ptr, grad_payload_ptr,
        N, D,
        BLOCK_D: tl.constexpr,
    ):
        gid = tl.program_id(0)
        source = tl.load(src_ptr + gid)
        if source >= 0:
            offs = tl.arange(0, BLOCK_D)
            dmask = offs < D
            g = tl.load(grad_out_ptr + gid * D + offs, mask=dmask, other=0.0)
            tl.atomic_add(grad_payload_ptr + source * D + offs, g, mask=dmask)


def can_use_triton_latest1_v2(
    write_addresses: Tensor,
    payloads: Tensor,
    read_addresses: Optional[Tensor],
    *,
    address_space: int,
    k: int,
    write_mask,
) -> bool:
    if not (_HAS_TRITON and k == 1 and write_mask is None):
        return False
    if not (payloads.is_cuda and payloads.dtype == torch.float32):
        return False
    B, T, C = write_addresses.shape
    return B * C * int(address_space) * (T + 1) <= torch.iinfo(torch.int32).max


class _Latest1V2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, write_addresses, payloads, read_addresses, address_space):
        B, T, C = write_addresses.shape
        D = payloads.shape[-1]
        device = payloads.device
        N = B * T * C

        group = (
            torch.arange(B * C, device=device, dtype=torch.int32).view(B, 1, C)
            * int(address_space)
        )
        write_base = group + write_addresses.to(torch.int32)
        read_base = (group + read_addresses.to(torch.int32)).reshape(-1).contiguous()
        times = torch.arange(T, device=device, dtype=torch.int32).view(1, T, 1)
        sorted_keys, order = torch.sort((write_base * (T + 1) + times).reshape(-1))
        order = order.to(torch.int32)

        payload_flat = payloads.reshape(N, D).contiguous()
        out = payloads.new_empty(B, T, C, D)
        src = torch.empty(N, device=device, dtype=torch.int32)
        mask = torch.empty(N, device=device, dtype=torch.int8)
        BLOCK_D = max(16, triton.next_power_of_2(D))
        _search_gather_kernel[(N,)](
            sorted_keys, order, read_base, payload_flat, out, src, mask,
            N, D, T + 1, T, C, BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(src)
        ctx.payload_shape = (B, T, C, D)
        return out, src, mask.view(B, T, C).to(torch.bool)

    @staticmethod
    def backward(ctx, grad_out, _grad_src, _grad_mask):
        (src,) = ctx.saved_tensors
        B, T, C, D = ctx.payload_shape
        N = B * T * C
        grad_payload = grad_out.new_zeros(N, D)
        BLOCK_D = max(16, triton.next_power_of_2(D))
        _scatter_grad_kernel[(N,)](
            src, grad_out.contiguous(), grad_payload, N, D, BLOCK_D=BLOCK_D,
        )
        return None, grad_payload.view(B, T, C, D), None, None


def triton_latest1_v2_lookup(
    write_addresses: Tensor,
    payloads: Tensor,
    read_addresses: Tensor,
    *,
    address_space: int,
) -> Tuple[Tensor, Tensor]:
    """Exact latest-1 lookup; returns values [B,T,C,1,D] and mask [B,T,C,1]."""
    out, _, mask = _Latest1V2.apply(write_addresses, payloads, read_addresses, int(address_space))
    return out.unsqueeze(3), mask.unsqueeze(3)
