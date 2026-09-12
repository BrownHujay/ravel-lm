"""Fused MPS latest-1 event-memory lookup.

Replaces the searchsorted + validate + gather + mask chain of
``causal_last_k_lookup`` (k=1, unmasked) with one Metal kernel after a single
``torch.sort``. Values are bit-identical to the torch path (pure gather); the
backward scatter-add uses float atomics, the same nondeterminism class as the
``index_add_`` issued by the torch path's ``index_select`` backward.
"""
from __future__ import annotations

from functools import lru_cache

import torch

Tensor = torch.Tensor

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// One thread per read element (b, t, c): binary-search the sorted write keys
// for the latest same-stream write strictly before t, then gather its payload.
kernel void latest1_search_gather(
    device const int* sorted_keys [[buffer(0)]],
    device const int* order [[buffer(1)]],
    device const int* read_base [[buffer(2)]],
    device const float* payload [[buffer(3)]],
    device float* out [[buffer(4)]],
    device int* src [[buffer(5)]],
    device bool* mask [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& D [[buffer(8)]],
    constant uint& Tplus1 [[buffer(9)]],
    constant uint& T [[buffer(10)]],
    constant uint& C [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    // read key = base * (T+1) + (t - 1); t = (gid / C) % T
    uint t = (gid / C) % T;
    int base = read_base[gid];
    int key = base * int(Tplus1) + int(t) - 1;
    // upper_bound(sorted_keys, key) - 1
    uint lo = 0, hi = N;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (sorted_keys[mid] <= key) lo = mid + 1; else hi = mid;
    }
    int idx = int(lo) - 1;
    bool valid = idx >= 0 && (sorted_keys[idx] / int(Tplus1)) == base;
    int source = valid ? order[idx] : -1;
    src[gid] = source;
    mask[gid] = valid;
    device float* dst = out + gid * D;
    if (valid) {
        device const float* row = payload + uint(source) * D;
        for (uint d = 0; d < D; ++d) dst[d] = row[d];
    } else {
        for (uint d = 0; d < D; ++d) dst[d] = 0.0f;
    }
}

// One thread per (read element, channel): atomically accumulate grad into the
// source payload row.
kernel void latest1_scatter_grad(
    device const int* src [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic_float* grad_payload [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    uint n = gid / D;
    if (n >= N) return;
    int source = src[n];
    if (source < 0) return;
    uint d = gid % D;
    float g = grad_out[n * D + d];
    if (g != 0.0f) {
        atomic_fetch_add_explicit(grad_payload + uint(source) * D + d, g, memory_order_relaxed);
    }
}
"""


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SRC)


def can_use_mps_latest1(
    write_addresses: Tensor,
    payloads: Tensor,
    read_addresses: Tensor,
    *,
    address_space: int,
    k: int,
    write_mask,
) -> bool:
    if k != 1 or write_mask is not None:
        return False
    if payloads.device.type != "mps" or payloads.dtype != torch.float32:
        return False
    if not hasattr(torch.mps, "compile_shader"):
        return False
    B, T, C = write_addresses.shape
    key_bound = B * C * int(address_space) * (T + 1)
    return key_bound <= torch.iinfo(torch.int32).max


@torch.library.custom_op("ravel::mps_latest1", mutates_args=())
def _latest1_forward(
    write_addresses: Tensor,
    payloads: Tensor,
    read_addresses: Tensor,
    address_space: int,
) -> tuple[Tensor, Tensor, Tensor]:
    B, T, C = write_addresses.shape
    D = payloads.shape[-1]
    device = write_addresses.device
    N = B * T * C

    group = (
        torch.arange(B * C, device=device, dtype=torch.int32).view(B, 1, C)
        * int(address_space)
    )
    write_base = group + write_addresses.to(torch.int32)
    read_base = (group + read_addresses.to(torch.int32)).reshape(-1).contiguous()
    times = torch.arange(T, device=device, dtype=torch.int32).view(1, T, 1)
    write_keys = (write_base * (T + 1) + times).reshape(-1)
    sorted_keys, order = torch.sort(write_keys)
    order = order.to(torch.int32)

    payload_flat = payloads.reshape(N, D).contiguous()
    out = payloads.new_empty(B, T, C, D)
    src = torch.empty(N, device=device, dtype=torch.int32)
    mask = torch.empty(N, device=device, dtype=torch.bool)
    _library().latest1_search_gather(
        sorted_keys, order, read_base, payload_flat, out, src, mask,
        N, D, T + 1, T, C,
        threads=N, group_size=256,
    )
    return out, src, mask.view(B, T, C)


@_latest1_forward.register_fake
def _fake_forward(write_addresses, payloads, read_addresses, address_space):
    B, T, C = write_addresses.shape
    D = payloads.shape[-1]
    return (
        payloads.new_empty(B, T, C, D),
        torch.empty(B * T * C, device=payloads.device, dtype=torch.int32),
        torch.empty(B, T, C, device=payloads.device, dtype=torch.bool),
    )


@torch.library.custom_op("ravel::mps_latest1_backward", mutates_args=())
def _latest1_backward(src: Tensor, grad_out: Tensor, n_rows: int, payload_dim: int) -> Tensor:
    grad_payload = grad_out.new_zeros(n_rows, payload_dim)
    N = src.numel()
    _library().latest1_scatter_grad(
        src, grad_out.contiguous(), grad_payload,
        N, payload_dim,
        threads=N * payload_dim, group_size=256,
    )
    return grad_payload


@_latest1_backward.register_fake
def _fake_backward(src, grad_out, n_rows, payload_dim):
    return grad_out.new_empty(n_rows, payload_dim)


def _setup_context(ctx, inputs, output):
    write_addresses, payloads, read_addresses, address_space = inputs
    _, src, mask = output
    ctx.save_for_backward(src)
    ctx.payload_shape = payloads.shape
    ctx.mark_non_differentiable(src, mask)


def _autograd_backward(ctx, grad_out, _grad_src, _grad_mask):
    (src,) = ctx.saved_tensors
    B, T, C, D = ctx.payload_shape
    grad_flat = _latest1_backward(src, grad_out, B * T * C, D)
    return None, grad_flat.view(B, T, C, D), None, None


_latest1_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


def mps_latest1_lookup(
    write_addresses: Tensor,
    payloads: Tensor,
    read_addresses: Tensor,
    *,
    address_space: int,
) -> tuple[Tensor, Tensor]:
    """Exact latest-1 lookup; returns values [B,T,C,1,D] and mask [B,T,C,1]."""
    out, _, mask = _latest1_forward(write_addresses, payloads, read_addresses, int(address_space))
    return out.unsqueeze(3), mask.unsqueeze(3)
