from __future__ import annotations

from typing import Optional, Tuple

import os
import torch


def _load_triton():
    try:
        import triton
        import triton.language as tl
    except Exception:
        return None, None
    return triton, tl


triton, tl = _load_triton()


if triton is not None:

    @triton.jit
    def _latest1_forward_kernel(
        write_addr,
        payload,
        read_addr,
        out,
        out_mask,
        source_idx,
        T: tl.constexpr,
        C: tl.constexpr,
        D: tl.constexpr,
        ADDRESS_SPACE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        d_block = tl.program_id(1)

        c = row % C
        tmp = row // C
        t = tmp % T
        b = tmp // T

        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        latest = tl.zeros((BLOCK_D,), tl.float32)
        latest_t = tl.full((), -1, tl.int32)
        filled = False
        read_a = tl.load(read_addr + row) % ADDRESS_SPACE

        for offset in tl.range(1, T + 1):
            s = t - offset
            in_range = s >= 0
            addr_idx = (b * T + s) * C + c
            write_a = tl.load(write_addr + addr_idx, mask=in_range, other=-1) % ADDRESS_SPACE
            take = in_range & (write_a == read_a) & ~filled
            value_idx = addr_idx * D + d_offsets
            candidate = tl.load(payload + value_idx, mask=d_mask & take, other=0.0).to(tl.float32)
            latest = tl.where(take, candidate, latest)
            latest_t = tl.where(take, s, latest_t)
            filled = filled | take

        out_idx = row * D + d_offsets
        tl.store(out + out_idx, latest, mask=d_mask & filled)
        tl.store(out + out_idx, 0.0, mask=d_mask & ~filled)
        if d_block == 0:
            tl.store(out_mask + row, filled)
            tl.store(source_idx + row, latest_t)


    @triton.jit
    def _latest1_backward_kernel(
        grad_out,
        source_idx,
        grad_payload,
        N: tl.constexpr,
        T: tl.constexpr,
        C: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        d_block = tl.program_id(1)

        d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        src_t = tl.load(source_idx + row)
        valid = src_t >= 0

        c = row % C
        tmp = row // C
        t = tmp % T
        b = tmp // T

        grad_offsets = row * D + d_offsets
        src_row = (b * T + src_t) * C + c
        payload_offsets = src_row * D + d_offsets
        g = tl.load(grad_out + grad_offsets, mask=d_mask & valid, other=0.0)
        tl.atomic_add(grad_payload + payload_offsets, g, sem="relaxed", mask=d_mask & valid)


    @triton.jit
    def _latest1_fused_linear_forward_kernel(
        write_addr,
        payload,
        read_addr,
        weight,
        out,
        source_idx,
        T: tl.constexpr,
        C: tl.constexpr,
        P: tl.constexpr,
        M: tl.constexpr,
        ADDRESS_SPACE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        row = tl.program_id(0)
        m_block = tl.program_id(1)

        t = row % T
        b = row // T
        m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        p_offsets = tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P
        acc = tl.zeros((BLOCK_M,), tl.float32)

        for c in tl.range(0, C):
            addr_row = row * C + c
            read_a = tl.load(read_addr + addr_row) % ADDRESS_SPACE
            latest_t = tl.full((), -1, tl.int32)
            filled = False

            for offset in tl.range(1, T + 1):
                s = t - offset
                in_range = s >= 0
                write_row = (b * T + s) * C + c
                write_a = tl.load(write_addr + write_row, mask=in_range, other=-1) % ADDRESS_SPACE
                take = in_range & (write_a == read_a) & ~filled
                latest_t = tl.where(take, s, latest_t)
                filled = filled | take

            if m_block == 0:
                tl.store(source_idx + addr_row, latest_t)

            payload_base = ((b * T + latest_t) * C + c) * P
            vals = tl.load(payload + payload_base + p_offsets, mask=p_mask & filled, other=0.0).to(tl.float32)
            w = tl.load(
                weight + m_offsets[:, None] * (C * P) + c * P + p_offsets[None, :],
                mask=m_mask[:, None] & p_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * vals[None, :], axis=1)

        tl.store(out + row * M + m_offsets, acc, mask=m_mask)


    @triton.jit
    def _latest1_fused_linear_backward_payload_kernel(
        grad_out,
        source_idx,
        weight,
        grad_payload,
        T: tl.constexpr,
        C: tl.constexpr,
        P: tl.constexpr,
        M: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        row_c = tl.program_id(0)
        p_block = tl.program_id(1)

        c = row_c % C
        row = row_c // C
        t = row % T
        b = row // T
        src_t = tl.load(source_idx + row * C + c)
        valid = src_t >= 0
        p_offsets = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P
        acc = tl.zeros((BLOCK_P,), tl.float32)

        for m0 in tl.range(0, M, BLOCK_M):
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            m_mask = m_offsets < M
            go = tl.load(grad_out + row * M + m_offsets, mask=m_mask, other=0.0).to(tl.float32)
            w = tl.load(
                weight + m_offsets[:, None] * (C * P) + c * P + p_offsets[None, :],
                mask=m_mask[:, None] & p_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(go[:, None] * w, axis=0)

        payload_row = (b * T + src_t) * C + c
        tl.atomic_add(
            grad_payload + payload_row * P + p_offsets,
            acc,
            sem="relaxed",
            mask=p_mask & valid,
        )


    @triton.jit
    def _latest1_fused_linear_backward_weight_kernel(
        grad_out,
        source_idx,
        payload,
        grad_weight,
        N: tl.constexpr,
        T: tl.constexpr,
        C: tl.constexpr,
        P: tl.constexpr,
        M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_CP: tl.constexpr,
    ):
        cp_block = tl.program_id(0)
        m_block = tl.program_id(1)
        n_block = tl.program_id(2)

        cp_offsets = cp_block * BLOCK_CP + tl.arange(0, BLOCK_CP)
        cp_mask = cp_offsets < (C * P)
        c_offsets = cp_offsets // P
        p_offsets = cp_offsets - c_offsets * P
        rows = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        src_t = tl.load(
            source_idx + rows[:, None] * C + c_offsets[None, :],
            mask=row_mask[:, None] & cp_mask[None, :],
            other=-1,
        )
        valid = row_mask[:, None] & cp_mask[None, :] & (src_t >= 0)
        b = rows // T
        payload_idx = ((b[:, None] * T + src_t) * C + c_offsets[None, :]) * P + p_offsets[None, :]
        vals = tl.load(payload + payload_idx, mask=valid, other=0.0).to(tl.float32)
        go = tl.load(
            grad_out + rows[:, None] * M + m_offsets[None, :],
            mask=row_mask[:, None] & m_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        partial = tl.dot(tl.trans(go), vals)
        tl.atomic_add(
            grad_weight + m_offsets[:, None] * (C * P) + cp_offsets[None, :],
            partial,
            sem="relaxed",
            mask=m_mask[:, None] & cp_mask[None, :],
        )


    @triton.jit
    def _latest1_source_gather_kernel(
        source_idx,
        payload,
        read_flat,
        T: tl.constexpr,
        C: tl.constexpr,
        P: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        row_c = tl.program_id(0)
        p_block = tl.program_id(1)
        c = row_c % C
        row = row_c // C
        b = row // T
        p_offsets = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P
        src_t = tl.load(source_idx + row * C + c)
        valid = src_t >= 0
        vals = tl.load(
            payload + ((b * T + src_t) * C + c) * P + p_offsets,
            mask=p_mask & valid,
            other=0.0,
        )
        tl.store(read_flat + row * (C * P) + c * P + p_offsets, vals, mask=p_mask)


    @triton.jit
    def _latest1_readgrad_scatter_kernel(
        grad_read,
        source_idx,
        grad_payload,
        T: tl.constexpr,
        C: tl.constexpr,
        P: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        row_c = tl.program_id(0)
        p_block = tl.program_id(1)
        c = row_c % C
        row = row_c // C
        b = row // T
        p_offsets = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P
        src_t = tl.load(source_idx + row * C + c)
        valid = src_t >= 0
        g = tl.load(grad_read + row * (C * P) + c * P + p_offsets, mask=p_mask, other=0.0)
        tl.atomic_add(
            grad_payload + ((b * T + src_t) * C + c) * P + p_offsets,
            g,
            sem="relaxed",
            mask=p_mask & valid,
        )


class _TritonLatest1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        write_addresses: torch.Tensor,
        payloads: torch.Tensor,
        read_addresses: torch.Tensor,
        address_space: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if triton is None:
            raise RuntimeError("triton is not available")
        if payloads.ndim != 4 or write_addresses.ndim != 3 or read_addresses.ndim != 3:
            raise ValueError("expected write/read [B,T,C] and payloads [B,T,C,D]")
        B, T, C, D = payloads.shape
        if write_addresses.shape != (B, T, C) or read_addresses.shape != (B, T, C):
            raise ValueError("address tensors must have shape [B,T,C]")
        if not payloads.is_cuda:
            raise RuntimeError("Triton latest1 requires a CUDA/HIP tensor")
        if not payloads.is_contiguous():
            payloads = payloads.contiguous()
        write_i64 = write_addresses.contiguous()
        read_i64 = read_addresses.contiguous()

        out = torch.empty_like(payloads)
        out_mask = torch.empty((B, T, C), device=payloads.device, dtype=torch.bool)
        source_idx = torch.empty((B, T, C), device=payloads.device, dtype=torch.int32)
        block_d = triton.next_power_of_2(D)
        block_d = min(max(block_d, 1), 64)
        grid = (B * T * C, triton.cdiv(D, block_d))
        _latest1_forward_kernel[grid](
            write_i64,
            payloads,
            read_i64,
            out,
            out_mask,
            source_idx,
            T,
            C,
            D,
            int(address_space),
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(source_idx)
        ctx.shape = (B, T, C, D)
        ctx.block_d = block_d
        return out, out_mask

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_mask: Optional[torch.Tensor]):
        (source_idx,) = ctx.saved_tensors
        B, T, C, D = ctx.shape
        grad_payload = torch.empty((B, T, C, D), device=grad_out.device, dtype=grad_out.dtype)
        grad_payload.zero_()
        grid = (B * T * C, triton.cdiv(D, ctx.block_d))
        _latest1_backward_kernel[grid](
            grad_out.contiguous(),
            source_idx,
            grad_payload,
            B * T * C,
            T,
            C,
            D,
            BLOCK_D=ctx.block_d,
        )
        return None, grad_payload, None, None


class _TritonLatest1FusedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        write_addresses: torch.Tensor,
        payloads: torch.Tensor,
        read_addresses: torch.Tensor,
        weight: torch.Tensor,
        address_space: int,
    ) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("triton is not available")
        if payloads.ndim != 4 or write_addresses.ndim != 3 or read_addresses.ndim != 3:
            raise ValueError("expected write/read [B,T,C] and payloads [B,T,C,P]")
        B, T, C, P = payloads.shape
        M = weight.shape[0]
        if weight.shape != (M, C * P):
            raise ValueError("weight must have shape [M, C * P]")
        if not payloads.is_cuda:
            raise RuntimeError("fused Triton latest1 requires a CUDA/HIP tensor")
        payloads = payloads.contiguous()
        weight = weight.contiguous()
        write_i64 = write_addresses.contiguous()
        read_i64 = read_addresses.contiguous()
        out = torch.empty((B, T, M), device=payloads.device, dtype=payloads.dtype)
        source_idx = torch.empty((B, T, C), device=payloads.device, dtype=torch.int32)
        block_p = min(max(triton.next_power_of_2(P), 1), 64)
        block_m = 128
        grid = (B * T, triton.cdiv(M, block_m))
        _latest1_fused_linear_forward_kernel[grid](
            write_i64,
            payloads,
            read_i64,
            weight,
            out,
            source_idx,
            T,
            C,
            P,
            M,
            int(address_space),
            BLOCK_P=block_p,
            BLOCK_M=block_m,
        )
        ctx.save_for_backward(source_idx, payloads, weight)
        ctx.shape = (B, T, C, P, M)
        ctx.block_p = block_p
        ctx.block_m = block_m
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        source_idx, payloads, weight = ctx.saved_tensors
        B, T, C, P, M = ctx.shape
        grad_out = grad_out.contiguous()
        N = B * T
        read_flat = torch.empty((N, C * P), device=payloads.device, dtype=payloads.dtype)
        grid_read = (N * C, triton.cdiv(P, ctx.block_p))
        _latest1_source_gather_kernel[grid_read](
            source_idx,
            payloads,
            read_flat,
            T,
            C,
            P,
            BLOCK_P=ctx.block_p,
        )

        grad_2d = grad_out.reshape(N, M)
        grad_weight = grad_2d.t().matmul(read_flat)
        grad_read = grad_2d.matmul(weight).contiguous()

        grad_payload = torch.empty_like(payloads)
        grad_payload.zero_()
        _latest1_readgrad_scatter_kernel[grid_read](
            grad_read,
            source_idx,
            grad_payload,
            T,
            C,
            P,
            BLOCK_P=ctx.block_p,
        )
        return None, grad_payload, None, grad_weight, None


def triton_causal_latest1_lookup(
    write_addresses: torch.Tensor,
    payloads: torch.Tensor,
    read_addresses: Optional[torch.Tensor] = None,
    *,
    address_space: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if read_addresses is None:
        read_addresses = write_addresses
    values, mask = _TritonLatest1.apply(write_addresses, payloads, read_addresses, int(address_space))
    return values.unsqueeze(3), mask.unsqueeze(3)


def triton_fused_latest1_linear(
    write_addresses: torch.Tensor,
    payloads: torch.Tensor,
    read_addresses: torch.Tensor,
    weight: torch.Tensor,
    *,
    address_space: int,
) -> torch.Tensor:
    return _TritonLatest1FusedLinear.apply(write_addresses, payloads, read_addresses, weight, int(address_space))


def can_use_triton_latest1(
    write_addresses: torch.Tensor,
    payloads: torch.Tensor,
    read_addresses: Optional[torch.Tensor],
    *,
    k: int,
    write_mask: Optional[torch.Tensor],
) -> bool:
    if os.environ.get("RAVEL_DISABLE_TRITON_MEMORY", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("RAVEL_ENABLE_TRITON_MEMORY", "").lower() not in {"1", "true", "yes"}:
        return False
    return (
        triton is not None
        and k == 1
        and write_mask is None
        and payloads.is_cuda
        and write_addresses.is_cuda
        and (read_addresses is None or read_addresses.is_cuda)
        and payloads.ndim == 4
        and write_addresses.ndim == 3
    )


def can_use_triton_fused_latest1(
    write_addresses: torch.Tensor,
    payloads: torch.Tensor,
    read_addresses: torch.Tensor,
    weight: torch.Tensor,
    *,
    write_mask: Optional[torch.Tensor],
    use_sum_read: bool,
    k: int,
) -> bool:
    if os.environ.get("RAVEL_DISABLE_TRITON_MEMORY", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("RAVEL_ENABLE_TRITON_FUSED_MEMORY", "").lower() not in {"1", "true", "yes"}:
        return False
    try:
        if torch.compiler.is_compiling():
            return False
    except Exception:
        pass
    return (
        triton is not None
        and k == 1
        and not use_sum_read
        and write_mask is None
        and payloads.is_cuda
        and write_addresses.is_cuda
        and read_addresses.is_cuda
        and weight.is_cuda
        and payloads.ndim == 4
        and write_addresses.ndim == 3
        and read_addresses.ndim == 3
        and weight.ndim == 2
    )
