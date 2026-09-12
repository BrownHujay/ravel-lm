from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .mps_memory import can_use_mps_latest1, mps_latest1_lookup
from .triton_latest1 import can_use_triton_latest1_v2, triton_latest1_v2_lookup
from .triton_memory import can_use_triton_latest1, triton_causal_latest1_lookup


Tensor = torch.Tensor


@torch.no_grad()
def _check_int_addresses(addresses: Tensor, address_space: int, name: str) -> None:
    if addresses.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError(f"{name} must be an integer tensor, got {addresses.dtype}")
    if torch.is_floating_point(addresses):
        raise TypeError(f"{name} must not be floating point")
    if addresses.numel() and (addresses.min() < 0 or addresses.max() >= address_space):
        raise ValueError(f"{name} values must be in [0, {address_space})")


def _composite_base(addresses: Tensor, address_space: int) -> Tensor:
    """Return a collision-free integer base id for (batch, memory_head, address).

    addresses: [B, T, C]
    base id:   [B, T, C] = (batch*C + head) * address_space + address
    """
    B, T, C = addresses.shape
    device = addresses.device
    dtype = addresses.dtype
    head_ids = torch.arange(C, device=device, dtype=dtype).view(1, 1, C)
    batch_ids = torch.arange(B, device=device, dtype=dtype).view(B, 1, 1)
    group = batch_ids * C + head_ids
    return group * int(address_space) + addresses


def _time_grid(B: int, T: int, C: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.arange(T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, C)


def _composite_key_dtype(B: int, T: int, C: int, address_space: int) -> torch.dtype:
    # Include one extra time slot because keys use a (T + 1) radix. The int64
    # fallback keeps large-address or large-batch configurations exact.
    key_bound = int(B) * int(C) * int(address_space) * (int(T) + 1)
    return torch.int32 if key_bound <= torch.iinfo(torch.int32).max else torch.int64


def causal_last_k_lookup(
    write_addresses: Tensor,
    payloads: Tensor,
    read_addresses: Optional[Tensor] = None,
    *,
    address_space: int,
    k: int = 1,
    write_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Exact causal last-k lookup over an event tape.

    This is the core RAVEL primitive used for exact training. It creates a virtual
    event tape of writes ``(batch, head, address, time, payload)`` and, for every
    read at time ``t``, returns the latest ``k`` payloads with the same
    ``(batch, head, address)`` and time ``< t``.

    No attention matrix is built. The implementation uses a fixed-width composite
    integer key and ``torch.sort`` + ``torch.searchsorted``. On GPU this maps to
    batchable sort/search/gather kernels; the fusion after this primitive is dense
    GEMM in ``model.py``.

    Args:
        write_addresses: [B, T, C] integer addresses for writes.
        payloads:        [B, T, C, D] payload tensor.
        read_addresses:  [B, T, C] integer read addresses. Defaults to writes.
        address_space:   number of legal addresses per memory head.
        k:               number of latest records to return.
        write_mask:      optional [B, T] boolean mask; false entries do not write.

    Returns:
        values: [B, T, C, k, D]
        mask:   [B, T, C, k] true where a causal record existed.
    """
    if read_addresses is None:
        read_addresses = write_addresses
    if write_addresses.shape != read_addresses.shape:
        raise ValueError("write_addresses and read_addresses must have the same shape")
    if write_addresses.ndim != 3:
        raise ValueError("addresses must have shape [B, T, C]")
    if payloads.ndim != 4:
        raise ValueError("payloads must have shape [B, T, C, D]")
    B, T, C = write_addresses.shape
    if payloads.shape[:3] != (B, T, C):
        raise ValueError("payloads leading dimensions must match addresses")
    if k <= 0:
        raise ValueError("k must be positive")
    if can_use_triton_latest1_v2(
        write_addresses, payloads, read_addresses,
        address_space=address_space, k=k, write_mask=write_mask,
    ):
        return triton_latest1_v2_lookup(
            write_addresses,
            payloads,
            read_addresses if read_addresses is not None else write_addresses,
            address_space=address_space,
        )
    if can_use_triton_latest1(write_addresses, payloads, read_addresses, k=k, write_mask=write_mask):
        return triton_causal_latest1_lookup(
            write_addresses,
            payloads,
            read_addresses,
            address_space=address_space,
        )
    if can_use_mps_latest1(
        write_addresses, payloads, read_addresses,
        address_space=address_space, k=k, write_mask=write_mask,
    ):
        return mps_latest1_lookup(
            write_addresses,
            payloads,
            read_addresses,
            address_space=address_space,
        )

    key_dtype = _composite_key_dtype(B, T, C, address_space)
    write_addresses = write_addresses.to(key_dtype).remainder(int(address_space))
    read_addresses = read_addresses.to(key_dtype).remainder(int(address_space))
    D = payloads.shape[-1]
    device = write_addresses.device

    # Composite key = ((batch/head/address) * (T+1)) + time.
    write_base = _composite_base(write_addresses, address_space)
    times = _time_grid(B, T, C, device, key_dtype)
    write_keys = write_base * (T + 1) + times

    flat_keys = write_keys.reshape(-1)
    flat_payloads = payloads.reshape(-1, D)
    if write_mask is not None:
        if write_mask.shape != (B, T):
            raise ValueError("write_mask must have shape [B, T]")
        flat_valid = write_mask.bool().view(B, T, 1).expand(B, T, C).reshape(-1)
        flat_keys = flat_keys[flat_valid]
        flat_payloads = flat_payloads[flat_valid]

    if flat_keys.numel() == 0:
        out = payloads.new_zeros(B, T, C, k, D)
        valid = torch.zeros(B, T, C, k, device=device, dtype=torch.bool)
        return out, valid

    sorted_keys, order = torch.sort(flat_keys)

    read_base = _composite_base(read_addresses, address_space)
    # Strictly causal: at time t, search key ends at t-1.
    read_keys = read_base * (T + 1) + (times - 1)
    flat_read_keys = read_keys.reshape(-1)
    flat_read_base = read_base.reshape(-1)

    latest_idx = torch.searchsorted(sorted_keys, flat_read_keys, right=True) - 1
    offsets = torch.arange(k, device=device, dtype=torch.long).view(1, k)
    candidate_idx = latest_idx.reshape(-1, 1) - offsets
    candidate_valid = candidate_idx >= 0
    safe_idx = candidate_idx.clamp_min(0)

    candidate_keys = sorted_keys.index_select(0, safe_idx.reshape(-1)).reshape(-1, k)
    candidate_base = candidate_keys // (T + 1)
    same_record_stream = candidate_base == flat_read_base.reshape(-1, 1)
    valid = candidate_valid & same_record_stream

    # Carry source indices through the key search and gather payloads once. The
    # former implementation first permuted the complete [B*T*C,D] payload tape,
    # then gathered from that copy, adding a dense gather and backward scatter.
    source_idx = order.index_select(0, safe_idx.reshape(-1))
    gathered = flat_payloads.index_select(0, source_idx).reshape(-1, k, D)
    gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
    return gathered.reshape(B, T, C, k, D), valid.reshape(B, T, C, k)


def causal_sum_lookup(
    write_addresses: Tensor,
    payloads: Tensor,
    read_addresses: Optional[Tensor] = None,
    *,
    address_space: int,
    write_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Exact causal segmented prefix-sum lookup over the event tape.

    For every read at time ``t``, returns the sum of all payloads with matching
    ``(batch, head, address)`` and time ``< t``. This is differentiable w.r.t.
    payloads and is useful for count/sum-style memory experiments.
    """
    if read_addresses is None:
        read_addresses = write_addresses
    if write_addresses.ndim != 3 or payloads.ndim != 4:
        raise ValueError("expected write_addresses [B,T,C] and payloads [B,T,C,D]")
    B, T, C = write_addresses.shape
    D = payloads.shape[-1]
    device = write_addresses.device
    key_dtype = _composite_key_dtype(B, T, C, address_space)
    write_addresses = write_addresses.to(key_dtype).remainder(int(address_space))
    read_addresses = read_addresses.to(key_dtype).remainder(int(address_space))

    write_base = _composite_base(write_addresses, address_space)
    times = _time_grid(B, T, C, device, key_dtype)
    write_keys = write_base * (T + 1) + times
    flat_keys = write_keys.reshape(-1)
    flat_payloads = payloads.reshape(-1, D)
    if write_mask is not None:
        flat_valid = write_mask.bool().view(B, T, 1).expand(B, T, C).reshape(-1)
        flat_keys = flat_keys[flat_valid]
        flat_payloads = flat_payloads[flat_valid]
    if flat_keys.numel() == 0:
        return payloads.new_zeros(B, T, C, D), torch.zeros(B, T, C, device=device, dtype=torch.bool)

    sorted_keys, order = torch.sort(flat_keys)
    sorted_payloads = flat_payloads.index_select(0, order)
    sorted_base = sorted_keys // (T + 1)

    # Segmented cumulative sum: global cumsum minus cumsum just before segment start.
    is_start = torch.ones_like(sorted_base, dtype=torch.bool)
    is_start[1:] = sorted_base[1:] != sorted_base[:-1]
    global_cumsum = torch.cumsum(sorted_payloads, dim=0)
    start_indices = torch.nonzero(is_start, as_tuple=False).flatten()
    seg_ids = torch.cumsum(is_start.long(), dim=0) - 1
    before_start = global_cumsum.new_zeros(start_indices.numel(), D)
    nonzero_start = start_indices > 0
    before_start[nonzero_start] = global_cumsum.index_select(0, start_indices[nonzero_start] - 1)
    segmented_cumsum = global_cumsum - before_start.index_select(0, seg_ids)

    read_base = _composite_base(read_addresses, address_space)
    read_keys = read_base * (T + 1) + (times - 1)
    flat_read_keys = read_keys.reshape(-1)
    flat_read_base = read_base.reshape(-1)
    idx = torch.searchsorted(sorted_keys, flat_read_keys, right=True) - 1
    valid = idx >= 0
    safe_idx = idx.clamp_min(0)
    candidate_base = sorted_base.index_select(0, safe_idx)
    valid = valid & (candidate_base == flat_read_base)
    gathered = segmented_cumsum.index_select(0, safe_idx) * valid.unsqueeze(-1).to(payloads.dtype)
    return gathered.reshape(B, T, C, D), valid.reshape(B, T, C)


@dataclass
class RavelLayerCache:
    """O(address_space * heads * payload_dim) latest-value cache for decoding.

    This cache implements the same latest-record semantics as
    ``causal_last_k_lookup(..., k=1)`` for incremental generation. It is deliberately
    tensorized: reads use ``gather`` and writes use ``scatter_``.
    """

    values: Tensor
    filled: Tensor

    @classmethod
    def empty(
        cls,
        batch_size: int,
        n_heads: int,
        address_space: int,
        payload_dim: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "RavelLayerCache":
        values = torch.zeros(batch_size, n_heads, address_space, payload_dim, device=device, dtype=dtype)
        filled = torch.zeros(batch_size, n_heads, address_space, device=device, dtype=torch.bool)
        return cls(values=values, filled=filled)

    def read(self, addresses: Tensor) -> Tuple[Tensor, Tensor]:
        """Read latest values for addresses [B, C]. Returns ([B,C,D], [B,C])."""
        B, C = addresses.shape
        if self.values.shape[:2] != (B, C):
            raise ValueError(f"cache shape {self.values.shape[:2]} does not match addresses {(B, C)}")
        idx = addresses.long().clamp_min(0).clamp_max(self.values.shape[2] - 1)
        gather_idx = idx.view(B, C, 1, 1).expand(B, C, 1, self.values.shape[-1])
        vals = torch.gather(self.values, 2, gather_idx).squeeze(2)
        mask = torch.gather(self.filled, 2, idx.view(B, C, 1)).squeeze(2)
        vals = vals * mask.unsqueeze(-1).to(vals.dtype)
        return vals, mask

    def write(self, addresses: Tensor, payloads: Tensor) -> None:
        """Write latest payloads. addresses [B,C], payloads [B,C,D]."""
        B, C = addresses.shape
        D = payloads.shape[-1]
        if self.values.shape[:2] != (B, C) or self.values.shape[-1] != D:
            raise ValueError("cache and payload shapes do not match")
        idx = addresses.long().clamp_min(0).clamp_max(self.values.shape[2] - 1)
        scatter_idx = idx.view(B, C, 1, 1).expand(B, C, 1, D)
        self.values.scatter_(2, scatter_idx, payloads.unsqueeze(2))
        self.filled.scatter_(2, idx.view(B, C, 1), torch.ones(B, C, 1, device=idx.device, dtype=torch.bool))
