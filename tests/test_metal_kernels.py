import math

import numpy as np
import pytest
import torch

from ravel_lm.ravel_memory import causal_last_k_lookup


Metal = pytest.importorskip("Metal")

from ravel_lm.metal_kernels import get_kernels, metal_available


pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal device is not available")


def test_metal_ravel_latest1_matches_torch_reference():
    rng = np.random.default_rng(0)
    B, T, C, D, A = 2, 7, 3, 4, 8
    write = rng.integers(0, A, size=(B, T, C), dtype=np.int32)
    read = rng.integers(0, A, size=(B, T, C), dtype=np.int32)
    payload = rng.normal(size=(B, T, C, D)).astype(np.float32)
    write_mask = rng.integers(0, 2, size=(B, T), dtype=np.uint8)

    kernels = get_kernels()
    ref_values, ref_mask = causal_last_k_lookup(
        torch.from_numpy(write).long(),
        torch.from_numpy(payload),
        torch.from_numpy(read).long(),
        address_space=A,
        k=1,
        write_mask=torch.from_numpy(write_mask.astype(bool)),
    )
    for implementation in ["auto", "scalar", "vector", "legacy"]:
        values, mask = kernels.ravel_latest1(
            write,
            payload,
            read,
            address_space=A,
            write_mask=write_mask,
            implementation=implementation,
        )
        assert np.array_equal(mask, ref_mask[..., 0].numpy())
        assert np.allclose(values, ref_values[..., 0, :].numpy(), atol=1e-6, rtol=1e-6)


def test_metal_ravel_scalar_payload_tail_matches_torch_reference():
    rng = np.random.default_rng(4)
    B, T, C, D, A = 1, 13, 2, 5, 16
    write = rng.integers(0, A, size=(B, T, C), dtype=np.int32)
    read = rng.integers(0, A, size=(B, T, C), dtype=np.int32)
    payload = rng.normal(size=(B, T, C, D)).astype(np.float32)

    values, mask = get_kernels().ravel_latest1(write, payload, read, address_space=A)
    ref_values, ref_mask = causal_last_k_lookup(
        torch.from_numpy(write).long(),
        torch.from_numpy(payload),
        torch.from_numpy(read).long(),
        address_space=A,
        k=1,
    )
    assert np.array_equal(mask, ref_mask[..., 0].numpy())
    assert np.allclose(values, ref_values[..., 0, :].numpy(), atol=1e-6, rtol=1e-6)


def test_metal_causal_softmax_attention_matches_torch_reference():
    rng = np.random.default_rng(1)
    B, H, T, D = 2, 2, 9, 8
    q = rng.normal(size=(B, H, T, D)).astype(np.float32)
    k = rng.normal(size=(B, H, T, D)).astype(np.float32)
    v = rng.normal(size=(B, H, T, D)).astype(np.float32)

    kernels = get_kernels()
    out = kernels.causal_softmax_attention(q, k, v)

    qt = torch.from_numpy(q)
    kt = torch.from_numpy(k)
    vt = torch.from_numpy(v)
    scores = torch.matmul(qt, kt.transpose(-2, -1)) / math.sqrt(D)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    ref = torch.softmax(scores, dim=-1).matmul(vt)

    assert np.allclose(out, ref.numpy(), atol=2e-5, rtol=2e-5)
