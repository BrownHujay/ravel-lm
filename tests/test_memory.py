import torch

from ravel_lm.ravel_memory import (
    RavelLayerCache,
    _composite_key_dtype,
    causal_last_k_lookup,
    causal_sum_lookup,
)


def test_composite_key_dtype_uses_int32_only_when_safe():
    assert _composite_key_dtype(1, 2048, 3, 1024) == torch.int32
    assert _composite_key_dtype(1024, 65536, 64, 65536) == torch.int64


def test_causal_latest_exact():
    addr = torch.tensor([[[1], [2], [1], [3], [1]]], dtype=torch.long)
    payload = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
    vals, mask = causal_last_k_lookup(addr, payload, address_space=8, k=2)
    assert not mask[0, 0, 0, 0]
    assert vals[0, 2, 0, 0, 0].item() == 0.0  # previous address-1 at t=0
    assert vals[0, 4, 0, 0, 0].item() == 2.0  # latest previous address-1 at t=2
    assert vals[0, 4, 0, 1, 0].item() == 0.0  # second latest previous at t=0


def test_causal_latest_separate_reads_mask_and_payload_gradient():
    write = torch.tensor([[[1], [2], [1], [3], [2], [1]]], dtype=torch.long)
    read = torch.tensor([[[1], [1], [2], [1], [2], [3]]], dtype=torch.long)
    write_mask = torch.tensor([[True, True, False, True, True, True]])
    payload = torch.randn(1, 6, 1, 3, requires_grad=True)

    actual, actual_mask = causal_last_k_lookup(
        write,
        payload,
        read,
        address_space=8,
        k=2,
        write_mask=write_mask,
    )

    expected = torch.zeros_like(actual)
    expected_mask = torch.zeros_like(actual_mask)
    history: dict[int, list[int]] = {}
    for t in range(write.shape[1]):
        prior = history.get(int(read[0, t, 0]), [])
        for j, source_t in enumerate(reversed(prior[-2:])):
            expected[0, t, 0, j] = payload[0, source_t, 0]
            expected_mask[0, t, 0, j] = True
        if write_mask[0, t]:
            history.setdefault(int(write[0, t, 0]), []).append(t)

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_mask, expected_mask)

    weights = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * weights).sum(), payload, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), payload)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_causal_sum_exact():
    addr = torch.tensor([[[1], [2], [1], [1]]], dtype=torch.long)
    payload = torch.tensor([1.0, 10.0, 3.0, 7.0]).view(1, 4, 1, 1)
    vals, mask = causal_sum_lookup(addr, payload, address_space=8)
    assert vals[0, 0, 0, 0].item() == 0.0
    assert vals[0, 2, 0, 0].item() == 1.0
    assert vals[0, 3, 0, 0].item() == 4.0


def test_decode_cache_latest_matches_semantics():
    cache = RavelLayerCache.empty(1, 2, 16, 3, device="cpu", dtype=torch.float32)
    addrs = torch.tensor([[5, 7]])
    vals, mask = cache.read(addrs)
    assert not mask.any()
    payload = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    cache.write(addrs, payload)
    vals, mask = cache.read(addrs)
    assert mask.all()
    assert torch.allclose(vals, payload)
