from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ravel_lm.ravel_memory import causal_last_k_lookup
from ravel_lm.triton_memory import triton, triton_fused_latest1_linear


@pytest.mark.skipif(triton is None or not torch.cuda.is_available(), reason="requires Triton on CUDA/HIP")
def test_triton_latest1_matches_torch_forward_backward(monkeypatch):
    monkeypatch.setenv("RAVEL_ENABLE_TRITON_MEMORY", "1")
    torch.manual_seed(7)
    device = torch.device("cuda")
    B, T, C, D, A = 2, 17, 3, 5, 11
    write = torch.randint(0, A, (B, T, C), device=device, dtype=torch.long)
    read = torch.randint(0, A, (B, T, C), device=device, dtype=torch.long)
    payload = torch.randn(B, T, C, D, device=device, dtype=torch.float32, requires_grad=True)

    with torch.no_grad():
        ref_values, ref_mask = causal_last_k_lookup(
            write.cpu(),
            payload.detach().cpu().requires_grad_(True),
            read.cpu(),
            address_space=A,
            k=1,
        )

    values, mask = causal_last_k_lookup(write, payload, read, address_space=A, k=1)
    assert values.shape == (B, T, C, 1, D)
    assert mask.shape == (B, T, C, 1)
    torch.testing.assert_close(values.detach().cpu(), ref_values, atol=1e-5, rtol=1e-5)
    assert torch.equal(mask.cpu(), ref_mask)

    grad = torch.randn_like(values)
    loss = (values * grad).sum()
    loss.backward()

    payload_ref = payload.detach().cpu().requires_grad_(True)
    ref_values_2, _ = causal_last_k_lookup(
        write.cpu(),
        payload_ref,
        read.cpu(),
        address_space=A,
        k=1,
    )
    (ref_values_2 * grad.cpu()).sum().backward()
    torch.testing.assert_close(payload.grad.cpu(), payload_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(triton is None or not torch.cuda.is_available(), reason="requires Triton on CUDA/HIP")
def test_triton_fused_latest1_linear_matches_lookup_linear(monkeypatch):
    monkeypatch.setenv("RAVEL_ENABLE_TRITON_FUSED_MEMORY", "1")
    torch.manual_seed(8)
    device = torch.device("cuda")
    B, T, C, P, M, A = 2, 13, 3, 7, 17, 11
    write = torch.randint(0, A, (B, T, C), device=device, dtype=torch.long)
    read = torch.randint(0, A, (B, T, C), device=device, dtype=torch.long)
    payload = torch.randn(B, T, C, P, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(M, C * P, device=device, dtype=torch.float32, requires_grad=True)

    values, _ = causal_last_k_lookup(write, payload, read, address_space=A, k=1)
    ref = F.linear(values.reshape(B, T, C * P), weight)
    fused = triton_fused_latest1_linear(write, payload, read, weight, address_space=A)
    torch.testing.assert_close(fused, ref, atol=1e-5, rtol=1e-5)

    grad = torch.randn_like(fused)
    fused.backward(grad)
    payload_grad = payload.grad.detach().clone()
    weight_grad = weight.grad.detach().clone()

    payload.grad = None
    weight.grad = None
    ref.backward(grad)
    torch.testing.assert_close(payload_grad, payload.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weight_grad, weight.grad, atol=1e-5, rtol=1e-5)
