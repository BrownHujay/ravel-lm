import pytest
import torch
import torch.nn.functional as F

from ravel_lm.triton_local import can_use_triton_local_gate, triton_local_gate

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")


def _reference(uv, weight, bias):
    D = uv.shape[-1] // 2
    u, v = uv[..., :D], uv[..., D:]
    windows = F.pad(u, (0, 0, 6, 0)).unfold(1, 7, 1)
    z = (windows * weight).sum(dim=-1) + bias
    return F.silu(z) * v


@cuda
def test_forward_matches_reference():
    torch.manual_seed(0)
    uv = torch.randn(2, 512, 320, device="cuda")
    w = torch.randn(160, 7, device="cuda")
    b = torch.randn(160, device="cuda")
    assert can_use_triton_local_gate(uv, w)
    out = triton_local_gate(uv, w, b)
    ref = _reference(uv, w, b)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@cuda
def test_backward_matches_reference():
    torch.manual_seed(1)
    uv = torch.randn(1, 256, 128, device="cuda", requires_grad=True)
    w = torch.randn(64, 7, device="cuda", requires_grad=True)
    b = torch.randn(64, device="cuda", requires_grad=True)
    gy = torch.randn(1, 256, 64, device="cuda")

    triton_local_gate(uv, w, b).backward(gy)
    guv, gw, gb = uv.grad.clone(), w.grad.clone(), b.grad.clone()
    uv.grad = w.grad = b.grad = None

    _reference(uv, w, b).backward(gy)
    torch.testing.assert_close(guv, uv.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gw, w.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gb, b.grad, rtol=1e-4, atol=1e-4)


def test_gate_declines_cpu():
    uv = torch.randn(1, 8, 16)
    w = torch.randn(8, 7)
    assert not can_use_triton_local_gate(uv, w)
