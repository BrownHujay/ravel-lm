import pytest
import torch
import torch.nn.functional as F

if not hasattr(torch.mps, "compile_shader"):
    pytest.skip("MPS shader compilation unavailable", allow_module_level=True)

from ravel_lm.mps_local import local_gate


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="MPS shader compilation unavailable",
)


def reference(uv, weight, bias):
    u, v = uv.chunk(2, dim=-1)
    x = F.pad(u.transpose(1, 2).contiguous(), (6, 0))
    conv = F.conv1d(x, weight[:, None, :], bias, groups=weight.shape[0])
    return F.silu(conv.transpose(1, 2)) * v


@pytest.mark.parametrize("shape", [(2, 1, 5), (2, 7, 17), (2, 33, 35), (1, 2048, 160)])
def test_forward_backward_matches_cpu(shape):
    torch.manual_seed(7)
    B, T, D = shape
    uv = torch.randn(B, T, D * 2, requires_grad=True)
    weight = (torch.randn(D, 7) * .03).requires_grad_()
    bias = (torch.randn(D) * .02).requires_grad_()
    grad = torch.randn(B, T, D)
    expected = reference(uv, weight, bias)
    expected_grads = torch.autograd.grad(expected, (uv, weight, bias), grad)
    inputs = tuple(x.detach().to("mps").requires_grad_() for x in (uv, weight, bias))
    actual = local_gate(*inputs)
    actual_grads = torch.autograd.grad(actual, inputs, grad.to("mps"))
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a.cpu(), e, atol=8e-5, rtol=3e-5)


def test_noncontiguous_inputs_and_upstream_gradient():
    torch.manual_seed(11)
    uv = torch.randn(2, 9, 68, device="mps")[:, :, ::2].requires_grad_()
    weight = torch.randn(7, 17, device="mps").T.requires_grad_()
    bias = torch.randn(34, device="mps")[::2].requires_grad_()
    grad = torch.randn(2, 9, 34, device="mps")[:, :, ::2]
    inputs = (uv, weight, bias)
    actual = local_gate(*inputs)
    expected = reference(*inputs)
    ag = torch.autograd.grad(actual, inputs, grad)
    eg = torch.autograd.grad(expected, inputs, grad)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=3e-5)
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, atol=3e-5, rtol=3e-5)


def test_compiled_matches_eager():
    torch.manual_seed(13)
    inputs = (torch.randn(2, 35, 64, device="mps", requires_grad=True),
              torch.randn(32, 7, device="mps", requires_grad=True),
              torch.randn(32, device="mps", requires_grad=True))
    grad = torch.randn(2, 35, 32, device="mps")
    expected = reference(*inputs)
    eg = torch.autograd.grad(expected, inputs, grad)
    compiled = torch.compile(local_gate, fullgraph=True, dynamic=False)
    actual = compiled(*inputs)
    ag = torch.autograd.grad(actual, inputs, grad)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, atol=5e-5, rtol=3e-5)


def test_hessian_vector_product_matches_reference():
    torch.manual_seed(17)
    cpu = (torch.randn(1, 9, 10).requires_grad_(),
           (torch.randn(5, 7) * .1).requires_grad_(),
           (torch.randn(5) * .1).requires_grad_())
    gpu = tuple(t.detach().to("mps").requires_grad_() for t in cpu)
    vectors = tuple(torch.randn_like(t) for t in cpu)

    def hvp(fn, inputs, vectors):
        loss = fn(*inputs).square().mean()
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        dot = sum((g * v).sum() for g, v in zip(first, vectors))
        return torch.autograd.grad(dot, inputs)

    expected = hvp(reference, cpu, vectors)
    actual = hvp(local_gate, gpu, tuple(v.to("mps") for v in vectors))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, atol=2e-5, rtol=3e-5)


def test_rejects_mixed_precision_and_odd_packing():
    uv = torch.randn(1, 9, 10, device="mps")
    weight = torch.randn(5, 7, device="mps")
    bias = torch.randn(5, device="mps")
    with pytest.raises(ValueError, match="FP32"):
        local_gate(uv, weight.half(), bias)
    with pytest.raises(ValueError, match="packed"):
        local_gate(torch.randn(1, 9, 11, device="mps"), weight, bias)
