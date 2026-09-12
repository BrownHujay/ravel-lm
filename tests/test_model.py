import torch
import torch.nn.functional as F

from ravel_lm.config import RavelConfig
from ravel_lm.model import CausalDepthwiseConv1d, RMSNorm, RavelLM


def test_rms_norm_matches_float32_reference_and_gradient():
    torch.manual_seed(4)
    norm = RMSNorm(17)
    x = torch.randn(2, 11, 17, requires_grad=True)
    actual = norm(x)
    scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + norm.eps)
    expected = (x.float() * scale).to(x.dtype) * norm.weight
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    grad = torch.randn_like(actual)
    actual_x_grad = torch.autograd.grad(actual, x, grad, retain_graph=True)[0]
    expected_x_grad = torch.autograd.grad(expected, x, grad)[0]
    torch.testing.assert_close(actual_x_grad, expected_x_grad, atol=2e-6, rtol=1e-5)


def test_grouped_causal_conv_matches_unfold_reference():
    torch.manual_seed(11)
    layer = CausalDepthwiseConv1d(channels=5, kernel_size=4)
    x = torch.randn(2, 13, 5, requires_grad=True)
    actual = layer(x)

    padded = F.pad(x, (0, 0, 3, 0))
    windows = padded.unfold(1, 4, 1)
    expected = (windows * layer.weight.view(1, 1, 5, 4)).sum(dim=-1) + layer.bias
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad, retain_graph=True)
    actual_x_grad = x.grad.detach().clone()
    actual_w_grad = layer.weight.grad.detach().clone()
    x.grad = None
    layer.weight.grad = None
    expected.backward(grad)
    torch.testing.assert_close(actual_x_grad, x.grad)
    torch.testing.assert_close(actual_w_grad, layer.weight.grad)


def tiny_cfg():
    return RavelConfig(
        vocab_size=260,
        block_size=32,
        d_model=32,
        n_layers=2,
        mlp_mult=1.5,
        conv_kernel=3,
        address_space=512,
        n_literal_heads=2,
        n_learned_heads=1,
        n_codebooks=2,
        codebook_size=8,
        payload_dim=8,
        last_k=1,
        dropout=0.0,
        memory_every=1,
    )


def test_model_forward_backward():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = RavelLM(cfg)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    out = model(x, y)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_incremental_step_matches_full_forward_eval():
    torch.manual_seed(123)
    cfg = tiny_cfg()
    model = RavelLM(cfg).eval()
    x = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        full = model(x)["logits"]
        cache = model.init_cache(batch_size=2, device="cpu", dtype=torch.float32)
        steps = []
        for t in range(x.shape[1]):
            steps.append(model.forward_step(x[:, t], cache))
        stepped = torch.stack(steps, dim=1)
    assert torch.allclose(full, stepped, atol=1e-5, rtol=1e-4)


def test_generate_runs():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = RavelLM(cfg).eval()
    prompt = torch.tensor([[257, 79, 110, 99, 101]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=3, temperature=0.0, use_cache=True)
    assert out.shape == (1, 8)
