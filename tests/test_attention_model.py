import torch

from ravel_lm.attention_model import AttentionLM, CausalSelfAttention
from ravel_lm.config import RavelConfig


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


def test_attention_model_forward_backward_and_probe():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = AttentionLM(cfg, n_heads=4)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    out = model(x, y)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    out["loss"].backward()
    probe = model.probe(x)
    assert len(probe.attentions) == cfg.n_layers
    assert probe.attentions[0].shape == (2, 4, 16, 16)


def test_fused_attention_matches_probe_path():
    torch.manual_seed(1)
    attention = CausalSelfAttention(tiny_cfg(), n_heads=4).eval()
    x = torch.randn(2, 13, 32)
    fused = attention(x)
    explicit, weights = attention(x, return_attn=True)
    torch.testing.assert_close(fused, explicit, rtol=2e-5, atol=2e-6)
    assert torch.all(weights.triu(diagonal=1) == 0)
