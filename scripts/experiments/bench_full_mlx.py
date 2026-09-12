"""Exact FP32 RAVEL graph experiment; compare full training across runtimes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.runtime import compiled_mps_clip_grad_norm_


def forward(p, frozen, tokens, cfg):
    B, T = tokens.shape
    D = cfg.d_model

    def linear(x, prefix):
        y = x @ p[prefix + ".weight"].T
        if prefix + ".bias" in p:
            y = y + p[prefix + ".bias"]
        return y

    def norm(x, prefix):
        return mx.fast.rms_norm(x, p[prefix + ".weight"], 1e-6)

    def silu(x):
        return x * mx.sigmoid(x)

    prev = mx.concatenate([mx.full((B, 1), cfg.bos_token_id), tokens[:, :-1]], axis=1)
    literal = mx.stack([tokens % cfg.address_space,
                        (prev * 257 + tokens * 131 + 17) % cfg.address_space], axis=-1)
    C = cfg.n_memory_heads
    group = (mx.arange(B)[:, None, None] * C + mx.arange(C)[None, None, :]) * cfg.address_space
    times = mx.arange(T)[None, :, None]
    x = p["tok_emb.weight"][tokens] + p["pos_emb.weight"][mx.arange(T)][None, :, :]
    for layer in range(cfg.n_layers):
        prefix = f"blocks.{layer}"
        xn = norm(x, prefix + ".local.norm")
        uv = linear(xn, prefix + ".local.in_proj")
        u, v = mx.split(uv, 2, axis=-1)
        conv_p = prefix + ".local.conv"
        padded = mx.pad(u, [(0, 0), (cfg.conv_kernel - 1, 0), (0, 0)])
        u = mx.conv1d(padded, p[conv_p + ".weight"][:, :, None], groups=D)
        u = u + p[conv_p + ".bias"]
        x = x + linear(silu(u) * v, prefix + ".local.out_proj")

        mp = prefix + ".memory"
        xn = norm(x, mp + ".norm")

        def address(name):
            logits = xn @ frozen[mp + f".{name}_addr.proj.weight"].T
            logits = logits.reshape(B, T, cfg.n_learned_heads, cfg.n_codebooks, cfg.codebook_size)
            codes = mx.argmax(logits, axis=-1).astype(mx.int32)
            multipliers = mx.array([cfg.codebook_size**j for j in range(cfg.n_codebooks)])
            return mx.stop_gradient(mx.concatenate([literal, mx.sum(codes * multipliers, axis=-1) % cfg.address_space], axis=-1))

        write = address("write")
        read = address("read")
        keys = ((group + write) * (T + 1) + times).reshape(-1)
        order = mx.argsort(keys)
        sorted_keys = keys[order]
        read_base = (group + read).reshape(-1)
        queries = ((group + read) * (T + 1) + times - 1).reshape(-1)
        index = mx.searchsorted(sorted_keys, queries, side="right").astype(mx.int32) - 1
        safe = mx.maximum(index, 0)
        valid = (index >= 0) & ((sorted_keys[safe] // (T + 1)) == read_base)
        payload = linear(xn, mp + ".payload").reshape(-1, cfg.payload_dim)
        values = payload[order[safe]] * valid[:, None]
        fused = linear(values.reshape(B, T, -1), mp + ".fuse")
        x = x + mx.sigmoid(linear(xn, mp + ".gate")) * fused

        fp = prefix + ".ffn"
        a, b = mx.split(linear(norm(x, fp + ".norm"), fp + ".w12"), 2, axis=-1)
        x = x + linear(silu(a) * b, fp + ".w3")
    return norm(x, "norm") @ p["tok_emb.weight"].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mx.set_cache_limit(128 * 1024 * 1024)
    torch.manual_seed(2026)
    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.block_size = 2048
    cfg.batch_size = 1
    assert cfg.n_literal_heads == 2 and cfg.last_k == 1 and not cfg.use_sum_read
    assert cfg.dropout == 0 and cfg.memory_every == 1 and cfg.tie_weights
    model = RavelLM(cfg).to("mps")
    for block in model.blocks:
        block.local.use_mps_local = False
    parameters = {}
    frozen = {}
    for name, p in model.named_parameters():
        dest = parameters if p.requires_grad else frozen
        dest[name] = mx.array(p.detach().cpu().numpy())
    x = torch.randint(cfg.vocab_size, (1, 2048), device="mps")
    y = torch.randint(cfg.vocab_size, (1, 2048), device="mps")
    xm, ym = mx.array(x.cpu().numpy().astype(np.int32)), mx.array(y.cpu().numpy().astype(np.int32))

    def loss(p, x, y):
        logits = forward(p, frozen, x, cfg)
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)
        mask = y != cfg.pad_token_id
        return mx.sum(mx.where(mask, nll, 0)) / mx.sum(mask)

    value_grad = mx.value_and_grad(loss)
    print("Checking logits, loss, and all gradients", flush=True)
    mlogits = forward(parameters, frozen, xm, cfg)
    mloss, mg = value_grad(parameters, xm, ym)
    mx.eval(mlogits, mloss, mg)
    reference = model(x, y)
    reference["loss"].backward()
    logits_error = float(np.max(np.abs(np.array(mlogits) - reference["logits"].detach().cpu().numpy())))
    gradient_error = max(float(np.max(np.abs(np.array(mg[n]) - p.grad.cpu().numpy())))
                         for n, p in model.named_parameters() if p.requires_grad)
    print("parity", logits_error, gradient_error, float(mloss), reference["loss"].item(), flush=True)
    if logits_error > 1e-4 or gradient_error > 1e-4:
        raise RuntimeError("Cross-runtime parity failed")

    lr, wd = 4e-4, cfg.weight_decay
    moment = {n: mx.zeros_like(p) for n, p in parameters.items()}
    variance = {n: mx.zeros_like(p) for n, p in parameters.items()}

    @mx.compile
    def train_step(p, m, v, step, x, y):
        value, grads = value_grad(p, x, y)
        norm = mx.sqrt(sum(mx.sum(g * g) for g in grads.values()))
        scale = mx.minimum(1 / (norm + 1e-6), 1.)
        grads = {n: g * scale for n, g in grads.items()}
        m = {n: .9 * m[n] + .1 * grads[n] for n in p}
        v = {n: .999 * v[n] + .001 * grads[n] ** 2 for n in p}
        bc1 = 1 - .9**step
        bc2 = 1 - .999**step
        p = {n: p[n] * (1 - lr * wd) - (lr / bc1) * m[n] / (mx.sqrt(v[n] / bc2) + 1e-8) for n in p}
        return value, p, m, v

    result = {"shape": {"batch": 1, "tokens": 2048, "params": model.num_parameters, "dtype": "float32"},
              "parity": {"logits_max_abs": logits_error, "gradient_max_abs": gradient_error}}
    samples = []
    for step in range(args.steps):
        start = time.perf_counter()
        value, parameters, moment, variance = train_step(parameters, moment, variance, mx.array(step + 1), xm, ym)
        mx.eval(value, parameters, moment, variance)
        elapsed = 1000 * (time.perf_counter() - start)
        if step >= args.warmup:
            samples.append(elapsed)
        if step in {0, args.warmup, args.steps - 1}:
            print("mlx", step, elapsed, float(value), flush=True)
    result["mlx"] = {"median_ms": statistics.median(samples), "samples_ms": samples, "final_loss": float(value)}

    compiled = torch.compile(model, fullgraph=True, dynamic=False, mode="reduce-overhead")
    ps = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(ps, lr=lr, weight_decay=wd, fused=True)
    samples = []
    for step in range(args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        value = compiled(x, y)["loss"]
        value.backward()
        compiled_mps_clip_grad_norm_(ps, 1.)
        optimizer.step()
        torch.mps.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        if step >= args.warmup:
            samples.append(elapsed)
        if step in {0, args.warmup, args.steps - 1}:
            print("torch", step, elapsed, value.item(), flush=True)
    result["torch"] = {"median_ms": statistics.median(samples), "samples_ms": samples, "final_loss": value.item()}
    print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
