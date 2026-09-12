#!/usr/bin/env python3
"""CHIME-X v1 on RAVEL: age-decorated reads for the literal-2 config.

Arm 'chime2': n_literal_heads=2, n_learned_heads=0, plus per-record read
features [valid mask, near flag, log2-age norm, mantissa, 2 harmonics]
concatenated into the fuse input. Same data protocol as routing_experiment.py
=> directly comparable to yesterday's arms (literal-2 baseline: 1.5038).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/Users/kotnewm/Documents/GitHub/ravel_lm_project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM, RavelMemoryLayer, literal_addresses_from_tokens
from ravel_lm.mps_memory import _latest1_forward
from ravel_lm.runtime import FlatAdamW

import routing_experiment as R

N_AGE_FEAT = 6  # mask, near, u_norm, mantissa, sin, cos


class ChimeMemoryLayer(RavelMemoryLayer):
    """Literal-only memory layer whose fuse sees CHIME-X v1 age features."""

    def forward(self, x, token_ids, write_mask=None, literal_addr=None):
        B, T, D = x.shape
        xn = self.norm(x)
        C = self.cfg.n_memory_heads
        packed = F.linear(xn, torch.cat([self.payload.weight, self.gate.weight], dim=0))
        n_pay = self.payload.weight.shape[0]
        payload = packed[..., :n_pay].reshape(B, T, C, self.cfg.payload_dim)
        gate_lin = packed[..., n_pay:] + self.gate.bias

        if literal_addr is None:
            literal_addr = literal_addresses_from_tokens(
                token_ids, address_space=self.cfg.address_space,
                n_heads=self.cfg.n_literal_heads, bos_token_id=self.cfg.bos_token_id,
            )
        values, src, mask = _latest1_forward(
            literal_addr, payload, literal_addr, int(self.cfg.address_space)
        )
        # CHIME-X v1 read pack from exact write times.
        src = src.view(B, T, C).long()
        t_write = (src.clamp_min(0) // C) % T
        t_read = torch.arange(T, device=x.device).view(1, T, 1)
        delta = (t_read - t_write).clamp_min(1).float()
        m = mask.view(B, T, C).float()
        u = torch.log2(1.0 + delta)
        feats = torch.stack([
            m,
            (delta <= 8).float() * m,
            (u / 12.0) * m,
            (u - u.floor()) * m,
            torch.sin(2 * math.pi * 0.5 * u) * m,
            torch.cos(2 * math.pi * 0.5 * u) * m,
        ], dim=-1).reshape(B, T, C * N_AGE_FEAT)

        read_flat = torch.cat([values.reshape(B, T, -1), feats], dim=-1)
        fused = self.fuse(read_flat)
        return x + self.dropout(torch.sigmoid(gate_lin) * fused)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "chime2"
    use_chime = variant in ("chime2", "nope_chime")
    use_pos = variant == "chime2"
    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.block_size = R.T
    cfg.batch_size = 1
    cfg.n_learned_heads = 0
    cfg.validate()
    torch.manual_seed(0)
    model = RavelLM(cfg).to(R.DEV)
    if not use_pos:
        model.pos_emb.weight.data.zero_()
        model.pos_emb.weight.requires_grad_(False)
    C = cfg.n_memory_heads
    for block in model.blocks:
        if block.memory is not None and use_chime:
            block.memory.__class__ = ChimeMemoryLayer
            old = block.memory.fuse
            new = nn.Linear(old.in_features + C * N_AGE_FEAT, old.out_features, bias=False).to(R.DEV)
            nn.init.normal_(new.weight, mean=0.0, std=0.02)
            block.memory.fuse = new

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"params: {sum(p.numel() for p in params):,}", flush=True)
    opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=R.LR, betas=(0.9, 0.95))
    exec_model = torch.compile(model, backend="inductor", mode="reduce-overhead", fullgraph=True, dynamic=False)

    train_tokens = R.load_bytes(R.TRAIN_CORPUS)
    eval_tokens = R.load_bytes(R.EVAL_CORPUS)
    offsets = R.make_offsets(len(train_tokens), R.STEPS, seed=2026)
    eval_offsets = R.make_offsets(len(eval_tokens), R.EVAL_WINDOWS, seed=7)

    curve = []
    t0 = time.perf_counter()
    for step in range(R.STEPS):
        lr = R.lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = R.batch_at(train_tokens, offsets[step])
        opt.zero_grad()
        loss = exec_model(x, y)["loss"]
        loss.backward()
        opt.clip_grad_norm_(1.0)
        opt.step()
        if (step + 1) % R.EVAL_EVERY == 0 or step == R.STEPS - 1:
            nll = R.eval_nll(exec_model, eval_tokens, eval_offsets)
            curve.append({"step": step + 1, "eval_nll": nll})
            print(f"[{variant}] step {step+1:5d} eval_nll {nll:.4f} train {float(loss):.4f}", flush=True)

    out = Path(__file__).parent / "routing_experiment_results.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    results[variant] = {
        "final_nll": curve[-1]["eval_nll"], "curve": curve,
        "params": sum(p.numel() for p in params), "wall_s": round(time.perf_counter() - t0, 1),
    }
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
