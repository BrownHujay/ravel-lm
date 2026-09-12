#!/usr/bin/env python3
"""15M-token ablation suite at two scales. Arms: orig, literal2, nope, nope_chime.

Phase 1: 3M params (d160 L10)  — matches the published multirate protocol.
Phase 2: ~9M params (d256 L12) — do the deletions still win with more capacity?

Same data offsets across arms within a phase. Incremental JSON saves per arm.
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

from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM, RavelMemoryLayer, literal_addresses_from_tokens
from ravel_lm.mps_memory import _latest1_forward
from ravel_lm.runtime import FlatAdamW

DEV = torch.device("mps")
TRAIN_CORPUS = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/train_corpus.txt"
EVAL_CORPUS = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/eval_corpus.txt"
T = 2048
STEPS = 7324          # 15.0M tokens, matches the published run
EVAL_EVERY = 1500
EVAL_WINDOWS = 16
LR = 3.5e-4
OUT = Path(__file__).parent / "scale_suite_results.json"

N_AGE_FEAT = 6


class ChimeMemoryLayer(RavelMemoryLayer):
    def forward(self, x, token_ids, write_mask=None, literal_addr=None):
        B, Tt, D = x.shape
        xn = self.norm(x)
        C = self.cfg.n_memory_heads
        packed = F.linear(xn, torch.cat([self.payload.weight, self.gate.weight], dim=0))
        n_pay = self.payload.weight.shape[0]
        payload = packed[..., :n_pay].reshape(B, Tt, C, self.cfg.payload_dim)
        gate_lin = packed[..., n_pay:] + self.gate.bias
        if literal_addr is None:
            literal_addr = literal_addresses_from_tokens(
                token_ids, address_space=self.cfg.address_space,
                n_heads=self.cfg.n_literal_heads, bos_token_id=self.cfg.bos_token_id,
            )
        values, src, mask = _latest1_forward(literal_addr, payload, literal_addr, int(self.cfg.address_space))
        src = src.view(B, Tt, C).long()
        t_write = (src.clamp_min(0) // C) % Tt
        t_read = torch.arange(Tt, device=x.device).view(1, Tt, 1)
        delta = (t_read - t_write).clamp_min(1).float()
        m = mask.view(B, Tt, C).float()
        u = torch.log2(1.0 + delta)
        feats = torch.stack([
            m, (delta <= 8).float() * m, (u / 12.0) * m, (u - u.floor()) * m,
            torch.sin(2 * math.pi * 0.5 * u) * m, torch.cos(2 * math.pi * 0.5 * u) * m,
        ], dim=-1).reshape(B, Tt, C * N_AGE_FEAT)
        fused = self.fuse(torch.cat([values.reshape(B, Tt, -1), feats], dim=-1))
        return x + self.dropout(torch.sigmoid(gate_lin) * fused)


def load_bytes(path):
    return np.frombuffer(path.read_bytes(), dtype=np.uint8)


def make_offsets(n_tokens, steps, seed):
    return np.random.default_rng(seed).integers(0, n_tokens - (T + 1), size=steps)


def batch_at(tokens, off):
    chunk = tokens[off : off + T + 1].astype(np.int64)
    return (torch.from_numpy(chunk[:-1]).view(1, T).to(DEV),
            torch.from_numpy(chunk[1:]).view(1, T).to(DEV))


def lr_at(step, warmup):
    if step < warmup:
        return LR * (step + 1) / warmup
    p = (step - warmup) / max(1, STEPS - warmup)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def eval_nll(model, eval_tokens, eval_offsets):
    model.eval()
    losses = [float(model(*batch_at(eval_tokens, off))["loss"]) for off in eval_offsets]
    model.train()
    return sum(losses) / len(losses)


def build(arm, big):
    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.block_size = T
    cfg.batch_size = 1
    if big:
        cfg.d_model = 256
        cfg.n_layers = 12
    if arm != "orig":
        cfg.n_learned_heads = 0
    cfg.validate()
    torch.manual_seed(0)
    model = RavelLM(cfg).to(DEV)
    if arm in ("nope", "nope_chime"):
        model.pos_emb.weight.data.zero_()
        model.pos_emb.weight.requires_grad_(False)
    if arm == "nope_chime":
        C = cfg.n_memory_heads
        for block in model.blocks:
            if block.memory is not None:
                block.memory.__class__ = ChimeMemoryLayer
                old = block.memory.fuse
                new = nn.Linear(old.in_features + C * N_AGE_FEAT, old.out_features, bias=False).to(DEV)
                nn.init.normal_(new.weight, mean=0.0, std=0.02)
                block.memory.fuse = new
    return model


def run_arm(arm, big, train_tokens, eval_tokens, offsets, eval_offsets):
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 64
    model = build(arm, big)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=LR, betas=(0.9, 0.95))
    em = torch.compile(model, backend="inductor", mode="reduce-overhead", fullgraph=True, dynamic=False)
    warmup = STEPS // 10
    tag = f"{'big' if big else 'base'}/{arm}"
    curve = []
    t0 = time.perf_counter()
    for step in range(STEPS):
        lr = lr_at(step, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch_at(train_tokens, offsets[step])
        opt.zero_grad()
        loss = em(x, y)["loss"]
        loss.backward()
        opt.clip_grad_norm_(1.0)
        opt.step()
        if (step + 1) % EVAL_EVERY == 0 or step == STEPS - 1:
            nll = eval_nll(em, eval_tokens, eval_offsets)
            curve.append({"step": step + 1, "eval_nll": nll})
            print(f"[{tag:15s}] step {step+1:5d} eval_nll {nll:.4f}", flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results[tag] = {"final_nll": curve[-1]["eval_nll"], "curve": curve,
                    "params": n_params, "wall_s": round(time.perf_counter() - t0, 1)}
    OUT.write_text(json.dumps(results, indent=2))


def main():
    train_tokens = load_bytes(TRAIN_CORPUS)
    eval_tokens = load_bytes(EVAL_CORPUS)
    offsets = make_offsets(len(train_tokens), STEPS, seed=2026)
    eval_offsets = make_offsets(len(eval_tokens), EVAL_WINDOWS, seed=7)
    done = set(json.loads(OUT.read_text()).keys()) if OUT.exists() else set()
    for big in (False, True):
        for arm in ("orig", "literal2", "nope", "nope_chime"):
            tag = f"{'big' if big else 'base'}/{arm}"
            if tag in done:
                print(f"skip {tag} (done)", flush=True)
                continue
            run_arm(arm, big, train_tokens, eval_tokens, offsets, eval_offsets)


if __name__ == "__main__":
    main()
