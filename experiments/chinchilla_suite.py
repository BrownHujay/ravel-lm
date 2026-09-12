#!/usr/bin/env python3
"""Chinchilla-scale ablation for the 9070XT (or any torch device).

Arms:
  orig        : published architecture (anchor)
  nope        : clean config — 2 literal heads, no learned head, no pos table
  nope_chime2 : nope + CHIME-X v2 — multiplicative age-gate, ZERO-INIT so the
                model is exactly `nope` at step 0 and the age pathway can only add.

Portable: write-times come back from the lookup itself by appending the
timestamp as an extra payload channel (works on CUDA/ROCm/MPS/CPU alike).

Usage:
  python3 chinchilla_suite.py [--steps 87890] [--arms nope,nope_chime2,orig]
                              [--big] [--device cuda]
  Defaults: --big (d256 L12, ~9M params), 87,890 steps = ~180M tokens (Chinchilla
  for 9M params), same fineweb corpus and data-order protocol as scale_suite.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM, RavelMemoryLayer, literal_addresses_from_tokens
from ravel_lm.ravel_memory import causal_last_k_lookup
from ravel_lm.runtime import FlatAdamW

TRAIN_CORPUS = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/train_corpus.txt"
EVAL_CORPUS = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/eval_corpus.txt"
T = 2048
LR = 3.5e-4
EVAL_WINDOWS = 16  # matches scale_suite so results are directly comparable
N_AGE_FEAT = 6


class ChimeV2MemoryLayer(RavelMemoryLayer):
    """CHIME-X v2: retrieved payloads modulated by a zero-init age gate.

    gate = 1 + tanh(W_age @ feats); W_age starts at zero => gate == 1 exactly,
    so this layer is bit-identical to the plain literal-only layer at init.
    Write time rides along as an extra payload channel (portable across backends).
    """

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
        # Append write-time as one extra channel; the lookup hands it back per record.
        t_chan = torch.arange(Tt, device=x.device, dtype=payload.dtype).view(1, Tt, 1, 1).expand(B, Tt, C, 1)
        payload_aug = torch.cat([payload, t_chan], dim=-1)
        values, mask = causal_last_k_lookup(
            literal_addr, payload_aug, literal_addr,
            address_space=self.cfg.address_space, k=1, write_mask=write_mask,
        )
        values = values.squeeze(3)                       # [B,T,C,P+1]
        retrieved = values[..., : self.cfg.payload_dim]  # payload
        t_write = values[..., self.cfg.payload_dim]      # write time (0 when miss)
        m = mask.squeeze(3).to(retrieved.dtype)          # [B,T,C]

        t_read = torch.arange(Tt, device=x.device, dtype=retrieved.dtype).view(1, Tt, 1)
        delta = (t_read - t_write).clamp_min(1.0)
        u = torch.log2(1.0 + delta)
        feats = torch.stack([
            m, (delta <= 8).to(u.dtype) * m, (u / 12.0) * m, (u - u.floor()) * m,
            torch.sin(2 * math.pi * 0.5 * u) * m, torch.cos(2 * math.pi * 0.5 * u) * m,
        ], dim=-1)                                       # [B,T,C,6]
        age_gate = 1.0 + torch.tanh(self.age_proj(feats))  # zero-init => exactly 1.0
        fused = self.fuse((retrieved * age_gate).reshape(B, Tt, -1))
        return x + self.dropout(torch.sigmoid(gate_lin) * fused)


def load_bytes(path):
    return np.frombuffer(path.read_bytes(), dtype=np.uint8)


def make_offsets(n_tokens, steps, seed):
    return np.random.default_rng(seed).integers(0, n_tokens - (T + 1), size=steps)


def batch_at(tokens, off, device):
    chunk = tokens[off : off + T + 1].astype(np.int64)
    return (torch.from_numpy(chunk[:-1]).view(1, T).to(device),
            torch.from_numpy(chunk[1:]).view(1, T).to(device))


def lr_at(step, warmup, steps):
    if step < warmup:
        return LR * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def eval_nll(model, eval_tokens, eval_offsets, device):
    model.eval()
    losses = [float(model(*batch_at(eval_tokens, off, device))["loss"]) for off in eval_offsets]
    model.train()
    return sum(losses) / len(losses)


def build(arm, big, device):
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
    model = RavelLM(cfg).to(device)
    if arm in ("nope", "nope_chime2"):
        model.pos_emb.weight.data.zero_()
        model.pos_emb.weight.requires_grad_(False)
    if arm == "nope_chime2":
        for block in model.blocks:
            if block.memory is not None:
                block.memory.__class__ = ChimeV2MemoryLayer
                proj = nn.Linear(N_AGE_FEAT, 1, bias=False).to(device)
                nn.init.zeros_(proj.weight)   # gate starts at exactly 1.0
                block.memory.age_proj = proj
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=87890)   # ~180M tokens
    p.add_argument("--arms", default="nope,nope_chime2,orig")
    p.add_argument("--big", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-every", type=int, default=10000)
    args = p.parse_args()
    device = torch.device(args.device)
    out = Path(__file__).parent / f"chinchilla_results_{'big' if args.big else 'base'}.json"

    train_tokens = load_bytes(TRAIN_CORPUS)
    eval_tokens = load_bytes(EVAL_CORPUS)
    offsets = make_offsets(len(train_tokens), args.steps, seed=2026)
    eval_offsets = make_offsets(len(eval_tokens), EVAL_WINDOWS, seed=7)
    done = set(json.loads(out.read_text()).keys()) if out.exists() else set()

    for arm in args.arms.split(","):
        if arm in done:
            print(f"skip {arm} (done)", flush=True)
            continue
        torch._dynamo.reset()
        torch._dynamo.config.recompile_limit = 64
        model = build(arm, args.big, device)
        params = [q for q in model.parameters() if q.requires_grad]
        opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=LR, betas=(0.9, 0.95))
        em = torch.compile(model, backend="inductor", mode="reduce-overhead",
                           fullgraph=True, dynamic=False) if args.compile else model
        warmup = args.steps // 10
        curve = []
        t0 = time.perf_counter()
        for step in range(args.steps):
            lr = lr_at(step, warmup, args.steps)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = batch_at(train_tokens, offsets[step], device)
            opt.zero_grad()
            loss = em(x, y)["loss"]
            loss.backward()
            opt.clip_grad_norm_(1.0)
            opt.step()
            if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
                nll = eval_nll(em, eval_tokens, eval_offsets, device)
                curve.append({"step": step + 1, "eval_nll": nll})
                print(f"[{arm:12s}] step {step+1:6d} eval_nll {nll:.4f}", flush=True)
        results = json.loads(out.read_text()) if out.exists() else {}
        results[arm] = {"final_nll": curve[-1]["eval_nll"], "curve": curve,
                        "params": sum(q.numel() for q in params),
                        "tokens": args.steps * T,
                        "wall_s": round(time.perf_counter() - t0, 1)}
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
