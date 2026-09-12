#!/usr/bin/env python3
"""Follow-ups: (A) ORDER trained with chrono-warp coverage (the spec's actual
recipe) instead of zero-shot extrapolation; (B) IDENT with sinusoid features
computed at float32 age precision (a real fp32 pipeline) vs CHIME-X int64 residues."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import chimex_mock as M


def train_order_warped(kind):
    rng = np.random.default_rng(42)
    model = M.Scorer(M.Features(kind), query_conditioned=False)
    opt = torch.optim.Adam(model.parameters(), lr=M.LR)
    for _ in range(M.STEPS):
        ages, q, target = M.sample_order(M.BATCH, 1, 2**40, rng)  # chrono-warp coverage
        loss = nn.functional.cross_entropy(model(ages, q), torch.from_numpy(target))
        opt.zero_grad(); loss.backward(); opt.step()
    for label, lo, hi in [("1..1e6", 1, int(1e6)), ("1e7..1e9", int(1e7), int(1e9)),
                          ("2^38..2^40", 2**38, 2**40), ("2^45..2^50 (beyond)", 2**45, 2**50)]:
        acc = M.accuracy(model, M.sample_order, lo, hi, rng)
        print(f"ORDER-warped {kind:13s} {label:20s} acc={acc:.3f}", flush=True)


class Fp32Features(M.Features):
    def __call__(self, ages):
        return super().__call__(ages.astype(np.float32).astype(np.int64))


def train_ident_fp32_sinusoid():
    rng = np.random.default_rng(42)
    model = M.Scorer(Fp32Features("sinusoid"), query_conditioned=True)
    opt = torch.optim.Adam(model.parameters(), lr=M.LR)
    for _ in range(M.STEPS):
        ages, q, target = M.sample_ident(M.BATCH, int(1e3), int(1e6), rng)
        loss = nn.functional.cross_entropy(model(ages, q), torch.from_numpy(target))
        opt.zero_grad(); loss.backward(); opt.step()
    for label, lo, hi in [("base 1e3..1e6", int(1e3), int(1e6)), ("base ~1e9", int(9e8), int(1e9))]:
        acc = M.accuracy(model, M.sample_ident, lo, hi, rng)
        print(f"IDENT sinusoid@fp32-age {label:16s} acc={acc:.3f}", flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    train_order_warped("chimex_nocrt")
    train_order_warped("chimex")
    train_ident_fp32_sinusoid()
