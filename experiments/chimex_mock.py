#!/usr/bin/env python3
"""CHIME-X feature-stack mock: does it deliver what the spec claims?

Task ORDER: among K retrieved records, pick the youngest. Trained on ages
  [1, 1e6], tested on ages [1e7, 1e9] — pure scale extrapolation, the
  chrono-warp generalization claim.
Task IDENT: query specifies an exact age; pick the record whose age matches
  exactly, among K records clustered within +-64 of a huge base age.
  Trained with bases in [1e3, 1e6], tested at bases ~1e9. Log features are
  blind here (identical mantissas at fp precision); CRT residues are
  scale-free (every residue value is seen in training at ANY scale).

Feature sets:
  chimex      : exact near-field flag + log2 pack (norm-u, dyadic bucket
                embed, mantissa, harmonics over u) + CRT residue embeds
  chimex_nocrt: same minus residues
  sinusoid    : transformer/RoPE-style sin/cos at linear age
  raw         : age / 1e6, clipped
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
DEV = torch.device("cpu")
PRIMES = [257, 263, 269, 271, 277, 281, 283, 293]
K = 8
EMB = 8          # per-bucket / per-residue embedding width
HARM = [0.5, 1.0, 2.0, 4.0]
N_BUCKETS = 64   # covers 2^64
STEPS = 4000
BATCH = 256
LR = 1e-3


class Features:
    """Turn int64 ages into fixed vectors + index features for embeddings."""

    def __init__(self, kind):
        self.kind = kind
        dim = 0
        self.n_embeds = 0
        if kind in ("chimex", "chimex_nocrt"):
            dim += 1 + 1 + 1 + 2 * len(HARM)  # near flag, norm-u, mantissa, harmonics
            self.n_embeds += 1                # dyadic bucket
            if kind == "chimex":
                self.n_embeds += len(PRIMES)  # residues
        elif kind == "sinusoid":
            dim += 2 * 16
        elif kind == "raw":
            dim += 1
        self.dense_dim = dim

    def __call__(self, ages):  # ages: int64 numpy [.., ]
        a = ages.astype(np.float64)
        dense, embeds = [], []
        if self.kind in ("chimex", "chimex_nocrt"):
            u = np.log2(1.0 + a)
            dense.append((a <= 32).astype(np.float64))       # near-field flag
            dense.append(u / 64.0)                           # normalized log-age
            dense.append(u - np.floor(u))                    # mantissa
            for f in HARM:
                dense.append(np.sin(2 * np.pi * f * u))
                dense.append(np.cos(2 * np.pi * f * u))
            embeds.append(np.minimum(np.floor(u), N_BUCKETS - 1).astype(np.int64))
            if self.kind == "chimex":
                for p in PRIMES:
                    embeds.append((ages % p).astype(np.int64))
        elif self.kind == "sinusoid":
            for i in range(16):
                lam = 10000.0 ** (i / 16.0)
                dense.append(np.sin(a / lam))
                dense.append(np.cos(a / lam))
        elif self.kind == "raw":
            dense.append(np.clip(a / 1e6, 0, 10))
        dense = np.stack(dense, axis=-1).astype(np.float32)
        return dense, embeds


class Scorer(nn.Module):
    def __init__(self, feats: Features, query_conditioned: bool):
        super().__init__()
        self.feats = feats
        self.embeds = nn.ModuleList()
        for i in range(feats.n_embeds):
            size = N_BUCKETS if i == 0 else PRIMES[i - 1]
            self.embeds.append(nn.Embedding(size, EMB))
        in_dim = feats.dense_dim + feats.n_embeds * EMB
        if query_conditioned:
            in_dim *= 2
        self.net = nn.Sequential(nn.Linear(in_dim, 96), nn.GELU(), nn.Linear(96, 96), nn.GELU(), nn.Linear(96, 1))

    def encode(self, ages):
        dense, embeds = self.feats(ages)
        parts = [torch.from_numpy(dense)]
        for e_idx, idx in enumerate(embeds):
            parts.append(self.embeds[e_idx](torch.from_numpy(idx)))
        return torch.cat(parts, dim=-1)

    def forward(self, record_ages, query_ages=None):
        x = self.encode(record_ages)                       # [B,K,F]
        if query_ages is not None:
            q = self.encode(query_ages)                    # [B,F]
            x = torch.cat([x, q.unsqueeze(1).expand_as(x)], dim=-1)
        return self.net(x).squeeze(-1)                     # [B,K]


def sample_order(batch, lo, hi, rng):
    # log-uniform distinct ages
    ages = np.exp(rng.uniform(math.log(lo), math.log(hi), size=(batch, K))).astype(np.int64)
    ages += np.arange(K, dtype=np.int64)[None, :]  # break exact ties
    target = ages.argmin(axis=1)
    return ages, None, target


def sample_ident(batch, lo, hi, rng):
    base = np.exp(rng.uniform(math.log(lo), math.log(hi), size=(batch, 1))).astype(np.int64)
    offsets = np.stack([rng.choice(128, size=K, replace=False) for _ in range(batch)])
    ages = base + offsets
    target = rng.integers(0, K, size=batch)
    query = ages[np.arange(batch), target]
    return ages, query, target


def accuracy(model, sampler, lo, hi, rng, n=4096):
    with torch.no_grad():
        ages, query, target = sampler(n, lo, hi, rng)
        scores = model(ages, query)
        return float((scores.argmax(dim=1).numpy() == target).mean())


def run(task, kind):
    rng = np.random.default_rng(42)
    sampler = sample_order if task == "ORDER" else sample_ident
    lo, hi = (1, int(1e6)) if task == "ORDER" else (int(1e3), int(1e6))
    model = Scorer(Features(kind), query_conditioned=(task == "IDENT"))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(STEPS):
        ages, query, target = sampler(BATCH, lo, hi, rng)
        loss = nn.functional.cross_entropy(model(ages, query), torch.from_numpy(target))
        opt.zero_grad(); loss.backward(); opt.step()
    ood_lo, ood_hi = (int(1e7), int(1e9)) if task == "ORDER" else (int(9e8), int(1e9))
    res = {
        "in_dist": accuracy(model, sampler, lo, hi, rng),
        "ood_1e9": accuracy(model, sampler, ood_lo, ood_hi, rng),
    }
    if task == "ORDER":
        res["ood_2^40"] = accuracy(model, sampler, 2**38, 2**40, rng)
    print(f"{task:5s} {kind:13s} " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    return res


def main():
    results = {}
    for task in ["ORDER", "IDENT"]:
        for kind in ["chimex", "chimex_nocrt", "sinusoid", "raw"]:
            results[f"{task}/{kind}"] = run(task, kind)
    out = Path(__file__).parent / "chimex_mock_results.json"
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
