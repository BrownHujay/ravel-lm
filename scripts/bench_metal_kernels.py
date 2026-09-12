#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.metal_kernels import get_kernels
from ravel_lm.ravel_memory import causal_last_k_lookup


def reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    B, H, T, D = q.shape
    qt = torch.from_numpy(q)
    kt = torch.from_numpy(k)
    vt = torch.from_numpy(v)
    scores = torch.matmul(qt, kt.transpose(-2, -1)) / (D**0.5)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    return torch.softmax(scores, dim=-1).matmul(vt).numpy()


def median(xs: list[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark raw Metal RAVEL and causal softmax attention kernels")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=4, help="attention heads")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--dim", type=int, default=64, help="attention head dimension")
    p.add_argument("--ravel-heads", type=int, default=4)
    p.add_argument("--payload-dim", type=int, default=16)
    p.add_argument("--address-space", type=int, default=512)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    kernels = get_kernels()

    B, T = args.batch, args.seq
    write = rng.integers(0, args.address_space, size=(B, T, args.ravel_heads), dtype=np.int32)
    read = rng.integers(0, args.address_space, size=(B, T, args.ravel_heads), dtype=np.int32)
    payload = rng.normal(size=(B, T, args.ravel_heads, args.payload_dim)).astype(np.float32)

    q = rng.normal(size=(B, args.heads, T, args.dim)).astype(np.float32)
    k = rng.normal(size=(B, args.heads, T, args.dim)).astype(np.float32)
    v = rng.normal(size=(B, args.heads, T, args.dim)).astype(np.float32)

    for _ in range(args.warmup):
        kernels.ravel_latest1(write, payload, read, address_space=args.address_space, implementation="legacy", return_timing=True)
        kernels.ravel_latest1(write, payload, read, address_space=args.address_space, return_timing=True)
        kernels.causal_softmax_attention(q, k, v, return_timing=True)

    legacy_wall: list[float] = []
    legacy_gpu: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _, _, gpu_ms = kernels.ravel_latest1(
            write,
            payload,
            read,
            address_space=args.address_space,
            implementation="legacy",
            return_timing=True,
        )
        legacy_wall.append((time.perf_counter() - t0) * 1000.0)
        legacy_gpu.append(gpu_ms)

    ravel_wall: list[float] = []
    ravel_gpu: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _, _, gpu_ms = kernels.ravel_latest1(write, payload, read, address_space=args.address_space, return_timing=True)
        ravel_wall.append((time.perf_counter() - t0) * 1000.0)
        ravel_gpu.append(gpu_ms)

    attn_wall: list[float] = []
    attn_gpu: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _, gpu_ms = kernels.causal_softmax_attention(q, k, v, return_timing=True)
        attn_wall.append((time.perf_counter() - t0) * 1000.0)
        attn_gpu.append(gpu_ms)

    rv, rm = kernels.ravel_latest1(write, payload, read, address_space=args.address_space)
    ref_rv, ref_rm = causal_last_k_lookup(
        torch.from_numpy(write).long(),
        torch.from_numpy(payload),
        torch.from_numpy(read).long(),
        address_space=args.address_space,
        k=1,
    )
    ravel_max_err = float(np.max(np.abs(rv - ref_rv[..., 0, :].numpy())))
    ravel_mask_ok = bool(np.array_equal(rm, ref_rm[..., 0].numpy()))

    attn = kernels.causal_softmax_attention(q, k, v)
    ref_attn = reference_attention(q, k, v)
    attn_max_err = float(np.max(np.abs(attn - ref_attn)))

    print("Raw Metal kernels; wall time includes buffer setup/readback, gpu time is command-buffer GPU time.")
    legacy_gpu_ms = median(legacy_gpu)
    legacy_wall_ms = median(legacy_wall)
    ravel_gpu_ms = median(ravel_gpu)
    ravel_wall_ms = median(ravel_wall)
    print(f"RAVEL latest1 shape: B={B} T={T} C={args.ravel_heads} Dv={args.payload_dim} A={args.address_space}")
    print(f"  legacy sweep gpu_ms={legacy_gpu_ms:.4f} wall_ms={legacy_wall_ms:.4f}")
    print(f"  indexed state gpu_ms={ravel_gpu_ms:.4f} wall_ms={ravel_wall_ms:.4f}")
    print(f"  speedup gpu={legacy_gpu_ms / ravel_gpu_ms:.2f}x wall={legacy_wall_ms / ravel_wall_ms:.2f}x")
    print(f"  correctness max_abs_err={ravel_max_err:.3e} mask_ok={ravel_mask_ok}")
    print(f"Attention shape: B={B} H={args.heads} T={T} D={args.dim}")
    print(f"  median gpu_ms={median(attn_gpu):.4f} median wall_ms={median(attn_wall):.4f}")
    print(f"  correctness max_abs_err={attn_max_err:.3e}")
    if args.json_out:
        result = {
            "shape": {"batch": B, "seq": T, "ravel_heads": args.ravel_heads, "payload_dim": args.payload_dim, "address_space": args.address_space},
            "legacy": {"gpu_ms": legacy_gpu_ms, "wall_ms": legacy_wall_ms},
            "indexed": {"gpu_ms": ravel_gpu_ms, "wall_ms": ravel_wall_ms},
            "speedup": {"gpu": legacy_gpu_ms / ravel_gpu_ms, "wall": legacy_wall_ms / ravel_wall_ms},
            "correctness": {"max_abs_err": ravel_max_err, "mask_ok": ravel_mask_ok},
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
