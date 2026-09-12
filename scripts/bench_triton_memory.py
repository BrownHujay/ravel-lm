#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.ravel_memory import causal_last_k_lookup


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(fn, *, warmup: int, iters: int, device: torch.device) -> tuple[float, torch.Tensor, torch.Tensor]:
    values = mask = None
    for _ in range(warmup):
        values, mask = fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        values, mask = fn()
    sync(device)
    dt = (time.perf_counter() - t0) / iters
    assert values is not None and mask is not None
    return dt * 1000.0, values, mask


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark RAVEL latest-1 lookup on CUDA/HIP.")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--payload-dim", type=int, default=24)
    p.add_argument("--address-space", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--pattern", choices=["random", "byte"], default="random")
    p.add_argument("--disable-triton", action="store_true")
    args = p.parse_args()
    if args.disable_triton:
        os.environ["RAVEL_DISABLE_TRITON_MEMORY"] = "1"
        os.environ.pop("RAVEL_ENABLE_TRITON_MEMORY", None)
    else:
        os.environ["RAVEL_ENABLE_TRITON_MEMORY"] = "1"

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP is not available")
    device = torch.device("cuda")
    torch.manual_seed(2026)
    B, T, C, D, A = args.batch, args.tokens, args.heads, args.payload_dim, args.address_space
    if args.pattern == "byte":
        toks = torch.randint(0, 256, (B, T), device=device, dtype=torch.long)
        prev = torch.cat([torch.full((B, 1), 257, device=device, dtype=torch.long), toks[:, :-1]], dim=1)
        pos = torch.arange(T, device=device, dtype=torch.long).view(1, T).expand(B, T)
        heads = [
            toks.remainder(A),
            (prev * 257 + toks * 131 + 17).remainder(A),
            (toks * 1013 + pos * 9179 + 97).remainder(A),
        ]
        write = torch.stack(heads[:C], dim=-1).contiguous()
        read = write.clone()
    else:
        write = torch.randint(0, A, (B, T, C), device=device, dtype=torch.long)
        read = torch.randint(0, A, (B, T, C), device=device, dtype=torch.long)
    payload = torch.randn(B, T, C, D, device=device, dtype=torch.float32, requires_grad=args.backward)
    grad = torch.randn(B, T, C, 1, D, device=device) if args.backward else None

    def run():
        if payload.grad is not None:
            payload.grad = None
        values, mask = causal_last_k_lookup(write, payload, read, address_space=A, k=1)
        if args.backward:
            (values * grad).sum().backward()
        return values, mask

    ms, values, mask = bench(run, warmup=args.warmup, iters=args.iters, device=device)
    records = B * T * C
    payload_values = records * D
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape B={B} T={T} C={C} D={D} A={A} backward={args.backward} pattern={args.pattern}")
    print(f"mean_ms={ms:.4f}")
    print(f"records_per_sec={records / (ms / 1000.0):,.0f}")
    print(f"payload_values_per_sec={payload_values / (ms / 1000.0):,.0f}")
    print(f"hit_rate={mask.float().mean().item():.4f}")
    print(f"checksum={values.float().sum().item():.6f}")


if __name__ == "__main__":
    main()
