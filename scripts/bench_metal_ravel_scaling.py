#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.metal_kernels import get_kernels


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legacy and indexed RAVEL Metal kernels across address spaces.")
    parser.add_argument("--out", default="runs/metal_ravel_scaling.csv")
    parser.add_argument("--address-spaces", default="64,128,256,512,1024,2048")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--payload-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    kernels = get_kernels()
    rows = []
    for address_space in [int(value) for value in args.address_spaces.split(",")]:
        shape = (args.batch, args.seq, args.heads)
        write = rng.integers(0, address_space, size=shape, dtype=np.int32)
        read = rng.integers(0, address_space, size=shape, dtype=np.int32)
        payload = rng.normal(size=shape + (args.payload_dim,)).astype(np.float32)
        for implementation in ["legacy", "auto"]:
            for _ in range(args.warmup):
                kernels.ravel_latest1(
                    write, payload, read, address_space=address_space, implementation=implementation, return_timing=True
                )
            gpu_times = []
            wall_times = []
            for _ in range(args.iters):
                start = time.perf_counter()
                _, _, gpu_ms = kernels.ravel_latest1(
                    write, payload, read, address_space=address_space, implementation=implementation, return_timing=True
                )
                wall_times.append((time.perf_counter() - start) * 1000.0)
                gpu_times.append(gpu_ms)
            rows.append(
                {
                    "address_space": address_space,
                    "implementation": "legacy sweep" if implementation == "legacy" else "indexed state",
                    "gpu_ms": float(np.median(gpu_times)),
                    "wall_ms": float(np.median(wall_times)),
                    "batch": args.batch,
                    "seq": args.seq,
                    "heads": args.heads,
                    "payload_dim": args.payload_dim,
                }
            )
            print(address_space, implementation, f"gpu={np.median(gpu_times):.4f}ms", f"wall={np.median(wall_times):.4f}ms")

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)


if __name__ == "__main__":
    main()
