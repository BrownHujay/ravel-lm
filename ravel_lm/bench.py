from __future__ import annotations

import argparse
import time

import torch

from .ravel_memory import causal_last_k_lookup


def dense_attention_flops(n: int, d: int) -> int:
    # QK^T and AV, ignoring softmax/exponentials.
    return 4 * n * n * d


def ravel_read_fuse_flops(n: int, reads: int, payload_dim: int, d: int) -> int:
    # Dense projection of gathered read payloads to d_model.
    return 2 * n * reads * payload_dim * d


def main() -> None:
    p = argparse.ArgumentParser(description="RAVEL memory primitive sanity benchmark")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--address-space", type=int, default=1024)
    p.add_argument("--payload-dim", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    B, T, C, D = 2, args.n, args.heads, args.payload_dim
    torch.manual_seed(0)
    addr = torch.randint(0, max(1, args.address_space - 1), (B, T, C), device=device)
    payload = torch.randn(B, T, C, D, device=device)
    # Force a known repeat for exactness. Use the last address and keep the
    # random background below it so no accidental newer record wins.
    forced_addr = args.address_space - 1
    addr[:, 10, :] = forced_addr
    addr[:, 100, :] = forced_addr
    payload[:, 10, :, :] = 42.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    vals, mask = causal_last_k_lookup(addr, payload, address_space=args.address_space, k=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ok = bool(torch.allclose(vals[:, 100, :, 0, :], torch.full_like(vals[:, 100, :, 0, :], 42.0)))
    print(f"exact_latest_check={ok}")
    print(f"lookup_time_seconds={dt:.6f} device={device}")
    print(f"dense_attention_core_flops_n={T}: {dense_attention_flops(T, 128):,}")
    print(f"ravel_fuse_flops_n={T}: {ravel_read_fuse_flops(T, C, D, 128):,}")


if __name__ == "__main__":
    main()
