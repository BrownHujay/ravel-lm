#!/usr/bin/env python3
"""HIP-graph-captured train-step benchmark for RDNA4/CUDA.

Captures the full step (fwd+bwd+clip+opt) as a single HIP graph and times
replay. This is the production-speed path on Windows ROCm, where per-launch
CPU overhead otherwise bounds the step. Deferred-dW (rocBLAS bmm) + HIP GEMM
+ fused sublayers, all graph-capturable and parity-verified.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.runtime import FlatAdamW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/byte/ravel_3m_byte.json"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    DEV = torch.device("cuda")
    cfg = RavelConfig.from_json(args.config)
    cfg.batch_size = args.batch_size
    cfg.block_size = args.block_size
    cfg.validate()
    torch.manual_seed(0)
    model = RavelLM(cfg).to(DEV)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=3e-4,
                    betas=(0.9, 0.999), capturable=True)
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=DEV)
    y = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=DEV)

    def step():
        opt.zero_grad()
        loss = model(x, y)["loss"]
        loss.backward()
        opt.clip_grad_norm_(1.0)
        opt.step()
        return loss

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_loss = step()

    for _ in range(15):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        g.replay()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.iters * 1000
    tokens = args.batch_size * args.block_size
    print(f"device={torch.cuda.get_device_name(0)}  params={model.num_parameters:,}")
    print(f"graphed_step_ms={ms:.3f}")
    print(f"tokens_per_sec={tokens / (ms / 1000):,.0f}")


if __name__ == "__main__":
    main()
