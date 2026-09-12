#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import os
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.runtime import FlatAdamW, adamw_fused_for


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    p = argparse.ArgumentParser(description="Lean RAVEL train-step timing without plotting dependencies.")
    p.add_argument("--config", default=str(ROOT / "configs" / "byte" / "ravel_1m_byte.json"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=384)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use full-graph torch.compile fusion (default: enabled on MPS)",
    )
    p.add_argument("--enable-triton-memory", action="store_true")
    p.add_argument("--enable-triton-fused-memory", action="store_true")
    p.add_argument("--mps-local", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.enable_triton_memory:
        os.environ["RAVEL_ENABLE_TRITON_MEMORY"] = "1"
    if args.enable_triton_fused_memory:
        os.environ["RAVEL_ENABLE_TRITON_FUSED_MEMORY"] = "1"

    device = torch.device(args.device)
    compile_model = args.compile if args.compile is not None else device.type == "mps"
    torch.manual_seed(2026)
    cfg = RavelConfig.from_json(args.config)
    cfg.batch_size = args.batch_size
    cfg.block_size = args.block_size
    cfg.learning_rate = args.lr
    cfg.validate()

    model = RavelLM(deepcopy(cfg)).to(device)
    for block in model.blocks:
        block.local.use_mps_local = args.mps_local
    exec_model = (
        torch.compile(
            model,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
        if compile_model
        else model
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = FlatAdamW(
        [{"params": trainable_params, "weight_decay": cfg.weight_decay}],
        lr=args.lr,
        betas=(0.9, 0.999),
        fused=adamw_fused_for(device),
    )
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=device)

    step_ms: list[float] = []
    for step in range(args.steps):
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = exec_model(x, y)["loss"]
        loss.backward()
        opt.clip_grad_norm_(1.0)
        opt.step()
        sync(device)
        dt = (time.perf_counter() - t0) * 1000.0
        if step >= args.warmup:
            step_ms.append(dt)
        if step in {0, args.warmup, args.steps - 1}:
            print(f"step={step + 1} loss={float(loss.detach().cpu()):.4f} step_ms={dt:.3f}")

    mean_ms = sum(step_ms) / max(1, len(step_ms))
    sorted_ms = sorted(step_ms)
    median_ms = sorted_ms[len(sorted_ms) // 2]
    tokens_per_step = args.batch_size * args.block_size
    print(f"device={torch.cuda.get_device_name(0) if device.type == 'cuda' else device}")
    print(f"params={model.num_parameters:,}")
    print(
        f"shape batch={args.batch_size} block={args.block_size} compile={compile_model} "
        f"triton_memory={args.enable_triton_memory} triton_fused_memory={args.enable_triton_fused_memory} "
        f"mps_local={args.mps_local and device.type == 'mps'}"
    )
    print(f"mean_step_ms={mean_ms:.3f}")
    print(f"median_step_ms={median_ms:.3f}")
    print(f"median_tokens_per_sec={tokens_per_step / (median_ms / 1000.0):,.0f}")


if __name__ == "__main__":
    main()
