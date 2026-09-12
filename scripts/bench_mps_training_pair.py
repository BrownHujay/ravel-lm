#!/usr/bin/env python3
"""Paired, synchronized full-step comparison at identical FP32 weights/data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.runtime import compiled_mps_clip_grad_norm_


def batches(path, count, tokens, seed):
    raw = path.read_bytes()
    data = np.frombuffer(raw, dtype=np.uint8)
    if len(data) <= tokens:
        raise ValueError("corpus is shorter than one sequence")
    offsets = np.random.default_rng(seed).integers(0, len(data) - tokens, size=count)
    result = []
    for offset in offsets:
        window = torch.from_numpy(data[offset:offset + tokens + 1].astype(np.int64)).to("mps")
        result.append((window[:-1].view(1, -1).contiguous(), window[1:].view(1, -1).contiguous()))
    return result, hashlib.sha256(raw).hexdigest(), offsets.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/fineweb_edu_3m_15m_ctx2048/ravel_final.pt")
    parser.add_argument("--corpus", type=Path, default=ROOT / "runs/fineweb_edu_3m_15m_ctx2048/train_corpus.txt")
    parser.add_argument("--eval-corpus", type=Path, default=ROOT / "runs/fineweb_edu_3m_15m_ctx2048/eval_corpus.txt")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/kernel_experiments/local_gate_pair.json")
    args = parser.parse_args()
    if args.steps <= 0 or args.rounds <= 0 or args.warmup < 1:
        parser.error("positive steps/rounds and at least one warmup step are required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = RavelConfig(**checkpoint["config"])
    assert cfg.block_size == 2048 and cfg.batch_size == 1 and cfg.dropout == 0
    base = RavelLM(cfg)
    base.load_state_dict(checkpoint["model"])
    count = args.warmup + args.steps * args.rounds
    data, digest, offsets = batches(args.corpus, count, cfg.block_size, 2026)
    evaluation, eval_digest, _ = batches(args.eval_corpus, 8, cfg.block_size, 271828)
    runs = {}
    for name, enabled in [("reference", False), ("fused", True)]:
        model = deepcopy(base).to("mps")
        for block in model.blocks:
            block.local.use_mps_local = enabled
        parameters = [p for p in model.parameters() if p.requires_grad]
        runs[name] = {
            "model": model,
            "compiled": torch.compile(model, fullgraph=True, dynamic=False, mode="reduce-overhead"),
            "parameters": parameters,
            "optimizer": torch.optim.AdamW(parameters, lr=cfg.learning_rate,
                                            betas=(cfg.beta1, cfg.beta2),
                                            weight_decay=cfg.weight_decay, fused=True),
            "samples": [], "losses": [], "rounds": [], "step": 0,
        }

    parity_outputs = []
    for name, run in runs.items():
        run["model"].zero_grad(set_to_none=True)
        out = run["compiled"](*evaluation[0])
        out["loss"].backward()
        parity_outputs.append((out["logits"].detach().cpu(), out["loss"].item(),
                               torch.cat([p.grad.detach().flatten().cpu() for p in run["parameters"]])))
    a, b = parity_outputs
    torch.testing.assert_close(a[0], b[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(a[2], b[2], atol=1e-5, rtol=1e-4)
    parity = {"logits_max_abs": (a[0] - b[0]).abs().max().item(),
              "loss_abs": abs(a[1] - b[1]), "gradient_max_abs": (a[2] - b[2]).abs().max().item(),
              "gradient_cosine": torch.nn.functional.cosine_similarity(a[2].double(), b[2].double(), dim=0).item()}
    print("initial parity", json.dumps(parity), flush=True)

    def step(run, timed):
        x, y = data[run["step"]]
        start = time.perf_counter()
        run["optimizer"].zero_grad(set_to_none=True)
        loss = run["compiled"](x, y)["loss"]
        loss.backward()
        compiled_mps_clip_grad_norm_(run["parameters"], cfg.grad_clip)
        run["optimizer"].step()
        torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        run["step"] += 1
        run["losses"].append(loss.item())
        if timed:
            run["samples"].append(elapsed)

    for run in runs.values():
        for _ in range(args.warmup):
            step(run, False)
    for r in range(args.rounds):
        order = ("reference", "fused") if r % 2 == 0 else ("fused", "reference")
        for name in order:
            run = runs[name]
            for _ in range(args.steps):
                step(run, True)
            median = statistics.median(run["samples"][-args.steps:])
            run["rounds"].append(median)
            print(f"round={r + 1} {name} full_step_ms={median:.4f}", flush=True)

    result = {
        "shape": {"batch": cfg.batch_size, "tokens": cfg.block_size, "params": base.num_parameters, "dtype": "float32"},
        "torch": torch.__version__, "device": "mps", "target_ms": 5.,
        "checkpoint": str(args.checkpoint), "corpus": str(args.corpus),
        "corpus_sha256": digest, "eval_sha256": eval_digest,
        "tokenization": "UTF-8 bytes sampled from persisted corpus text",
        "training_offsets": offsets, "warmup_steps": args.warmup,
        "optimizer": {"name": "fused AdamW", "lr": cfg.learning_rate,
                      "betas": [cfg.beta1, cfg.beta2], "weight_decay": cfg.weight_decay, "clip": cfg.grad_clip},
        "timing": "preloaded identical batches; forward, loss, backward, global clipping, AdamW, MPS synchronize; excludes compilation and loss readback",
        "parity": parity,
    }
    for name, run in runs.items():
        losses = []
        with torch.no_grad():
            for batch in evaluation:
                losses.append(run["model"](*batch)["loss"].item())
        median = statistics.median(run["samples"])
        result[name] = {"median_ms": median, "tokens_per_second": cfg.block_size * 1000 / median,
                        "round_medians_ms": run["rounds"], "samples_ms": run["samples"],
                        "train_losses": run["losses"], "eval_losses": losses,
                        "mean_eval_nll": statistics.mean(losses)}
    result["speedup"] = result["reference"]["median_ms"] / result["fused"]["median_ms"]
    result["target_achieved"] = result["fused"]["median_ms"] <= 5.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["shape", "parity", "speedup", "target_achieved"]}), flush=True)
    for name in runs:
        print(name, result[name]["median_ms"], result[name]["mean_eval_nll"], flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
