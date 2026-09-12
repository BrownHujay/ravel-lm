"""Matched raw-byte training for the original and multirate RAVEL candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multirate_model import MultirateRavelLM
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.runtime import compiled_mps_clip_grad_norm_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=15_000_000)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/multirate_15m")
    parser.add_argument("--eval-every", type=int, default=500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.block_size = 2048
    cfg.batch_size = 1
    steps = args.tokens // cfg.block_size
    warmup = max(1, int(.1 * steps))
    paths = {name: ROOT / f"runs/fineweb_edu_3m_15m_ctx2048/{name}_corpus.txt" for name in ("train", "eval")}
    raw = {name: p.read_bytes() for name, p in paths.items()}
    pools = {name: np.frombuffer(data, dtype=np.uint8) for name, data in raw.items()}
    rng = np.random.default_rng(2026)
    starts = rng.integers(0, len(pools["train"]) - cfg.block_size, size=steps)

    def batch(name, offset):
        v = torch.from_numpy(pools[name][offset:offset + cfg.block_size + 1].astype(np.int64)).to("mps")
        return v[:-1].view(1, -1).contiguous(), v[1:].view(1, -1).contiguous()

    erng = np.random.default_rng(271828)
    eval_offsets = erng.integers(0, len(pools["eval"]) - cfg.block_size, size=64)
    eval_batches = [batch("eval", offset) for offset in eval_offsets]
    specs = [("original", None), ("stride8_core2", {"stride": 8, "core_width": 384, "core_layers": 2}),
             ("stride16_core1", {"stride": 16, "core_width": 512, "core_layers": 1})]
    runs = {}
    for name, variant in specs:
        torch.manual_seed(2026)
        model = RavelLM(cfg) if variant is None else MultirateRavelLM(cfg, **variant)
        model = model.to("mps")
        params = [p for p in model.parameters() if p.requires_grad]
        runs[name] = {"model": model, "parameters": params,
                      "compiled": torch.compile(model, fullgraph=True, dynamic=False, mode="reduce-overhead"),
                      "optimizer": torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
                                                     weight_decay=cfg.weight_decay, fused=True),
                      "times": [], "losses": [], "eval": [], "variant": variant}
        print(name, "parameters", model.num_parameters, flush=True)

    metadata = {"config": cfg.to_dict(), "requested_training_bytes": args.tokens,
                "optimizer_tokens_per_model": steps * cfg.block_size, "steps": steps, "warmup": warmup,
                "seed": 2026, "dtype": "float32", "torch": torch.__version__,
                "corpus_paths": {k: str(v) for k, v in paths.items()},
                "corpus_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
                "tokenization": "raw UTF-8 bytes of persisted corpus; every next-byte target is scored",
                "batch_order": "identical offsets for every model; runtime order rotates each step",
                "timing": "forward, loss, backward, clipping, AdamW and synchronize; data staging and logging excluded",
                "specs": {name: {"variant": variant, "trainable_parameters": runs[name]["model"].num_parameters,
                                  "all_parameters": sum(p.numel() for p in runs[name]["model"].parameters())}
                          for name, variant in specs}}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.save(args.out / "training_offsets.npy", starts)
    np.save(args.out / "eval_offsets.npy", eval_offsets)

    @torch.no_grad()
    def evaluate(run, count):
        run["model"].eval()
        losses = [run["model"](x, y)["loss"].item() for x, y in eval_batches[:count]]
        run["model"].train()
        return losses

    metrics_file = args.out / "metrics.jsonl"
    with metrics_file.open("w") as log:
        for name, run in runs.items():
            values = evaluate(run, 8)
            row = {"model": name, "step": 0, "tokens": 0, "eval_nll": statistics.mean(values)}
            run["eval"].append(row)
            log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        names = list(runs)
        for step, offset in enumerate(starts):
            x, y = batch("train", int(offset))
            if step < warmup:
                lr = cfg.learning_rate * (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, steps - warmup - 1)
                lr = cfg.learning_rate * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            order = names[step % len(names):] + names[:step % len(names)]
            for name in order:
                run = runs[name]
                for group in run["optimizer"].param_groups:
                    group["lr"] = lr
                t0 = time.perf_counter()
                run["optimizer"].zero_grad(set_to_none=True)
                loss = run["compiled"](x, y)["loss"]
                loss.backward()
                compiled_mps_clip_grad_norm_(run["parameters"], cfg.grad_clip)
                run["optimizer"].step()
                torch.mps.synchronize()
                elapsed = 1000 * (time.perf_counter() - t0)
                run["losses"].append(loss.item())
                if step >= 20:
                    run["times"].append(elapsed)
            if (step + 1) % args.eval_every == 0 or step == steps - 1:
                for name, run in runs.items():
                    values = evaluate(run, 64 if step == steps - 1 else 8)
                    row = {"model": name, "step": step + 1, "tokens": (step + 1) * cfg.block_size,
                           "lr": lr, "train_nll": statistics.mean(run["losses"][-100:]),
                           "eval_nll": statistics.mean(values), "eval_batches": values,
                           "median_step_ms": statistics.median(run["times"][-args.eval_every:])}
                    run["eval"].append(row)
                    log.write(json.dumps(row) + "\n")
                    print(json.dumps({k: v for k, v in row.items() if k != "eval_batches"}), flush=True)
                log.flush()

    result = {"metadata": metadata, "models": {}}
    baseline = np.array(runs["original"]["eval"][-1]["eval_batches"])
    for name, run in runs.items():
        final = run["eval"][-1]
        diff = np.array(final["eval_batches"]) - baseline
        result["models"][name] = {"parameters": run["model"].num_parameters,
                                   "median_step_ms": statistics.median(run["times"]),
                                   "samples_ms": run["times"], "train_losses": run["losses"],
                                   "evaluations": run["eval"], "final_eval_nll": final["eval_nll"],
                                   "paired_nll_delta": float(diff.mean()),
                                   "paired_nll_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                   "meets_5ms_and_mean_nll": statistics.median(run["times"]) <= 5 and diff.mean() <= 0}
        torch.save({"model": {k: v.detach().cpu() for k, v in run["model"].state_dict().items()},
                    "config": cfg.to_dict(), "variant": run["variant"], "steps": steps}, args.out / f"{name}_final.pt")
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Finished", args.out, flush=True)


if __name__ == "__main__":
    main()
