#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.attention_model import AttentionLM
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.tokenizers import ByteTokenizer


def encode_corpus(path: Path) -> np.ndarray:
    tokenizer = ByteTokenizer()
    records = [record.strip() for record in path.read_text(encoding="utf-8").split("\n\n") if record.strip()]
    ids: list[int] = []
    for record in records:
        ids.extend(tokenizer.encode(record, add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.int64)


def make_batches(tokens: np.ndarray, *, batches: int, batch_size: int, block_size: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(batches):
        starts = rng.integers(0, tokens.size - block_size - 1, size=batch_size)
        x = np.stack([tokens[s : s + block_size] for s in starts])
        y = np.stack([tokens[s + 1 : s + block_size + 1] for s in starts])
        result.append((x, y))
    return result


def load_model(path: Path, architecture: str, device: torch.device) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RavelConfig(**checkpoint["config"])
    if architecture == "ravel":
        model = RavelLM(cfg)
    else:
        model = AttentionLM(cfg, n_heads=4)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), int(checkpoint["params"])


@torch.inference_mode()
def evaluate(model: torch.nn.Module, batches: list[tuple[np.ndarray, np.ndarray]], device: torch.device) -> list[float]:
    losses = []
    for x_np, y_np in batches:
        x = torch.as_tensor(x_np, dtype=torch.long, device=device)
        y = torch.as_tensor(y_np, dtype=torch.long, device=device)
        losses.append(float(model(x, y)["loss"].detach().cpu()))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two checkpoint pairs on identical held-out FineWeb windows.")
    parser.add_argument("--one-m-dir", default="runs/fineweb_edu_comparison_1m_3m_tokens")
    parser.add_argument("--three-m-dir", default="runs/fineweb_edu_comparison_3m_m3")
    parser.add_argument("--one-label", default="1M")
    parser.add_argument("--three-label", default="3M")
    parser.add_argument("--out-dir", default="runs/fineweb_edu_scaling_1m_to_3m")
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    one_m_dir = Path(args.one_m_dir)
    three_m_dir = Path(args.three_m_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = encode_corpus(three_m_dir / "eval_corpus.txt")
    batches = make_batches(
        tokens,
        batches=args.batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.seed,
    )
    device = torch.device(args.device)
    specs = [
        (args.one_label, "ravel", one_m_dir / "ravel_final.pt"),
        (args.one_label, "attention", one_m_dir / "attention_final.pt"),
        (args.three_label, "ravel", three_m_dir / "ravel_final.pt"),
        (args.three_label, "attention", three_m_dir / "attention_final.pt"),
    ]
    rows = []
    summaries = []
    for scale, architecture, path in specs:
        model, params = load_model(path, architecture, device)
        losses = evaluate(model, batches, device)
        mean = float(np.mean(losses))
        std = float(np.std(losses, ddof=1))
        se = std / math.sqrt(len(losses))
        summaries.append(
            {
                "scale": scale,
                "model": architecture,
                "params": params,
                "mean_nll": mean,
                "std_batch_nll": std,
                "se_nll": se,
                "ci95_low": mean - 1.96 * se,
                "ci95_high": mean + 1.96 * se,
            }
        )
        rows.extend({"scale": scale, "model": architecture, "batch": i, "nll": loss} for i, loss in enumerate(losses))
        print(f"{scale:>2s} {architecture:9s} nll={mean:.4f} +/- {1.96 * se:.4f} params={params:,}")
        model.cpu()
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    pd.DataFrame(rows).to_csv(out_dir / "shared_eval_batches.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "shared_eval_summary.csv", index=False)
    metadata = {
        "eval_source": str(three_m_dir / "eval_corpus.txt"),
        "eval_pool_tokens": int(tokens.size),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "evaluated_tokens_per_model": args.batches * args.batch_size * args.block_size,
        "seed": args.seed,
        "device": args.device,
        "labels": [args.one_label, args.three_label],
    }
    (out_dir / "shared_eval_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
