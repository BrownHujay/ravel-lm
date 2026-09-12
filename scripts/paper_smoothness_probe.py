#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
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


def encode_eval_tokens(run_dir: Path) -> np.ndarray:
    text = (run_dir / "eval_corpus.txt").read_text()
    tok = ByteTokenizer()
    return np.asarray(tok.encode(text, add_bos=True, add_eos=True), dtype=np.int64)


def sample_batch(tokens: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, tokens.size - block_size - 1, size=batch_size)
    x = np.stack([tokens[s : s + block_size] for s in starts])
    y = np.stack([tokens[s + 1 : s + block_size + 1] for s in starts])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def eval_loss(model: torch.nn.Module, tokens: np.ndarray, *, batch_size: int, block_size: int, eval_batches: int, seed: int) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(eval_batches):
        x, y = sample_batch(tokens, batch_size, block_size, rng)
        losses.append(float(model(x, y)["loss"].detach().cpu()))
    return float(np.mean(losses))


def trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def parameter_norm(params: list[torch.nn.Parameter]) -> float:
    return float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in params)).cpu())


@torch.no_grad()
def random_unit_direction(params: list[torch.nn.Parameter], rng: torch.Generator) -> list[torch.Tensor]:
    direction = [torch.randn(p.shape, generator=rng, dtype=p.dtype, device=p.device) for p in params]
    norm = torch.sqrt(sum((d.float() ** 2).sum() for d in direction))
    return [d / norm.to(d.dtype) for d in direction]


@torch.no_grad()
def add_direction(params: list[torch.nn.Parameter], direction: list[torch.Tensor], scale: float) -> None:
    for p, d in zip(params, direction):
        p.add_(d, alpha=scale)


def load_models(run_dir: Path) -> dict[str, torch.nn.Module]:
    ravel_ckpt = torch.load(run_dir / "ravel_final.pt", map_location="cpu")
    attention_ckpt = torch.load(run_dir / "attention_final.pt", map_location="cpu")

    ravel_cfg = RavelConfig(**ravel_ckpt["config"])
    attention_cfg = RavelConfig(**attention_ckpt["config"])
    ravel = RavelLM(ravel_cfg)
    attention = AttentionLM(attention_cfg, n_heads=4)
    ravel.load_state_dict(ravel_ckpt["model"])
    attention.load_state_dict(attention_ckpt["model"])
    return {"ravel": ravel, "attention": attention}


def main() -> None:
    p = argparse.ArgumentParser(description="Probe local loss smoothness around trained RAVEL/attention checkpoints")
    p.add_argument("--run-dir", default="runs/tinystories_comparison_200k_compiled_speedfix")
    p.add_argument("--out", default="paper/full/smoothness_probe.csv")
    p.add_argument("--directions", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    run_dir = ROOT / args.run_dir
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    tokens = encode_eval_tokens(run_dir)
    models = load_models(run_dir)
    radii_pct = [0.25, 0.5, 1.0, 2.0]
    rows: list[dict] = []

    for model_name, model in models.items():
        params = trainable_params(model)
        weight_norm = parameter_norm(params)
        base_loss = eval_loss(
            model,
            tokens,
            batch_size=args.batch_size,
            block_size=args.block_size,
            eval_batches=args.eval_batches,
            seed=args.seed,
        )
        rows.append(
            {
                "model": model_name,
                "direction": -1,
                "radius_pct": 0.0,
                "signed_radius_pct": 0.0,
                "loss": base_loss,
                "delta_loss": 0.0,
                "weight_norm": weight_norm,
            }
        )
        print(f"{model_name}: base loss {base_loss:.4f}, weight norm {weight_norm:.2f}")

        torch_rng = torch.Generator(device="cpu")
        torch_rng.manual_seed(args.seed + (0 if model_name == "ravel" else 1000))
        for direction_idx in range(args.directions):
            direction = random_unit_direction(params, torch_rng)
            for radius_pct in radii_pct:
                scale = weight_norm * radius_pct / 100.0
                for sign in [-1.0, 1.0]:
                    add_direction(params, direction, sign * scale)
                    loss = eval_loss(
                        model,
                        tokens,
                        batch_size=args.batch_size,
                        block_size=args.block_size,
                        eval_batches=args.eval_batches,
                        seed=args.seed + 100 * direction_idx + int(radius_pct * 1000) + (1 if sign > 0 else 2),
                    )
                    add_direction(params, direction, -sign * scale)
                    rows.append(
                        {
                            "model": model_name,
                            "direction": direction_idx,
                            "radius_pct": radius_pct,
                            "signed_radius_pct": sign * radius_pct,
                            "loss": loss,
                            "delta_loss": loss - base_loss,
                            "weight_norm": weight_norm,
                        }
                    )
            print(f"  direction {direction_idx + 1}/{args.directions}")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
