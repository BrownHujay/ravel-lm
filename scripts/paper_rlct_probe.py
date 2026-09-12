#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import random
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


def encode_tokens(path: Path) -> np.ndarray:
    tok = ByteTokenizer()
    return np.asarray(tok.encode(path.read_text(), add_bos=True, add_eos=True), dtype=np.int64)


def fixed_batches(
    tokens: np.ndarray,
    *,
    batch_size: int,
    block_size: int,
    num_batches: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_batches):
        starts = rng.integers(0, tokens.size - block_size - 1, size=batch_size)
        x = np.stack([tokens[s : s + block_size] for s in starts])
        y = np.stack([tokens[s + 1 : s + block_size + 1] for s in starts])
        batches.append((torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)))
    return batches


def load_base_models(run_dir: Path) -> dict[str, torch.nn.Module]:
    ravel_ckpt = torch.load(run_dir / "ravel_final.pt", map_location="cpu")
    attention_ckpt = torch.load(run_dir / "attention_final.pt", map_location="cpu")
    ravel_cfg = RavelConfig(**ravel_ckpt["config"])
    attention_cfg = RavelConfig(**attention_ckpt["config"])
    ravel = RavelLM(ravel_cfg)
    attention = AttentionLM(attention_cfg, n_heads=4)
    ravel.load_state_dict(ravel_ckpt["model"])
    attention.load_state_dict(attention_ckpt["model"])
    return {"ravel": ravel, "attention": attention}


def trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def clone_centers(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


@torch.no_grad()
def trainable_norm(params: list[torch.nn.Parameter]) -> float:
    total = sum(float((p.detach().float() ** 2).sum().cpu()) for p in params)
    return math.sqrt(total)


@torch.no_grad()
def delta_norm_sq(params: list[torch.nn.Parameter], centers: list[torch.Tensor]) -> torch.Tensor:
    total = None
    for p, c in zip(params, centers):
        val = ((p - c).float() ** 2).sum()
        total = val if total is None else total + val
    assert total is not None
    return total


@torch.no_grad()
def restore(params: list[torch.nn.Parameter], centers: list[torch.Tensor]) -> None:
    for p, c in zip(params, centers):
        p.copy_(c)


@torch.no_grad()
def full_loss(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in batches:
        out = model(x, y)
        tokens = y.numel()
        total_loss += float(out["loss"].detach().cpu()) * tokens
        total_tokens += tokens
    return total_loss / total_tokens


def sgld_chain(
    base_model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    model_name: str,
    chain: int,
    beta: float,
    n_tokens: int,
    steps: int,
    burn: int,
    eval_every: int,
    step_size: float,
    local_radius: float,
    seed: int,
) -> tuple[list[dict], list[float]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = deepcopy(base_model)
    model.train()
    params = trainable_params(model)
    centers = clone_centers(params)
    theta_norm = trainable_norm(params)
    radius_abs = max(local_radius * theta_norm, 1e-12)
    base = full_loss(model, batches)
    losses = [base]
    records: list[dict] = []

    for step in range(1, steps + 1):
        x, y = batches[(step + chain) % len(batches)]
        model.train()
        for p in params:
            p.grad = None
        loss = model(x, y)["loss"]
        loss.backward()

        with torch.no_grad():
            for p, c in zip(params, centers):
                grad = beta * n_tokens * p.grad
                grad = grad + (p - c) / (radius_abs**2)
                noise = torch.randn_like(p) * math.sqrt(step_size)
                p.add_(grad, alpha=-0.5 * step_size)
                p.add_(noise)

        if step % eval_every == 0:
            cur_loss = full_loss(model, batches)
            losses.append(cur_loss)
            records.append(
                {
                    "model": model_name,
                    "chain": chain,
                    "step": step,
                    "phase": "sample" if step > burn else "burn",
                    "loss": cur_loss,
                    "base_checkpoint_loss": base,
                    "delta_from_checkpoint": cur_loss - base,
                    "delta_norm_pct": 100.0 * math.sqrt(float(delta_norm_sq(params, centers).cpu())) / max(theta_norm, 1e-12),
                }
            )
            print(
                f"{model_name:9s} chain={chain} step={step:04d} loss={cur_loss:.4f} "
                f"move={records[-1]['delta_norm_pct']:.2f}%",
                flush=True,
            )

    restore(params, centers)
    return records, losses


def summarize(samples: pd.DataFrame, *, beta: float, n_tokens: int) -> pd.DataFrame:
    rows = []
    for (model, chain), sub in samples[samples["phase"] == "sample"].groupby(["model", "chain"]):
        all_model = samples[(samples["model"] == model) & (samples["chain"] == chain)]
        loss_star = min(float(all_model["loss"].min()), float(all_model["base_checkpoint_loss"].iloc[0]))
        posterior_mean = float(sub["loss"].mean())
        llc = beta * n_tokens * (posterior_mean - loss_star)
        rows.append(
            {
                "model": model,
                "chain": chain,
                "n_tokens": n_tokens,
                "beta": beta,
                "loss_star": loss_star,
                "posterior_mean_loss": posterior_mean,
                "llc_lambda_hat": llc,
                "mean_delta_norm_pct": float(sub["delta_norm_pct"].mean()),
            }
        )
    chain_df = pd.DataFrame(rows)
    agg = (
        chain_df.groupby("model", as_index=False)
        .agg(
            llc_lambda_hat=("llc_lambda_hat", "mean"),
            llc_lambda_std=("llc_lambda_hat", "std"),
            loss_star=("loss_star", "mean"),
            posterior_mean_loss=("posterior_mean_loss", "mean"),
            mean_delta_norm_pct=("mean_delta_norm_pct", "mean"),
            n_tokens=("n_tokens", "first"),
            beta=("beta", "first"),
        )
        .fillna(0.0)
    )
    return chain_df, agg


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate local learning coefficient / RLCT proxy using WBIC-temperature SGLD")
    p.add_argument("--run-dir", default="runs/tinystories_comparison_200k_compiled_speedfix")
    p.add_argument("--out-dir", default="paper/full/rlct")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--chains", type=int, default=3)
    p.add_argument("--steps", type=int, default=260)
    p.add_argument("--burn", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--step-size", type=float, default=1e-8)
    p.add_argument("--local-radius", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=4242)
    args = p.parse_args()

    run_dir = ROOT / args.run_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = encode_tokens(run_dir / "train_corpus.txt")
    batches = fixed_batches(
        tokens,
        batch_size=args.batch_size,
        block_size=args.block_size,
        num_batches=args.num_batches,
        seed=args.seed,
    )
    n_tokens = args.batch_size * args.block_size * args.num_batches
    beta = 1.0 / math.log(float(n_tokens))
    models = load_base_models(run_dir)
    records: list[dict] = []

    for model_name, model in models.items():
        for chain in range(args.chains):
            chain_records, _ = sgld_chain(
                model,
                batches,
                model_name=model_name,
                chain=chain,
                beta=beta,
                n_tokens=n_tokens,
                steps=args.steps,
                burn=args.burn,
                eval_every=args.eval_every,
                step_size=args.step_size,
                local_radius=args.local_radius,
                seed=args.seed + 1000 * chain + (0 if model_name == "ravel" else 100),
            )
            records.extend(chain_records)

    samples = pd.DataFrame(records)
    chain_summary, summary = summarize(samples, beta=beta, n_tokens=n_tokens)
    samples.to_csv(out_dir / "rlct_samples.csv", index=False)
    chain_summary.to_csv(out_dir / "rlct_chain_summary.csv", index=False)
    summary.to_csv(out_dir / "rlct_summary.csv", index=False)
    meta = vars(args) | {"n_tokens": n_tokens, "beta": beta}
    (out_dir / "rlct_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
