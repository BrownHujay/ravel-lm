#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.attention_model import AttentionLM
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def match_attention_cfg(base_cfg: RavelConfig, target_params: int, n_heads: int) -> tuple[RavelConfig, int]:
    best_cfg = deepcopy(base_cfg)
    best_gap = float("inf")
    best_params = -1
    for d_model in range(n_heads, 257, n_heads):
        trial = deepcopy(base_cfg)
        trial.d_model = d_model
        try:
            params = AttentionLM(trial, n_heads=n_heads).num_parameters
        except ValueError:
            continue
        gap = abs(params - target_params)
        if gap < best_gap:
            best_cfg = trial
            best_gap = gap
            best_params = params
    return best_cfg, best_params


def median(xs: list[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def iqr(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def attention_score_mib(batch_size: int, block_size: int, n_heads: int, n_layers: int) -> float:
    bytes_per_score_tensor = batch_size * n_heads * block_size * block_size * 4
    return bytes_per_score_tensor * n_layers / (1024**2)


def one_model(cfg: RavelConfig, model_name: str, attention_heads: int):
    if model_name == "ravel":
        return RavelLM(deepcopy(cfg))
    ravel_for_count = RavelLM(deepcopy(cfg))
    attention_cfg, _ = match_attention_cfg(deepcopy(cfg), ravel_for_count.num_parameters, attention_heads)
    return AttentionLM(attention_cfg, n_heads=attention_heads)


def run_train_step_benchmark(
    *,
    cfg: RavelConfig,
    model_name: str,
    batch_size: int,
    block_size: int,
    attention_heads: int,
    compile_model: bool,
    compile_mode: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = one_model(cfg, model_name, attention_heads)
    params = trainable_parameters(model)
    exec_model = torch.compile(model, mode=compile_mode) if compile_model else model
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.1)
    x = torch.randint(0, cfg.vocab_size, (batch_size, block_size), dtype=torch.long)
    y = torch.randint(0, cfg.vocab_size, (batch_size, block_size), dtype=torch.long)

    compile_or_warmup_ms = []
    for _ in range(warmup):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = exec_model(x, y)["loss"]
        loss.backward()
        optimizer.step()
        compile_or_warmup_ms.append((time.perf_counter() - t0) * 1000.0)

    step_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = exec_model(x, y)["loss"]
        loss.backward()
        optimizer.step()
        step_ms.append((time.perf_counter() - t0) * 1000.0)

    med_ms = median(step_ms)
    tokens = batch_size * block_size
    return {
        "model": model_name,
        "compiled": bool(compile_model),
        "mode": "compiled" if compile_model else "eager",
        "batch_size": batch_size,
        "block_size": block_size,
        "tokens_per_step": tokens,
        "params": model.num_parameters,
        "step_ms_median": med_ms,
        "step_ms_iqr": iqr(step_ms),
        "tokens_per_sec": tokens / max(med_ms / 1000.0, 1e-9),
        "warmup_ms_first": float(compile_or_warmup_ms[0]) if compile_or_warmup_ms else math.nan,
        "warmup_ms_median": median(compile_or_warmup_ms) if compile_or_warmup_ms else math.nan,
        "attention_score_mib_per_fwd": attention_score_mib(batch_size, block_size, attention_heads, cfg.n_layers),
    }


def plot_context_scaling(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["suite"] == "context"].copy()
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for mode, marker in [("eager", "o"), ("compiled", "s")]:
        m = sub[sub["mode"] == mode]
        sns.lineplot(data=m, x="block_size", y="tokens_per_sec", hue="model", marker=marker, ax=axes[0], legend=mode == "eager")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Throughput vs context length")
    axes[0].set_xlabel("sequence length T")
    axes[0].set_ylabel("tokens/sec")

    sns.lineplot(data=sub, x="block_size", y="step_ms_median", hue="model", style="mode", marker="o", ax=axes[1])
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_title("Train step time")
    axes[1].set_xlabel("sequence length T")
    axes[1].set_ylabel("median ms")

    pivot = sub.pivot_table(index=["block_size", "mode"], columns="model", values="tokens_per_sec")
    speedup_rows = []
    for (block, mode), row in pivot.iterrows():
        if "ravel" in row and "attention" in row:
            speedup_rows.append({"block_size": block, "mode": mode, "ravel_vs_attention": row["ravel"] / row["attention"]})
    speed = pd.DataFrame(speedup_rows)
    sns.lineplot(data=speed, x="block_size", y="ravel_vs_attention", hue="mode", marker="o", ax=axes[2])
    axes[2].axhline(1.0, color="black", linewidth=1)
    axes[2].set_xscale("log", base=2)
    axes[2].set_title("RAVEL throughput ratio")
    axes[2].set_xlabel("sequence length T")
    axes[2].set_ylabel("RAVEL / attention")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_token_scaling(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["suite"] == "tokens"].copy()
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    sns.lineplot(data=sub, x="tokens_per_step", y="tokens_per_sec", hue="model", style="mode", marker="o", ax=axes[0])
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Throughput vs tokens/step")
    axes[0].set_xlabel("tokens per train step")
    axes[0].set_ylabel("tokens/sec")

    sns.lineplot(data=sub, x="tokens_per_step", y="step_ms_median", hue="model", style="mode", marker="o", ax=axes[1])
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_title("Step time at fixed T=512")
    axes[1].set_xlabel("tokens per train step")
    axes[1].set_ylabel("median ms")

    pivot = sub.pivot_table(index=["tokens_per_step", "mode"], columns="model", values="tokens_per_sec")
    speedup_rows = []
    for (tokens, mode), row in pivot.iterrows():
        if "ravel" in row and "attention" in row:
            speedup_rows.append({"tokens_per_step": tokens, "mode": mode, "ravel_vs_attention": row["ravel"] / row["attention"]})
    speed = pd.DataFrame(speedup_rows)
    sns.lineplot(data=speed, x="tokens_per_step", y="ravel_vs_attention", hue="mode", marker="o", ax=axes[2])
    axes[2].axhline(1.0, color="black", linewidth=1)
    axes[2].set_xscale("log", base=2)
    axes[2].set_title("RAVEL throughput ratio")
    axes[2].set_xlabel("tokens per train step")
    axes[2].set_ylabel("RAVEL / attention")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_memory_pressure(df: pd.DataFrame, out: Path) -> None:
    context = df[(df["suite"] == "context") & (df["model"] == "attention") & (df["mode"] == "compiled")].copy()
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(context["block_size"], context["tokens_per_sec"], marker="o", label="attention throughput")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("sequence length T")
    ax1.set_ylabel("attention tokens/sec")
    ax2 = ax1.twinx()
    ax2.plot(context["block_size"], context["attention_score_mib_per_fwd"], marker="s", color="#dd8452", label="score tensor MiB")
    ax2.set_ylabel("attention score tensor MiB / forward")
    ax1.set_title("Quadratic score tensor pressure")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def speedup_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["suite", "mode", "batch_size", "block_size", "tokens_per_step"],
        columns="model",
        values=["tokens_per_sec", "step_ms_median"],
    )
    rows = []
    for idx, row in pivot.iterrows():
        suite, mode, batch_size, block_size, tokens_per_step = idx
        if ("tokens_per_sec", "ravel") not in row or ("tokens_per_sec", "attention") not in row:
            continue
        r_tps = float(row[("tokens_per_sec", "ravel")])
        a_tps = float(row[("tokens_per_sec", "attention")])
        r_ms = float(row[("step_ms_median", "ravel")])
        a_ms = float(row[("step_ms_median", "attention")])
        rows.append(
            {
                "suite": suite,
                "mode": mode,
                "batch_size": int(batch_size),
                "block_size": int(block_size),
                "tokens_per_step": int(tokens_per_step),
                "ravel_tokens_per_sec": r_tps,
                "attention_tokens_per_sec": a_tps,
                "ravel_step_ms": r_ms,
                "attention_step_ms": a_ms,
                "ravel_speedup": r_tps / a_tps,
                "ravel_step_time_ratio": r_ms / a_ms,
            }
        )
    return pd.DataFrame(rows).sort_values(["suite", "mode", "block_size", "tokens_per_step"])


def plot_speedup_summary(speed: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    context = speed[speed["suite"] == "context"].copy()
    sns.barplot(data=context, x="block_size", y="ravel_speedup", hue="mode", ax=axes[0])
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].set_title("Context sweep speedup")
    axes[0].set_xlabel("sequence length T")
    axes[0].set_ylabel("RAVEL / attention throughput")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2fx", fontsize=8, padding=2)

    tokens = speed[speed["suite"] == "tokens"].copy()
    sns.barplot(data=tokens, x="tokens_per_step", y="ravel_speedup", hue="mode", ax=axes[1])
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_title("Fixed T=512 token-count sweep")
    axes[1].set_xlabel("tokens per step")
    axes[1].set_ylabel("RAVEL / attention throughput")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2fx", fontsize=8, padding=2)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_speedup_markdown(speed: pd.DataFrame, out: Path) -> None:
    lines = [
        "# RAVEL Speedup Table",
        "",
        "| suite | mode | B | T | tokens/step | RAVEL tok/s | attention tok/s | speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in speed.itertuples(index=False):
        lines.append(
            f"| {row.suite} | {row.mode} | {row.batch_size} | {row.block_size} | {row.tokens_per_step} | "
            f"{row.ravel_tokens_per_sec:,.0f} | {row.attention_tokens_per_sec:,.0f} | {row.ravel_speedup:.2f}x |"
        )
    out.write_text("\n".join(lines) + "\n")


def write_report(out_dir: Path, df: pd.DataFrame, args: argparse.Namespace, meta: dict) -> None:
    lines = [
        "# RAVEL Speed Scaling Notes",
        "",
        "## Setup",
        "",
        f"- Config: `{args.config}`",
        f"- Host: `{meta['platform']}`",
        f"- PyTorch: `{meta['torch_version']}`",
        f"- Threads: `{meta['torch_threads']}`",
        f"- Attention heads: `{args.attention_heads}`",
        f"- Warmup/measured steps: `{args.warmup}` / `{args.iters}`",
        f"- Compile modes measured: `{', '.join(args.modes)}`",
        "",
        "## Figures",
        "",
        "![Context scaling](paper_context_scaling.png)",
        "",
        "![Token scaling](paper_token_scaling.png)",
        "",
        "![Attention memory pressure](paper_attention_memory_pressure.png)",
        "",
        "![Speedup summary](paper_speedup_summary.png)",
        "",
        "## Reading",
        "",
        "These are synthetic-token train-step timings, not quality results. They isolate implementation speed while keeping memory bounded. The context sweep keeps roughly 1024 tokens per step and increases sequence length, which is the cleanest place to see the expected RAVEL/attention crossover. The token sweep fixes `T=512` and increases batch/token count, which checks whether the crossover is only a batch-size artifact.",
        "",
    ]
    for suite in ["context", "tokens"]:
        sub = df[df["suite"] == suite]
        pivot = sub.pivot_table(index=[sub["block_size"] if suite == "context" else sub["tokens_per_step"], "mode"], columns="model", values="tokens_per_sec")
        best_rows = []
        for (x, mode), row in pivot.iterrows():
            if "ravel" in row and "attention" in row:
                best_rows.append((x, mode, row["ravel"] / row["attention"]))
        if best_rows:
            strongest = max(best_rows, key=lambda item: item[2])
            weakest = min(best_rows, key=lambda item: item[2])
            label = "T" if suite == "context" else "tokens/step"
            lines.extend(
                [
                    f"- `{suite}` sweep strongest RAVEL ratio: `{strongest[2]:.2f}x` at {label} `{int(strongest[0])}` in `{strongest[1]}` mode.",
                    f"- `{suite}` sweep weakest RAVEL ratio: `{weakest[2]:.2f}x` at {label} `{int(weakest[0])}` in `{weakest[1]}` mode.",
                ]
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `speed_scaling.csv`",
            "- `speedup_table.csv`",
            "- `speedup_table.md`",
            "- `speed_scaling_summary.json`",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-oriented RAVEL vs attention speed scaling benchmark")
    p.add_argument("--config", default=str(ROOT / "configs" / "byte" / "ravel_200k_byte.json"))
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "paper_speed_scaling"))
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument("--modes", nargs="+", choices=["eager", "compiled"], default=["eager", "compiled"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--max-attention-score-mib", type=float, default=384.0)
    args = p.parse_args()

    torch.set_num_threads(args.torch_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = RavelConfig.from_json(args.config)

    # Keep these intentionally conservative for a laptop. The context sweep holds
    # B*T near 1024; the token sweep fixes T=512 and increases batch size.
    planned = [
        ("context", 8, 128),
        ("context", 4, 256),
        ("context", 2, 512),
        ("context", 1, 1024),
        ("tokens", 1, 512),
        ("tokens", 2, 512),
        ("tokens", 4, 512),
    ]

    rows = []
    for suite, batch_size, block_size in planned:
        score_mib = attention_score_mib(batch_size, block_size, args.attention_heads, base_cfg.n_layers)
        if score_mib > args.max_attention_score_mib:
            print(f"skip {suite} B={batch_size} T={block_size}: score tensor estimate {score_mib:.1f} MiB")
            continue
        cfg = deepcopy(base_cfg)
        cfg.batch_size = batch_size
        cfg.block_size = block_size
        cfg.validate()
        for mode in args.modes:
            for model_name in ["ravel", "attention"]:
                print(f"bench suite={suite:7s} mode={mode:8s} model={model_name:9s} B={batch_size} T={block_size}")
                row = run_train_step_benchmark(
                    cfg=cfg,
                    model_name=model_name,
                    batch_size=batch_size,
                    block_size=block_size,
                    attention_heads=args.attention_heads,
                    compile_model=mode == "compiled",
                    compile_mode=args.compile_mode,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed + len(rows),
                )
                row["suite"] = suite
                rows.append(row)
                print(f"  median_ms={row['step_ms_median']:.3f} tok/s={row['tokens_per_sec']:.0f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "speed_scaling.csv", index=False)
    speed = speedup_table(df)
    speed.to_csv(out_dir / "speedup_table.csv", index=False)
    write_speedup_markdown(speed, out_dir / "speedup_table.md")
    plot_context_scaling(df, out_dir / "paper_context_scaling.png")
    plot_token_scaling(df, out_dir / "paper_token_scaling.png")
    plot_memory_pressure(df, out_dir / "paper_attention_memory_pressure.png")
    plot_speedup_summary(speed, out_dir / "paper_speedup_summary.png")
    meta = {
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "args": vars(args),
    }
    (out_dir / "speed_scaling_summary.json").write_text(json.dumps(meta, indent=2))
    write_report(out_dir, df, args, meta)
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
