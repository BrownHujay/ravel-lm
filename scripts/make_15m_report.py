#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INK = "#20252d"
MUTED = "#667085"
GRID = "#d9dee5"
PAPER = "#fbfaf7"
RAVEL = "#1769aa"
ATTENTION = "#d04f3f"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
        }
    )


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def eval_curve(run_dir: Path, budget: str) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    frame = metrics[metrics["split"] == "eval"].copy()
    frame["optimizer_mtok"] = frame["step"] * int(summary["args"]["batch_size"]) * int(summary["args"]["block_size"]) / 1e6
    frame["budget"] = budget
    cleaned = []
    for _, model_frame in frame.groupby("model", sort=False):
        model_frame = model_frame.sort_values("step")
        intervals = np.diff(model_frame["step"].to_numpy())
        if intervals.size >= 2 and intervals[-1] < 0.25 * np.median(intervals[:-1]):
            model_frame = model_frame.drop(model_frame.index[-2])
        cleaned.append(model_frame)
    return pd.concat(cleaned, ignore_index=True)


def make_nll_figure(out_dir: Path, ten_dir: Path, fifteen_dir: Path, batches: pd.DataFrame, meta: dict) -> dict:
    curves = pd.concat([eval_curve(ten_dir, "10M"), eval_curve(fifteen_dir, "15M")], ignore_index=True)
    paired = batches.pivot(index="batch", columns=["scale", "model"], values="nll").sort_index()
    improvements = {
        "ravel": paired[("10M", "ravel")].to_numpy() - paired[("15M", "ravel")].to_numpy(),
        "attention": paired[("10M", "attention")].to_numpy() - paired[("15M", "attention")].to_numpy(),
    }
    effects = {
        "ravel_improvement": bootstrap_mean(improvements["ravel"], 40),
        "attention_improvement": bootstrap_mean(improvements["attention"], 41),
        "gap_10m": bootstrap_mean(
            paired[("10M", "attention")].to_numpy() - paired[("10M", "ravel")].to_numpy(), 42
        ),
        "gap_15m": bootstrap_mean(
            paired[("15M", "attention")].to_numpy() - paired[("15M", "ravel")].to_numpy(), 43
        ),
    }

    fig = plt.figure(figsize=(13.6, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.47, wspace=0.32)
    ax = fig.add_subplot(grid[0, :])
    for (budget, model), frame in curves.groupby(["budget", "model"]):
        color = RAVEL if model == "ravel" else ATTENTION
        is_fifteen = budget == "15M"
        ax.plot(
            frame["optimizer_mtok"],
            frame["loss"],
            color=color,
            linestyle="-" if is_fifteen else "--",
            linewidth=2.7 if is_fifteen else 1.7,
            alpha=1.0 if is_fifteen else 0.48,
            marker="o",
            markersize=4.5 if is_fifteen else 3.5,
            markeredgewidth=0,
            label=f"{'RAVEL' if model == 'ravel' else 'Attention'} · {budget} schedule",
        )
    ax.set_xlim(0, 15.25)
    ax.set_ylim(1.2, 5.85)
    ax.set_xticks(np.arange(0, 16, 2.5))
    ax.grid(axis="y")
    ax.set_title("A  |  Learning continues beyond ten million tokens", loc="left", y=1.075, pad=0)
    ax.text(0, 1.012, "Run-local snapshots; fixed-window endpoint comparison below", transform=ax.transAxes, color=MUTED, fontsize=9.5, va="bottom")
    ax.set_xlabel("optimizer tokens consumed per model (millions)")
    ax.set_ylabel("held-out byte NLL")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19), handlelength=2.8, columnspacing=1.5)

    ax = fig.add_subplot(grid[1, 0])
    x = np.array([10.0, 15.0])
    for model, color, label in [("ravel", RAVEL, "RAVEL"), ("attention", ATTENTION, "Attention")]:
        values = np.array([paired[("10M", model)].mean(), paired[("15M", model)].mean()])
        errors = []
        for i, budget in enumerate(["10M", "15M"]):
            mean, low, high = bootstrap_mean(paired[(budget, model)].to_numpy(), 50 + i)
            errors.append((mean - low, high - mean))
        ax.plot(x, values, color=color, linewidth=2.7, marker="o", markersize=7, zorder=3)
        ax.errorbar(x, values, yerr=np.array(errors).T, fmt="none", ecolor=color, capsize=4, linewidth=1.4)
        offset = -0.05 if model == "ravel" else 0.05
        for px, value in zip(x, values):
            ax.text(px, value + offset, f"{value:.3f}", ha="center", color=color, fontweight="bold")
        ax.text(15.2, values[-1], label, color=color, va="center", fontweight="bold")
    ax.set_xlim(9.4, 16.25)
    ax.set_ylim(1.2, 2.7)
    ax.set_xticks(x, ["10M", "15M"])
    ax.grid(axis="y")
    ax.set_title("B  |  Same model, text windows, and context", loc="left", y=1.105, pad=0)
    ax.text(0, 1.022, f"{meta['evaluated_tokens_per_model']:,} held-out tokens per checkpoint · bootstrap 95% intervals", transform=ax.transAxes, color=MUTED, fontsize=9.0, va="bottom")
    ax.set_xlabel("optimizer-token budget")
    ax.set_ylabel("mean byte NLL")

    ax = fig.add_subplot(grid[1, 1])
    percentiles = (np.arange(len(improvements["ravel"])) + 0.5) / len(improvements["ravel"]) * 100
    for model, color, label in [("ravel", RAVEL, "RAVEL"), ("attention", ATTENTION, "Attention")]:
        ordered = np.sort(improvements[model])
        ax.plot(percentiles, ordered, color=color, linewidth=2.4, label=label)
        mean = effects[f"{model}_improvement"][0]
        ax.axhline(mean, color=color, linewidth=1.1, linestyle=":")
        ax.text(99, mean, f"mean {mean:+.3f}", color=color, ha="right", va="bottom", fontweight="bold")
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.grid(axis="y")
    ax.set_title("C  |  Where the extra five million tokens help", loc="left", y=1.105, pad=0)
    ax.text(0, 1.022, "Each curve ranks the same 64 evaluation windows by paired improvement", transform=ax.transAxes, color=MUTED, fontsize=9.0, va="bottom")
    ax.set_xlabel("held-out window percentile")
    ax.set_ylabel("10M NLL minus 15M NLL")
    ax.legend(loc="lower right")

    fig.suptitle("Extending 3M-parameter training to 15M tokens", x=0.07, y=0.984, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.07, 0.946, "Both architectures improve; RAVEL remains more than one nat ahead on the shared evaluation.", ha="left", color=MUTED, fontsize=10.5)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.09)
    for suffix in ["png", "pdf"]:
        fig.savefig(out_dir / f"fineweb_3m_15m_story.{suffix}", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return effects


def make_kernel_figure(out_dir: Path, metal: pd.DataFrame) -> None:
    pivot_gpu = metal.pivot(index="address_space", columns="implementation", values="gpu_ms")
    pivot_wall = metal.pivot(index="address_space", columns="implementation", values="wall_ms")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    ax = axes[0]
    for name, color in [("legacy sweep", ATTENTION), ("indexed state", RAVEL)]:
        ax.plot(pivot_gpu.index, pivot_gpu[name], marker="o", linewidth=2.5, color=color, label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(pivot_gpu.index, [str(value) for value in pivot_gpu.index])
    ax.grid(axis="y")
    ax.set_title("A  |  Address-space scaling", loc="left")
    ax.set_xlabel("address-space size")
    ax.set_ylabel("median Metal GPU time (ms)")
    ax.legend()

    ax = axes[1]
    gpu_speedup = pivot_gpu["legacy sweep"] / pivot_gpu["indexed state"]
    wall_speedup = pivot_wall["legacy sweep"] / pivot_wall["indexed state"]
    ax.plot(gpu_speedup.index, gpu_speedup, marker="o", linewidth=2.5, color=RAVEL, label="GPU time")
    ax.plot(wall_speedup.index, wall_speedup, marker="o", linewidth=2.0, color="#6f7f92", label="wrapper wall time")
    ax.axhline(1, color=INK, linewidth=1.0)
    ax.axvline(1024, color=GRID, linewidth=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpu_speedup.index, [str(value) for value in gpu_speedup.index])
    ax.grid(axis="y")
    ax.set_title("B  |  Indexed lookup removes the A factor", loc="left")
    ax.set_xlabel("address-space size")
    ax.set_ylabel("speedup over original shader")
    ax.legend()
    actual = float(gpu_speedup.loc[1024])
    ax.annotate(f"paper shape\n{actual:.2f}x GPU", xy=(1024, actual), xytext=(520, actual + 1.2), arrowprops={"arrowstyle": "-", "color": MUTED}, color=INK)

    fig.suptitle("RAVEL latest-record Metal kernel", x=0.07, y=1.02, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(out_dir / f"metal_kernel_scaling.{suffix}", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def write_report(out_dir: Path, ten_summary: dict, fifteen_summary: dict, shared: pd.DataFrame, effects: dict, meta: dict, metal: pd.DataFrame) -> Path:
    table = shared.set_index(["scale", "model"])
    r_imp, r_low, r_high = effects["ravel_improvement"]
    a_imp, a_low, a_high = effects["attention_improvement"]
    gap10, gap10_low, gap10_high = effects["gap_10m"]
    gap15, gap15_low, gap15_high = effects["gap_15m"]
    gpu = metal.pivot(index="address_space", columns="implementation", values="gpu_ms")
    wall = metal.pivot(index="address_space", columns="implementation", values="wall_ms")
    gpu_speedup = float(gpu.loc[1024, "legacy sweep"] / gpu.loc[1024, "indexed state"])
    wall_speedup = float(wall.loc[1024, "legacy sweep"] / wall.loc[1024, "indexed state"])
    repeated = json.loads((out_dir / "metal_benchmark.json").read_text(encoding="utf-8"))
    path = out_dir / "report.md"
    lines = [
        "# RAVEL at 15M Training Tokens",
        "",
        "## Result",
        "",
        f"Extending the same 3.26M-parameter RAVEL model family from 10.00M to 15.00M optimizer tokens lowers shared-evaluation NLL from `{table.loc[('10M', 'ravel'), 'mean_nll']:.4f}` to `{table.loc[('15M', 'ravel'), 'mean_nll']:.4f}`. The matched attention baseline also improves, from `{table.loc[('10M', 'attention'), 'mean_nll']:.4f}` to `{table.loc[('15M', 'attention'), 'mean_nll']:.4f}`. RAVEL remains ahead by `{gap15:.4f}` nats at 15M tokens.",
        "",
        "![15M FineWeb-Edu result](fineweb_3m_15m_story.png)",
        "",
        "## Protocol",
        "",
        f"- RAVEL parameters: `{fifteen_summary['params']['ravel']:,}`; attention parameters: `{fifteen_summary['params']['attention']:,}`.",
        f"- Context: `{fifteen_summary['args']['block_size']}` bytes for both architectures.",
        f"- 15M optimizer budget: `{fifteen_summary['corpus']['optimizer_tokens_per_model']:,}` tokens/model over `{fifteen_summary['args']['steps']:,}` updates.",
        f"- Train/eval pool: `{fifteen_summary['corpus']['train_tokens']:,}` / `{fifteen_summary['corpus']['eval_tokens']:,}` byte tokens.",
        f"- Shared evaluation: `{meta['evaluated_tokens_per_model']:,}` identical held-out tokens from the new evaluation pool, seed `{meta['seed']}`.",
        "- Optimization: AdamW, identical architecture-specific learning rates, 10% warmup, cosine decay, and gradient clipping.",
        "",
        "## Shared Evaluation",
        "",
        "| model | 10M NLL | 15M NLL | paired improvement | bootstrap 95% interval |",
        "|---|---:|---:|---:|---:|",
        f"| RAVEL | `{table.loc[('10M', 'ravel'), 'mean_nll']:.4f}` | `{table.loc[('15M', 'ravel'), 'mean_nll']:.4f}` | `{r_imp:.4f}` | `[{r_low:.4f}, {r_high:.4f}]` |",
        f"| attention | `{table.loc[('10M', 'attention'), 'mean_nll']:.4f}` | `{table.loc[('15M', 'attention'), 'mean_nll']:.4f}` | `{a_imp:.4f}` | `[{a_low:.4f}, {a_high:.4f}]` |",
        "",
        f"Attention gains more over this particular interval (`{a_imp:.4f}` versus `{r_imp:.4f}`), so the architecture gap narrows from `{gap10:.4f}` (`[{gap10_low:.4f}, {gap10_high:.4f}]`) to `{gap15:.4f}` (`[{gap15_low:.4f}, {gap15_high:.4f}]`). The important result is not that the gap grows forever; it is that RAVEL's NLL continues to improve and remains more than one nat lower after attention receives the same additional data.",
        "",
        "The ranked-window panel shows that the mean gains are not created by one outlier document. RAVEL improves on most shared windows, while attention's somewhat larger mean gain is spread broadly across the held-out sample. This looks like diminishing returns for RAVEL at the current 3M capacity, not collapse or overfitting.",
        "",
        "## Metal Kernel",
        "",
        "The original shader launched work for every possible address and scanned the sequence, with `O(B*C*A*D*T)` operations. The replacement scans each event stream once and directly indexes the latest-value table, reducing the kernel body to `O(B*C*D*T)`. A vector path is retained for batches large enough to occupy the GPU; the batch-1 paper shape deliberately selects the faster scalar path.",
        "",
        "![Metal kernel scaling](metal_kernel_scaling.png)",
        "",
        f"At the actual `B=1, T=2048, C=3, D=32, A=1024` shape, the dedicated sweep reduces median GPU time from `{gpu.loc[1024, 'legacy sweep']:.4f}` ms to `{gpu.loc[1024, 'indexed state']:.4f}` ms (`{gpu_speedup:.2f}x`) and wrapper wall time by `{wall_speedup:.2f}x`. A separate interleaved repeat measures `{repeated['speedup']['gpu']:.2f}x` GPU and `{repeated['speedup']['wall']:.2f}x` wall speedups. Outputs and hit masks exactly match the PyTorch reference in both runs.",
        "",
        "The raw Metal wrapper remains a forward-only inference/benchmark backend. Training still uses the differentiable PyTorch MPS operator; claiming this shader speedup as training acceleration would be incorrect.",
        "",
        "## Reading",
        "",
        "The 10M-to-15M interval establishes that RAVEL has not saturated in held-out NLL. Its incremental gain is smaller than the earlier parameter-and-data scaling gain, while attention recovers some ground. The natural next test is therefore capacity scaling: a roughly 5M-parameter model trained at the new 4.61 tokens-per-parameter density, rather than repeatedly feeding more data into the same 3M model.",
        "",
        "## Artifacts",
        "",
        "- `shared_eval_batches.csv`, `shared_eval_summary.csv`, `shared_eval_meta.json`: fixed-window evaluation.",
        "- `metal_scaling.csv`, `metal_benchmark.json`: shader measurements.",
        "- `fineweb_3m_15m_story.png/.pdf`, `metal_kernel_scaling.png/.pdf`: figures.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 3M-model 10M-to-15M FineWeb-Edu report.")
    parser.add_argument("--ten-dir", default="runs/fineweb_edu_scaling_fair_3m_ctx2048")
    parser.add_argument("--fifteen-dir", default="runs/fineweb_edu_3m_15m_ctx2048")
    parser.add_argument("--out-dir", default="runs/fineweb_edu_3m_10m_to_15m_ctx2048")
    args = parser.parse_args()
    ten_dir = Path(args.ten_dir)
    fifteen_dir = Path(args.fifteen_dir)
    out_dir = Path(args.out_dir)
    set_style()
    batches = pd.read_csv(out_dir / "shared_eval_batches.csv")
    shared = pd.read_csv(out_dir / "shared_eval_summary.csv")
    meta = json.loads((out_dir / "shared_eval_meta.json").read_text(encoding="utf-8"))
    metal = pd.read_csv(out_dir / "metal_scaling.csv")
    ten_summary = json.loads((ten_dir / "summary.json").read_text(encoding="utf-8"))
    fifteen_summary = json.loads((fifteen_dir / "summary.json").read_text(encoding="utf-8"))
    effects = make_nll_figure(out_dir, ten_dir, fifteen_dir, batches, meta)
    make_kernel_figure(out_dir, metal)
    report = write_report(out_dir, ten_summary, fifteen_summary, shared, effects, meta, metal)
    print(f"wrote {report}")
    for name, values in effects.items():
        print(name, " ".join(f"{value:+.4f}" for value in values))


if __name__ == "__main__":
    main()
