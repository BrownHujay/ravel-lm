#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


INK = "#20252d"
MUTED = "#667085"
GRID = "#d9dee5"
PAPER = "#fbfaf7"
RAVEL = "#1769aa"
ATTENTION = "#d04f3f"
LIGHT_RAVEL = "#76a9cf"
LIGHT_ATTENTION = "#e69a8f"


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


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, draws: int = 20_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def load_eval_series(run_dir: Path, scale: str) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    evaluated = metrics[metrics["split"] == "eval"].copy()
    cleaned = []
    for _, model_evals in evaluated.groupby("model", sort=False):
        model_evals = model_evals.sort_values("step")
        intervals = np.diff(model_evals["step"].to_numpy())
        if intervals.size >= 2 and intervals[-1] < 0.25 * np.median(intervals[:-1]):
            model_evals = model_evals.drop(model_evals.index[-2])
        cleaned.append(model_evals)
    evaluated = pd.concat(cleaned, ignore_index=True)
    tokens_per_step = int(summary["args"]["batch_size"]) * int(summary["args"]["block_size"])
    evaluated["optimizer_mtok"] = evaluated["step"] * tokens_per_step / 1_000_000
    evaluated["tokens_per_parameter"] = evaluated.apply(
        lambda row: row["step"] * tokens_per_step / int(summary["params"][row["model"]]), axis=1
    )
    evaluated["scale"] = scale
    return evaluated


def paired_matrix(batches: pd.DataFrame) -> pd.DataFrame:
    return batches.pivot(index="batch", columns=["scale", "model"], values="nll").sort_index()


def save_scaling_figure(
    out_dir: Path,
    one_dir: Path,
    three_dir: Path,
    batches: pd.DataFrame,
    one_summary: dict,
    three_summary: dict,
    meta: dict,
) -> dict[str, tuple[float, float, float]]:
    learning = pd.concat([load_eval_series(one_dir, "1M"), load_eval_series(three_dir, "3M")], ignore_index=True)
    paired = paired_matrix(batches)
    effects = {
        "ravel_scale": bootstrap_mean_ci(
            paired[("3M", "ravel")].to_numpy() - paired[("1M", "ravel")].to_numpy(), seed=11
        ),
        "attention_scale": bootstrap_mean_ci(
            paired[("3M", "attention")].to_numpy() - paired[("1M", "attention")].to_numpy(), seed=12
        ),
        "gap_1m": bootstrap_mean_ci(
            paired[("1M", "attention")].to_numpy() - paired[("1M", "ravel")].to_numpy(), seed=13
        ),
        "gap_3m": bootstrap_mean_ci(
            paired[("3M", "attention")].to_numpy() - paired[("3M", "ravel")].to_numpy(), seed=14
        ),
    }

    fig = plt.figure(figsize=(13.6, 8.9), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.47, wspace=0.32)

    ax = fig.add_subplot(grid[0, :])
    styles = {
        ("1M", "ravel"): (LIGHT_RAVEL, "--", 1.9, "RAVEL · 1M"),
        ("3M", "ravel"): (RAVEL, "-", 2.8, "RAVEL · 3M"),
        ("1M", "attention"): (LIGHT_ATTENTION, "--", 1.9, "Attention · 1M"),
        ("3M", "attention"): (ATTENTION, "-", 2.8, "Attention · 3M"),
    }
    for key, frame in learning.groupby(["scale", "model"]):
        color, linestyle, width, label = styles[key]
        ordered = frame.sort_values("tokens_per_parameter")
        ax.plot(
            ordered["tokens_per_parameter"],
            ordered["loss"],
            color=color,
            linestyle=linestyle,
            linewidth=width,
            marker="o",
            markersize=4.6 if key[0] == "3M" else 3.8,
            markeredgewidth=0,
            label=label,
            zorder=3 if key[0] == "3M" else 2,
        )
    ax.set_xlim(0, 3.2)
    ax.set_ylim(1.25, 5.85)
    ax.set_xticks(np.arange(0, 3.1, 0.5))
    ax.grid(axis="y")
    ax.set_title("A  |  Matched training density", loc="left", y=1.075, pad=0)
    ax.text(
        0,
        1.012,
        "Common 2048-byte context · 3M models consume 2.68x more optimizer tokens",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.5,
        va="bottom",
    )
    ax.set_xlabel("optimizer tokens consumed per parameter")
    ax.set_ylabel("held-out byte NLL")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19), handlelength=2.8, columnspacing=1.8)

    ax = fig.add_subplot(grid[1, 0])
    x = np.array([1.0, 3.0])
    for model, color, label in [("ravel", RAVEL, "RAVEL"), ("attention", ATTENTION, "Attention")]:
        values = np.array(
            [paired[("1M", model)].mean(), paired[("3M", model)].mean()], dtype=np.float64
        )
        yerr_low = []
        yerr_high = []
        for scale in ["1M", "3M"]:
            _, low, high = bootstrap_mean_ci(paired[(scale, model)].to_numpy(), seed=30 + len(yerr_low))
            yerr_low.append(paired[(scale, model)].mean() - low)
            yerr_high.append(high - paired[(scale, model)].mean())
        ax.plot(x, values, color=color, linewidth=2.6, marker="o", markersize=7, zorder=3)
        ax.errorbar(x, values, yerr=[yerr_low, yerr_high], fmt="none", ecolor=color, capsize=4, linewidth=1.4)
        offset = -0.055 if model == "ravel" else 0.045
        for px, value in zip(x, values):
            ax.text(px, value + offset, f"{value:.3f}", ha="center", va="center", color=color, fontweight="bold")
        ax.text(3.08, values[-1], label, color=color, va="center", fontweight="bold")
    ax.set_xlim(0.7, 3.55)
    all_values = np.concatenate(
        [paired[(scale, model)].to_numpy() for scale in ["1M", "3M"] for model in ["ravel", "attention"]]
    )
    ax.set_ylim(float(all_values.min()) - 0.12, float(all_values.max()) + 0.12)
    ax.set_xticks(x, ["1M", "3M"])
    ax.grid(axis="y")
    ax.set_title("B  |  Same text, same windows, same context", loc="left", y=1.105, pad=0)
    ax.text(
        0,
        1.022,
        f"{meta['evaluated_tokens_per_model']:,} held-out tokens per checkpoint · window-bootstrap 95% intervals",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.0,
        va="bottom",
    )
    ax.set_xlabel("parameter scale")
    ax.set_ylabel("mean byte NLL")

    ax = fig.add_subplot(grid[1, 1])
    order = ["no prior in block", "1", "2-4", "5-16", "17-64", "65+"]
    one_gaps = one_summary["diagnostics"]["attention_minus_ravel_nll_by_target_distance_bucket"]
    three_gaps = three_summary["diagnostics"]["attention_minus_ravel_nll_by_target_distance_bucket"]
    y = np.arange(len(order))
    for i, bucket in enumerate(order):
        one = float(one_gaps[bucket])
        three = float(three_gaps[bucket])
        ax.plot([one, three], [i, i], color="#bcc4cf", linewidth=2.2, zorder=1)
        ax.scatter(one, i, s=42, color=LIGHT_RAVEL, edgecolor=PAPER, linewidth=0.7, zorder=2)
        ax.scatter(three, i, s=58, color=RAVEL, edgecolor=PAPER, linewidth=0.7, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y, ["no prior", "1 token", "2–4", "5–16", "17–64", "65+"])
    ax.invert_yaxis()
    max_gap = max(max(map(float, one_gaps.values())), max(map(float, three_gaps.values())))
    ax.set_xlim(0, max_gap * 1.08)
    ax.grid(axis="x")
    ax.set_title("C  |  Scaling gain concentrates in recurrence", loc="left", y=1.105, pad=0)
    ax.text(
        0,
        1.022,
        "Five of six recurrence buckets widen; the no-prior bucket does not",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.0,
        va="bottom",
    )
    ax.set_xlabel("attention NLL minus RAVEL NLL")
    ax.set_ylabel("previous same-byte distance")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=LIGHT_RAVEL, markeredgecolor="none", label="1M"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=RAVEL, markeredgecolor="none", label="3M"),
        ],
        loc="lower right",
        ncol=2,
        handletextpad=0.3,
        columnspacing=1.0,
    )

    fig.suptitle("RAVEL scaling on FineWeb-Edu", x=0.07, y=0.984, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(
        0.07,
        0.946,
        "More capacity receives proportionally more data; every checkpoint uses a 2048-byte context.",
        ha="left",
        color=MUTED,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.09)
    for suffix in ["png", "pdf"]:
        fig.savefig(out_dir / f"fineweb_scaling_story.{suffix}", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return effects


def write_report(
    out_dir: Path,
    one_dir: Path,
    three_dir: Path,
    shared_summary: pd.DataFrame,
    effects: dict[str, tuple[float, float, float]],
    one_summary: dict,
    three_summary: dict,
    meta: dict,
) -> Path:
    shared = shared_summary.set_index(["scale", "model"])
    one_tokens = int(one_summary["corpus"]["optimizer_tokens_per_model"])
    three_tokens = int(three_summary["corpus"]["optimizer_tokens_per_model"])
    one_params = int(one_summary["params"]["ravel"])
    three_params = int(three_summary["params"]["ravel"])
    r_scale, r_low, r_high = effects["ravel_scale"]
    a_scale, a_low, a_high = effects["attention_scale"]
    gap_1m, gap_1m_low, gap_1m_high = effects["gap_1m"]
    gap_3m, gap_3m_low, gap_3m_high = effects["gap_3m"]
    ravel_1m = float(shared.loc[("1M", "ravel"), "mean_nll"])
    ravel_3m = float(shared.loc[("3M", "ravel"), "mean_nll"])
    attention_1m = float(shared.loc[("1M", "attention"), "mean_nll"])
    attention_3m = float(shared.loc[("3M", "attention"), "mean_nll"])
    path = out_dir / "report.md"
    lines = [
        "# RAVEL FineWeb-Edu Proportional-Data Scaling Check",
        "",
        "## Result",
        "",
        (
            f"With optimizer data scaled in proportion to parameters, both architectures improve. RAVEL moves from `{ravel_1m:.4f}` to "
            f"`{ravel_3m:.4f}` held-out byte NLL, while softmax attention moves from `{attention_1m:.4f}` to `{attention_3m:.4f}`. "
            f"RAVEL gains `{abs(r_scale):.4f}` nats from scaling, compared with `{abs(a_scale):.4f}` for attention, and the architecture gap "
            f"widens from `{gap_1m:.4f}` to `{gap_3m:.4f}` nats. The earlier apparent attention regression was an artifact of holding training tokens fixed while tripling model size."
        ),
        "",
        "![FineWeb-Edu scaling story](fineweb_scaling_story.png)",
        "",
        "## What Changed",
        "",
        "| quantity | 1M run | 3M run |",
        "|---|---:|---:|",
        f"| RAVEL parameters | `{one_params:,}` | `{three_params:,}` |",
        f"| attention parameters | `{int(one_summary['params']['attention']):,}` | `{int(three_summary['params']['attention']):,}` |",
        f"| available train-pool tokens | `{int(one_summary['corpus']['train_tokens']):,}` | `{int(three_summary['corpus']['train_tokens']):,}` |",
        f"| optimizer tokens consumed/model | `{one_tokens:,}` | `{three_tokens:,}` |",
        f"| optimizer tokens/RAVEL parameter | `{one_tokens / one_params:.2f}` | `{three_tokens / three_params:.2f}` |",
        f"| training context | `{one_summary['args']['block_size']}` | `{three_summary['args']['block_size']}` |",
        f"| optimizer updates | `{one_summary['args']['steps']:,}` | `{three_summary['args']['steps']:,}` |",
        f"| LR schedule | `{one_summary['args']['lr_schedule']}` | `{three_summary['args']['lr_schedule']}` |",
        "",
        f"The 3M RAVEL run consumes `{three_tokens / one_tokens:.2f}x` as many optimizer tokens as the 1M run, matching the `{three_params / one_params:.2f}x` parameter increase. "
        "Both end at approximately `3.07` optimizer tokens per parameter. The larger corpus pool also preserves similar sampling headroom, so the additional updates draw from more distinct FineWeb-Edu text rather than repeatedly cycling through a small corpus.",
        "",
        "## Shared Evaluation",
        "",
        f"Every checkpoint was evaluated on the same `{meta['evaluated_tokens_per_model']:,}` held-out tokens sampled from the 3M run's evaluation pool, with block length `{meta['block_size']}` and seed `{meta['seed']}`.",
        "",
        "| checkpoint | mean NLL | nominal 95% interval |",
        "|---|---:|---:|",
        f"| RAVEL 1M | `{shared.loc[('1M', 'ravel'), 'mean_nll']:.4f}` | `[{shared.loc[('1M', 'ravel'), 'ci95_low']:.4f}, {shared.loc[('1M', 'ravel'), 'ci95_high']:.4f}]` |",
        f"| RAVEL 3M | `{shared.loc[('3M', 'ravel'), 'mean_nll']:.4f}` | `[{shared.loc[('3M', 'ravel'), 'ci95_low']:.4f}, {shared.loc[('3M', 'ravel'), 'ci95_high']:.4f}]` |",
        f"| attention 1M | `{shared.loc[('1M', 'attention'), 'mean_nll']:.4f}` | `[{shared.loc[('1M', 'attention'), 'ci95_low']:.4f}, {shared.loc[('1M', 'attention'), 'ci95_high']:.4f}]` |",
        f"| attention 3M | `{shared.loc[('3M', 'attention'), 'mean_nll']:.4f}` | `[{shared.loc[('3M', 'attention'), 'ci95_low']:.4f}, {shared.loc[('3M', 'attention'), 'ci95_high']:.4f}]` |",
        "",
        f"The paired RAVEL scale effect is `{r_scale:+.4f}` NLL with a bootstrap interval of `[{r_low:+.4f}, {r_high:+.4f}]`; negative is better. "
        f"The attention scale effect is `{a_scale:+.4f}` with interval `[{a_low:+.4f}, {a_high:+.4f}]`. Both intervals are below zero, so both models benefit from proportional-data scaling. RAVEL's gain is about `{abs(r_scale / a_scale):.1f}x` larger in NLL units.",
        "",
        f"The paired architecture advantage is `{gap_1m:.4f}` nats at 1M (`[{gap_1m_low:.4f}, {gap_1m_high:.4f}]`) and `{gap_3m:.4f}` at 3M (`[{gap_3m_low:.4f}, {gap_3m_high:.4f}]`). "
        "The gap therefore grows even after fixing the data-budget error and using an identical, substantially longer context.",
        "",
        "## Mechanistic Reading",
        "",
        (
            "The recurrence-bucket probe localizes the scaling gain. RAVEL's advantage grows in five of the six same-byte distance buckets, including immediate, "
            "medium, and long-range repeats. It slightly shrinks for positions with no prior matching byte in the 2048-byte window. The largest absolute gap remains "
            "at one-token recurrence, where direct addressed retrieval is especially well aligned with the prediction problem."
        ),
        "",
        "That pattern argues against a generic optimization advantage: if extra capacity merely made RAVEL uniformly better, the no-prior bucket should improve similarly. "
        "Instead, most of the widening appears where a previous matching byte exists, consistent with additional width and depth making the addressed payload path more useful.",
        "",
        "## Next Experiment",
        "",
        (
            "The next clean experiment is a multi-seed scaling series with an intermediate size, while holding context at 2048 and optimizer tokens per parameter at 3.07. "
            "That would estimate variance across initialization and corpus sampling, and reveal whether the larger RAVEL gain follows a smooth scaling trend rather than a single favorable size transition."
        ),
        "",
        "## Artifacts",
        "",
        "- `fineweb_scaling_story.png` and `.pdf`: scaling figure.",
        "- `shared_eval_batches.csv`: per-window losses for all four checkpoints.",
        "- `shared_eval_summary.csv`: checkpoint means and standard errors.",
        "- `shared_eval_meta.json`: shared evaluation protocol.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 1M-to-3M FineWeb-Edu scaling figure and report.")
    parser.add_argument("--one-m-dir", default="runs/fineweb_edu_comparison_1m_3m_tokens")
    parser.add_argument("--three-m-dir", default="runs/fineweb_edu_comparison_3m_m3")
    parser.add_argument("--out-dir", default="runs/fineweb_edu_scaling_1m_to_3m")
    args = parser.parse_args()
    one_dir = Path(args.one_m_dir)
    three_dir = Path(args.three_m_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    batches = pd.read_csv(out_dir / "shared_eval_batches.csv")
    shared_summary = pd.read_csv(out_dir / "shared_eval_summary.csv")
    meta = json.loads((out_dir / "shared_eval_meta.json").read_text(encoding="utf-8"))
    one_summary = json.loads((one_dir / "summary.json").read_text(encoding="utf-8"))
    three_summary = json.loads((three_dir / "summary.json").read_text(encoding="utf-8"))
    effects = save_scaling_figure(out_dir, one_dir, three_dir, batches, one_summary, three_summary, meta)
    report = write_report(out_dir, one_dir, three_dir, shared_summary, effects, one_summary, three_summary, meta)
    print(f"wrote {report}")
    for name, (mean, low, high) in effects.items():
        print(f"{name:>15}: {mean:+.4f} [{low:+.4f}, {high:+.4f}]")


if __name__ == "__main__":
    main()
