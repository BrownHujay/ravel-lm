#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INK = "#20242C"
MUTED = "#657080"
BLUE = "#2F6FBB"
GREEN = "#2F9D74"
CORAL = "#D95F4F"
GOLD = "#C69214"
PURPLE = "#6F5AA8"
GRID = "#D8DDE6"
PAPER = "#FBFBF8"
BLACK = INK
RED = CORAL
GRAY = MUTED


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 13,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#293241",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "lines.linewidth": 2.3,
        }
    )


def save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", facecolor=PAPER)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def box(ax: plt.Axes, xy, wh, label: str, fc: str = "#FFFFFF", ec: str | None = None, fontsize: float = 10, lw: float = 0.0, color: str | None = None):
    x, y = xy
    w, h = wh
    if ec is None:
        ec = fc
    if color is None:
        hfc = fc.lstrip("#")
        if len(hfc) == 6:
            r, g, b = [int(hfc[i : i + 2], 16) / 255 for i in (0, 2, 4)]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color = INK if lum > 0.62 else "white"
        else:
            color = INK
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, wrap=True, color=color, fontweight="bold", linespacing=1.08)
    return patch


def arrow(ax: plt.Axes, a, b, color=INK, lw: float = 1.7, rad: float = 0.0):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def fig_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.97, "RAVEL-LM replaces dense similarity attention with exact addressed event memory", ha="center", va="top", fontsize=15, fontweight="bold")

    box(ax, (0.04, 0.68), (0.13, 0.12), "byte tokens", fc="#F3F6FA", ec=GRAY)
    box(ax, (0.22, 0.68), (0.16, 0.12), "token + position\nembedding", fc="#F3F6FA", ec=GRAY)
    box(ax, (0.43, 0.68), (0.22, 0.12), "RAVEL block\n(repeated L times)", fc="#E8F1F8", ec=BLUE, fontsize=10.5, lw=2)
    box(ax, (0.72, 0.68), (0.13, 0.12), "RMSNorm", fc="#F3F6FA", ec=GRAY)
    box(ax, (0.89, 0.68), (0.08, 0.12), "logits", fc="#F3F6FA", ec=GRAY)
    arrow(ax, (0.17, 0.74), (0.22, 0.74))
    arrow(ax, (0.38, 0.74), (0.43, 0.74))
    arrow(ax, (0.65, 0.74), (0.72, 0.74))
    arrow(ax, (0.85, 0.74), (0.89, 0.74))

    box(ax, (0.12, 0.37), (0.18, 0.13), "local mixer\ncausal depthwise conv", fc="#FFF9E8", ec="#B88400", fontsize=9.5)
    box(ax, (0.41, 0.37), (0.20, 0.13), "event memory\nlatest matching address", fc="#EAF6EF", ec=GREEN, fontsize=9.5, lw=2)
    box(ax, (0.72, 0.37), (0.16, 0.13), "SwiGLU FFN", fc="#FFF9E8", ec="#B88400", fontsize=9.5)
    arrow(ax, (0.30, 0.435), (0.41, 0.435))
    arrow(ax, (0.61, 0.435), (0.72, 0.435))
    ax.text(0.50, 0.56, "inside each block", ha="center", fontsize=10.5, color=GRAY)
    arrow(ax, (0.54, 0.68), (0.50, 0.50), color=BLUE, rad=0.12)

    box(ax, (0.15, 0.09), (0.18, 0.12), "address\nliteral byte / bigram\n+ hard product code", fc="#FFFFFF", ec=GREEN, fontsize=8.8)
    box(ax, (0.42, 0.09), (0.16, 0.12), "payload\nlinear projection", fc="#FFFFFF", ec=GREEN, fontsize=8.8)
    box(ax, (0.67, 0.09), (0.19, 0.12), "read previous record\nsame address, time < t", fc="#FFFFFF", ec=GREEN, fontsize=8.8)
    arrow(ax, (0.33, 0.15), (0.42, 0.15), color=GREEN)
    arrow(ax, (0.58, 0.15), (0.67, 0.15), color=GREEN)
    arrow(ax, (0.51, 0.37), (0.51, 0.21), color=GREEN)
    ax.text(0.50, 0.015, "No QK^T matrix is formed. Long-range recurrence is a key lookup over an event tape.", ha="center", fontsize=10.5, color=BLACK)
    save(fig, out_dir / "fig_architecture")


def fig_event_lookup(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.text(5, 4.05, "Exact latest-record memory is a causal address lookup", ha="center", va="top", fontsize=14, fontweight="bold")

    times = np.arange(8)
    labels = ["A", "B", "A", "C", "B", "A", "D", "A"]
    xs = np.linspace(1.0, 8.2, len(times))
    y = 2.5
    for i, (x, lab) in enumerate(zip(xs, labels)):
        color = BLUE if lab == "A" else "#E5E7EB"
        ec = BLUE if lab == "A" else GRAY
        box(ax, (x - 0.32, y - 0.22), (0.64, 0.44), f"{lab}\nt={i}", fc=color if lab == "A" else "#FFFFFF", ec=ec, fontsize=8.5)
    ax.plot([xs[0] - 0.55, xs[-1] + 0.55], [y, y], color=GRAY, linewidth=1)
    ax.text(0.35, y, "event tape", ha="right", va="center", color=GRAY)
    query_x = 8.95
    box(ax, (query_x - 0.38, y - 0.28), (0.76, 0.56), "read A\nat t=8", fc="#EAF6EF", ec=GREEN, fontsize=9.5, lw=2)
    arrow(ax, (query_x - 0.40, y), (xs[7] + 0.35, y), color=GREEN, rad=0.18)
    ax.text(7.8, 3.25, "latest previous A", color=GREEN, ha="center", fontsize=10, fontweight="bold")
    ax.text(5, 0.98, r"composite key = ((batch, head, address) $\times$ (T+1)) + time", ha="center", fontsize=11)
    ax.text(5, 0.48, r"sort keys $\rightarrow$ search predecessor $\rightarrow$ gather payload", ha="center", fontsize=11, color=BLACK)
    save(fig, out_dir / "fig_event_lookup")


def fig_loss_and_training(metrics: pd.DataFrame, out_dir: Path) -> None:
    eval_df = metrics[metrics["split"] == "eval"].copy()
    train_df = metrics[metrics["split"] == "train"].copy()
    fig = plt.figure(figsize=(11.2, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 0.95], wspace=0.16)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("The loss gap opens immediately, then persists", loc="left", pad=12, fontweight="bold")
    sub_r = eval_df[eval_df["model"] == "ravel"].sort_values("step")
    sub_a = eval_df[eval_df["model"] == "attention"].sort_values("step")
    steps = sub_r["step"].to_numpy()
    r_loss_arr = sub_r["loss"].to_numpy()
    a_loss_arr = sub_a["loss"].to_numpy()
    ax.fill_between(steps, r_loss_arr, a_loss_arr, where=a_loss_arr >= r_loss_arr, color=BLUE, alpha=0.13)
    for model, color in [("ravel", BLUE), ("attention", RED)]:
        sub = eval_df[eval_df["model"] == model].sort_values("step")
        ax.plot(sub["step"], sub["loss"], marker="o", markersize=5.5, linewidth=3.0, color=color, label=model)
    ax.set_xlabel("training step")
    ax.set_ylabel("eval cross-entropy")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    r_final = float(eval_df[(eval_df["model"] == "ravel")].sort_values("step").tail(1)["loss"].iloc[0])
    a_final = float(eval_df[(eval_df["model"] == "attention")].sort_values("step").tail(1)["loss"].iloc[0])
    ax.annotate(
        f"{a_final-r_final:.2f} nat gap",
        xy=(800, r_final),
        xytext=(535, 3.05),
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.5},
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.08,
        "same corpus, same tokenizer,\nmatched parameter budget",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steady = train_df[train_df["step"] > 1].copy()
    med = steady.groupby("model")["tokens_per_sec"].median()
    ax.text(0.02, 0.93, "run summary", fontsize=15, fontweight="bold", ha="left")
    box(ax, (0.04, 0.64), (0.88, 0.19), f"RAVEL final loss\n{r_final:.3f}", fc=BLUE, fontsize=15)
    box(ax, (0.04, 0.40), (0.88, 0.19), f"attention final loss\n{a_final:.3f}", fc=CORAL, fontsize=15)
    speed_ratio = float(med["ravel"] / med["attention"])
    ax.text(0.04, 0.27, "compiled steady-state throughput", fontsize=9.5, color=MUTED, ha="left")
    ax.plot([0.06, 0.88], [0.18, 0.18], color="#C9D1DD", linewidth=7, solid_capstyle="round")
    ax.plot([0.06, 0.06 + 0.82 * min(speed_ratio / 1.25, 1.0)], [0.18, 0.18], color=GREEN, linewidth=7, solid_capstyle="round")
    ax.text(0.06, 0.08, f"attention\n{med['attention']/1000:.1f}k tok/s", color=CORAL, fontsize=9.5, fontweight="bold", ha="left", linespacing=1.05)
    ax.text(0.88, 0.08, f"RAVEL\n{med['ravel']/1000:.1f}k", color=BLUE, fontsize=9.5, fontweight="bold", ha="right", linespacing=1.05)
    ax.text(0.47, 0.205, f"{speed_ratio:.2f}x", color=GREEN, fontsize=13, fontweight="bold", ha="center")
    save(fig, out_dir / "fig_loss_training")


def fig_mechanism_diagnostics(diag: pd.DataFrame, out_dir: Path) -> None:
    order = ["no prior in block", "1", "2-4", "5-16", "17-64", "65+"]
    class_order = ["space", "lowercase", "uppercase", "punctuation", "whitespace", "special"]
    fig = plt.figure(figsize=(11.2, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylim(-0.7, 1.55)
    ax.axis("off")
    bucket = diag.pivot_table(index="read_distance_bucket", columns="model", values="nll", aggfunc="mean").reindex(order)
    gap = bucket["attention"] - bucket["ravel"]
    freq = diag.groupby("read_distance_bucket").size().reindex(order).fillna(0)
    freq = freq / max(float(freq.max()), 1.0)
    ax.text(-0.65, 1.42, "recurrence atlas", fontsize=15, fontweight="bold", ha="left")
    ax.text(-0.65, 1.25, "circle size = how often the read situation occurs\ncircle color/value = attention NLL - RAVEL NLL", fontsize=9.5, color=MUTED, ha="left")
    ax.plot([0, len(order) - 1], [0.42, 0.42], color="#C8CFDA", linewidth=2)
    for i, label in enumerate(order):
        val = float(gap.loc[label])
        size = 520 + 4200 * float(freq.loc[label])
        ax.scatter([i], [0.42], s=size, color=BLUE if val >= 0 else CORAL, edgecolor=PAPER, linewidth=2.2, zorder=3)
        ax.text(i, 0.42, f"{val:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(i, -0.02, "none" if label == "no prior in block" else label, ha="center", va="top", fontsize=9.5, fontweight="bold")
    ax.annotate(
        "strongest gain when the\nread key just recurred",
        xy=(1, 0.67),
        xytext=(2.15, 1.05),
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.5},
        fontsize=10,
        color=INK,
    )

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(class_order) + 1.0)
    ax.axis("off")
    cls = diag.pivot_table(index="token_class", columns="model", values="nll", aggfunc="mean").reindex(class_order).dropna()
    cls_gap = cls["attention"] - cls["ravel"]
    ax.text(0.02, len(class_order) + 0.75, "byte-class signature", fontsize=15, fontweight="bold", ha="left")
    max_abs = max(float(cls_gap.abs().max()), 1e-6)
    for row, (label, val) in enumerate(cls_gap.items()):
        y = len(class_order) - row - 0.1
        ax.text(0.02, y, label, ha="left", va="center", fontsize=10.5, fontweight="bold")
        ax.plot([0.34, 0.86], [y, y], color="#D8DEE8", linewidth=9, solid_capstyle="round")
        if val >= 0:
            ax.plot([0.34, 0.34 + 0.52 * val / max_abs], [y, y], color=BLUE, linewidth=9, solid_capstyle="round")
        else:
            ax.plot([0.34, 0.34 + 0.52 * abs(val) / max_abs], [y, y], color=CORAL, linewidth=9, solid_capstyle="round")
        ax.text(0.98, y, f"{val:+.2f}", ha="right", va="center", fontsize=10, fontweight="bold", color=BLUE if val >= 0 else CORAL)
    ax.text(0.34, 0.35, "RAVEL better", color=BLUE, fontsize=9.5, fontweight="bold")
    ax.text(0.92, 0.35, "larger gap", color=MUTED, fontsize=9.5, ha="right")
    save(fig, out_dir / "fig_mechanism_diagnostics")


def fig_context_crossover(speed: pd.DataFrame, out_dir: Path) -> None:
    context = speed[speed["suite"] == "context"].copy()
    fig = plt.figure(figsize=(11.2, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Attention step time\nbends upward with context", loc="left", pad=12, fontweight="bold")
    for mode, alpha, ls in [("eager", 0.55, "--"), ("compiled", 1.0, "-")]:
        sub = context[context["mode"] == mode].sort_values("block_size")
        ax.plot(sub["block_size"], sub["attention_step_ms"], color=CORAL, linestyle=ls, alpha=alpha, marker="o", label=f"attention {mode}")
        ax.plot(sub["block_size"], sub["ravel_step_ms"], color=BLUE, linestyle=ls, alpha=alpha, marker="o", label=f"RAVEL {mode}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512, 1024])
    ax.set_xticklabels(["128", "256", "512", "1024"])
    ax.set_xlabel("context length T, with B x T near 1024")
    ax.set_ylabel("median train step time (ms)")
    ax.grid(axis="y", color=GRID)
    ax.text(128, 43, "solid = compiled\ndashed = eager", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.18))

    ax = fig.add_subplot(gs[0, 1])
    sub_c = context[context["mode"] == "compiled"].sort_values("block_size")
    x = sub_c["block_size"].to_numpy()
    y = sub_c["ravel_speedup"].to_numpy()
    ax.set_title("Crossover follows\ncontext length", loc="left", pad=12, fontweight="bold")
    ax.axhspan(1, max(4.25, y.max() + 0.2), color=BLUE, alpha=0.10)
    ax.axhspan(0.65, 1, color=CORAL, alpha=0.12)
    ax.plot(x, y, color=INK, marker="o", linewidth=3.2)
    ax.axhline(1, color=INK, linewidth=1.2)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.11, f"{yi:.2f}x", ha="center", fontsize=10, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512, 1024])
    ax.set_xticklabels(["128", "256", "512", "1024"])
    ax.set_xlabel("context length T")
    ax.set_ylabel("throughput ratio: RAVEL / attention")
    ax.grid(axis="y", color=GRID)
    ax.text(135, 1.17, "RAVEL faster", color=BLUE, fontsize=9.5, fontweight="bold")
    ax.text(135, 0.82, "attention faster", color=CORAL, fontsize=9.5, fontweight="bold")
    save(fig, out_dir / "fig_context_crossover")


def fig_fixed_t_scaling(speed: pd.DataFrame, out_dir: Path) -> None:
    token = speed[(speed["suite"] == "tokens") & (speed["mode"] == "compiled")].sort_values("tokens_per_step").copy()
    fig = plt.figure(figsize=(10.8, 4.6))
    ax = fig.add_subplot(111)
    ax.set_title("At fixed T=512, batching exposes the linear-memory advantage", loc="left", pad=14, fontweight="bold")
    x = token["tokens_per_step"].to_numpy()
    r = token["ravel_tokens_per_sec"].to_numpy() / 1000.0
    a = token["attention_tokens_per_sec"].to_numpy() / 1000.0
    ax.fill_between(x, a, r, color=BLUE, alpha=0.12)
    ax.plot(x, r, marker="o", color=BLUE, linewidth=3.0)
    ax.plot(x, a, marker="o", color=CORAL, linewidth=3.0)
    for xi, ri, ai, sp in zip(x, r, a, token["ravel_speedup"]):
        ax.text(xi, ri + 4, f"{sp:.2f}x", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(x[-1] + 35, r[-1], "RAVEL", color=BLUE, fontweight="bold", va="center")
    ax.text(x[-1] + 35, a[-1], "attention", color=CORAL, fontweight="bold", va="center")
    ax.set_xlabel("tokens per step at fixed context length")
    ax.set_ylabel("thousand tokens/sec")
    ax.set_xticks(x)
    ax.grid(axis="y", color=GRID)
    ax.spines["right"].set_visible(False)
    save(fig, out_dir / "fig_fixed_t_scaling")


def fig_attention_pressure(speed: pd.DataFrame, out_dir: Path) -> None:
    context = speed[(speed["suite"] == "context") & (speed["mode"] == "compiled")].sort_values("block_size").copy()
    T = context["block_size"].to_numpy()
    # Matches the existing pressure figure: three layers, four heads, fp32 scores, B*T ~= 1024.
    score_mib = np.array([6, 12, 24, 48], dtype=float)
    fig, ax = plt.subplots(figsize=(10.3, 4.5))
    ax.set_title("The object RAVEL avoids is the quadratic score tensor", loc="left", pad=14, fontweight="bold")
    ax.fill_between(T, 0, score_mib, color=CORAL, alpha=0.14)
    ax.plot(T, score_mib, color=CORAL, marker="o", linewidth=3.2)
    for xi, yi in zip(T, score_mib):
        ax.text(xi, yi + 1.5, f"{int(yi)} MiB", ha="center", color=CORAL, fontsize=11, fontweight="bold")
    ax.text(145, 38, "B x T is held near 1024,\nso the growth comes from T itself.", color=MUTED, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512, 1024])
    ax.set_xticklabels(["128", "256", "512", "1024"])
    ax.set_xlabel("context length T")
    ax.set_ylabel("attention score tensor MiB / forward")
    ax.grid(axis="y", color=GRID)
    save(fig, out_dir / "fig_attention_pressure")


def fig_rlct_probe(samples: pd.DataFrame, chain_summary: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(11.2, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.25], wspace=0.22)
    colors = {"ravel": BLUE, "attention": CORAL}

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("WBIC local-learning coefficient", loc="left", pad=12, fontweight="bold")
    ax.set_xlim(0, 0.36)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["RAVEL", "attention"], fontweight="bold")
    ax.set_xlabel(r"$\hat{\lambda}=\beta n(\mathbb{E}_{\beta}[L]-L^\star)$")
    ax.grid(axis="x", color=GRID)
    for y, model in enumerate(["ravel", "attention"]):
        sub = chain_summary[chain_summary["model"] == model]
        vals = sub["llc_lambda_hat"].to_numpy()
        mean = float(summary[summary["model"] == model]["llc_lambda_hat"].iloc[0])
        std = float(summary[summary["model"] == model]["llc_lambda_std"].iloc[0])
        ax.scatter(vals, np.full_like(vals, y, dtype=float), s=58, color=colors[model], alpha=0.58, edgecolor=PAPER, linewidth=1.2, zorder=3)
        ax.errorbar(mean, y, xerr=std, fmt="o", color=INK, ecolor=colors[model], elinewidth=3.0, capsize=6, markersize=7, zorder=4)
        ax.text(mean + std + 0.018, y, f"{mean:.3f}", va="center", fontsize=11, fontweight="bold", color=colors[model])
    ax.text(
        0.02,
        1.37,
        r"$\beta=1/\log n,\ n=2048$ tokens" + "\n3 SGLD chains per model",
        fontsize=9.5,
        color=MUTED,
    )

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Posterior samples stay in a local basin", loc="left", pad=12, fontweight="bold")
    for model in ["ravel", "attention"]:
        model_samples = samples[samples["model"] == model].copy()
        loss_star = float(summary[summary["model"] == model]["loss_star"].iloc[0])
        model_samples["excess"] = model_samples["loss"] - loss_star
        for chain, sub in model_samples.groupby("chain"):
            sub = sub.sort_values("step")
            alpha = 0.32 if chain else 0.48
            ax.plot(sub["step"], sub["excess"], color=colors[model], alpha=alpha, linewidth=1.8)
        mean = model_samples.groupby("step", as_index=False)["excess"].mean().sort_values("step")
        ax.plot(mean["step"], mean["excess"], color=colors[model], linewidth=3.2)
        last = mean.iloc[-1]
        ax.text(float(last["step"]) + 4, float(last["excess"]), model, color=colors[model], va="center", fontweight="bold")
    ax.axvline(100, color=INK, linewidth=1.0, alpha=0.55)
    ax.text(103, ax.get_ylim()[1] * 0.88, "burn-in ends", color=MUTED, fontsize=9.2)
    ax.set_xlabel("SGLD step")
    ax.set_ylabel(r"$L(\theta)-L^\star$ on fixed training tokens")
    ax.grid(axis="y", color=GRID)
    save(fig, out_dir / "fig_rlct_probe")


def copy_speed_figures(speed_dir: Path, out_dir: Path) -> None:
    for name in ["fig_context_crossover", "fig_fixed_t_scaling", "fig_attention_pressure"]:
        for suffix in [".pdf", ".png"]:
            shutil.copy2(speed_dir / f"{name}{suffix}", out_dir / f"{name}{suffix}")


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_full_tex(
    out_dir: Path,
    summary: dict,
    speed: pd.DataFrame,
    metrics: pd.DataFrame,
    diag: pd.DataFrame,
    rlct_summary: pd.DataFrame | None = None,
) -> None:
    eval_df = metrics[metrics["split"] == "eval"].copy()
    train_df = metrics[(metrics["split"] == "train") & (metrics["step"] > 1)].copy()
    final = eval_df.sort_values("step").groupby("model").tail(1).set_index("model")
    r_loss = float(final.loc["ravel", "loss"])
    a_loss = float(final.loc["attention", "loss"])
    r_tps = float(train_df[train_df["model"] == "ravel"]["tokens_per_sec"].median())
    a_tps = float(train_df[train_df["model"] == "attention"]["tokens_per_sec"].median())
    context_c = speed[(speed["suite"] == "context") & (speed["mode"] == "compiled")].set_index("block_size")
    token_c = speed[(speed["suite"] == "tokens") & (speed["mode"] == "compiled")].set_index("tokens_per_step")
    read_gap = diag.pivot_table(index="read_distance_bucket", columns="model", values="nll", aggfunc="mean")
    read_gap = read_gap["attention"] - read_gap["ravel"]
    class_gap = diag.pivot_table(index="token_class", columns="model", values="nll", aggfunc="mean")
    class_gap = class_gap["attention"] - class_gap["ravel"]
    rlct_section = ""
    if rlct_summary is not None:
        r_rlct = float(rlct_summary[rlct_summary["model"] == "ravel"]["llc_lambda_hat"].iloc[0])
        r_rlct_std = float(rlct_summary[rlct_summary["model"] == "ravel"]["llc_lambda_std"].iloc[0])
        a_rlct = float(rlct_summary[rlct_summary["model"] == "attention"]["llc_lambda_hat"].iloc[0])
        a_rlct_std = float(rlct_summary[rlct_summary["model"] == "attention"]["llc_lambda_std"].iloc[0])
        n_tokens = int(rlct_summary["n_tokens"].iloc[0])
        beta = float(rlct_summary["beta"].iloc[0])
        rlct_section = rf"""
\subsection{{Local learning coefficient / RLCT estimate}}

To test whether the loss result is hiding a sharper or more fragile basin, we ran a WBIC-temperature local learning coefficient estimator around both trained checkpoints. The sampler uses fixed training tokens, inverse temperature $\beta=1/\log n$, and the estimator
\[
  \hat{{\lambda}} = \beta n\left(\mathbb{{E}}_\beta[L(\theta)] - L^\star\right),
\]
where $L^\star$ is the best loss observed by the local chains. We used $n={n_tokens}$ byte-token targets, $\beta={beta:.3f}$, three SGLD chains per model, 260 steps per chain, and discarded the first 100 steps as burn-in.

The estimate is {r_rlct:.3f}$\pm${r_rlct_std:.3f} for RAVEL and {a_rlct:.3f}$\pm${a_rlct_std:.3f} for attention (Figure~\ref{{fig:rlct}}). In this run, RAVEL has lower task loss and a slightly lower local learning coefficient estimate. The result argues against the boring explanation that RAVEL only wins by landing in a sharper local basin.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{fig_rlct_probe.pdf}}
  \caption{{WBIC/local-learning-coefficient estimate around the trained checkpoints. Points show chain estimates; thick intervals show mean $\pm$ chain standard deviation. The right panel shows sampled excess loss after burn-in.}}
  \label{{fig:rlct}}
\end{{figure}}
"""

    tex = rf"""
\documentclass[10pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{microtype}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\usepackage{{xcolor}}
\usepackage{{float}}

\hypersetup{{colorlinks=true, linkcolor=black, citecolor=black, urlcolor=blue}}
\setcounter{{secnumdepth}}{{2}}

\title{{RAVEL-LM: Exact Addressed Event Memory as Linear-Time Recurrence for Byte-Level Language Modeling}}
\author{{}}
\date{{}}

\begin{{document}}
\maketitle
\vspace{{-2em}}

\begin{{abstract}}
Softmax attention implements dense content addressing by scoring every query against every previous key, giving the core operation $O(BT^2d)$ time and $O(BhT^2)$ score storage. RAVEL-LM replaces that all-pairs similarity problem with exact predecessor search over discrete event streams. Each token writes payloads to memory heads indexed by literal and product-code addresses; each later token reads the latest previous payload with the same address. The operator is not approximate attention. It is a different recurrence primitive: exact same-address event retrieval followed by dense fusion. This paper gives the RAVEL algorithms, their time and storage complexity, and an empirical audit at approximately 200k trainable parameters. On TinyStories bytes, RAVEL reaches {r_loss:.3f} eval cross-entropy versus {a_loss:.3f} for a parameter-matched attention baseline. In speed scaling, compiled RAVEL is {context_c.loc[512, 'ravel_speedup']:.2f}$\times$ faster at $T=512$ and {context_c.loc[1024, 'ravel_speedup']:.2f}$\times$ faster at $T=1024$. The result is straightforward: when the useful computation is recurrence over repeated symbols and local byte events, exact event memory gives the model a direct route to that computation without building a quadratic attention matrix.
\end{{abstract}}

\section{{Introduction}}

The standard transformer attention layer computes a dense causal interaction graph. For a sequence of length $T$, every query position compares itself to every previous key position. This is a useful default because it lets the model discover arbitrary token-to-token dependencies. It is also a blunt default: the layer forms a score tensor whose size grows quadratically with context length.

RAVEL-LM uses a stricter primitive. It does not ask which previous token is most similar. It asks which previous event has the same discrete address. Each memory head defines an address stream, each token writes a payload to that stream, and a future token retrieves the latest previous payload at the same address. Long-range recurrence becomes exact predecessor lookup rather than dense similarity search.

That swap changes both the computation and the inductive bias. Attention spends work on all candidate pairs and learns a soft weighting over them. RAVEL spends work on routing and fusion, then gives the model a hard recurrence channel for repeated bytes, byte n-grams, and learned hash families. This is the right bias when byte-level text contains reusable local events: names, spaces, punctuation, quote boundaries, word fragments, and repeated story entities. The benchmark results follow from that mechanism rather than from curve fitting theater.

The paper is organized around the algorithm. Section~\ref{{sec:operator}} defines exact event memory. Section~\ref{{sec:algorithms}} gives the training/prefill and decode algorithms. Section~\ref{{sec:complexity}} gives the complexity ledger against causal softmax attention. The experiments then audit the claim: lower loss on TinyStories bytes, diagnostic wins exactly where recurrence should help, and a runtime crossover as $T$ grows.

\section{{Architecture}}

Figure~\ref{{fig:architecture}} summarizes the model. A RAVEL block contains three residual sublayers: a causal depthwise-convolution local mixer, an exact event-memory layer, and a SwiGLU feed-forward network. The local mixer handles short-range composition. The memory layer provides long-range recurrence through address lookup.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{fig_architecture.pdf}}
  \caption{{RAVEL-LM block structure. Local mixing is convolutional; long-range recurrence is handled by exact addressed event memory rather than dense softmax attention.}}
  \label{{fig:architecture}}
\end{{figure}}

At each position $t$, every memory head emits an integer address $a_t$ and a payload vector $v_t$. The event tape stores tuples
\[
  (\mathrm{{batch}}, \mathrm{{head}}, a_t, t, v_t).
\]
A read at time $t$ returns the payload from the latest previous event with the same batch, head, and address:
\[
  \operatorname{{read}}(t, a_t)
  =
  v_{{t^\star}}, \quad
  t^\star = \max \{{s < t : a_s = a_t\}}.
\]
If no previous event matches, the read returns zero and a false mask. In training and prefill, the current implementation constructs composite integer keys, sorts them, uses predecessor search, and gathers payloads. In incremental decoding, the latest-record case is a fixed-size gather/scatter cache indexed by address.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\linewidth]{{fig_event_lookup.pdf}}
  \caption{{Exact latest-record lookup. The model-defined address selects an event stream; the read retrieves the most recent previous payload in that stream.}}
  \label{{fig:event-lookup}}
\end{{figure}}

\section{{Exact event memory}}
\label{{sec:operator}}

Let $x_t\in\mathbb{{R}}^d$ be the normalized residual state at position $t$. A RAVEL memory layer with $C$ heads and payload dimension $p$ computes
\begin{{align}}
  a^w_{{t,c}} &= A^w_c(x_t, y_t) \in \{{0,\ldots,M-1\}},\\
  a^r_{{t,c}} &= A^r_c(x_t, y_t) \in \{{0,\ldots,M-1\}},\\
  v_{{t,c}} &= P_c x_t \in \mathbb{{R}}^p,
\end{{align}}
where $y_t$ is the input byte token and $M$ is the address-space size. The write address $a^w$ chooses the stream receiving the payload. The read address $a^r$ chooses the stream queried by the same position. In the current configuration, $C=3$, $p=16$, $M=512$, with two literal heads and one product-code head.

The latest-record read is
\begin{{equation}}
  r_{{t,c}} =
  \begin{{cases}}
    v_{{s^\star,c}}, &
    s^\star = \max \{{s<t: a^w_{{s,c}}=a^r_{{t,c}}\}},\\
    0, & \text{{if no such }}s\text{{ exists.}}
  \end{{cases}}
  \label{{eq:latest-read}}
\end{{equation}}
The read vectors are concatenated and projected back into the residual stream:
\begin{{equation}}
  \operatorname{{Mem}}(x_t)
  =
  x_t
  +
  \sigma(Gx_t+b_g)\odot
  W_f [r_{{t,1}};\ldots;r_{{t,C}}].
  \label{{eq:memory-fusion}}
\end{{equation}}
The gate is a learned write-back gate, not an attention probability. Attention normalizes across previous positions. RAVEL selects exactly one previous record per head, then lets the fusion matrix decide how the retrieved payload should move the residual state.

Literal address heads are deterministic:
\begin{{align}}
  A_1(y_t) &= y_t \bmod M,\\
  A_2(y_{{t-1}},y_t) &= (257y_{{t-1}}+131y_t+17)\bmod M.
\end{{align}}
These heads give byte identity and bigram recurrence immediately. The product-code head maps $x_t$ into codebook logits, takes an argmax in each codebook, and packs the selected codes into an address. In this implementation the product-code projections are frozen because hard argmax routing gives no useful gradient to the router; the trainable parts are the payloads, fusion, gates, local mixer, feed-forward layers, embeddings, and output head.

\section{{Algorithms}}
\label{{sec:algorithms}}

\begin{{figure}}[H]
\small
\fbox{{\begin{{minipage}}{{0.94\linewidth}}
\textbf{{Algorithm 1: RAVEL block forward for one training batch}}\\
\textbf{{Input:}} token ids $y\in\mathbb{{N}}^{{B\times T}}$, residual states $x\in\mathbb{{R}}^{{B\times T\times d}}$\\
\textbf{{Output:}} updated residual states
\begin{{enumerate}}
  \item Apply RMSNorm, gated causal depthwise convolution, and residual add.
  \item Normalize the residual stream for memory.
  \item Compute literal byte and bigram addresses from $y$.
  \item Compute hard product-code write/read addresses from the normalized residual.
  \item Project payloads $v\in\mathbb{{R}}^{{B\times T\times C\times p}}$.
  \item Run Algorithm~2 to retrieve latest same-address payloads.
  \item Concatenate retrieved payloads, apply dense fusion, gate the fused update, and add it to the residual stream.
  \item Apply RMSNorm, SwiGLU feed-forward, and residual add.
\end{{enumerate}}
\end{{minipage}}}}
\caption{{The RAVEL block separates local mixing, exact event-memory recurrence, and channel mixing.}}
\label{{alg:block}}
\end{{figure}}

\begin{{figure}}[H]
\small
\fbox{{\begin{{minipage}}{{0.94\linewidth}}
\textbf{{Algorithm 2: exact latest-record lookup for training and prefill}}\\
\textbf{{Input:}} write addresses $a^w$, read addresses $a^r$, payloads $v$, address-space size $M$\\
\textbf{{Output:}} latest causal payloads $r$ and validity mask $m$
\begin{{enumerate}}
  \item For every batch $b$, time $t$, and head $c$, form the stream id
  \[
    g_{{b,t,c}} = ((bC+c)M + a^w_{{b,t,c}}).
  \]
  \item Form collision-free event keys
  \[
    k^w_{{b,t,c}} = g_{{b,t,c}}(T+1)+t.
  \]
  \item Flatten event keys and payloads; sort keys and permute payloads by the same order.
  \item For each read, form
  \[
    k^r_{{b,t,c}} = ((bC+c)M+a^r_{{b,t,c}})(T+1)+(t-1).
  \]
  \item Use predecessor search in the sorted keys to find the largest key $\leq k^r_{{b,t,c}}$.
  \item Accept the candidate only if its stream id equals $((bC+c)M+a^r_{{b,t,c}})$; otherwise return zero.
  \item Gather the accepted payloads and reshape to $B\times T\times C\times p$.
\end{{enumerate}}
\end{{minipage}}}}
\caption{{Training uses exact sort/search/gather over event keys. This computes Equation~\ref{{eq:latest-read}} for every position without a $T\times T$ attention matrix.}}
\label{{alg:lookup}}
\end{{figure}}

\begin{{figure}}[H]
\small
\fbox{{\begin{{minipage}}{{0.94\linewidth}}
\textbf{{Algorithm 3: one-token decode with latest-address cache}}\\
\textbf{{State per layer:}} cache values $V\in\mathbb{{R}}^{{B\times C\times M\times p}}$ and filled bits $F\in\{{0,1\}}^{{B\times C\times M}}$\\
\textbf{{Input:}} current token $y_t$, residual state $x_t$
\begin{{enumerate}}
  \item Compute literal and product-code read/write addresses.
  \item Read $r_{{b,c}}=V_{{b,c,a^r_{{b,c}}}}$ with a tensor gather; zero it if $F_{{b,c,a^r_{{b,c}}}}=0$.
  \item Fuse, gate, and add the memory update to the residual stream.
  \item Compute payloads $v_{{b,c}}$ for the current token.
  \item Write $V_{{b,c,a^w_{{b,c}}}}\leftarrow v_{{b,c}}$ and set $F_{{b,c,a^w_{{b,c}}}}\leftarrow 1$.
\end{{enumerate}}
\end{{minipage}}}}
\caption{{Decode has no dependence on generated context length for the latest-record memory. The cache is indexed by address, not by time.}}
\label{{alg:decode}}
\end{{figure}}

\section{{Time and storage complexity}}
\label{{sec:complexity}}

Let $B$ be batch size, $T$ sequence length, $d$ model width, $h$ attention heads, $C$ RAVEL memory heads, $p$ payload dimension, $M$ address-space size, and $K$ the local convolution width. Ignoring constants shared by both language models, causal softmax attention pays
\begin{{equation}}
  \operatorname{{Attention}}_{{\text{{core}}}}
  =
  O(BT^2d)
  \quad\text{{time}},\qquad
  O(BhT^2)
  \quad\text{{score storage}}.
\end{{equation}}
The RAVEL memory core pays
\begin{{equation}}
  \operatorname{{RAVEL}}_{{\text{{memory}}}}
  =
  O(BTdCp)
  +
  O(BTC\log(BTC))
  +
  O(BTCpd)
\end{{equation}}
with the portable PyTorch sort implementation. The sort term is the exact predecessor-search implementation. A fused fixed-width radix sort or segmented latest-record kernel turns that term into fixed-pass linear work in the number of records, giving
\begin{{equation}}
  \operatorname{{RAVEL}}_{{\text{{memory, fused}}}}
  =
  O(BT(dCp+Cpd+C)).
\end{{equation}}
The local mixer adds $O(BTdK)$ and the feed-forward block adds the usual $O(BTdH)$, where $H$ is the MLP hidden width. These terms are linear in $T$.

\begin{{table}}[H]
  \centering
  \small
  \caption{{Core sequence-operation complexity per layer. Projection and MLP terms are shown separately from the operation that distinguishes the architectures.}}
  \label{{tab:complexity}}
  \begin{{tabular}}{{llll}}
    \toprule
    operation & training/prefill time & decode time per token & sequence storage \\
    \midrule
    softmax attention core & $O(BT^2d)$ & $O(BT d)$ with KV cache & $O(BhT^2)$ scores during train \\
    RAVEL sort lookup & $O(BTC\log(BTC))$ & $O(BCp+BCpd)$ & $O(BTCp)$ temporary events \\
    RAVEL fused/latest kernel & $O(BTC)$ lookup core & $O(BCp+BCpd)$ & $O(BCMp)$ decode cache \\
    \bottomrule
  \end{{tabular}}
\end{{table}}

For the measured 200k-parameter configuration, the memory dimensions are small: $d=68$, $C=3$, $p=16$, and $M=512$. Attention's distinguishing term grows with $T^2d$. RAVEL's distinguishing term grows with $TC$ records plus fusion. That is why the runtime comparison is allowed to be close at short contexts and still separate sharply at $T=512$ and $T=1024$.

\section{{Experimental setup}}

The learning experiment uses 5000 TinyStories records with a byte tokenizer, block size 128, batch size 4, and 800 optimization steps. The attention baseline is parameter-matched after excluding non-trainable hard-address weights: RAVEL has {summary['params']['ravel']:,} trainable parameters and attention has {summary['params']['attention']:,}. Both models are trained with AdamW. The attention baseline uses a 100-step learning-rate warmup, because earlier runs showed larger early gradient variation for attention.

The speed experiments use synthetic token batches to isolate operator runtime from corpus effects. Warmup and compile steps are excluded from reported medians. We report both eager and \texttt{{torch.compile}} measurements. The context sweep keeps $BT \approx 1024$ while increasing $T$; the fixed-context sweep holds $T=512$ and increases tokens per step by increasing batch size.

\section{{Results}}

\subsection{{Learning dynamics}}

RAVEL learns faster in the TinyStories run. Figure~\ref{{fig:loss}} shows the evaluation cross-entropy over training. The curve separates by step 100 and remains separated through the final evaluation. The final loss is {r_loss:.3f} for RAVEL and {a_loss:.3f} for attention, a gap of {a_loss-r_loss:.3f} nats at matched scale.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{fig_loss_training.pdf}}
  \caption{{TinyStories comparison. RAVEL reaches lower evaluation loss than a parameter-matched softmax-attention baseline and is faster in the compiled steady-state training loop for this run.}}
  \label{{fig:loss}}
\end{{figure}}

\subsection{{Where the loss advantage comes from}}

The diagnostic slices in Figure~\ref{{fig:diagnostics}} ask whether the loss gap is consistent with the memory mechanism. At prediction position $j$, RAVEL reads memory using the current input byte and its address variants. The largest current-byte gap occurs when the read key appeared one position earlier: attention-minus-RAVEL NLL is {read_gap.loc['1']:.2f}. The token-class slice shows the largest advantage on punctuation ({class_gap.loc['punctuation']:.2f}), with spaces and lowercase bytes also favoring RAVEL. This supports a mechanism-level interpretation: the model is learning reusable continuation and boundary records quickly.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{fig_mechanism_diagnostics.pdf}}
  \caption{{Mechanism diagnostics. Positive values mean RAVEL assigns lower negative log-likelihood than attention. The advantage is largest when the current read key has just recurred and on boundary-like byte classes such as punctuation and space.}}
  \label{{fig:diagnostics}}
\end{{figure}}

{rlct_section}

\subsection{{Speed scaling}}

The speed results follow the complexity ledger. At very short contexts, runtime is dominated by constant factors and backend kernel choices. As $T$ grows, attention's quadratic term becomes visible and RAVEL's event lookup stays much flatter. Compiled RAVEL reaches {context_c.loc[512, 'ravel_speedup']:.2f}$\times$ attention throughput at $T=512$ and {context_c.loc[1024, 'ravel_speedup']:.2f}$\times$ at $T=1024$ (Figure~\ref{{fig:speed-context}}).

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{fig_context_crossover.pdf}}
  \caption{{Runtime crossover under a constant-token context sweep. The benchmark holds $BT \approx 1024$ and increases $T$. RAVEL is near parity at short contexts but reaches {context_c.loc[512, 'ravel_speedup']:.2f}$\times$ attention throughput at $T=512$ and {context_c.loc[1024, 'ravel_speedup']:.2f}$\times$ at $T=1024$ in compiled mode.}}
  \label{{fig:speed-context}}
\end{{figure}}

The fixed-context sweep checks that this is not merely a batch-size artifact. At $T=512$, the compiled RAVEL advantage grows from {token_c.loc[512, 'ravel_speedup']:.2f}$\times$ at 512 tokens/step to {token_c.loc[2048, 'ravel_speedup']:.2f}$\times$ at 2048 tokens/step (Figure~\ref{{fig:speed-tokens}}).

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\linewidth]{{fig_fixed_t_scaling.pdf}}
  \caption{{Fixed-context throughput at $T=512$. As more tokens are packed into each training step, RAVEL's advantage grows rather than disappearing.}}
  \label{{fig:speed-tokens}}
\end{{figure}}

The attention score tensor explains the direction of the crossover. In the context sweep, $BT$ is held near 1024, but the attention score tensor still grows from 6 MiB at $T=128$ to 48 MiB at $T=1024$ per forward pass across layers (Figure~\ref{{fig:pressure}}). RAVEL does not form this tensor.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.86\linewidth]{{fig_attention_pressure.pdf}}
  \caption{{Attention's causal score tensor grows quadratically with context length. In the context sweep, $BT$ is held near 1024, so the growth comes from $T$ itself rather than from processing more tokens per step.}}
  \label{{fig:pressure}}
\end{{figure}}

\section{{Discussion}}

The experiments support three claims. First, exact addressed memory trains cleanly on natural byte-level text and beats the parameter-matched attention baseline in this controlled 200k-parameter run. Second, the gain has the mechanism signature RAVEL should have: it concentrates around repeated read keys and boundary-like bytes where exact recurrence is immediately useful. Third, the runtime separates in the direction predicted by the complexity analysis: attention pays for all-pairs scores, while RAVEL pays for event records and fusion.

The next engineering targets are also clear. The current training path uses portable PyTorch sort/search/gather rather than a fused latest-record kernel. The product-code router is hard and frozen; a stronger version should use straight-through routing, differentiable address learning, or explicitly randomized hash families. The present paper establishes the operator, the algorithms, and the first scaling audit. The obvious next version is a fused event-memory kernel plus a trainable router.

\section{{Conclusion}}

RAVEL-LM replaces dense attention with exact addressed event memory. In byte-level experiments, this gives faster learning and long-context speedups while preserving a clear mechanism: retrieve the latest previous payload with the same address. The central result is algorithmic as much as empirical. Useful recurrence can be implemented as event lookup and fusion, without constructing a quadratic attention matrix.

\end{{document}}
"""
    (out_dir / "ravel_lm_paper.tex").write_text(textwrap.dedent(tex).strip() + "\n")


def compile_tex(out_dir: Path) -> None:
    subprocess.run(["tectonic", "ravel_lm_paper.tex"], cwd=out_dir, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the full RAVEL paper artifact")
    p.add_argument("--comparison-run", default="runs/tinystories_comparison_200k_compiled_speedfix")
    p.add_argument("--speed-run", default="runs/paper_speed_scaling_200k")
    p.add_argument("--speed-fig-dir", default="paper/speed")
    p.add_argument("--rlct-dir", default="paper/full/rlct")
    p.add_argument("--out-dir", default="paper/full")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    setup_matplotlib()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = ROOT / args.comparison_run
    speed_run = ROOT / args.speed_run
    speed_fig_dir = ROOT / args.speed_fig_dir
    metrics = pd.read_csv(comparison / "metrics.csv")
    diag = pd.read_csv(comparison / "token_diagnostics.csv")
    summary = json.loads((comparison / "summary.json").read_text())
    speed = pd.read_csv(speed_run / "speedup_table.csv")
    rlct_dir = ROOT / args.rlct_dir
    rlct_samples = pd.read_csv(rlct_dir / "rlct_samples.csv") if (rlct_dir / "rlct_samples.csv").exists() else None
    rlct_chain = pd.read_csv(rlct_dir / "rlct_chain_summary.csv") if (rlct_dir / "rlct_chain_summary.csv").exists() else None
    rlct_summary = pd.read_csv(rlct_dir / "rlct_summary.csv") if (rlct_dir / "rlct_summary.csv").exists() else None

    fig_architecture(out_dir)
    fig_event_lookup(out_dir)
    fig_loss_and_training(metrics, out_dir)
    fig_mechanism_diagnostics(diag, out_dir)
    fig_context_crossover(speed, out_dir)
    fig_fixed_t_scaling(speed, out_dir)
    fig_attention_pressure(speed, out_dir)
    if rlct_samples is not None and rlct_chain is not None and rlct_summary is not None:
        fig_rlct_probe(rlct_samples, rlct_chain, rlct_summary, out_dir)
        for name in ["rlct_samples.csv", "rlct_chain_summary.csv", "rlct_summary.csv", "rlct_meta.json"]:
            src = rlct_dir / name
            if src.exists() and src.resolve() != (out_dir / name).resolve():
                shutil.copy2(src, out_dir / name)
    write_full_tex(out_dir, summary, speed, metrics, diag, rlct_summary)
    shutil.copy2(speed_run / "speedup_table.csv", out_dir / "speedup_table.csv")

    if not args.no_compile:
        compile_tex(out_dir)
    print(f"wrote full paper to {out_dir}")


if __name__ == "__main__":
    main()
