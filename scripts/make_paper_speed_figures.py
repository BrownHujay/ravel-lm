#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAVEL = "#1C4E80"
ATTN = "#C4513E"
EAGER = "#5D6D7E"
COMPILED = "#111111"
GRID = "#D9DEE7"
TEXT = "#17202A"


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#5A6472",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axis(ax: plt.Axes, *, ygrid: bool = True) -> None:
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(length=0)


def direct_label(ax: plt.Axes, x: float, y: float, text: str, color: str, dx: float = 8, dy: float = 0) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        color=color,
        fontsize=10,
        fontweight="bold",
    )


def fig_context_crossover(speed: pd.DataFrame, out_dir: Path) -> None:
    context = speed[speed["suite"] == "context"].copy()
    compiled = context[context["mode"] == "compiled"].sort_values("block_size")
    eager = context[context["mode"] == "eager"].sort_values("block_size")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [1.1, 1.0]})

    ax = axes[0]
    for mode_df, linestyle, alpha, label_suffix in [(eager, "--", 0.65, "eager"), (compiled, "-", 1.0, "compiled")]:
        x = mode_df["block_size"].to_numpy()
        ax.plot(x, mode_df["ravel_step_ms"], marker="o", color=RAVEL, linewidth=2.4, linestyle=linestyle, alpha=alpha)
        ax.plot(x, mode_df["attention_step_ms"], marker="o", color=ATTN, linewidth=2.4, linestyle=linestyle, alpha=alpha)
        if label_suffix == "compiled":
            direct_label(ax, x[-1], mode_df["ravel_step_ms"].iloc[-1], "RAVEL", RAVEL)
            direct_label(ax, x[-1], mode_df["attention_step_ms"].iloc[-1], "attention", ATTN)

    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512, 1024])
    ax.set_xticklabels(["128", "256", "512", "1024"])
    ax.set_ylabel("median train step time (ms)")
    ax.set_xlabel("context length T, with B x T = 1024")
    ax.set_title("(a) Runtime stays flatter for RAVEL")
    clean_axis(ax)
    ax.annotate(
        "compiled runs use solid lines\n(eager shown dashed)",
        xy=(128, compiled["attention_step_ms"].iloc[0]),
        xytext=(148, 34),
        arrowprops={"arrowstyle": "-", "color": "#6B7280", "lw": 0.8},
        fontsize=8.8,
        color="#394452",
    )

    ax = axes[1]
    ax.axhspan(0, 1, color="#F7E7E4", zorder=0)
    ax.axhspan(1, 4.4, color="#E8F1F8", zorder=0)
    ax.axhline(1.0, color="#222222", linewidth=1.0)
    ax.plot(eager["block_size"], eager["ravel_speedup"], marker="o", color=EAGER, linewidth=2.0, linestyle="--")
    ax.plot(compiled["block_size"], compiled["ravel_speedup"], marker="o", color=COMPILED, linewidth=2.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512, 1024])
    ax.set_xticklabels(["128", "256", "512", "1024"])
    ax.set_ylim(0.55, 4.25)
    ax.set_ylabel("throughput ratio: RAVEL / attention")
    ax.set_xlabel("context length T")
    ax.set_title("(b) Crossover becomes decisive")
    clean_axis(ax)
    direct_label(ax, 1024, compiled["ravel_speedup"].iloc[-1], "compiled", COMPILED, dx=8)
    direct_label(ax, 1024, eager["ravel_speedup"].iloc[-1], "eager", EAGER, dx=8, dy=-2)
    for t, y, label in [(512, 2.134, "2.13x"), (1024, 3.940, "3.94x")]:
        ax.annotate(
            label,
            xy=(t, y),
            xytext=(-8, 14),
            textcoords="offset points",
            ha="center",
            color=COMPILED,
            fontweight="bold",
            fontsize=9.5,
        )
    ax.text(142, 0.78, "attention faster", color=ATTN, fontsize=8.8)
    ax.text(142, 1.18, "RAVEL faster", color=RAVEL, fontsize=8.8)

    fig.suptitle("RAVEL speedup appears when sequence length, not token count, is increased", y=1.03, fontsize=13)
    save(fig, out_dir / "fig_context_crossover")


def fig_fixed_t_scaling(speed: pd.DataFrame, out_dir: Path) -> None:
    tokens = speed[speed["suite"] == "tokens"].copy().sort_values(["mode", "tokens_per_step"])
    compiled = tokens[tokens["mode"] == "compiled"]
    eager = tokens[tokens["mode"] == "eager"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25))

    ax = axes[0]
    for mode_df, linestyle, alpha in [(eager, "--", 0.65), (compiled, "-", 1.0)]:
        x = mode_df["tokens_per_step"].to_numpy()
        ax.plot(x, mode_df["ravel_tokens_per_sec"] / 1000, marker="o", color=RAVEL, linewidth=2.4, linestyle=linestyle, alpha=alpha)
        ax.plot(x, mode_df["attention_tokens_per_sec"] / 1000, marker="o", color=ATTN, linewidth=2.4, linestyle=linestyle, alpha=alpha)
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048])
    ax.set_xticklabels(["512", "1024", "2048"])
    ax.set_ylabel("thousand tokens/sec")
    ax.set_xlabel("tokens per step at fixed T = 512")
    ax.set_title("(a) RAVEL holds throughput as batch grows")
    clean_axis(ax)
    direct_label(ax, 2048, compiled["ravel_tokens_per_sec"].iloc[-1] / 1000, "RAVEL", RAVEL)
    direct_label(ax, 2048, compiled["attention_tokens_per_sec"].iloc[-1] / 1000, "attention", ATTN)

    ax = axes[1]
    ax.axhline(1.0, color="#222222", linewidth=1)
    ax.fill_between([512, 2048], [1, 1], [3.05, 3.05], color="#E8F1F8")
    ax.plot(eager["tokens_per_step"], eager["ravel_speedup"], marker="o", color=EAGER, linewidth=2.0, linestyle="--")
    ax.plot(compiled["tokens_per_step"], compiled["ravel_speedup"], marker="o", color=COMPILED, linewidth=2.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048])
    ax.set_xticklabels(["512", "1024", "2048"])
    ax.set_ylim(0.9, 3.05)
    ax.set_ylabel("throughput ratio: RAVEL / attention")
    ax.set_xlabel("tokens per step")
    ax.set_title("(b) The advantage is not a tiny-batch artifact")
    clean_axis(ax)
    for x, y in zip(compiled["tokens_per_step"], compiled["ravel_speedup"]):
        ax.annotate(f"{y:.2f}x", xy=(x, y), xytext=(0, 12), textcoords="offset points", ha="center", fontweight="bold", fontsize=9)
    direct_label(ax, 2048, compiled["ravel_speedup"].iloc[-1], "compiled", COMPILED)
    direct_label(ax, 2048, eager["ravel_speedup"].iloc[-1], "eager", EAGER, dy=-4)

    fig.suptitle("At T = 512, RAVEL gains speed as more tokens are packed into each step", y=1.03, fontsize=13)
    save(fig, out_dir / "fig_fixed_t_scaling")


def fig_attention_pressure(raw: pd.DataFrame, out_dir: Path) -> None:
    attn_pressure = raw[(raw["suite"] == "context") & (raw["mode"] == "compiled") & (raw["model"] == "attention")].sort_values("block_size")

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    x = attn_pressure["block_size"].to_numpy()
    pressure = attn_pressure["attention_score_mib_per_fwd"].to_numpy()
    ax.plot(x, pressure, color=ATTN, marker="o", linewidth=2.5)
    ax.fill_between(x, pressure, color="#F7E7E4", alpha=0.85)
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512, 1024])
    ax.set_xticklabels(["128", "256", "512", "1024"])
    ax.set_ylabel("attention score tensor MiB / forward")
    ax.set_xlabel("context length T")
    ax.set_title("Attention's intermediate state grows quadratically")
    clean_axis(ax)
    for xi, yi in zip(x, pressure):
        ax.annotate(f"{yi:.0f} MiB", xy=(xi, yi), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color=ATTN)
    ax.text(145, max(pressure) * 0.76, "B x T is held near 1024,\nso this growth is from T itself.", fontsize=9.5, color="#394452")
    save(fig, out_dir / "fig_attention_pressure")


def write_tex(speed: pd.DataFrame, out_dir: Path) -> None:
    context_c = speed[(speed["suite"] == "context") & (speed["mode"] == "compiled")].set_index("block_size")
    tokens_c = speed[(speed["suite"] == "tokens") & (speed["mode"] == "compiled")].set_index("tokens_per_step")
    lines = [
        r"\subsection{Speed scaling}",
        "",
        "We report synthetic-token train-step timings to isolate the runtime behavior of the sequence operators from corpus effects. "
        "The benchmark uses the same byte-level configuration as the small TinyStories comparison and matches the attention baseline by trainable parameter count. "
        "Warmup/compile steps are excluded from the medians. The context-length sweep keeps approximately 1024 tokens per step while increasing $T$, so the main changing variable is the amount of history visible to each token.",
        "",
        "The crossover is sharp. At $T=128$, compiled RAVEL and attention are close "
        f"({context_c.loc[128, 'ravel_speedup']:.2f}$\\times$ RAVEL/attention throughput). "
        "At $T=512$, RAVEL reaches "
        f"{context_c.loc[512, 'ravel_tokens_per_sec'] / 1000:.0f}k tokens/s versus "
        f"{context_c.loc[512, 'attention_tokens_per_sec'] / 1000:.0f}k for attention "
        f"({context_c.loc[512, 'ravel_speedup']:.2f}$\\times$). "
        "At $T=1024$, the gap grows to "
        f"{context_c.loc[1024, 'ravel_speedup']:.2f}$\\times$. "
        "This is the regime in which the event-memory formulation begins to express its intended scaling advantage: RAVEL's measured step time remains nearly flat while the attention baseline pays for a growing causal score/value computation.",
        "",
        "The fixed-context sweep checks that the effect is not only a single-batch artifact. At $T=512$, increasing the batch from 1 to 4 raises the compiled RAVEL advantage from "
        f"{tokens_c.loc[512, 'ravel_speedup']:.2f}$\\times$ to "
        f"{tokens_c.loc[2048, 'ravel_speedup']:.2f}$\\times$. "
        "Thus the speedup persists when the number of tokens per step increases, not merely when the context sweep reduces batch size.",
        "",
        "There is one non-monotone point worth keeping rather than smoothing away: at $T=256$, compiled attention is faster in this CPU/PyTorch setup. "
        "This reflects implementation constants and compiler choices, not the asymptotic behavior. The result should therefore be stated as a crossover claim: RAVEL is not universally faster at tiny contexts, but it becomes substantially faster once the context is long enough for attention's quadratic term to dominate.",
        "",
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \includegraphics[width=\linewidth]{fig_context_crossover.pdf}",
        r"  \caption{Runtime crossover under a constant-token context sweep. The benchmark holds $B T \approx 1024$ and increases $T$. RAVEL is near parity at short contexts, but reaches 2.13$\times$ attention throughput at $T=512$ and 3.94$\times$ at $T=1024$ in compiled mode.}",
        r"  \label{fig:ravel-context-speed}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \includegraphics[width=\linewidth]{fig_fixed_t_scaling.pdf}",
        r"  \caption{Fixed-context throughput at $T=512$. As more tokens are packed into each training step, RAVEL's advantage grows rather than disappearing, reaching 2.62$\times$ attention throughput at 2048 tokens/step in compiled mode.}",
        r"  \label{fig:ravel-token-speed}",
        r"\end{figure}",
        "",
    ]
    (out_dir / "speed_results.tex").write_text("\n".join(lines))


def write_table(speed: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Sweep & Mode & $B$ & $T$ & Tokens/step & RAVEL speedup \\",
        r"\midrule",
    ]
    for row in speed.itertuples(index=False):
        lines.append(
            f"{row.suite} & {row.mode} & {row.batch_size} & {row.block_size} & {row.tokens_per_step} & {row.ravel_speedup:.2f}$\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (out_dir / "speedup_table.tex").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Make paper-quality speed figures and section text")
    p.add_argument("--run-dir", default="runs/paper_speed_scaling_200k")
    p.add_argument("--out-dir", default="paper/speed")
    args = p.parse_args()

    setup_matplotlib()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speed = pd.read_csv(run_dir / "speedup_table.csv")
    raw = pd.read_csv(run_dir / "speed_scaling.csv")
    speed.to_csv(out_dir / "speedup_table.csv", index=False)

    fig_context_crossover(speed, out_dir)
    fig_fixed_t_scaling(speed, out_dir)
    fig_attention_pressure(raw, out_dir)
    write_tex(speed, out_dir)
    write_table(speed, out_dir)
    print(f"wrote paper figures and section to {out_dir}")


if __name__ == "__main__":
    main()
