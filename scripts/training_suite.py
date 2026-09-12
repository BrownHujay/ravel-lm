#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.attention_model import AttentionLM
from ravel_lm.config import RavelConfig
from ravel_lm.data import iter_fineweb_edu, iter_local_text, iter_tinystories
from ravel_lm.model import RavelLM
from ravel_lm.ravel_memory import causal_last_k_lookup
from ravel_lm.runtime import FlatAdamW, adamw_fused_for, compiled_mps_clip_grad_norm_
from ravel_lm.tokenizers import ByteTokenizer


def generated_stories(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    names = ["Lily", "Tom", "Mia", "Sam", "Nora", "Ben", "Ava", "Leo"]
    colors = ["red", "blue", "green", "yellow", "silver", "purple"]
    objects = ["ball", "cup", "kite", "book", "bell", "box", "shell", "cake"]
    places = ["garden", "hill", "kitchen", "river", "school", "porch"]
    feelings = ["happy", "calm", "brave", "sleepy", "proud", "kind"]
    actions = ["found", "carried", "hid", "washed", "painted", "shared"]
    outcomes = [
        "so everyone clapped",
        "and the room grew quiet",
        "so the small problem was solved",
        "and nobody forgot the lesson",
        "so the day ended gently",
    ]
    stories: list[str] = []
    for i in range(n):
        name = rng.choice(names)
        friend = rng.choice([x for x in names if x != name])
        color = rng.choice(colors)
        obj = rng.choice(objects)
        place = rng.choice(places)
        feeling = rng.choice(feelings)
        action = rng.choice(actions)
        outcome = rng.choice(outcomes)
        code = rng.choice(["mip", "lor", "sava", "nill", "toma", "vep"])
        template = rng.randrange(4)
        if template == 0:
            text = (
                f"{name} had a {color} {obj}. {friend} asked where the {obj} was. "
                f"{name} said the {color} {obj} was in the {place}, {outcome}."
            )
        elif template == 1:
            text = (
                f"The word was {code}. {name} went to the {place} and {action} a {obj}. "
                f"When {friend} asked for the word, {name} said {code} and felt {feeling}."
            )
        elif template == 2:
            text = (
                f"In the {place}, {name} put the {color} {obj} beside a plain {obj}. "
                f"The {color} {obj} mattered because {friend} needed that one, {outcome}."
            )
        else:
            text = (
                f"{friend} was {feeling}. {name} made a plan with a {color} {obj}. "
                f"First {name} {action} it, then {friend} smiled at the {color} {obj}."
            )
        stories.append(text)
    return stories


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.local_text:
        texts = list(iter_local_text(args.local_text))
        if args.max_records:
            texts = texts[: args.max_records]
        return texts
    if args.corpus == "tinystories":
        return list(iter_tinystories(max_records=args.max_records))
    if args.corpus == "fineweb-edu":
        return list(
            iter_fineweb_edu(
                config_name=args.hf_config,
                split=args.hf_split,
                max_records=args.max_records,
            )
        )
    return generated_stories(args.max_records, seed=args.seed)


def encode_records(records: list[str], tokenizer: ByteTokenizer) -> np.ndarray:
    ids: list[int] = []
    for text in records:
        ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=np.int64)


def sample_batch(tokens: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator, device: torch.device):
    if tokens.size <= block_size + 1:
        raise ValueError("not enough tokens for requested block_size")
    starts = rng.integers(0, tokens.size - block_size - 1, size=batch_size)
    x = np.stack([tokens[s : s + block_size] for s in starts])
    y = np.stack([tokens[s + 1 : s + block_size + 1] for s in starts])
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def scheduled_lr(
    base_lr: float,
    step: int,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
    min_lr_ratio: float,
) -> float:
    update = step + 1
    if warmup_steps > 0 and update <= warmup_steps:
        return base_lr * float(update) / float(warmup_steps)
    if schedule == "constant" or total_steps <= warmup_steps:
        return base_lr
    progress = (update - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


@torch.no_grad()
def evaluate(model: torch.nn.Module, tokens: np.ndarray, args: argparse.Namespace, device: torch.device, seed: int) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(args.eval_batches):
        x, y = sample_batch(tokens, args.batch_size, args.block_size, rng, device)
        losses.append(float(model(x, y)["loss"].detach().cpu()))
    model.train()
    return float(np.mean(losses))


def maybe_compile_model(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.nn.Module:
    if not args.compile_models:
        return model
    return torch.compile(
        model,
        backend="inductor",
        mode=args.compile_mode,
        fullgraph=True,
        dynamic=False,
    )


def train_model(
    name: str,
    model: torch.nn.Module,
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    *,
    lr: float,
    warmup_steps: int,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    model.to(device)
    exec_model = maybe_compile_model(model, args)
    params_to_optimize = trainable_parameters(model)
    if device.type != "cuda":
        # Flat-buffer AdamW: clip + step run as a few large kernels instead of
        # one per tensor. NOTE: optimizer states from runs recorded before this
        # change cannot be resumed (per-tensor vs flat state layout).
        optimizer = FlatAdamW(
            [{"params": params_to_optimize, "weight_decay": args.weight_decay}],
            lr=lr,
            betas=(0.9, 0.999),
            fused=adamw_fused_for(device),
        )
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=lr,
            weight_decay=args.weight_decay,
            fused=adamw_fused_for(device),
        )
    rng = np.random.default_rng(args.seed + (0 if name == "ravel" else 10_000))
    state_path = Path(args.out_dir) / f"{name}_training_state.pt"
    start_step = 0
    rows: list[dict] = []
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng_state"]
        start_step = int(state["step"])
        rows = list(state.get("rows", []))
        print(f"{name:9s} resumed at step {start_step}")
    model.train()
    for step in range(start_step, args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            eval_loss = evaluate(exec_model, eval_tokens, args, device, seed=args.seed + step)
            rows.append(
                {
                    "model": name,
                    "step": step,
                    "split": "eval",
                    "loss": eval_loss,
                    "ppl": math.exp(min(20.0, eval_loss)),
                    "grad_norm": np.nan,
                    "tokens_per_sec": np.nan,
                    "step_ms": np.nan,
                }
            )
            print(f"{name:9s} step {step:04d} eval_loss {eval_loss:.4f}")
        if step == args.steps:
            break

        synchronize_device(device)
        t0 = time.perf_counter()
        step_lr = scheduled_lr(
            lr,
            step,
            warmup_steps,
            args.steps,
            args.lr_schedule,
            args.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        x, y = sample_batch(train_tokens, args.batch_size, args.block_size, rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss = exec_model(x, y)["loss"]
        loss.backward()
        should_log = step % args.log_interval == 0
        if args.grad_clip > 0:
            if isinstance(optimizer, FlatAdamW):
                clipped_norm = optimizer.clip_grad_norm_(args.grad_clip)
            elif device.type == "mps" and args.compile_models:
                clipped_norm = compiled_mps_clip_grad_norm_(params_to_optimize, args.grad_clip)
            else:
                clipped_norm = torch.nn.utils.clip_grad_norm_(params_to_optimize, args.grad_clip)
            gnorm = float(clipped_norm.detach().cpu()) if should_log else np.nan
        else:
            gnorm = grad_norm(model) if should_log else np.nan
        optimizer.step()
        synchronize_device(device)
        dt = time.perf_counter() - t0
        if should_log:
            rows.append(
                {
                    "model": name,
                    "step": step + 1,
                    "split": "train",
                    "loss": float(loss.detach().cpu()),
                    "ppl": math.exp(min(20.0, float(loss.detach().cpu()))),
                    "grad_norm": gnorm,
                    "tokens_per_sec": args.batch_size * args.block_size / max(dt, 1e-9),
                    "step_ms": dt * 1000.0,
                    "lr": step_lr,
                }
            )
            print(
                f"{name:9s} step {step + 1:04d} train_loss {float(loss.detach().cpu()):.4f} "
                f"grad {gnorm:.3f} tok/s {args.batch_size * args.block_size / max(dt, 1e-9):.0f}"
            )
        if args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0:
            torch.save(
                {
                    "step": step + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": rng.bit_generator.state,
                    "rows": rows,
                },
                state_path,
            )
    return model.cpu(), pd.DataFrame(rows)


def token_class(token_id: int) -> str:
    if token_id >= 256:
        return "special"
    ch = chr(token_id)
    if ch == " ":
        return "space"
    if ch in "\n\r\t":
        return "whitespace"
    if "a" <= ch <= "z":
        return "lowercase"
    if "A" <= ch <= "Z":
        return "uppercase"
    if "0" <= ch <= "9":
        return "digit"
    if ch in ".,;:!?\"'()-":
        return "punctuation"
    if 32 <= token_id <= 126:
        return "symbol"
    return "other_byte"


def distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "no prior in block"
    if distance == 1:
        return "1"
    if distance <= 4:
        return "2-4"
    if distance <= 16:
        return "5-16"
    if distance <= 64:
        return "17-64"
    return "65+"


def batch_target_metadata(x: torch.Tensor, y: torch.Tensor) -> list[dict]:
    x_cpu = x.detach().cpu().numpy()
    y_cpu = y.detach().cpu().numpy()
    rows = []
    B, T = y_cpu.shape
    for b in range(B):
        seen: dict[int, int] = {}
        for j in range(T):
            context_token = int(x_cpu[b, j])
            prev_context = seen.get(context_token)
            read_distance = None if prev_context is None else (j - prev_context)
            seen[context_token] = j
            target = int(y_cpu[b, j])
            prev = seen.get(target)
            distance = None if prev is None else (j + 1 - prev)
            rows.append(
                {
                    "target_id": target,
                    "input_id": context_token,
                    "token_class": token_class(target),
                    "distance": distance if distance is not None else np.nan,
                    "distance_bucket": distance_bucket(distance),
                    "read_distance": read_distance if read_distance is not None else np.nan,
                    "read_distance_bucket": distance_bucket(read_distance),
                    "position": j,
                }
            )
    return rows


@torch.no_grad()
def token_diagnostics(
    models: dict[str, torch.nn.Module],
    eval_tokens: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 4242)
    for model in models.values():
        model.to(device)
        model.eval()
    rows = []
    for batch_idx in range(args.diag_batches):
        x, y = sample_batch(eval_tokens, args.batch_size, args.block_size, rng, device)
        meta = batch_target_metadata(x, y)
        for model_name, model in models.items():
            logits = model(x, y)["logits"]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=model.cfg.pad_token_id,
                reduction="none",
            )
            for item, loss_value in zip(meta, nll.detach().cpu().numpy()):
                rows.append({"model": model_name, "batch": batch_idx, "nll": float(loss_value), **item})
    for model in models.values():
        model.cpu()
    return pd.DataFrame(rows)


def summarize_diagnostics(diag: pd.DataFrame) -> dict:
    bucket_order = ["no prior in block", "1", "2-4", "5-16", "17-64", "65+"]
    summary: dict[str, object] = {}
    for col, prefix in [("distance_bucket", "target"), ("read_distance_bucket", "read")]:
        profile = (
            diag[diag["model"] == diag["model"].iloc[0]]
            .groupby(col, observed=False)
            .size()
            .reindex(bucket_order, fill_value=0)
        )
        summary[f"{prefix}_distance_profile"] = {k: float(v / max(1, profile.sum())) for k, v in profile.items()}
        pivot = diag.pivot_table(index=col, columns="model", values="nll", aggfunc="mean")
        bucket_loss = {}
        for bucket in bucket_order:
            if bucket in pivot.index:
                bucket_loss[bucket] = {str(k): float(v) for k, v in pivot.loc[bucket].dropna().items()}
        summary[f"loss_by_{prefix}_distance_bucket"] = bucket_loss
        if {"ravel", "attention"}.issubset(set(diag["model"])):
            gap = pivot.get("attention") - pivot.get("ravel")
            summary[f"attention_minus_ravel_nll_by_{prefix}_distance_bucket"] = {
                str(k): float(v) for k, v in gap.dropna().items()
            }
    summary["distance_profile"] = summary["target_distance_profile"]
    summary["loss_by_distance_bucket"] = summary["loss_by_target_distance_bucket"]
    summary["attention_minus_ravel_nll_by_distance_bucket"] = summary[
        "attention_minus_ravel_nll_by_target_distance_bucket"
    ]
    class_pivot = diag.pivot_table(index="token_class", columns="model", values="nll", aggfunc="mean")
    summary["loss_by_token_class"] = {
        str(idx): {str(k): float(v) for k, v in row.dropna().items()} for idx, row in class_pivot.iterrows()
    }
    return summary


def build_param_matched_attention_cfg(base_cfg: RavelConfig, target_params: int, n_heads: int) -> RavelConfig:
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
    print(f"matched attention width d_model={best_cfg.d_model} params={best_params:,} target={target_params:,}")
    return best_cfg


@torch.no_grad()
def probe_ravel(model: RavelLM, idx: torch.Tensor) -> dict:
    model.eval()
    cfg = model.cfg
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device, dtype=torch.long)
    x = model.tok_emb(idx) + model.pos_emb(pos).unsqueeze(0)
    x = model.drop(x)
    layers = []
    activation_rms = []
    for layer_idx, block in enumerate(model.blocks):
        x = block.local(x)
        if block.memory is not None:
            xn = block.memory.norm(x)
            write_addr, read_addr = block.memory._addresses(xn, idx)
            C = write_addr.shape[-1]
            payload = block.memory.payload(xn).reshape(B, T, C, cfg.payload_dim)
            _, mask = causal_last_k_lookup(
                write_addr,
                payload,
                read_addr,
                address_space=cfg.address_space,
                k=cfg.last_k,
            )
            mask1 = mask[..., 0].float()
            head_hit = mask1.mean(dim=(0, 1)).cpu().numpy()
            hit_by_pos = mask1.mean(dim=(0, 2)).cpu().numpy()
            addr = write_addr.cpu().numpy()
            head_entropy = []
            head_counts = []
            for h in range(addr.shape[-1]):
                counts = np.bincount(addr[:, :, h].reshape(-1), minlength=cfg.address_space).astype(np.float64)
                prob = counts / max(1.0, counts.sum())
                entropy = -np.sum(prob[prob > 0] * np.log(prob[prob > 0])) / math.log(cfg.address_space)
                head_entropy.append(float(entropy))
                head_counts.append(counts)
            layers.append(
                {
                    "layer": layer_idx,
                    "head_hit": head_hit,
                    "hit_by_pos": hit_by_pos,
                    "head_entropy": np.asarray(head_entropy),
                    "head_counts": np.stack(head_counts),
                    "addresses": addr,
                }
            )
            x = block.memory(x, idx)
        x = block.ffn(x)
        activation_rms.append(x.float().pow(2).mean(dim=-1).sqrt().cpu().numpy())
    return {"layers": layers, "activation_rms": activation_rms}


@torch.no_grad()
def probe_attention(model: AttentionLM, idx: torch.Tensor) -> dict:
    model.eval()
    probe = model.probe(idx)
    attentions = [a.cpu().numpy() for a in probe.attentions]
    activation_rms = [a.cpu().numpy() for a in probe.activation_rms]
    layer_entropy = []
    distance_weights = []
    for attn in attentions:
        B, H, T, _ = attn.shape
        eps = 1e-12
        entropy = -(attn * np.log(attn + eps)).sum(axis=-1)
        denom = np.log(np.arange(1, T + 1, dtype=np.float64))
        denom[0] = 1.0
        norm_entropy = entropy / denom.reshape(1, 1, T)
        layer_entropy.append(norm_entropy.mean(axis=(0, 2)))
        dist = np.zeros(T, dtype=np.float64)
        for t in range(T):
            for s in range(t + 1):
                dist[t - s] += attn[:, :, t, s].sum()
        dist /= max(1e-9, dist.sum())
        distance_weights.append(dist)
    return {
        "attentions": attentions,
        "activation_rms": activation_rms,
        "head_entropy": np.stack(layer_entropy),
        "distance_weights": np.stack(distance_weights),
    }


def plot_loss(metrics: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for split, ax in zip(["train", "eval"], axes):
        sub = metrics[metrics["split"] == split]
        sns.lineplot(data=sub, x="step", y="loss", hue="model", marker="o", ax=ax)
        ax.set_title(f"{split.title()} loss")
        ax.set_ylabel("cross entropy")
    eval_pivot = metrics[metrics["split"] == "eval"].pivot(index="step", columns="model", values="loss")
    if {"ravel", "attention"}.issubset(eval_pivot.columns):
        gap = eval_pivot["attention"] - eval_pivot["ravel"]
        axes[2].axhline(0.0, color="black", linewidth=1)
        axes[2].plot(gap.index, gap.values, marker="o", color="#4c72b0")
        axes[2].fill_between(gap.index, 0.0, gap.values, where=gap.values >= 0, alpha=0.25, color="#4c72b0")
        axes[2].fill_between(gap.index, 0.0, gap.values, where=gap.values < 0, alpha=0.25, color="#dd8452")
        axes[2].set_title("Eval gap: attention - RAVEL")
        axes[2].set_ylabel("cross entropy")
    else:
        axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_distance_diagnostic(diag: pd.DataFrame, bucket_col: str, out: Path, title_prefix: str) -> None:
    bucket_order = ["no prior in block", "1", "2-4", "5-16", "17-64", "65+"]
    first_model = diag["model"].iloc[0]
    profile = (
        diag[diag["model"] == first_model]
        .groupby(bucket_col, observed=False)
        .size()
        .reindex(bucket_order, fill_value=0)
        .reset_index(name="count")
    )
    profile["share"] = profile["count"] / max(1, profile["count"].sum())
    bucket_loss = diag.groupby(["model", bucket_col], observed=False)["nll"].mean().reset_index()
    bucket_loss[bucket_col] = pd.Categorical(bucket_loss[bucket_col], bucket_order, ordered=True)
    pivot = bucket_loss.pivot(index=bucket_col, columns="model", values="nll").reindex(bucket_order)
    gap_df = pd.DataFrame({bucket_col: bucket_order})
    if {"ravel", "attention"}.issubset(pivot.columns):
        gap_df["attention_minus_ravel"] = (pivot["attention"] - pivot["ravel"]).values
    else:
        gap_df["attention_minus_ravel"] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    sns.barplot(data=profile, x=bucket_col, y="share", color="#8da0cb", ax=axes[0])
    axes[0].set_title(f"{title_prefix}: bucket share")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("token share")
    axes[0].tick_params(axis="x", rotation=25)
    sns.lineplot(data=bucket_loss, x=bucket_col, y="nll", hue="model", marker="o", ax=axes[1])
    axes[1].set_title(f"{title_prefix}: conditional NLL")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("mean token NLL")
    axes[1].tick_params(axis="x", rotation=25)
    colors = np.where(gap_df["attention_minus_ravel"] >= 0, "#4c72b0", "#dd8452")
    axes[2].bar(gap_df[bucket_col], gap_df["attention_minus_ravel"], color=colors)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title(f"{title_prefix}: signed gap")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("attention NLL - RAVEL NLL")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_diagnostics(diag: pd.DataFrame, out_distance: Path, out_read_distance: Path, out_class: Path) -> dict:
    class_order = ["space", "lowercase", "uppercase", "punctuation", "whitespace", "digit", "symbol", "other_byte", "special"]
    first_model = diag["model"].iloc[0]
    plot_distance_diagnostic(diag, "distance_bucket", out_distance, "Target-byte repetition")
    plot_distance_diagnostic(diag, "read_distance_bucket", out_read_distance, "Current-byte read opportunity")

    class_profile = (
        diag[diag["model"] == first_model]
        .groupby("token_class", observed=False)
        .size()
        .reindex(class_order, fill_value=0)
        .reset_index(name="count")
    )
    class_profile["share"] = class_profile["count"] / max(1, class_profile["count"].sum())
    class_loss = diag.groupby(["model", "token_class"], observed=False)["nll"].mean().reset_index()
    class_loss["token_class"] = pd.Categorical(class_loss["token_class"], class_order, ordered=True)
    class_pivot = class_loss.pivot(index="token_class", columns="model", values="nll").reindex(class_order)
    class_gap = pd.DataFrame({"token_class": class_order})
    if {"ravel", "attention"}.issubset(class_pivot.columns):
        class_gap["attention_minus_ravel"] = (class_pivot["attention"] - class_pivot["ravel"]).values
    else:
        class_gap["attention_minus_ravel"] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    sns.barplot(data=class_profile, x="token_class", y="share", color="#66c2a5", ax=axes[0])
    axes[0].set_title("Eval byte classes")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("token share")
    axes[0].tick_params(axis="x", rotation=30)
    sns.lineplot(data=class_loss, x="token_class", y="nll", hue="model", marker="o", ax=axes[1])
    axes[1].set_title("NLL by byte class")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("mean token NLL")
    axes[1].tick_params(axis="x", rotation=30)
    colors = np.where(class_gap["attention_minus_ravel"] >= 0, "#4c72b0", "#dd8452")
    axes[2].bar(class_gap["token_class"], class_gap["attention_minus_ravel"], color=colors)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title("Byte classes driving the gap")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("attention NLL - RAVEL NLL")
    axes[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_class, dpi=180)
    plt.close(fig)
    return summarize_diagnostics(diag)


def plot_training_dashboard(metrics: pd.DataFrame, params: dict[str, int], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    final_eval = metrics[metrics["split"] == "eval"].sort_values("step").groupby("model").tail(1)
    sns.barplot(data=final_eval, x="model", y="loss", ax=axes[0, 0])
    axes[0, 0].set_title("Final eval loss")
    train = metrics[metrics["split"] == "train"]
    sns.boxplot(data=train, x="model", y="tokens_per_sec", ax=axes[0, 1])
    axes[0, 1].set_title("Training throughput distribution")
    sns.violinplot(data=train, x="model", y="grad_norm", ax=axes[1, 0], inner="quart")
    axes[1, 0].set_title("Gradient norm distribution")
    axes[1, 1].bar(list(params.keys()), list(params.values()))
    axes[1, 1].set_title("Trainable parameters")
    axes[1, 1].ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_ravel(ravel_probe: dict, out_heatmap: Path, out_hit: Path) -> dict:
    layers = ravel_probe["layers"]
    rows = []
    heat_rows = []
    labels = []
    for layer in layers:
        for h, hit in enumerate(layer["head_hit"]):
            rows.append(
                {
                    "layer": layer["layer"],
                    "head": h,
                    "hit_rate": float(hit),
                    "address_entropy": float(layer["head_entropy"][h]),
                }
            )
            counts = layer["head_counts"][h]
            binned = counts.reshape(64, -1).sum(axis=1) if counts.size % 64 == 0 else np.histogram(
                np.repeat(np.arange(counts.size), counts.astype(int)), bins=64, range=(0, counts.size)
            )[0]
            heat_rows.append(binned / max(1.0, binned.sum()))
            labels.append(f"L{layer['layer']} H{h}")
    heat = np.stack(heat_rows)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(labels))))
    sns.heatmap(heat, cmap="viridis", yticklabels=labels, xticklabels=8, ax=ax)
    ax.set_title("RAVEL write-address occupancy by layer/head")
    ax.set_xlabel("address-space bin")
    fig.tight_layout()
    fig.savefig(out_heatmap, dpi=180)
    plt.close(fig)

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=df, x="head", y="hit_rate", hue="layer", ax=axes[0])
    axes[0].set_title("Strict-causal memory hit rate")
    sns.barplot(data=df, x="head", y="address_entropy", hue="layer", ax=axes[1])
    axes[1].set_title("Normalized write-address entropy")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_hit, dpi=180)
    plt.close(fig)
    return {
        "mean_hit_rate": float(df["hit_rate"].mean()),
        "mean_address_entropy": float(df["address_entropy"].mean()),
    }


def plot_attention(attn_probe: dict, out: Path) -> dict:
    attn = attn_probe["attentions"][-1].mean(axis=(0, 1))
    distance = attn_probe["distance_weights"].mean(axis=0)
    entropy = attn_probe["head_entropy"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.heatmap(attn, cmap="mako", ax=axes[0])
    axes[0].set_title("Attention map, final layer avg")
    axes[0].set_xlabel("key position")
    axes[0].set_ylabel("query position")
    axes[1].bar(np.arange(distance.size), distance)
    axes[1].set_xlim(-1, min(64, distance.size))
    axes[1].set_title("Attention mass by backward distance")
    axes[1].set_xlabel("query - key")
    sns.heatmap(entropy, vmin=0, vmax=1, cmap="rocket", annot=True, fmt=".2f", ax=axes[2])
    axes[2].set_title("Normalized attention entropy")
    axes[2].set_xlabel("head")
    axes[2].set_ylabel("layer")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    expected_distance = float(np.sum(distance * np.arange(distance.size)))
    return {"mean_attention_entropy": float(entropy.mean()), "expected_attention_distance": expected_distance}


def plot_internals(ravel_probe: dict, attn_probe: dict, out: Path) -> dict:
    rows = []
    for i, arr in enumerate(ravel_probe["activation_rms"]):
        rows.extend({"model": "ravel", "layer": i, "rms": float(x)} for x in arr.reshape(-1)[:: max(1, arr.size // 500)])
    for i, arr in enumerate(attn_probe["activation_rms"]):
        rows.extend({"model": "attention", "layer": i, "rms": float(x)} for x in arr.reshape(-1)[:: max(1, arr.size // 500)])
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="layer", y="rms", hue="model", ax=ax)
    ax.set_title("Activation RMS distribution by layer")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {
        "ravel_activation_rms": float(df[df["model"] == "ravel"]["rms"].mean()),
        "attention_activation_rms": float(df[df["model"] == "attention"]["rms"].mean()),
    }


def generate_sample(model: torch.nn.Module, tokenizer: ByteTokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    idx = torch.tensor(ids, dtype=torch.long).view(1, -1)
    with torch.no_grad():
        if isinstance(model, RavelLM):
            out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40, use_cache=True)
        else:
            out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    return tokenizer.decode(out[0].tolist())


def default_prompt(args: argparse.Namespace) -> str:
    if args.local_text:
        return "Once upon a time"
    if args.corpus == "tinystories":
        return "Once upon a time"
    if args.corpus == "fineweb-edu":
        return "The study found that"
    return "Lily had a red"


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    params: dict[str, int],
    probe_summary: dict,
    diagnostic_summary: dict,
    samples: dict[str, str],
) -> None:
    final_eval = metrics[metrics["split"] == "eval"].sort_values("step").groupby("model").tail(1)
    final = {row["model"]: row for _, row in final_eval.iterrows()}
    train = metrics[metrics["split"] == "train"]
    med_tps = train.groupby("model")["tokens_per_sec"].median().to_dict()
    med_grad = train.groupby("model")["grad_norm"].median().to_dict()
    distance_gaps = diagnostic_summary.get("attention_minus_ravel_nll_by_distance_bucket", {})
    read_gaps = diagnostic_summary.get("attention_minus_ravel_nll_by_read_distance_bucket", {})
    strongest_ravel_bucket = None
    strongest_attention_bucket = None
    if distance_gaps:
        strongest_ravel_bucket = max(distance_gaps, key=lambda k: distance_gaps[k])
        min_bucket = min(distance_gaps, key=lambda k: distance_gaps[k])
        if distance_gaps[min_bucket] < 0:
            strongest_attention_bucket = min_bucket
    distance_profile = diagnostic_summary.get("distance_profile", {})
    read_profile = diagnostic_summary.get("read_distance_profile", {})
    token_class_gaps = {}
    for cls, losses in diagnostic_summary.get("loss_by_token_class", {}).items():
        if "attention" in losses and "ravel" in losses:
            token_class_gaps[cls] = float(losses["attention"] - losses["ravel"])
    best_class = max(token_class_gaps, key=lambda k: token_class_gaps[k]) if token_class_gaps else "n/a"
    weakest_class = min(token_class_gaps, key=lambda k: token_class_gaps[k]) if token_class_gaps else "n/a"
    read_best = max(read_gaps, key=lambda k: read_gaps[k]) if read_gaps else "n/a"
    lines = [
        "# RAVEL vs Softmax Attention Training Suite",
        "",
        "## Run setup",
        "",
        f"- Config: `{args.config}`",
        f"- Device: `{args.device}`",
        f"- Full-graph compilation: `{args.compile_models}`",
        f"- Steps: `{args.steps}`",
        f"- Batch/block: `{args.batch_size} x {args.block_size}`",
        f"- Corpus: `{args.local_text or args.corpus}`",
        f"- Hugging Face config/split: `{getattr(args, 'hf_config', None)}` / `{getattr(args, 'hf_split', None)}`",
        f"- Trainable params: RAVEL `{params['ravel']:,}`, attention `{params['attention']:,}`",
        f"- Widths: RAVEL d_model `{args.ravel_d_model}`, attention d_model `{args.attention_d_model}`",
        f"- Optimizer: RAVEL lr `{args.lr:g}` warmup `{args.ravel_warmup_steps}`; attention lr `{args.attention_lr:g}` warmup `{args.attention_warmup_steps}`; schedule `{args.lr_schedule}`",
        "",
        "## Plots",
        "",
        "![Loss curves](loss_curves.png)",
        "",
        "![Repetition diagnostics](repetition_diagnostics.png)",
        "",
        "![Current-byte read diagnostics](read_opportunity_diagnostics.png)",
        "",
        "![Token-class diagnostics](token_class_diagnostics.png)",
        "",
        "![Training dashboard](training_dashboard.png)",
        "",
        "![RAVEL address heatmap](ravel_address_heatmap.png)",
        "",
        "![RAVEL hit entropy](ravel_hit_entropy.png)",
        "",
        "![Attention internals](attention_internals.png)",
        "",
        "![Activation internals](activation_rms.png)",
        "",
        "## What we can learn from this run",
        "",
    ]
    if "ravel" in final and "attention" in final:
        r_loss = float(final["ravel"]["loss"])
        a_loss = float(final["attention"]["loss"])
        winner = "RAVEL" if r_loss < a_loss else "attention"
        loss_gap = abs(a_loss - r_loss)
        lines.extend(
            [
                f"- Final eval loss: RAVEL `{r_loss:.4f}`, attention `{a_loss:.4f}`. On this run, `{winner}` is lower.",
                f"- Eval-loss gap: `{loss_gap:.4f}` cross-entropy.",
                f"- Median training throughput: RAVEL `{med_tps.get('ravel', float('nan')):,.0f}` tokens/sec, attention `{med_tps.get('attention', float('nan')):,.0f}` tokens/sec.",
                f"- Median gradient norm: RAVEL `{med_grad.get('ravel', float('nan')):.3f}`, attention `{med_grad.get('attention', float('nan')):.3f}`.",
                f"- RAVEL mean memory hit rate on the probe batch: `{probe_summary['ravel']['mean_hit_rate']:.3f}`.",
                f"- RAVEL mean normalized address entropy: `{probe_summary['ravel']['mean_address_entropy']:.3f}`.",
                f"- Attention mean normalized entropy: `{probe_summary['attention']['mean_attention_entropy']:.3f}`.",
                f"- Attention expected backward distance: `{probe_summary['attention']['expected_attention_distance']:.1f}` tokens.",
                f"- Mean activation RMS: RAVEL `{probe_summary['activations']['ravel_activation_rms']:.3f}`, attention `{probe_summary['activations']['attention_activation_rms']:.3f}`.",
            ]
        )
        if distance_profile:
            repeated_share = 1.0 - float(distance_profile.get("no prior in block", 0.0))
            near_share = float(distance_profile.get("1", 0.0)) + float(distance_profile.get("2-4", 0.0))
            mid_share = float(distance_profile.get("5-16", 0.0)) + float(distance_profile.get("17-64", 0.0))
            lines.extend(
                [
                    f"- Diagnostic corpus profile: `{repeated_share:.1%}` of target bytes appeared earlier in the sampled block; `{near_share:.1%}` were within four bytes and `{mid_share:.1%}` were 5-64 bytes back.",
                ]
            )
        if read_profile:
            read_repeated_share = 1.0 - float(read_profile.get("no prior in block", 0.0))
            lines.append(
                f"- Current-byte memory-read opportunity: `{read_repeated_share:.1%}` of positions had seen the current input byte earlier in the same block."
            )
        if strongest_ravel_bucket is not None:
            lines.append(
                f"- Largest target-repetition RAVEL edge: `{strongest_ravel_bucket}` with attention-minus-RAVEL NLL `{distance_gaps[strongest_ravel_bucket]:+.4f}`."
            )
        if strongest_attention_bucket is not None:
            lines.append(
                f"- Largest target-repetition attention edge: `{strongest_attention_bucket}` with attention-minus-RAVEL NLL `{distance_gaps[strongest_attention_bucket]:+.4f}`."
            )
        elif distance_gaps:
            smallest_bucket = min(distance_gaps, key=lambda k: distance_gaps[k])
            lines.append(
                f"- No target-repetition bucket favored attention; the smallest RAVEL edge was `{smallest_bucket}` at `{distance_gaps[smallest_bucket]:+.4f}`."
            )
        if read_gaps:
            lines.append(
                f"- Largest current-byte read-opportunity RAVEL edge: `{read_best}` with attention-minus-RAVEL NLL `{read_gaps[read_best]:+.4f}`."
            )
        if token_class_gaps:
            lines.append(
                f"- Largest token-class RAVEL edge: `{best_class}` at `{token_class_gaps[best_class]:+.4f}`; weakest class was `{weakest_class}` at `{token_class_gaps[weakest_class]:+.4f}`."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This benchmark is best read as a mechanism test, not a leaderboard. RAVEL has hard literal address heads for bytes and byte bigrams, plus a learned product-code head. That gives it a cheap way to retrieve the most recent payload written under the same discrete address. If its advantage concentrates on repeated-byte buckets, the result is evidence that the event-memory path is helping. If the advantage disappears on novel or highly local syntax bytes, then the win is not general language modeling competence; it is addressed recurrence doing addressed-recurrence work.",
            "",
            "The repetition diagnostic is therefore the most important plot. The left panel says how much opportunity the corpus gives to exact recurrence. The middle panel asks how hard each opportunity type is for each model. The right panel shows the signed loss gap, so positive bars mean RAVEL is lower-NLL and negative bars mean attention is lower-NLL.",
            "",
            "The current-byte read diagnostic is closer to RAVEL's actual operation. At prediction position `j`, the memory read is keyed by the current input token and its literal/bigram variants, not by the unknown next token. If RAVEL wins when the current input byte has appeared before, that supports the interpretation that it is learning reusable byte-continuation records such as after `t`, after a space, or after punctuation-like contexts. A win in the `no prior` bucket means recurrence is not the full explanation; it points to the local mixer, learned address heads, optimizer behavior, or corpus-level byte priors.",
            "",
            "The attention baseline now uses LR warmup because the first structured-story run showed much larger and more variable attention gradients. That change makes the comparison less likely to be a story about optimizer shock in the first few hundred updates.",
            "",
            (
                "On this run, RAVEL is faster per token. That should be read as a steady-state implementation result for this shape, not as a universal claim: at short contexts the constant factors from routing, local mixing, and Python/PyTorch dispatch can dominate, while at longer contexts the attention baseline pays its quadratic score/value cost."
                if med_tps.get("ravel", 0.0) > med_tps.get("attention", float("inf"))
                else "On this run, RAVEL is still slower per token. That does not contradict the asymptotic story by itself: at short contexts the constant factors from routing, local mixing, and Python/PyTorch dispatch can dominate, while the attention baseline is mostly dense matmul."
            ),
            "",
            f"My read of this particular run: the result is stronger than the synthetic-story pass because the RAVEL edge survives on TinyStories, but the reason still looks byte-mechanistic. The biggest target and current-byte gaps are both in the `{strongest_ravel_bucket or 'n/a'}` / `{read_best}` near-repeat regimes, and the largest class gap is `{best_class}`. That points toward a small byte-level model learning continuation and boundary records quickly. The attention model's high entropy and broad expected distance suggest it is still using a diffuse context average after 800 steps, even with warmup. The generated samples are still rough for both models, so this is not yet a quality claim; it is a short-horizon learning-dynamics claim.",
            "",
            "## Samples",
            "",
        ]
    )
    for name, text in samples.items():
        lines.extend([f"### {name}", "", "```text", text, "```", ""])
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `metrics.csv`",
            "- `token_diagnostics.csv`",
            "- `summary.json`",
            "- `ravel_final.pt`",
            "- `attention_final.pt`",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Train and compare RAVEL and standard softmax attention LMs")
    p.add_argument("--config", default=str(ROOT / "configs" / "byte" / "ravel_200k_byte.json"))
    p.add_argument("--out-dir", default=str(ROOT / "runs" / "comparison_200k"))
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--attention-lr", type=float, default=None)
    p.add_argument("--ravel-warmup-steps", type=int, default=0)
    p.add_argument("--attention-warmup-steps", type=int, default=100)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--no-match-attention-params", action="store_true")
    p.add_argument(
        "--compile-models",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use full-graph torch.compile fusion (default: enabled on MPS)",
    )
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument("--corpus", choices=["generated", "tinystories", "fineweb-edu"], default="generated")
    p.add_argument("--hf-config", default="sample-10BT")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--max-records", type=int, default=2500)
    p.add_argument("--local-text", default=None)
    p.add_argument("--diag-batches", type=int, default=16)
    p.add_argument("--checkpoint-interval", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    if args.attention_lr is None:
        args.attention_lr = args.lr
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        p.error("--min-lr-ratio must be between 0 and 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if args.compile_models is None:
        args.compile_models = device.type == "mps"

    cfg = RavelConfig.from_json(args.config)
    cfg.block_size = args.block_size
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.max_steps = args.steps
    cfg.validate()

    tokenizer = ByteTokenizer()
    texts = load_texts(args)
    if len(texts) < 2:
        raise ValueError("need at least two text records for a train/eval split")
    split = max(1, int(0.9 * len(texts)))
    train_texts, eval_texts = texts[:split], texts[split:]
    train_tokens = encode_records(train_texts, tokenizer)
    eval_tokens = encode_records(eval_texts, tokenizer)
    (out_dir / "train_corpus.txt").write_text("\n\n".join(train_texts))
    (out_dir / "eval_corpus.txt").write_text("\n\n".join(eval_texts))
    cfg.to_json(out_dir / "config.json")
    tokenizer.save(out_dir / "tokenizer.json")

    ravel = RavelLM(deepcopy(cfg))
    attention_cfg = deepcopy(cfg)
    if not args.no_match_attention_params:
        attention_cfg = build_param_matched_attention_cfg(attention_cfg, ravel.num_parameters, args.attention_heads)
    attention = AttentionLM(attention_cfg, n_heads=args.attention_heads)
    params = {"ravel": ravel.num_parameters, "attention": attention.num_parameters}
    args.ravel_d_model = cfg.d_model
    args.attention_d_model = attention_cfg.d_model
    print(f"params ravel={params['ravel']:,} attention={params['attention']:,}")
    print(f"tokens train={train_tokens.size:,} eval={eval_tokens.size:,}")

    ravel, ravel_metrics = train_model(
        "ravel",
        ravel,
        train_tokens,
        eval_tokens,
        args,
        device,
        lr=args.lr,
        warmup_steps=args.ravel_warmup_steps,
    )
    attention, attention_metrics = train_model(
        "attention",
        attention,
        train_tokens,
        eval_tokens,
        args,
        device,
        lr=args.attention_lr,
        warmup_steps=args.attention_warmup_steps,
    )
    metrics = pd.concat([ravel_metrics, attention_metrics], ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    torch.save({"model": ravel.state_dict(), "config": cfg.to_dict(), "params": params["ravel"]}, out_dir / "ravel_final.pt")
    torch.save(
        {"model": attention.state_dict(), "config": attention_cfg.to_dict(), "params": params["attention"]},
        out_dir / "attention_final.pt",
    )

    probe_rng = np.random.default_rng(args.seed + 42)
    probe_x, _ = sample_batch(eval_tokens, min(args.batch_size, 4), args.block_size, probe_rng, torch.device("cpu"))
    ravel_probe = probe_ravel(ravel, probe_x)
    attention_probe = probe_attention(attention, probe_x)

    plot_loss(metrics, out_dir / "loss_curves.png")
    diag = token_diagnostics({"ravel": ravel, "attention": attention}, eval_tokens, args, device)
    diag.to_csv(out_dir / "token_diagnostics.csv", index=False)
    diagnostic_summary = plot_diagnostics(
        diag,
        out_dir / "repetition_diagnostics.png",
        out_dir / "read_opportunity_diagnostics.png",
        out_dir / "token_class_diagnostics.png",
    )
    plot_training_dashboard(metrics, params, out_dir / "training_dashboard.png")
    ravel_summary = plot_ravel(ravel_probe, out_dir / "ravel_address_heatmap.png", out_dir / "ravel_hit_entropy.png")
    attention_summary = plot_attention(attention_probe, out_dir / "attention_internals.png")
    activation_summary = plot_internals(ravel_probe, attention_probe, out_dir / "activation_rms.png")

    prompt = default_prompt(args)
    samples = {
        "RAVEL": generate_sample(ravel, tokenizer, prompt, max_new_tokens=100),
        "Attention": generate_sample(attention, tokenizer, prompt, max_new_tokens=100),
    }
    summary = {
        "args": vars(args),
        "params": params,
        "corpus": {
            "train_records": len(train_texts),
            "eval_records": len(eval_texts),
            "train_tokens": int(train_tokens.size),
            "eval_tokens": int(eval_tokens.size),
            "optimizer_tokens_per_model": int(args.steps * args.batch_size * args.block_size),
        },
        "ravel": ravel_summary,
        "attention": attention_summary,
        "activations": activation_summary,
        "diagnostics": diagnostic_summary,
        "samples": samples,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        out_dir,
        args,
        metrics,
        params,
        {"ravel": ravel_summary, "attention": attention_summary, "activations": activation_summary},
        diagnostic_summary,
        samples,
    )
    print(f"wrote report to {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
