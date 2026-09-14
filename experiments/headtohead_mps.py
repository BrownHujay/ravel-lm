#!/usr/bin/env python3
"""Head-to-head: proper-recipe softmax transformer vs RAVEL, same data, same
recipe, matched params. Runs on MPS (laptop). Only difference between the two
models is attention vs RAVEL memory — identical FFN (SwiGLU), norm (RMSNorm),
embeddings, tying.

Recipe (byte-LM standard): batch 64, LR = 0.005*B^-0.5 = 6.25e-4, AdamW
(0.9,0.95), warmup 1% -> cosine to 10%, weight decay 0.1, grad clip 1.0,
dropout 0.0. Byte-level, context 1024.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.attention_model import AttentionLM

DEV = torch.device("mps")
TRAIN = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/train_corpus.txt"
EVAL = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/eval_corpus.txt"
OUT = Path(__file__).parent / "headtohead_results.json"


def build(kind, T):
    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.block_size = T; cfg.d_model = 160; cfg.n_layers = 10; cfg.dropout = 0.0
    if kind == "ravel":
        cfg.n_learned_heads = 0  # clean config: NoPE + 2 literal heads
        cfg.validate()
        torch.manual_seed(0)
        m = RavelLM(cfg).to(DEV)
        m.pos_emb.weight.data.zero_(); m.pos_emb.weight.requires_grad_(False)  # NoPE
        return m, cfg
    cfg.validate()
    torch.manual_seed(0)
    m = AttentionLM(cfg, n_heads=8).to(DEV)  # proper transformer: 8 heads, SwiGLU, pre-LN
    return m, cfg


def load_bytes(p): return np.frombuffer(Path(p).read_bytes(), dtype=np.uint8)


def batch(tok, offs, T):
    xs = np.stack([tok[o:o+T+1].astype(np.int64) for o in offs])
    x = torch.from_numpy(xs[:, :-1]).to(DEV)
    y = torch.from_numpy(xs[:, 1:]).to(DEV)
    return x, y


def lr_at(step, steps, warmup, peak):
    if step < warmup: return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(m, tok, rng, T, n=64, bs=16):
    m.eval(); losses = []
    for _ in range(n // bs):
        offs = rng.integers(0, len(tok) - (T + 1), size=bs)
        x, y = batch(tok, offs, T)
        losses.append(float(m(x, y)["loss"]))
    m.train(); return sum(losses) / len(losses)


def run(kind, args, train_tok, eval_tok):
    m, cfg = build(kind, args.T)
    nparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
    params = [p for p in m.parameters() if p.requires_grad]
    peak = 0.005 * (args.batch ** -0.5)
    opt = torch.optim.AdamW(params, lr=peak, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.1, fused=True)
    em = m
    if args.compile:
        em = torch.compile(m, backend="inductor", mode="default", fullgraph=False, dynamic=False)
    rng = np.random.default_rng(2026)
    ev_rng = np.random.default_rng(7)
    warmup = max(1, args.steps // 100)
    micro = args.batch // args.accum
    curve = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        lr = lr_at(step, args.steps, warmup, peak)
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            offs = rng.integers(0, len(train_tok) - (args.T + 1), size=micro)
            x, y = batch(train_tok, offs, args.T)
            loss = em(x, y)["loss"] / args.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            nll = evaluate(em, eval_tok, ev_rng, args.T)
            el = time.perf_counter() - t0
            curve.append({"step": step, "eval_nll": nll})
            print(f"[{kind:5s}] step {step:5d}/{args.steps} nll {nll:.4f} bpb {nll/math.log(2):.4f}"
                  f"  lr {lr:.2e}  ({el/max(1,step+1)*1000:.0f} ms/step, {el/60:.1f} min)", flush=True)
            res = json.loads(OUT.read_text()) if OUT.exists() else {}
            res[kind] = {"final_nll": nll, "final_bpb": nll/math.log(2), "curve": curve,
                         "params": nparams, "batch": args.batch, "T": args.T,
                         "peak_lr": peak, "steps": args.steps}
            OUT.write_text(json.dumps(res, indent=2))
    print(f"[{kind}] DONE nll {curve[-1]['eval_nll']:.4f} bpb {curve[-1]['eval_nll']/math.log(2):.4f} "
          f"params {nparams:,}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4)   # micro-batch = batch/accum
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--models", default="attention,ravel")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    train_tok = load_bytes(TRAIN); eval_tok = load_bytes(EVAL)
    print(f"recipe: batch={args.batch} (micro {args.batch//args.accum}x{args.accum}) T={args.T} "
          f"peak_lr={0.005*args.batch**-0.5:.2e} steps={args.steps} "
          f"tokens={args.steps*args.batch*args.T/1e6:.0f}M", flush=True)
    for kind in args.models.split(","):
        run(kind, args, train_tok, eval_tok)


if __name__ == "__main__":
    main()
