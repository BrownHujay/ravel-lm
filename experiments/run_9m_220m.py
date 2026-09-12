#!/usr/bin/env python3
"""9M-param, 220M-token comparison run on RDNA4, HIP-graph captured.

Arm A 'full3'  : 2 literal + 1 learned head, with position embeddings (baseline).
Arm B 'nope_chimex': NoPE, 2 literal, 0 learned, CHIME-X v2 age-gate (ablation winner).

Graph-captures fwd+bwd+clip+opt; LR schedule applied via an in-place lr tensor
(capturable AdamW). Per-arm eager fallback if capture fails. Saves JSON + prints.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.runtime import FlatAdamW
from experiments.chinchilla_suite import ChimeV2MemoryLayer

DEV = torch.device("cuda")
T = 2048
LR = 3.5e-4
TRAIN = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/train_corpus.txt"
EVAL = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/eval_corpus.txt"
OUT = Path(__file__).parent / "run_9m_220m_results.json"
EVAL_WINDOWS = 32


def build(arm):
    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.block_size = T; cfg.batch_size = 1; cfg.d_model = 256; cfg.n_layers = 12
    if arm == "nope_chimex":
        cfg.n_learned_heads = 0
    cfg.validate()
    torch.manual_seed(0)
    model = RavelLM(cfg).to(DEV)
    if arm == "nope_chimex":
        model.pos_emb.weight.data.zero_(); model.pos_emb.weight.requires_grad_(False)
        for b in model.blocks:
            if b.memory is not None:
                b.memory.__class__ = ChimeV2MemoryLayer
                proj = torch.nn.Linear(6, 1, bias=False).to(DEV)
                torch.nn.init.zeros_(proj.weight)
                b.memory.age_proj = proj
    return model, cfg


def lr_at(step, steps, warmup):
    if step < warmup:
        return LR * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def load_bytes(p):
    return np.frombuffer(Path(p).read_bytes(), dtype=np.uint8)


def run_arm(arm, steps, train_tok, eval_tok, offsets, eval_offsets):
    model, cfg = build(arm)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=LR, betas=(0.9, 0.95), capturable=True)
    lr_t = torch.tensor(LR, device=DEV)
    opt._inner.param_groups[0]["lr"] = lr_t  # in-place updatable under graph

    x = torch.zeros(1, T, dtype=torch.long, device=DEV)
    y = torch.zeros(1, T, dtype=torch.long, device=DEV)

    def load(src, o):
        chunk = src[o:o + T + 1].astype(np.int64)
        x.copy_(torch.from_numpy(chunk[:-1]).view(1, T))
        y.copy_(torch.from_numpy(chunk[1:]).view(1, T))

    def step():
        opt.zero_grad()
        loss = model(x, y)["loss"]
        loss.backward()
        opt.clip_grad_norm_(1.0)
        opt.step()
        return loss

    load(train_tok, int(offsets[0]))
    graphed = True
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                step()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        # reset to init after warmup
        _m2, _ = build(arm)
        with torch.no_grad():
            for pn, pf in zip(model.parameters(), _m2.parameters()):
                pn.copy_(pf)
        del _m2
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_loss = step()
        replay = g.replay
    except Exception as e:
        print(f"[{arm}] graph capture failed ({str(e)[:80]}); eager fallback", flush=True)
        graphed = False

    @torch.no_grad()
    def evaluate():
        model.eval()
        ls = []
        for o in eval_offsets:
            load(eval_tok, int(o))
            ls.append(float(model(x, y)["loss"]))
        model.train()
        return sum(ls) / len(ls)

    warmup = steps // 20
    curve = []
    t0 = time.perf_counter()
    for stp in range(steps):
        lr_t.fill_(lr_at(stp, steps, warmup))
        load(train_tok, int(offsets[stp]))
        if graphed:
            replay()
        else:
            step()
        if stp % 5000 == 0 or stp == steps - 1:
            nll = evaluate()
            curve.append({"step": stp, "eval_nll": nll})
            el = time.perf_counter() - t0
            print(f"[{arm}] step {stp:6d}/{steps} eval_nll {nll:.4f}  ({el/max(1,stp+1)*1000:.2f} ms/step, {el/60:.1f} min elapsed)", flush=True)
            _partial = json.loads(OUT.read_text()) if OUT.exists() else {}
            _partial[arm] = {"arm": arm, "final_nll": nll, "curve": curve,
                             "params": model.num_parameters, "step": stp, "graphed": graphed}
            OUT.write_text(json.dumps(_partial, indent=2))
    wall = time.perf_counter() - t0
    res = {"arm": arm, "final_nll": curve[-1]["eval_nll"], "curve": curve,
           "params": model.num_parameters, "steps": steps, "tokens": steps * T,
           "graphed": graphed, "wall_s": round(wall, 1),
           "ms_per_step": round(wall / steps * 1000, 3)}
    all_res = json.loads(OUT.read_text()) if OUT.exists() else {}
    all_res[arm] = res
    OUT.write_text(json.dumps(all_res, indent=2))
    print(f"[{arm}] DONE final_nll {res['final_nll']:.4f}  {res['ms_per_step']:.2f} ms/step  {wall/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=220_000_000)
    ap.add_argument("--arms", default="nope_chimex,full3")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    steps = args.smoke if args.smoke else args.tokens // T
    train_tok = load_bytes(TRAIN); eval_tok = load_bytes(EVAL)
    rng = np.random.default_rng(2026)
    offsets = rng.integers(0, len(train_tok) - (T + 1), size=steps)
    eval_offsets = np.random.default_rng(7).integers(0, len(eval_tok) - (T + 1), size=EVAL_WINDOWS)
    done = set(json.loads(OUT.read_text()).keys()) if (OUT.exists() and not args.smoke) else set()
    for arm in args.arms.split(","):
        if arm in done:
            print(f"skip {arm} (done)", flush=True); continue
        run_arm(arm, steps, train_tok, eval_tok, offsets, eval_offsets)


if __name__ == "__main__":
    main()
