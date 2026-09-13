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
from ravel_lm.model import RavelMemoryLayer, literal_addresses_from_tokens
from ravel_lm.ravel_memory import causal_last_k_lookup
from experiments.chinchilla_suite import ChimeV2MemoryLayer
import torch.nn.functional as F
import math as _math


class ChimeFullMemoryLayer(RavelMemoryLayer):
    """CHIME-X v2 age gate on top of FULL addressing (literal + learned heads).

    Same age-gate mechanism as ChimeV2 (zero-init -> identity at start) but keeps
    the learned product-code heads, so it works with n_learned_heads > 0."""

    def forward(self, x, token_ids, write_mask=None, literal_addr=None):
        B, Tt, D = x.shape
        xn = self.norm(x)
        C = self.cfg.n_memory_heads
        P = self.cfg.payload_dim
        payload = self.payload(xn).reshape(B, Tt, C, P)
        gate_lin = self.gate(xn)
        if literal_addr is None:
            literal_addr = literal_addresses_from_tokens(
                token_ids, address_space=self.cfg.address_space,
                n_heads=self.cfg.n_literal_heads, bos_token_id=self.cfg.bos_token_id,
            )
        with torch.no_grad():
            lw = self.write_addr(xn)
            lr = self.read_addr(xn)
        write_addr = torch.cat([literal_addr, lw], dim=-1)
        read_addr = torch.cat([literal_addr, lr], dim=-1)
        t_chan = torch.arange(Tt, device=x.device, dtype=payload.dtype).view(1, Tt, 1, 1).expand(B, Tt, C, 1)
        payload_aug = torch.cat([payload, t_chan], dim=-1)
        values, mask = causal_last_k_lookup(
            write_addr, payload_aug, read_addr,
            address_space=self.cfg.address_space, k=1, write_mask=write_mask,
        )
        values = values.squeeze(3)
        retrieved = values[..., :P]
        t_write = values[..., P]
        m = mask.squeeze(3).to(retrieved.dtype)
        t_read = torch.arange(Tt, device=x.device, dtype=retrieved.dtype).view(1, Tt, 1)
        delta = (t_read - t_write).clamp_min(1.0)
        u = torch.log2(1.0 + delta)
        feats = torch.stack([
            m, (delta <= 8).to(u.dtype) * m, (u / 12.0) * m, (u - u.floor()) * m,
            torch.sin(2 * _math.pi * 0.5 * u) * m, torch.cos(2 * _math.pi * 0.5 * u) * m,
        ], dim=-1)
        age_gate = 1.0 + torch.tanh(self.age_proj(feats))
        fused = self.fuse((retrieved * age_gate).reshape(B, Tt, -1))
        return x + self.dropout(torch.sigmoid(gate_lin) * fused)

DEV = torch.device("cuda")
EAGER = False  # set True to skip HIP graph capture (reliable in detached/no-TTY runs)
T = 2048
LR = 3.5e-4
TRAIN = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/train_corpus.txt"
EVAL = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/eval_corpus.txt"
OUT = Path(__file__).parent / "run_9m_220m_results.json"
EVAL_WINDOWS = 32


def _add_age_proj(model):
    for b in model.blocks:
        if b.memory is not None:
            proj = torch.nn.Linear(6, 1, bias=False).to(DEV)
            torch.nn.init.zeros_(proj.weight)
            b.memory.age_proj = proj


def build(arm):
    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.block_size = T; cfg.batch_size = 1; cfg.d_model = 256; cfg.n_layers = 12
    if arm in ("nope_chimex", "nope_chimex_big"):
        cfg.n_learned_heads = 0
    if arm == "nope_chimex_big":
        # param-match to full3 (~9.08M): +1 layer + a small width bump.
        cfg.n_layers = 13
    cfg.validate()
    torch.manual_seed(0)
    model = RavelLM(cfg).to(DEV)
    if arm in ("nope_chimex", "nope_chimex_big"):
        model.pos_emb.weight.data.zero_(); model.pos_emb.weight.requires_grad_(False)
        for b in model.blocks:
            if b.memory is not None:
                b.memory.__class__ = ChimeV2MemoryLayer
        _add_age_proj(model)
    elif arm == "full3_chimex":
        for b in model.blocks:
            if b.memory is not None:
                b.memory.__class__ = ChimeFullMemoryLayer
        _add_age_proj(model)
    return model, cfg


def lr_at(step, steps, warmup):
    if step < warmup:
        return LR * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def load_bytes(p):
    return np.frombuffer(Path(p).read_bytes(), dtype=np.uint8)


def run_arm(arm, steps, train_tok, eval_tok, offsets, eval_offsets, chunk=0):
    model, cfg = build(arm)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=LR, betas=(0.9, 0.95), capturable=True)
    lr_t = torch.tensor(LR, device=DEV)
    opt._inner.param_groups[0]["lr"] = lr_t  # in-place updatable under graph
    ckpt_path = Path(__file__).parent / f"{arm}_ckpt.pt"
    start_step = 0
    curve = []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEV)
        model.load_state_dict(ck["model"])
        opt._inner.load_state_dict(ck["opt"])
        start_step = ck["step"]
        curve = ck["curve"]
        print(f"[{arm}] resumed at step {start_step}", flush=True)

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
    if EAGER:
        graphed = False
    try:
      if not EAGER:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                step()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        if start_step == 0:
            # reset to init after warmup (only on a fresh run)
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
    end_step = min(steps, start_step + chunk) if chunk else steps
    t0 = time.perf_counter()
    for stp in range(start_step, end_step):
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
    done_full = (end_step >= steps)
    # checkpoint
    torch.save({"model": model.state_dict(), "opt": opt._inner.state_dict(),
                "step": end_step, "curve": curve}, ckpt_path)
    all_res = json.loads(OUT.read_text()) if OUT.exists() else {}
    all_res[arm] = {"arm": arm, "final_nll": curve[-1]["eval_nll"] if curve else None,
                    "curve": curve, "params": model.num_parameters, "step": end_step,
                    "steps": steps, "graphed": graphed, "done": done_full,
                    "ms_per_step_chunk": round(wall / max(1, end_step - start_step) * 1000, 3)}
    OUT.write_text(json.dumps(all_res, indent=2))
    tag = "DONE" if done_full else f"chunk -> step {end_step}"
    print(f"[{arm}] {tag}  nll {curve[-1]['eval_nll'] if curve else 0:.4f}  "
          f"{wall/max(1,end_step-start_step)*1000:.2f} ms/step  {wall/60:.1f} min", flush=True)
    return done_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=220_000_000)
    ap.add_argument("--arms", default="nope_chimex,full3")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--one-chunk", action="store_true")
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()
    global EAGER
    EAGER = args.eager
    steps = args.smoke if args.smoke else args.tokens // T
    train_tok = load_bytes(TRAIN); eval_tok = load_bytes(EVAL)
    rng = np.random.default_rng(2026)
    offsets = rng.integers(0, len(train_tok) - (T + 1), size=steps)
    eval_offsets = np.random.default_rng(7).integers(0, len(eval_tok) - (T + 1), size=EVAL_WINDOWS)
    res = json.loads(OUT.read_text()) if (OUT.exists() and not args.smoke) else {}
    for arm in args.arms.split(","):
        if res.get(arm, {}).get("done"):
            print(f"skip {arm} (done)", flush=True); continue
        finished = run_arm(arm, steps, train_tok, eval_tok, offsets, eval_offsets, chunk=args.chunk)
        if args.one_chunk and not finished:
            print(f"[driver] {arm} chunk complete, more remain", flush=True)
            break


if __name__ == "__main__":
    main()
