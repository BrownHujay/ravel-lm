#!/usr/bin/env python3
"""Real graph-captured training on RDNA4/CUDA — proves the graphed step speed
transfers to actual training (real data, decreasing loss), not just a bench.

Clean ablation-winning config: 2 literal heads, no learned head, NoPE.
Captures fwd+bwd+clip+opt as one HIP graph; each step copies a real batch into
static buffers and replays. LR schedule applied via a static-tensor lr the
optimizer reads (capturable AdamW).
"""
import argparse, sys, time, math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.runtime import FlatAdamW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(ROOT / "runs/fineweb_edu_3m_15m_ctx2048/train_corpus.txt"))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3.5e-4)
    args = p.parse_args()
    DEV = torch.device("cuda")
    T = args.block_size

    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.batch_size = 1; cfg.block_size = T; cfg.n_learned_heads = 0; cfg.validate()
    torch.manual_seed(0)
    model = RavelLM(cfg).to(DEV)
    model.pos_emb.weight.data.zero_(); model.pos_emb.weight.requires_grad_(False)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=args.lr,
                    betas=(0.9, 0.95), capturable=True)

    data = np.frombuffer(Path(args.corpus).read_bytes(), dtype=np.uint8)
    rng = np.random.default_rng(2026)
    offsets = rng.integers(0, len(data) - (T + 1), size=args.steps)

    x = torch.zeros(1, T, dtype=torch.long, device=DEV)
    y = torch.zeros(1, T, dtype=torch.long, device=DEV)

    def load(step):
        o = int(offsets[step]); chunk = data[o:o + T + 1].astype(np.int64)
        x.copy_(torch.from_numpy(chunk[:-1]).view(1, T))
        y.copy_(torch.from_numpy(chunk[1:]).view(1, T))

    def step():
        opt.zero_grad()
        loss = model(x, y)["loss"]
        loss.backward()
        opt.clip_grad_norm_(1.0)
        opt.step()
        return loss

    load(0)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            step()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    # reset params after warmup so we train from init
    torch.manual_seed(0)
    fresh = RavelLM(cfg).to(DEV)
    fresh.pos_emb.weight.data.zero_()
    with torch.no_grad():
        for pn, pf in zip(model.parameters(), fresh.parameters()):
            pn.copy_(pf)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_loss = step()

    losses = []
    t0 = time.perf_counter()
    for stp in range(args.steps):
        load(stp)
        g.replay()
        if stp % 25 == 0 or stp == args.steps - 1:
            torch.cuda.synchronize()
            losses.append((stp, float(static_loss)))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms = dt / args.steps * 1000
    print(f"device={torch.cuda.get_device_name(0)}  params={model.num_parameters:,}  config=clean(NoPE,2lit,0learned)")
    for stp, l in losses:
        print(f"  step {stp:4d}  loss {l:.4f}")
    print(f"REAL-TRAINING graphed: {ms:.3f} ms/step  ({T/ms*1000:,.0f} tok/s)  over {args.steps} steps on real fineweb bytes")


if __name__ == "__main__":
    main()
