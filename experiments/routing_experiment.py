#!/usr/bin/env python3
"""Do RAVEL's learned addressors deserve gradients?

Arms (identical data order, same protocol as the repo's comparisons):
  A frozen : current model — learned addressors are frozen random projections
  B st     : straight-through routing — forward identical (scale==1), but
             selection log-probs carry gradient into the addressor projections
  C literal: n_learned_heads=0 (slightly fewer params; tests whether the
             learned head earns its keep at all)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/Users/kotnewm/Documents/GitHub/ravel_lm_project")
sys.path.insert(0, str(ROOT))

from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM, RavelMemoryLayer, literal_addresses_from_tokens
from ravel_lm.ravel_memory import causal_last_k_lookup
from ravel_lm.runtime import FlatAdamW

DEV = torch.device("mps")
TRAIN_CORPUS = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/train_corpus.txt"
EVAL_CORPUS = ROOT / "runs/fineweb_edu_3m_15m_ctx2048/eval_corpus.txt"
STEPS = 2500
EVAL_EVERY = 500
EVAL_WINDOWS = 16
LR = 3.5e-4
WARMUP = 100
T = 2048
OUT = Path(__file__).parent / "routing_experiment_results.json"


class STMemoryLayer(RavelMemoryLayer):
    """Straight-through routed variant. Forward values are bit-identical to
    RavelMemoryLayer (the ST scale is exactly 1.0 in forward); backward sends
    (grad . value) credit into the selection log-probs of both addressors."""

    def forward(self, x, token_ids, write_mask=None, literal_addr=None):
        B, Tt, D = x.shape
        xn = self.norm(x)
        C = self.cfg.n_memory_heads
        n_pay = self.payload.weight.shape[0]
        n_gate = self.gate.weight.shape[0]
        weights = [self.payload.weight, self.gate.weight,
                   self.write_addr.proj.weight, self.read_addr.proj.weight]
        packed = F.linear(xn, torch.cat(weights, dim=0))
        payload = packed[..., :n_pay].reshape(B, Tt, C, self.cfg.payload_dim)
        gate_lin = packed[..., n_pay : n_pay + n_gate] + self.gate.bias

        if literal_addr is None:
            literal_addr = literal_addresses_from_tokens(
                token_ids, address_space=self.cfg.address_space,
                n_heads=self.cfg.n_literal_heads, bos_token_id=self.cfg.bos_token_id,
            )
        n_addr = self.write_addr.proj.weight.shape[0]
        addr_logits = packed[..., n_pay + n_gate :]  # NOT detached: ST needs grad

        def decode_st(addressor, logits):
            shp = logits.shape[:-1] + (addressor.n_heads, addressor.n_codebooks, addressor.codebook_size)
            lg = logits.reshape(*shp)
            codes = lg.argmax(dim=-1)
            mult = addressor.multipliers.view(*([1] * (codes.ndim - 1)), addressor.n_codebooks)
            addr = (codes * mult).sum(dim=-1).remainder(addressor.address_space)
            logp = lg.log_softmax(dim=-1).gather(-1, codes.unsqueeze(-1)).squeeze(-1).sum(dim=-1)  # [B,T,H]
            return addr, logp

        w_addr_l, w_logp = decode_st(self.write_addr, addr_logits[..., :n_addr])
        r_addr_l, r_logp = decode_st(self.read_addr, addr_logits[..., n_addr:])
        write_addr = torch.cat([literal_addr, w_addr_l], dim=-1)
        read_addr = torch.cat([literal_addr, r_addr_l], dim=-1)

        # ST scale on the learned head's written payload (write credit).
        n_lit = self.cfg.n_literal_heads
        w_p = w_logp.exp()
        st_w = (w_p / w_p.detach()).unsqueeze(-1)  # [B,T,H,1], forward == 1
        payload = torch.cat([payload[..., :n_lit, :], payload[..., n_lit:, :] * st_w], dim=2)

        last_values, _ = causal_last_k_lookup(
            write_addr, payload, read_addr,
            address_space=self.cfg.address_space, k=self.cfg.last_k, write_mask=write_mask,
        )
        # ST scale on the learned head's retrieved values (read credit).
        r_p = r_logp.exp()
        st_r = (r_p / r_p.detach()).unsqueeze(-1).unsqueeze(-1)  # [B,T,H,1,1]
        last_values = torch.cat(
            [last_values[..., :n_lit, :, :], last_values[..., n_lit:, :, :] * st_r], dim=2
        )
        fused = self.fuse(last_values.reshape(B, Tt, -1))
        return x + self.dropout(torch.sigmoid(gate_lin) * fused)


def load_bytes(path):
    return np.frombuffer(path.read_bytes(), dtype=np.uint8)


def make_offsets(n_tokens, steps, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tokens - (T + 1), size=steps)


def batch_at(tokens, off):
    chunk = tokens[off : off + T + 1].astype(np.int64)
    x = torch.from_numpy(chunk[:-1]).view(1, T).to(DEV)
    y = torch.from_numpy(chunk[1:]).view(1, T).to(DEV)
    return x, y


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def eval_nll(model, eval_tokens, eval_offsets):
    model.eval()
    losses = []
    for off in eval_offsets:
        x, y = batch_at(eval_tokens, off)
        losses.append(float(model(x, y)["loss"]))
    model.train()
    return sum(losses) / len(losses)


def run_arm(name, train_tokens, eval_tokens, offsets, eval_offsets):
    cfg = RavelConfig.from_json(ROOT / "configs/byte/ravel_3m_byte.json")
    cfg.block_size = T
    cfg.batch_size = 1
    if name == "literal":
        cfg.n_learned_heads = 0
    if name == "literal3":
        cfg.n_learned_heads = 0
        cfg.n_literal_heads = 3
    cfg.validate()
    torch.manual_seed(0)
    model = RavelLM(cfg).to(DEV)
    if name == "st":
        for block in model.blocks:
            if block.memory is not None:
                block.memory.__class__ = STMemoryLayer
                block.memory.write_addr.proj.weight.requires_grad_(True)
                block.memory.read_addr.proj.weight.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=LR, betas=(0.9, 0.95))
    exec_model = torch.compile(model, backend="inductor", mode="reduce-overhead", fullgraph=True, dynamic=False)

    curve = []
    t0 = time.perf_counter()
    for step in range(STEPS):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch_at(train_tokens, offsets[step])
        opt.zero_grad()
        loss = exec_model(x, y)["loss"]
        loss.backward()
        opt.clip_grad_norm_(1.0)
        opt.step()
        if (step + 1) % EVAL_EVERY == 0 or step == STEPS - 1:
            nll = eval_nll(exec_model, eval_tokens, eval_offsets)
            curve.append({"step": step + 1, "eval_nll": nll})
            print(f"[{name:8s}] step {step+1:5d} eval_nll {nll:.4f} train {float(loss):.4f}", flush=True)
    wall = time.perf_counter() - t0

    # learned-head address diversity on one eval window (routing health probe)
    diversity = None
    if cfg.n_learned_heads > 0:
        with torch.no_grad():
            x, _ = batch_at(eval_tokens, eval_offsets[0])
            xe = model.tok_emb(x) + model.pos_emb.weight[:T].unsqueeze(0)
            uniq = []
            for block in model.blocks:
                mem = block.memory
                xn = mem.norm(xe)
                logits = F.linear(xn, mem.read_addr.proj.weight)
                shp = logits.shape[:-1] + (mem.read_addr.n_heads, mem.read_addr.n_codebooks, mem.read_addr.codebook_size)
                codes = logits.reshape(*shp).argmax(-1)
                mult = mem.read_addr.multipliers.view(1, 1, 1, -1)
                addr = (codes * mult).sum(-1).remainder(cfg.address_space)
                uniq.append(int(addr.unique().numel()))
                xe = block(xe, x)
            diversity = uniq
    return {"final_nll": curve[-1]["eval_nll"], "curve": curve, "params": n_params,
            "wall_s": round(wall, 1), "read_addr_unique_per_layer": diversity}


def main():
    train_tokens = load_bytes(TRAIN_CORPUS)
    eval_tokens = load_bytes(EVAL_CORPUS)
    offsets = make_offsets(len(train_tokens), STEPS, seed=2026)
    eval_offsets = make_offsets(len(eval_tokens), EVAL_WINDOWS, seed=7)
    print(f"train {len(train_tokens)/1e6:.1f}M bytes, eval {len(eval_tokens)/1e6:.1f}M bytes, "
          f"{STEPS} steps x {T} tokens = {STEPS*T/1e6:.1f}M train tokens/arm", flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    arms = sys.argv[1:] or ["frozen", "st", "literal"]
    for arm in arms:
        results[arm] = run_arm(arm, train_tokens, eval_tokens, offsets, eval_offsets)
        OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {"final_nll": v["final_nll"], "params": v["params"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
