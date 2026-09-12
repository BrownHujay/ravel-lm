from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .config import RavelConfig, load_config
from .data import (
    MemmapTokenDataset,
    ShuffleBuffer,
    StreamingTokenDataset,
    infinite_loader,
    iter_tinystories,
    toy_stories,
)
from .model import RavelLM
from .runtime import FlatAdamW, adamw_fused_for
from .tokenizers import ByteTokenizer, load_tokenizer


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_text_iter(args):
    if args.toy:
        texts = toy_stories()
        # Repeat forever through IterableDataset packing.
        def forever():
            while True:
                for t in texts:
                    yield t

        return forever()
    if args.local_text:
        return iter_tinystories(local_path=args.local_text, max_records=args.max_records)
    return iter_tinystories(split=args.split, streaming=True, max_records=args.max_records)


def build_loader(args, cfg: RavelConfig, tokenizer, split: str = "train"):
    if args.token_bin:
        ds = MemmapTokenDataset(args.token_bin, block_size=cfg.block_size, dtype=args.token_dtype)
        return DataLoader(ds, batch_size=args.batch_size or cfg.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    texts = build_text_iter(args)
    if args.shuffle_buffer > 0:
        texts = ShuffleBuffer(texts, args.shuffle_buffer, seed=args.seed)
    ds = StreamingTokenDataset(texts, tokenizer, block_size=cfg.block_size, add_bos=args.add_bos, add_eos=True)
    return DataLoader(ds, batch_size=args.batch_size or cfg.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def configure_optimizer(model: RavelLM, cfg: RavelConfig, device: torch.device):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "emb" not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused_ok = adamw_fused_for(device) and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    if device.type != "cuda":
        # Flat-buffer AdamW turns clip + step into a handful of large kernels;
        # the stock path dispatches one kernel per parameter tensor. CUDA keeps
        # the stock optimizer for GradScaler compatibility.
        return FlatAdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=fused_ok)
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=fused_ok)


def lr_at_step(step: int, cfg: RavelConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.1 + 0.9 * cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: RavelLM, loader, device: torch.device, eval_batches: int) -> float:
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(eval_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        out = model(x, y)
        losses.append(out["loss"].item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train a RAVEL language model")
    p.add_argument("--config", required=True, help="Config JSON path or name under configs/")
    p.add_argument("--tokenizer", default="byte", help="'byte' or tokenizer JSON path")
    p.add_argument("--out-dir", default="runs/ravel", help="Checkpoint/output directory")
    p.add_argument("--toy", action="store_true", help="Use built-in tiny offline toy stories")
    p.add_argument("--local-text", default=None, help="Local txt/jsonl corpus instead of HF TinyStories")
    p.add_argument("--split", default="train")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--token-bin", default=None, help="Pre-tokenized flat .bin file")
    p.add_argument("--token-dtype", default="uint16")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--add-bos", action="store_true")
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use full-graph torch.compile fusion (default: enabled on MPS)",
    )
    p.add_argument("--device", default=default_device())
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval

    tokenizer = ByteTokenizer() if args.tokenizer == "byte" else load_tokenizer(args.tokenizer)
    if tokenizer.vocab_size != cfg.vocab_size:
        print(f"[info] resizing cfg.vocab_size {cfg.vocab_size} -> tokenizer.vocab_size {tokenizer.vocab_size}")
        cfg.vocab_size = tokenizer.vocab_size
        cfg.validate()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "config.json")
    if hasattr(tokenizer, "save"):
        tokenizer.save(out_dir / "tokenizer.json")

    model = RavelLM(cfg).to(device)
    print(f"model parameters: {model.num_parameters:,}")
    compile_model = args.compile if args.compile is not None else device.type == "mps"
    exec_model = (
        torch.compile(
            model,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
        if compile_model
        else model
    )
    optimizer = configure_optimizer(model, cfg, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    train_loader = build_loader(args, cfg, tokenizer)
    train_iter = infinite_loader(train_loader)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    exec_model.train()
    running_loss = 0.0
    for step in range(cfg.max_steps):
        lr = lr_at_step(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_ctx:
                out = exec_model(x, y)
                loss = out["loss"] / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            total_loss += float(loss.item())
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            if isinstance(optimizer, FlatAdamW):
                optimizer.clip_grad_norm_(cfg.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss = 0.95 * running_loss + 0.05 * total_loss if step else total_loss
        if step % args.log_interval == 0:
            print(f"step {step:05d} loss {total_loss:.4f} ema {running_loss:.4f} lr {lr:.2e}")
        if cfg.eval_interval > 0 and step > 0 and step % cfg.eval_interval == 0:
            # Reuse train source for a cheap smoke eval when no validation source is configured.
            eval_loss = estimate_loss(exec_model, train_loader, device, cfg.eval_batches)
            print(f"eval loss {eval_loss:.4f}")
        if args.save_interval > 0 and step > 0 and step % args.save_interval == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(),
                "step": step,
            }
            torch.save(ckpt, out_dir / f"ckpt_{step:06d}.pt")

    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": cfg.max_steps}, out_dir / "final.pt")
    print(f"saved final checkpoint to {out_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
