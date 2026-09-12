from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from .config import RavelConfig, load_config
from .model import RavelLM
from .tokenizers import ByteTokenizer, load_tokenizer


def load_model(config: str, checkpoint: Optional[str], device: torch.device) -> RavelLM:
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        cfg = RavelConfig(**ckpt.get("config", {})) if "config" in ckpt else load_config(config)
        model = RavelLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        return model
    return RavelLM(load_config(config)).to(device)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Generate text with RAVEL")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tokenizer", default="byte")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    tokenizer = ByteTokenizer() if args.tokenizer == "byte" else load_tokenizer(args.tokenizer)
    model = load_model(args.config, args.checkpoint, device)
    model.eval()
    ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    idx = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
    out = model.generate(
        idx,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
