#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.data import iter_tinystories, write_token_bin
from ravel_lm.tokenizers import ByteTokenizer, BytePairTokenizer, load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare TinyStories/local text into flat token .bin")
    p.add_argument("--out", required=True, help="Output token .bin path")
    p.add_argument("--tokenizer", default="byte", help="byte or tokenizer JSON path")
    p.add_argument("--local-text", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    p.add_argument("--add-bos", action="store_true")
    args = p.parse_args()

    tokenizer = ByteTokenizer() if args.tokenizer == "byte" else load_tokenizer(args.tokenizer)
    if args.dtype == "uint16" and tokenizer.vocab_size > 65535:
        raise ValueError("uint16 cannot store this tokenizer's vocab; use --dtype uint32")
    texts = iter_tinystories(local_path=args.local_text, split=args.split, max_records=args.max_records)
    n = write_token_bin(texts, tokenizer, args.out, dtype=args.dtype, add_bos=args.add_bos, add_eos=True)
    print(f"wrote {n:,} tokens to {args.out} using vocab_size={tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
