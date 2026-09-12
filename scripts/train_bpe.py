#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.data import iter_tinystories, toy_stories
from ravel_lm.tokenizers import BytePairTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Train dependency-free byte-level BPE tokenizer")
    p.add_argument("--out", required=True, help="Output tokenizer JSON")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--local-text", default=None)
    p.add_argument("--tinystories", action="store_true")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--max-texts", type=int, default=10000)
    p.add_argument("--min-pair-count", type=int, default=2)
    args = p.parse_args()

    if args.toy:
        texts = toy_stories()
    elif args.local_text:
        texts = iter_tinystories(local_path=args.local_text, max_records=args.max_texts)
    else:
        # HF TinyStories by default. This requires datasets + internet/cache.
        texts = iter_tinystories(max_records=args.max_texts)

    tok = BytePairTokenizer.train(
        texts,
        vocab_size=args.vocab_size,
        max_texts=args.max_texts if not args.local_text else None,
        min_pair_count=args.min_pair_count,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"saved {args.out} with vocab_size={tok.vocab_size}")


if __name__ == "__main__":
    main()
