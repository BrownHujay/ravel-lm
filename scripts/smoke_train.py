#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ravel_lm.train import main


if __name__ == "__main__":
    main([
        "--config", str(ROOT / "configs" / "byte" / "ravel_smoke_byte.json"),
        "--tokenizer", "byte",
        "--toy",
        "--batch-size", "2",
        "--max-steps", "2",
        "--eval-interval", "0",
        "--out-dir", str(ROOT / "runs" / "smoke"),
        "--log-interval", "1",
        "--save-interval", "0",
    ])
