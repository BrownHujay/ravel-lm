from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import json


@dataclass
class ByteTokenizer:
    """UTF-8 byte tokenizer.

    IDs 0..255 are raw bytes. Specials live at 256+ by default, matching the
    default RavelConfig. This tokenizer is deterministic, reversible, fast, and
    excellent for tiny parameter-count configs because the vocab is tiny.
    """

    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258
    unk_token_id: int = 259

    @property
    def vocab_size(self) -> int:
        return 260

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend(text.encode("utf-8", errors="replace"))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        raw = bytearray()
        chunks: List[str] = []
        for idx in ids:
            idx = int(idx)
            if 0 <= idx <= 255:
                raw.append(idx)
            elif not skip_special:
                if raw:
                    chunks.append(raw.decode("utf-8", errors="replace"))
                    raw.clear()
                chunks.append(f"<|{idx}|>")
        if raw:
            chunks.append(raw.decode("utf-8", errors="replace"))
        return "".join(chunks)

    def batch_encode(self, texts: Iterable[str], *, add_bos: bool = False, add_eos: bool = True) -> List[List[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def save(self, path: str | Path) -> None:
        data = {
            "type": "byte",
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        data = json.loads(Path(path).read_text())
        if data.get("type") != "byte":
            raise ValueError(f"not a byte tokenizer file: {path}")
        return cls(
            pad_token_id=data.get("pad_token_id", 256),
            bos_token_id=data.get("bos_token_id", 257),
            eos_token_id=data.get("eos_token_id", 258),
            unk_token_id=data.get("unk_token_id", 259),
        )
