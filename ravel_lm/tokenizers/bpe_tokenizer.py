from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple
import json

Pair = Tuple[int, int]


def _bytes_to_word(text: str) -> Tuple[int, ...]:
    return tuple(text.encode("utf-8", errors="replace"))


@dataclass
class BytePairTokenizer:
    """Small pure-Python byte-level BPE tokenizer.

    It starts with byte IDs 0..255 and learns merge tokens from text. The trainer
    is intentionally dependency-free so the repo works without Hugging Face
    tokenizers. For full-scale corpus preprocessing, the scripts stream text in
    chunks; for very large BPE jobs, installing ``tokenizers`` and swapping in a
    Rust-backed trainer would be faster, but this implementation is complete and
    reproducible.
    """

    merges: List[Pair] = field(default_factory=list)
    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258
    unk_token_id: int = 259

    def __post_init__(self) -> None:
        self.base_vocab = 260
        self.merge_to_id: Dict[Pair, int] = {tuple(pair): self.base_vocab + i for i, pair in enumerate(self.merges)}
        self.id_to_merge: Dict[int, Pair] = {self.base_vocab + i: tuple(pair) for i, pair in enumerate(self.merges)}
        self.rank: Dict[Pair, int] = {tuple(pair): i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return self.base_vocab + len(self.merges)

    def _encode_bytes(self, data: bytes) -> List[int]:
        tokens = list(data)
        if not self.merges or len(tokens) < 2:
            return tokens
        # Greedy by merge rank, like standard BPE. This is simple and exact for
        # the learned merge list; speed is fine for small TinyStories configs.
        while True:
            best_pair = None
            best_rank = None
            for a, b in zip(tokens, tokens[1:]):
                r = self.rank.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best_pair = (a, b)
                    best_rank = r
            if best_pair is None:
                break
            new_id = self.merge_to_id[best_pair]
            merged: List[int] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
            if len(tokens) < 2:
                break
        return tokens

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        ids.extend(self._encode_bytes(text.encode("utf-8", errors="replace")))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def _decode_id(self, idx: int, out: bytearray) -> None:
        if 0 <= idx <= 255:
            out.append(idx)
        elif idx in self.id_to_merge:
            a, b = self.id_to_merge[idx]
            self._decode_id(a, out)
            self._decode_id(b, out)
        # Specials are skipped by default.

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        out = bytearray()
        chunks: List[str] = []
        for idx in ids:
            idx = int(idx)
            if idx in (self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id):
                if not skip_special:
                    if out:
                        chunks.append(out.decode("utf-8", errors="replace"))
                        out.clear()
                    chunks.append(f"<|{idx}|>")
                continue
            self._decode_id(idx, out)
        if out:
            chunks.append(out.decode("utf-8", errors="replace"))
        return "".join(chunks)

    def save(self, path: str | Path) -> None:
        data = {
            "type": "byte_bpe",
            "merges": [[int(a), int(b)] for a, b in self.merges],
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        data = json.loads(Path(path).read_text())
        if data.get("type") != "byte_bpe":
            raise ValueError(f"not a byte-level BPE tokenizer file: {path}")
        return cls(
            merges=[tuple(pair) for pair in data["merges"]],
            pad_token_id=data.get("pad_token_id", 256),
            bos_token_id=data.get("bos_token_id", 257),
            eos_token_id=data.get("eos_token_id", 258),
            unk_token_id=data.get("unk_token_id", 259),
        )

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 1024,
        max_texts: int | None = None,
        min_pair_count: int = 2,
    ) -> "BytePairTokenizer":
        """Train byte-level BPE merges from an iterable of texts.

        The trainer stores a corpus as token sequences while learning. For full
        TinyStories, use ``scripts/train_bpe.py --max-texts`` or stream a sampled
        shard. The learned tokenizer can still encode the whole corpus afterward.
        """
        if vocab_size < 260:
            raise ValueError("vocab_size must be at least 260")
        corpus: List[List[int]] = []
        for i, text in enumerate(texts):
            if max_texts is not None and i >= max_texts:
                break
            ids = list(text.encode("utf-8", errors="replace"))
            if ids:
                corpus.append(ids)
        merges: List[Pair] = []
        next_id = 260
        while next_id < vocab_size:
            pair_counts: Counter[Pair] = Counter()
            for ids in corpus:
                pair_counts.update(zip(ids, ids[1:]))
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_count:
                break
            merges.append(best_pair)
            # Apply the merge to every stored sequence.
            for row_idx, ids in enumerate(corpus):
                if len(ids) < 2:
                    continue
                merged: List[int] = []
                j = 0
                while j < len(ids):
                    if j + 1 < len(ids) and ids[j] == best_pair[0] and ids[j + 1] == best_pair[1]:
                        merged.append(next_id)
                        j += 2
                    else:
                        merged.append(ids[j])
                        j += 1
                corpus[row_idx] = merged
            next_id += 1
        return cls(merges=merges)


def load_tokenizer(path_or_kind: str | Path):
    """Load tokenizer from JSON or construct by name: ``byte``."""
    from .byte_tokenizer import ByteTokenizer

    if str(path_or_kind) == "byte":
        return ByteTokenizer()
    path = Path(path_or_kind)
    data = json.loads(path.read_text())
    if data.get("type") == "byte":
        return ByteTokenizer.load(path)
    if data.get("type") == "byte_bpe":
        return BytePairTokenizer.load(path)
    raise ValueError(f"unknown tokenizer type in {path}")
