from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class RavelConfig:
    """Configuration for the RAVEL language model.

    The defaults intentionally describe a small byte-level model so smoke tests are
    fast. JSON configs in ``configs/`` override these values for the named sizes.
    """

    # Token/model shape
    vocab_size: int = 260
    block_size: int = 256
    d_model: int = 64
    n_layers: int = 3
    mlp_mult: float = 2.0
    dropout: float = 0.0
    tie_weights: bool = True

    # Local non-attention mixer
    conv_kernel: int = 5

    # RAVEL exact event-memory layer
    address_space: int = 512
    n_literal_heads: int = 2  # token-id and rolling-bigram by default
    n_learned_heads: int = 1
    n_codebooks: int = 2
    codebook_size: int = 16
    payload_dim: int = 16
    last_k: int = 1
    memory_every: int = 1
    use_sum_read: bool = False

    # Optimization/runtime defaults used by scripts
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 1000
    warmup_steps: int = 100
    eval_interval: int = 100
    eval_batches: int = 20

    # Special tokens. ByteTokenizer uses 0..255 for raw bytes and starts specials at 256.
    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258
    unk_token_id: int = 259

    # Metadata only; useful when loading configs from scripts.
    name: str = "ravel"
    tokenizer: str = "byte"
    notes: str = ""

    def validate(self) -> None:
        if self.vocab_size <= max(self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id):
            raise ValueError("vocab_size must include the configured special token ids")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("d_model and n_layers must be positive")
        if self.conv_kernel < 1:
            raise ValueError("conv_kernel must be positive")
        if self.address_space <= 1:
            raise ValueError("address_space must be > 1")
        if self.n_literal_heads < 0 or self.n_learned_heads < 0:
            raise ValueError("head counts must be non-negative")
        if self.n_literal_heads + self.n_learned_heads <= 0:
            raise ValueError("at least one RAVEL memory head is required")
        if self.payload_dim <= 0:
            raise ValueError("payload_dim must be positive")
        if self.last_k <= 0:
            raise ValueError("last_k must be positive")
        if self.codebook_size <= 1 or self.n_codebooks <= 0:
            raise ValueError("codebook_size and n_codebooks must be valid")
        if self.codebook_size ** self.n_codebooks > 2**31:
            raise ValueError("product-code address space is too large for int64 composite keys")
        if self.address_space * self.block_size > 2**40:
            raise ValueError("address_space * block_size is too large for current composite-key implementation")

    @property
    def n_memory_heads(self) -> int:
        return self.n_literal_heads + self.n_learned_heads

    @property
    def mlp_hidden_dim(self) -> int:
        # Round to a multiple of 8 for tensor-core-friendly shapes on GPU.
        return max(8, int((self.d_model * self.mlp_mult + 7) // 8 * 8))

    @classmethod
    def from_json(cls, path: str | Path) -> "RavelConfig":
        data = json.loads(Path(path).read_text())
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def load_config(path_or_name: str | Path) -> RavelConfig:
    """Load a config from an explicit path or from the repository configs folder."""
    path = Path(path_or_name)
    if path.exists():
        return RavelConfig.from_json(path)

    root = Path(__file__).resolve().parents[1]
    candidates = list((root / "configs").rglob(f"{path_or_name}.json"))
    if not candidates:
        candidates = list((root / "configs").rglob(str(path_or_name)))
    if not candidates:
        raise FileNotFoundError(f"Could not find config: {path_or_name}")
    return RavelConfig.from_json(candidates[0])
