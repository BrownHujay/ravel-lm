from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RavelConfig
from .model import RMSNorm, SwiGLUFFN, count_parameters, sample_logits

Tensor = torch.Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: RavelConfig, n_heads: int = 4) -> None:
        super().__init__()
        if cfg.d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = cfg.d_model // self.n_heads
        self.scale = self.head_dim**-0.5
        self.norm = RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor, *, return_attn: bool = False):
        B, T, D = x.shape
        xn = self.norm(x)
        qkv = self.qkv(xn).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if not return_attn:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            y = y.transpose(1, 2).reshape(B, T, D)
            return x + self.dropout(self.out_proj(y))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        y = x + self.dropout(self.out_proj(y))
        if return_attn:
            return y, attn
        return y


class AttentionBlock(nn.Module):
    def __init__(self, cfg: RavelConfig, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg, n_heads=n_heads)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: Tensor, *, return_attn: bool = False):
        if return_attn:
            x, attn = self.attn(x, return_attn=True)
            x = self.ffn(x)
            return x, attn
        x = self.attn(x)
        return self.ffn(x)


@dataclass
class AttentionProbe:
    logits: Tensor
    attentions: list[Tensor]
    activation_rms: list[Tensor]


class AttentionLM(nn.Module):
    """Standard causal softmax-attention LM for controlled RAVEL comparisons."""

    def __init__(self, cfg: RavelConfig, n_heads: int = 4) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.n_heads = int(n_heads)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([AttentionBlock(cfg, n_heads=n_heads) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        return count_parameters(self)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> dict:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [B,T]")
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
            )
        return out

    @torch.no_grad()
    def probe(self, idx: Tensor) -> AttentionProbe:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [B,T]")
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
        attentions: list[Tensor] = []
        activation_rms: list[Tensor] = []
        for block in self.blocks:
            x, attn = block(x, return_attn=True)
            attentions.append(attn.detach())
            activation_rms.append(x.float().pow(2).mean(dim=-1).sqrt().detach())
        x = self.norm(x)
        logits = self.lm_head(x)
        return AttentionProbe(logits=logits, attentions=attentions, activation_rms=activation_rms)

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits = self(idx_cond)["logits"][:, -1, :]
            next_id = sample_logits(logits, temperature=temperature, top_k=top_k)
            idx = torch.cat([idx, next_id[:, None]], dim=1)
        return idx
