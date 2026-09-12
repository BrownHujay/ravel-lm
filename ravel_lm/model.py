from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RavelConfig
from .ravel_memory import RavelLayerCache, causal_last_k_lookup, causal_sum_lookup
from .triton_block import fast_linear, fast_linear_packed, gate_residual, rms_norm, silu_gate
from .triton_fusedblock import fused_ffn, fused_memory, fused_memory_ok, fused_mixer, fused_ok
from .triton_local import can_use_triton_local_gate, triton_local_gate
from .triton_memory import can_use_triton_fused_latest1, triton_fused_latest1_linear

if hasattr(torch.mps, "compile_shader"):
    from .mps_local import local_gate
else:
    local_gate = None

Tensor = torch.Tensor


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal conv with a matching one-token ``step`` path."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(channels, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,T,C]
        if self.kernel_size == 1:
            return x * self.weight[:, 0] + self.bias
        # Grouped conv2d is the same causal cross-correlation as conv1d while
        # retaining the faster complete training path on MPS.
        channels_first = x.transpose(1, 2).contiguous()
        padded = F.pad(channels_first, (self.kernel_size - 1, 0))
        y = F.conv2d(
            padded.unsqueeze(2),
            self.weight.view(self.channels, 1, 1, self.kernel_size),
            self.bias,
            groups=self.channels,
        )
        if y.device.type == "mps" and y.requires_grad:
            y.register_hook(lambda grad: grad.contiguous())
        return y.squeeze(2).transpose(1, 2).contiguous()

    def init_state(self, batch_size: int, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
        if self.kernel_size == 1:
            return torch.empty(batch_size, 0, self.channels, device=device, dtype=dtype)
        return torch.zeros(batch_size, self.kernel_size - 1, self.channels, device=device, dtype=dtype)

    def step(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [B,C], state: [B,K-1,C]
        if self.kernel_size == 1:
            return x * self.weight[:, 0] + self.bias, state
        window = torch.cat([state, x.unsqueeze(1)], dim=1)  # [B,K,C]
        y = (window * self.weight.t().unsqueeze(0)).sum(dim=1) + self.bias
        new_state = window[:, 1:, :].contiguous()
        return y, new_state


class LocalMixer(nn.Module):
    """Non-attention local mixer: RMSNorm -> gated depthwise causal conv -> projection."""

    def __init__(self, cfg: RavelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm = RMSNorm(d)
        self.in_proj = nn.Linear(d, 2 * d, bias=False)
        self.conv = CausalDepthwiseConv1d(d, cfg.conv_kernel)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.use_mps_local = True

    def forward(self, x: Tensor) -> Tensor:
        if (x.is_cuda and x.dtype == torch.float32 and self.dropout.p == 0.0
                and fused_ok(self.conv.kernel_size, x.shape[-1])):
            return fused_mixer(x, self.norm.weight, self.in_proj.weight,
                               self.conv.weight, self.conv.bias,
                               self.out_proj.weight, self.norm.eps)
        uv = fast_linear(self.norm(x), self.in_proj.weight)
        if self.conv.kernel_size == 7 and can_use_triton_local_gate(uv, self.conv.weight):
            y = triton_local_gate(uv, self.conv.weight, self.conv.bias)
        elif (self.use_mps_local and local_gate is not None and uv.device.type == "mps"
                and uv.dtype == torch.float32 and self.conv.kernel_size == 7):
            y = local_gate(uv, self.conv.weight, self.conv.bias)
        else:
            u, v = uv.chunk(2, dim=-1)
            u = self.conv(u)
            y = F.silu(u) * v
        return x + self.dropout(fast_linear(y, self.out_proj.weight))

    def init_state(self, batch_size: int, *, device: torch.device | str, dtype: torch.dtype) -> Tensor:
        return self.conv.init_state(batch_size, device=device, dtype=dtype)

    def step(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        u, v = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u, new_state = self.conv.step(u, state)
        y = F.silu(u) * v
        return x + self.out_proj(y), new_state


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: RavelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        h = cfg.mlp_hidden_dim
        self.norm = RMSNorm(d)
        self.w12 = nn.Linear(d, 2 * h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cuda and x.dtype == torch.float32 and self.dropout.p == 0.0:
            return fused_ffn(x, self.norm.weight, self.w12.weight,
                             self.w3.weight, self.norm.eps)
        y = silu_gate(fast_linear(self.norm(x), self.w12.weight))
        return x + self.dropout(fast_linear(y, self.w3.weight))

    def step(self, x: Tensor) -> Tensor:
        a, b = self.w12(self.norm(x)).chunk(2, dim=-1)
        y = F.silu(a) * b
        return x + self.w3(y)


class ProductCodeAddressor(nn.Module):
    """Hard product-code address predictor.

    This is deliberately not dot-product attention: it emits discrete address codes.
    The selected address is non-smooth, but selected-address probabilities and the
    payload/fusion path remain trainable. Literal heads provide guaranteed exact
    addressability for token IDs when ``address_space >= vocab_size``.
    """

    def __init__(self, d_model: int, n_heads: int, n_codebooks: int, codebook_size: int, address_space: int) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.address_space = int(address_space)
        if self.n_heads > 0:
            self.proj = nn.Linear(d_model, n_heads * n_codebooks * codebook_size, bias=False)
            self.proj.weight.requires_grad_(False)
        else:
            self.proj = None
        multipliers = []
        m = 1
        for _ in range(n_codebooks):
            multipliers.append(m)
            m *= codebook_size
        self.register_buffer("multipliers", torch.tensor(multipliers, dtype=torch.long), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,T,D] or [B,D]. Return [B,T,H] or [B,H].
        if self.n_heads == 0:
            shape = x.shape[:-1] + (0,)
            return torch.empty(*shape, device=x.device, dtype=torch.long)
        logits = self.proj(x)
        out_shape = x.shape[:-1] + (self.n_heads, self.n_codebooks, self.codebook_size)
        logits = logits.reshape(*out_shape)
        codes = logits.argmax(dim=-1).long()
        addr = (codes * self.multipliers.view(*([1] * (codes.ndim - 1)), self.n_codebooks)).sum(dim=-1)
        return addr.remainder(self.address_space)

    def logits(self, x: Tensor) -> Tensor:
        if self.proj is None:
            return torch.empty(*x.shape[:-1], 0, self.n_codebooks, self.codebook_size, device=x.device, dtype=x.dtype)
        return self.proj(x).reshape(*x.shape[:-1], self.n_heads, self.n_codebooks, self.codebook_size)


def literal_addresses_from_tokens(
    token_ids: Tensor,
    *,
    address_space: int,
    n_heads: int,
    bos_token_id: int,
) -> Tensor:
    """Deterministic literal address families from token IDs.

    Heads, in order:
      0. current token ID modulo address space
      1. rolling bigram hash(prev_token, token)
      2. rolling trigram hash(prev2, prev1, token)
      3+. token/position mixed hashes
    """
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [B,T]")
    B, T = token_ids.shape
    device = token_ids.device
    toks = token_ids.long()
    heads = []
    if n_heads >= 1:
        heads.append(toks.remainder(address_space))
    if n_heads >= 2:
        prev = torch.cat([torch.full((B, 1), bos_token_id, device=device, dtype=torch.long), toks[:, :-1]], dim=1)
        heads.append((prev * 257 + toks * 131 + 17).remainder(address_space))
    if n_heads >= 3:
        prev1 = torch.cat([torch.full((B, 1), bos_token_id, device=device, dtype=torch.long), toks[:, :-1]], dim=1)
        prev2 = torch.cat([torch.full((B, 2), bos_token_id, device=device, dtype=torch.long), toks[:, :-2]], dim=1)
        heads.append((prev2 * 65537 + prev1 * 257 + toks * 131 + 31).remainder(address_space))
    for j in range(3, n_heads):
        pos = torch.arange(T, device=device, dtype=torch.long).view(1, T).expand(B, T)
        # Keep multipliers modest to avoid int64 overflow for long contexts.
        heads.append((toks * (1009 + 2 * j) + pos * (9176 + j) + j * 97).remainder(address_space))
    if not heads:
        return torch.empty(B, T, 0, device=device, dtype=torch.long)
    return torch.stack(heads, dim=-1)


def literal_addresses_step(
    token_ids: Tensor,
    token_history: Tensor,
    *,
    address_space: int,
    n_heads: int,
    position: int,
    bos_token_id: int,
) -> Tensor:
    """Step version of ``literal_addresses_from_tokens``.

    token_ids: [B]
    token_history: [B,2] storing previous two token IDs as [prev2, prev1]
    """
    B = token_ids.shape[0]
    toks = token_ids.long()
    heads = []
    if n_heads >= 1:
        heads.append(toks.remainder(address_space))
    prev1 = token_history[:, -1].long() if token_history.numel() else torch.full_like(toks, bos_token_id)
    prev2 = token_history[:, -2].long() if token_history.shape[1] >= 2 else torch.full_like(toks, bos_token_id)
    if n_heads >= 2:
        heads.append((prev1 * 257 + toks * 131 + 17).remainder(address_space))
    if n_heads >= 3:
        heads.append((prev2 * 65537 + prev1 * 257 + toks * 131 + 31).remainder(address_space))
    for j in range(3, n_heads):
        heads.append((toks * (1009 + 2 * j) + int(position) * (9176 + j) + j * 97).remainder(address_space))
    if not heads:
        return torch.empty(B, 0, device=token_ids.device, dtype=torch.long)
    return torch.stack(heads, dim=-1)


class RavelMemoryLayer(nn.Module):
    """Exact addressed event-memory layer.

    During training/prefill it uses ``causal_last_k_lookup`` over the whole sequence.
    During decode it uses ``RavelLayerCache`` with gather/scatter latest semantics.
    """

    def __init__(self, cfg: RavelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        C = cfg.n_memory_heads
        self.norm = RMSNorm(d)
        self.payload = nn.Linear(d, C * cfg.payload_dim, bias=False)
        self.write_addr = ProductCodeAddressor(d, cfg.n_learned_heads, cfg.n_codebooks, cfg.codebook_size, cfg.address_space)
        self.read_addr = ProductCodeAddressor(d, cfg.n_learned_heads, cfg.n_codebooks, cfg.codebook_size, cfg.address_space)
        read_slots = cfg.last_k + (1 if cfg.use_sum_read else 0)
        self.fuse = nn.Linear(C * read_slots * cfg.payload_dim, d, bias=False)
        self.gate = nn.Linear(d, d, bias=True)
        self.dropout = nn.Dropout(cfg.dropout)

    def _addresses(self, x: Tensor, token_ids: Tensor) -> Tuple[Tensor, Tensor]:
        literal = literal_addresses_from_tokens(
            token_ids,
            address_space=self.cfg.address_space,
            n_heads=self.cfg.n_literal_heads,
            bos_token_id=self.cfg.bos_token_id,
        )
        # Address selection is hard argmax routing, so gradients do not flow
        # through these projections. Avoid building dead autograd graphs.
        with torch.no_grad():
            learned_write = self.write_addr(x)
            learned_read = self.read_addr(x)
        write = torch.cat([literal, learned_write], dim=-1)
        read = torch.cat([literal, learned_read], dim=-1)
        return write, read

    def _decode_codes(self, addressor: ProductCodeAddressor, logits: Tensor) -> Tensor:
        shape = logits.shape[:-1] + (addressor.n_heads, addressor.n_codebooks, addressor.codebook_size)
        codes = logits.reshape(*shape).argmax(dim=-1).long()
        mult = addressor.multipliers.view(*([1] * (codes.ndim - 1)), addressor.n_codebooks)
        return (codes * mult).sum(dim=-1).remainder(addressor.address_space)

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        write_mask: Optional[Tensor] = None,
        literal_addr: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, D = x.shape
        xn = self.norm(x)
        C = self.cfg.n_memory_heads
        # One packed GEMM for every projection fed by xn: payload, gate, and the
        # frozen product-code addressors. MPS GEMMs at these widths are launch-
        # and tile-bound, so fewer, wider matmuls are markedly faster. The packed
        # matmul computes the same dot products as the separate Linears.
        n_pay = self.payload.weight.shape[0]
        n_gate = self.gate.weight.shape[0]
        weights = [self.payload.weight, self.gate.weight]
        if self.write_addr.proj is not None:
            weights.append(self.write_addr.proj.weight)
            weights.append(self.read_addr.proj.weight)
        packed = fast_linear_packed(xn, weights)
        payload = packed[..., :n_pay].reshape(B, T, C, self.cfg.payload_dim)
        gate_lin = packed[..., n_pay : n_pay + n_gate] + self.gate.bias

        if literal_addr is None:
            literal_addr = literal_addresses_from_tokens(
                token_ids,
                address_space=self.cfg.address_space,
                n_heads=self.cfg.n_literal_heads,
                bos_token_id=self.cfg.bos_token_id,
            )
        if self.write_addr.proj is not None:
            n_addr = self.write_addr.proj.weight.shape[0]
            addr_logits = packed[..., n_pay + n_gate :].detach()
            learned_write = self._decode_codes(self.write_addr, addr_logits[..., :n_addr])
            learned_read = self._decode_codes(self.read_addr, addr_logits[..., n_addr:])
            write_addr = torch.cat([literal_addr, learned_write], dim=-1)
            read_addr = torch.cat([literal_addr, learned_read], dim=-1)
        else:
            write_addr = literal_addr
            read_addr = literal_addr
        if can_use_triton_fused_latest1(
            write_addr,
            payload,
            read_addr,
            self.fuse.weight,
            write_mask=write_mask,
            use_sum_read=self.cfg.use_sum_read,
            k=self.cfg.last_k,
        ):
            fused = triton_fused_latest1_linear(
                write_addr,
                payload,
                read_addr,
                self.fuse.weight,
                address_space=self.cfg.address_space,
            )
        else:
            last_values, _ = causal_last_k_lookup(
                write_addr,
                payload,
                read_addr,
                address_space=self.cfg.address_space,
                k=self.cfg.last_k,
                write_mask=write_mask,
            )
            pieces = [last_values.reshape(B, T, -1)]
            if self.cfg.use_sum_read:
                summed, _ = causal_sum_lookup(
                    write_addr,
                    payload,
                    read_addr,
                    address_space=self.cfg.address_space,
                    write_mask=write_mask,
                )
                pieces.append(summed.reshape(B, T, -1))
            read_flat = torch.cat(pieces, dim=-1)
            fused = fast_linear(read_flat, self.fuse.weight)
        if self.dropout.p == 0.0:
            return gate_residual(x, gate_lin, fused)
        return x + self.dropout(torch.sigmoid(gate_lin) * fused)

    def init_cache(self, batch_size: int, *, device: torch.device | str, dtype: torch.dtype) -> RavelLayerCache:
        return RavelLayerCache.empty(
            batch_size,
            self.cfg.n_memory_heads,
            self.cfg.address_space,
            self.cfg.payload_dim,
            device=device,
            dtype=dtype,
        )

    def step(self, x: Tensor, token_ids: Tensor, token_history: Tensor, position: int, cache: RavelLayerCache) -> Tensor:
        if self.cfg.last_k != 1 or self.cfg.use_sum_read:
            raise NotImplementedError("incremental cache currently supports last_k=1 and use_sum_read=False")
        B, D = x.shape
        xn = self.norm(x)
        literal = literal_addresses_step(
            token_ids,
            token_history,
            address_space=self.cfg.address_space,
            n_heads=self.cfg.n_literal_heads,
            position=position,
            bos_token_id=self.cfg.bos_token_id,
        )
        with torch.no_grad():
            learned_write = self.write_addr(xn)
            learned_read = self.read_addr(xn)
        write_addr = torch.cat([literal, learned_write], dim=-1)
        read_addr = torch.cat([literal, learned_read], dim=-1)
        C = write_addr.shape[-1]
        payload = self.payload(xn).reshape(B, C, self.cfg.payload_dim)
        values, _ = cache.read(read_addr)
        fused = self.fuse(values.reshape(B, -1))
        out = x + torch.sigmoid(self.gate(xn)) * fused
        cache.write(write_addr, payload)
        return out


@dataclass
class RavelBlockCache:
    conv_state: Tensor
    memory_cache: Optional[RavelLayerCache]


class RavelBlock(nn.Module):
    def __init__(self, cfg: RavelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.local = LocalMixer(cfg)
        self.memory = RavelMemoryLayer(cfg) if (layer_idx % cfg.memory_every == 0) else None
        self.ffn = SwiGLUFFN(cfg)

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor,
        write_mask: Optional[Tensor] = None,
        literal_addr: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.local(x)
        if self.memory is not None:
            x = self.memory(x, token_ids, write_mask=write_mask, literal_addr=literal_addr)
        x = self.ffn(x)
        return x

    def init_cache(self, batch_size: int, *, device: torch.device | str, dtype: torch.dtype) -> RavelBlockCache:
        conv_state = self.local.init_state(batch_size, device=device, dtype=dtype)
        memory_cache = self.memory.init_cache(batch_size, device=device, dtype=dtype) if self.memory is not None else None
        return RavelBlockCache(conv_state=conv_state, memory_cache=memory_cache)

    def step(self, x: Tensor, token_ids: Tensor, token_history: Tensor, position: int, cache: RavelBlockCache) -> Tensor:
        x, cache.conv_state = self.local.step(x, cache.conv_state)
        if self.memory is not None:
            assert cache.memory_cache is not None
            x = self.memory.step(x, token_ids, token_history, position, cache.memory_cache)
        x = self.ffn.step(x)
        return x


@dataclass
class RavelGenerationCache:
    block_caches: List[RavelBlockCache]
    token_history: Tensor
    position: int = 0


class RavelLM(nn.Module):
    """Tiny-to-small causal LM using RAVEL instead of attention."""

    def __init__(self, cfg: RavelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([RavelBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        return count_parameters(self)

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        *,
        write_mask: Optional[Tensor] = None,
    ) -> dict:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [B,T]")
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        x = self.drop(x)
        # Literal addresses depend only on token ids; hash them once for all layers.
        literal_addr = None
        if any(block.memory is not None for block in self.blocks):
            literal_addr = literal_addresses_from_tokens(
                idx,
                address_space=self.cfg.address_space,
                n_heads=self.cfg.n_literal_heads,
                bos_token_id=self.cfg.bos_token_id,
            )
        for block in self.blocks:
            x = block(x, idx, write_mask=write_mask, literal_addr=literal_addr)
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
            )
            out["loss"] = loss
        return out

    def init_cache(self, batch_size: int, *, device: Optional[torch.device | str] = None, dtype: Optional[torch.dtype] = None) -> RavelGenerationCache:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        block_caches = [block.init_cache(batch_size, device=device, dtype=dtype) for block in self.blocks]
        token_history = torch.full((batch_size, 2), self.cfg.bos_token_id, device=device, dtype=torch.long)
        return RavelGenerationCache(block_caches=block_caches, token_history=token_history, position=0)

    @torch.no_grad()
    def forward_step(self, token_ids: Tensor, cache: RavelGenerationCache) -> Tensor:
        """One-token decode path with exact latest-address cache."""
        if token_ids.ndim != 1:
            raise ValueError("token_ids must have shape [B]")
        if cache.position >= self.cfg.block_size:
            # Keep positional embedding bounded for simple demos; real serving would use
            # rotary/ALiBi-style unbounded positions or paged position embeddings.
            pos_id = cache.position % self.cfg.block_size
        else:
            pos_id = cache.position
        pos = torch.full_like(token_ids, pos_id)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        for block, block_cache in zip(self.blocks, cache.block_caches):
            x = block.step(x, token_ids, cache.token_history, cache.position, block_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        # Update token history only after every layer has read causal memory.
        cache.token_history = torch.cat([cache.token_history[:, 1:], token_ids.view(-1, 1)], dim=1)
        cache.position += 1
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tensor:
        self.eval()
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.block_size :]
                logits = self(idx_cond)["logits"][:, -1, :]
                next_id = sample_logits(logits, temperature=temperature, top_k=top_k)
                idx = torch.cat([idx, next_id[:, None]], dim=1)
            return idx

        B = idx.shape[0]
        cache = self.init_cache(B, device=idx.device, dtype=next(self.parameters()).dtype)
        logits = None
        # Prefill through the exact incremental memory path.
        for t in range(idx.shape[1]):
            logits = self.forward_step(idx[:, t], cache)
        assert logits is not None
        for _ in range(max_new_tokens):
            next_id = sample_logits(logits, temperature=temperature, top_k=top_k)
            idx = torch.cat([idx, next_id[:, None]], dim=1)
            logits = self.forward_step(next_id, cache)
        return idx


def sample_logits(logits: Tensor, *, temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        cutoff = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
