"""Causal byte LM with a full-rate small RAVEL path and a slower-rate core."""
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelBlock, RMSNorm, SwiGLUFFN, count_parameters


class MultirateRavelLM(nn.Module):
    def __init__(self, cfg: RavelConfig, *, stride=8, byte_width=32,
                 core_width=384, core_layers=2, parameter_budget=3_255_840):
        super().__init__()
        if stride < 1 or cfg.dropout != 0:
            raise ValueError("prototype requires positive stride and zero dropout")
        self.cfg = deepcopy(cfg)
        self.stride = stride
        self.core_width = core_width
        self.core_layers = core_layers
        self.byte_width = byte_width
        self.parameter_budget = parameter_budget
        self.byte_embedding = nn.Embedding(cfg.vocab_size, byte_width)
        self.byte_position = nn.Embedding(cfg.block_size, byte_width)
        byte_cfg = deepcopy(cfg)
        byte_cfg.d_model = byte_width
        byte_cfg.payload_dim = 8
        byte_cfg.mlp_mult = 2.
        self.byte_block = RavelBlock(byte_cfg, 0)
        self.byte_head = SwiGLUFFN(byte_cfg)
        self.byte_norm = RMSNorm(byte_width)
        self.patch_projection = nn.Linear(stride * byte_width, core_width, bias=False)
        self.core_position = nn.Embedding((cfg.block_size + stride - 1) // stride, core_width)
        core_cfg = deepcopy(cfg)
        core_cfg.d_model = core_width
        core_cfg.n_layers = core_layers
        core_cfg.mlp_mult = 8 / core_width
        core_cfg.block_size = (cfg.block_size + stride - 1) // stride
        self.core = nn.ModuleList([RavelBlock(core_cfg, i) for i in range(core_layers)])
        self.core_norm = RMSNorm(core_width)
        self.bridge = nn.Linear(core_width, byte_width, bias=False)

        # Allocate remaining parameters to active FFN units, never dummy storage.
        unit_cost = 3 * core_width * 8
        units = round((parameter_budget - count_parameters(self)) / unit_cost)
        if units < 0:
            raise ValueError("fixed architecture already exceeds parameter budget")
        per_layer, remainder = divmod(units, core_layers)
        self.hidden_dims = []
        for i, block in enumerate(self.core):
            hidden = 8 * (1 + per_layer + int(i < remainder))
            ffn_cfg = deepcopy(core_cfg)
            ffn_cfg.mlp_mult = hidden / core_width
            block.ffn = SwiGLUFFN(ffn_cfg)
            self.hidden_dims.append(block.ffn.w3.in_features)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def num_parameters(self):
        return count_parameters(self)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if not 0 < T <= self.cfg.block_size:
            raise ValueError("invalid context length")
        S = self.stride
        G = (T + S - 1) // S
        pos = torch.arange(T, device=idx.device)
        byte = self.byte_embedding(idx) + self.byte_position(pos)[None]
        byte = self.byte_block(byte, idx)

        # Core group g sees only the completed group g-1. This delay is what
        # prevents any target inside a group from leaking through the core.
        previous = F.pad(byte, (0, 0, S, 0))[:, :G * S]
        core = self.patch_projection(previous.reshape(B, G, S * self.byte_width))
        core = core + self.core_position(torch.arange(G, device=idx.device))[None]
        previous_tokens = F.pad(idx, (S, 0), value=self.cfg.bos_token_id)
        core_ids = previous_tokens[:, S - 1:S - 1 + G * S:S]
        for block in self.core:
            core = block(core, core_ids)
        context = self.bridge(self.core_norm(core))
        context = context[:, :, None, :].expand(B, G, S, self.byte_width).reshape(B, G * S, self.byte_width)[:, :T]
        byte = self.byte_head(byte + context)
        logits = self.byte_norm(byte) @ self.byte_embedding.weight.T
        result = {"logits": logits}
        if targets is not None:
            result["loss"] = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1),
                                             ignore_index=self.cfg.pad_token_id)
        return result
