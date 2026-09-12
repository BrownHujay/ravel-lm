"""Experimental contiguous FP32 parameters; not used by the training runtime."""
from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call


class FlatTrainingModel(nn.Module):
    """One optimizer parameter; original named weights remain views of its storage.

    Functional splitting gives autograd a single concatenated gradient instead of
    one accumulation and optimizer launch sequence per named parameter.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not named:
            raise ValueError("model has no trainable parameters")
        first = named[0][1]
        if any(p.dtype != first.dtype or p.device != first.device for _, p in named):
            raise ValueError("flat training requires one parameter dtype and device")
        self.flat = nn.Parameter(torch.cat([p.detach().reshape(-1) for _, p in named]))
        self.names = tuple(name for name, _ in named)
        self.shapes = tuple(p.shape for _, p in named)
        self.sizes = tuple(p.numel() for _, p in named)
        # Keep the source module out of this wrapper's optimizer parameter list.
        object.__setattr__(self, "source", model)
        with torch.no_grad():
            for (_, p), view in zip(named, self.flat.split(self.sizes)):
                p.data = view.view(p.shape).detach()

    def forward(self, *args, **kwargs):
        views = self.flat.split(self.sizes)
        parameters = {
            name: value.view(shape)
            for name, shape, value in zip(self.names, self.shapes, views)
        }
        return functional_call(self.source, parameters, args, kwargs, tie_weights=True)
