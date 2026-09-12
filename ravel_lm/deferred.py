"""Deferred, batched weight gradients for same-shape Linear families.

On stacks where every GEMM call carries a large fixed cost (e.g. Windows
ROCm), the ~50 per-layer weight-gradient GEMMs dominate backward. Weight
gradients are order-independent, so backward can return only the input
gradient while stashing ``(weight, input, grad_out)``; ``flush_deferred``
then computes each shape-family's weight grads as one ``bmm`` and accumulates
into ``.grad``. Call ``flush_deferred()`` after ``loss.backward()`` and
before clipping/stepping.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor

_STASH: List[Tuple[nn.Parameter, Tensor, Tensor]] = []


class _DeferredLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return x.matmul(weight.t())

    @staticmethod
    def backward(ctx, gy):
        x, weight = ctx.saved_tensors
        _STASH.append((weight, x, gy))
        return gy.matmul(weight), None


def defer_linear_wgrads(module: nn.Module, names=("in_proj", "out_proj", "w12", "w3", "fuse")) -> int:
    """Route the named bias-free Linears through deferred weight grads."""
    count = 0
    for m in module.modules():
        for name in names:
            lin = getattr(m, name, None)
            if isinstance(lin, nn.Linear) and lin.bias is None:
                def make_fwd(l):
                    def fwd(x):
                        return _DeferredLinearFn.apply(x, l.weight)
                    return fwd
                lin.forward = make_fwd(lin)
                count += 1
    return count


@torch.no_grad()
def flush_deferred() -> None:
    if not _STASH:
        return
    groups = {}
    for weight, x, gy in _STASH:
        if isinstance(weight, (list, tuple)):
            key = ("packed",) + tuple(w.shape[0] for w in weight) + (weight[0].shape[1],)
        else:
            key = tuple(weight.shape)
        groups.setdefault(key, []).append((weight, x, gy))
    _STASH.clear()
    for key, items in groups.items():
        xs = torch.stack([x.reshape(-1, x.shape[-1]) for _, x, _ in items])
        gys = torch.stack([gy.reshape(-1, gy.shape[-1]) for _, _, gy in items])
        dws = torch.bmm(gys.transpose(1, 2), xs)  # [N, out_total, in]
        for (weight, _, _), dw in zip(items, dws):
            if isinstance(weight, (list, tuple)):
                row = 0
                for w in weight:
                    n = w.shape[0]
                    if w.requires_grad:
                        piece = dw[row : row + n]
                        if w.grad is None:
                            w.grad = piece.clone()
                        else:
                            w.grad.add_(piece)
                    row += n
            elif weight.grad is None:
                weight.grad = dw.clone()
            else:
                weight.grad.add_(dw)
