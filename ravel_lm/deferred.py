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

# When True, weight grads are computed inline via HIP GEMM straight into the
# static ``.grad`` buffers (graph-capture friendly). When False, they are
# stashed and batched by ``flush_deferred`` (lower launch count in eager mode).
INLINE_DW = False


def _ensure_grad(w: Tensor) -> Tensor:
    if w.grad is None:
        w.grad = torch.zeros_like(w)
    return w.grad


def accum_dw(weight, x2: Tensor, gy2: Tensor) -> None:
    """Accumulate dW = gy2^T @ x2 into weight.grad (single weight or a list of
    row-stacked weights sharing input x2). Inline HIP path or deferred stash."""
    if not INLINE_DW:
        _STASH.append((weight, x2, gy2))
        return
    from .hip_gemm import gemm_full

    M, K = x2.shape
    if isinstance(weight, (list, tuple)):
        row = 0
        for w in weight:
            n = w.shape[0]
            if w.requires_grad:
                g = _ensure_grad(w)
                gcol = gy2[:, row:row + n].contiguous()
                # dW[n,K] = gcol[M,n]^T @ x2[M,K]
                gemm_full(gcol, x2, n, K, M,
                          gcol.stride(1), gcol.stride(0), x2.stride(0), x2.stride(1),
                          g, accumulate=True)
            row += n
    else:
        g = _ensure_grad(weight)
        gemm_full(gy2, x2, weight.shape[0], K, M,
                  gy2.stride(1), gy2.stride(0), x2.stride(0), x2.stride(1),
                  g, accumulate=True)


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
