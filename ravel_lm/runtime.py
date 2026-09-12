from __future__ import annotations

from collections.abc import Iterable

import torch


@torch.no_grad()
def _global_l2_clip_(grads: tuple[torch.Tensor, ...], max_norm: float) -> torch.Tensor:
    norms = torch.stack([torch.linalg.vector_norm(grad, 2.0) for grad in grads])
    total_norm = torch.linalg.vector_norm(norms, 2.0)
    scale = torch.clamp(float(max_norm) / (total_norm + 1e-6), max=1.0)
    for grad in grads:
        grad.mul_(scale)
    return total_norm


_compiled_global_l2_clip = torch.compile(
    _global_l2_clip_,
    backend="inductor",
    fullgraph=True,
    dynamic=False,
)


def compiled_mps_clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Full-precision global L2 clipping fused for a fixed MPS model shape."""
    grads = tuple(param.grad for param in parameters if param.grad is not None)
    if not grads:
        return torch.tensor(0.0)
    if any(grad.device.type != "mps" for grad in grads):
        raise ValueError("compiled_mps_clip_grad_norm_ requires MPS gradients")
    return _compiled_global_l2_clip(grads, float(max_norm))


def adamw_fused_for(device: torch.device) -> bool:
    """Use the fused AdamW implementation on backends that provide it."""
    return device.type in {"cuda", "mps"}


class FlatAdamW:
    """AdamW + global-norm clipping over per-group flat parameter buffers.

    Parameters in each group are re-viewed into one contiguous buffer, and their
    ``.grad`` fields into a matching flat gradient buffer, so backward accumulates
    directly into flat storage. Clipping is then one reduction and one scale, and
    AdamW runs as a single fused kernel per group instead of one per tensor.

    The update is elementwise-identical to ``torch.optim.AdamW``; only the
    floating-point reduction order of the global grad norm differs from
    ``clip_grad_norm_``. Call ``zero_grad()`` (not ``optimizer.zero_grad``) each
    step so the flat gradient views survive.
    """

    def __init__(
        self,
        param_groups: list[dict],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        fused: bool = True,
    ) -> None:
        self._flat_params: list[torch.nn.Parameter] = []
        self._flat_grads: list[torch.Tensor] = []
        self._group_specs: list[dict] = []
        inner_groups = []
        for group in param_groups:
            params = [p for p in group["params"] if p.requires_grad]
            if not params:
                continue
            device = params[0].device
            dtype = params[0].dtype
            total = sum(p.numel() for p in params)
            flat = torch.empty(total, device=device, dtype=dtype)
            flat_grad = torch.zeros(total, device=device, dtype=dtype)
            offset = 0
            for p in params:
                n = p.numel()
                flat[offset : offset + n].copy_(p.detach().reshape(-1))
                p.data = flat[offset : offset + n].view(p.shape)
                p.grad = flat_grad[offset : offset + n].view(p.shape)
                offset += n
            flat_param = torch.nn.Parameter(flat)
            flat_param.grad = flat_grad
            self._flat_params.append(flat_param)
            self._flat_grads.append(flat_grad)
            self._group_specs.append({"params": params, "weight_decay": group.get("weight_decay", 0.0)})
            inner_groups.append(
                {"params": [flat_param], "weight_decay": group.get("weight_decay", 0.0)}
            )
        self._inner = torch.optim.AdamW(inner_groups, lr=lr, betas=betas, eps=eps, fused=fused)

    @property
    def param_groups(self):
        return self._inner.param_groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        del set_to_none  # flat grad views must stay alive; always zero in place
        for flat_grad in self._flat_grads:
            flat_grad.zero_()

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        # torch.dot is ~12x faster than aten::linalg_vector_norm on MPS for
        # large 1-D buffers and computes the same sum of squares.
        sq = self._flat_grads[0].dot(self._flat_grads[0])
        for g in self._flat_grads[1:]:
            sq = sq + g.dot(g)
        total_norm = sq.sqrt()
        scale = torch.clamp(float(max_norm) / (total_norm + 1e-6), max=1.0)
        for flat_grad in self._flat_grads:
            flat_grad.mul_(scale)
        return total_norm

    def step(self) -> None:
        self._inner.step()

    def state_dict(self) -> dict:
        return self._inner.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._inner.load_state_dict(state)
