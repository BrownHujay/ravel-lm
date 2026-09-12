"""Fused FP32 seven-tap causal convolution and SwiGLU gating on MPS."""
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_name("metal").joinpath("local_gate.metal").read_text()
    return torch.mps.compile_shader(source)


@torch.library.custom_op("ravel::local_gate", mutates_args=())
def _forward(uv: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, DD = uv.shape
    D = DD // 2
    out = uv.new_empty(B, T, D)
    conv = torch.empty_like(out)
    if D % 4 == 0:
        _library().local_gate_forward_v4(
            uv.contiguous(), weight.contiguous(), bias.contiguous(), out, conv,
            T, D, threads=B * T * (D // 4), group_size=256,
        )
    else:
        _library().local_gate_forward(
            uv.contiguous(), weight.contiguous(), bias.contiguous(), out, conv,
            T, D, 7, threads=B * T * D, group_size=256,
        )
    return out, conv


@_forward.register_fake
def _fake_forward(uv, weight, bias):
    shape = (*uv.shape[:-1], uv.shape[-1] // 2)
    return uv.new_empty(shape), uv.new_empty(shape)


@torch.library.custom_op("ravel::local_gate_backward", mutates_args=())
def _backward_kernel(uv: torch.Tensor, weight: torch.Tensor, conv: torch.Tensor,
                     grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, DD = uv.shape
    D = DD // 2
    grad_uv = uv.new_empty(uv.shape)
    partial = uv.new_empty(B * ((T + 31) // 32), D, 8)
    _library().local_gate_backward7(
        uv.contiguous(), weight.contiguous(), conv, grad.contiguous(), grad_uv, partial,
        T, D, threads=[((D + 31) // 32) * 256, (T + 31) // 32, B],
        group_size=[256, 1, 1],
    )
    return grad_uv, partial


@_backward_kernel.register_fake
def _fake_backward(uv, weight, conv, grad):
    B, T, DD = uv.shape
    return uv.new_empty(uv.shape), uv.new_empty(B * ((T + 31) // 32), DD // 2, 8)


def _setup_context(ctx, inputs, output):
    uv, weight, bias = inputs
    _, conv = output
    ctx.save_for_backward(uv, weight, bias, conv)
    ctx.mark_non_differentiable(conv)


def _autograd_backward(ctx, grad, _):
    uv, weight, bias, conv = ctx.saved_tensors
    if torch.is_grad_enabled():
        # Retain differentiability for Hessian probes and other higher derivatives.
        u, v = uv.chunk(2, dim=-1)
        T = uv.shape[1]
        windows = F.pad(u, (0, 0, 6, 0)).unfold(1, 7, 1)
        z = (windows * weight).sum(dim=-1) + bias
        s = z.sigmoid()
        gz = grad * v * s * (1 + z * (1 - s))
        gv = grad * z * s
        future = F.pad(gz, (0, 0, 0, 6))
        gu = sum(future[:, 6 - k:6 - k + T] * weight[:, k] for k in range(7))
        return torch.cat((gu, gv), dim=-1), (windows * gz[..., None]).sum((0, 1)), gz.sum((0, 1))
    grad_uv, partial = _backward_kernel(uv, weight, conv, grad)
    total = partial.sum(dim=0)
    return grad_uv, total[:, :7], total[:, 7]


_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


def local_gate(uv: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Return silu(causal_depthwise_conv(uv[..., :D])) * uv[..., D:]."""
    if (uv.device.type != "mps" or uv.dtype != torch.float32
            or any(t.device != uv.device or t.dtype != uv.dtype for t in (weight, bias))):
        raise ValueError("local_gate requires FP32 MPS inputs")
    if (uv.ndim != 3 or any(s == 0 for s in uv.shape) or uv.shape[-1] % 2 != 0
            or weight.shape != (uv.shape[-1] // 2, 7) or bias.shape != weight.shape[:1]):
        raise ValueError("local_gate requires packed [B,T,2D] input and [D,7] weights")
    return _forward(uv, weight, bias)[0]
