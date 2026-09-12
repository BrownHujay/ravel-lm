"""Triton block kernels for CUDA/ROCm: fast linears and fused glue.

On the Windows ROCm SDK, rocBLAS carries a 50-200us fixed cost per small GEMM
and every eltwise aten op costs ~20-25us. These kernels replace both:

- ``fast_linear``: bias-free Linear whose forward and input-grad run as Triton
  GEMMs (1.5-2.5x rocBLAS at RAVEL shapes on RX 9070 XT); weight grads are
  deferred to the batched ``ravel_lm.deferred`` path.
- ``rms_norm``: fused forward/backward RMSNorm.
- ``silu_gate``: fused silu(a) * b forward/backward (FFN gate).
- ``gate_residual``: fused x + sigmoid(g) * f forward/backward (memory epilogue).

All backward formulas are hand-derived and parity-tested against the aten
reference in ``tests/test_triton_block.py``.
"""
from __future__ import annotations

import torch

from .deferred import _STASH

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

Tensor = torch.Tensor


if _HAS_TRITON:

    @triton.jit
    def _gemm_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BM)
        grid_n = tl.cdiv(N, BN)
        width = GROUP * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP, GROUP)
        pid_m = group_id * GROUP + (pid % group_size)
        pid_n = (pid % width) // group_size
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                        mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
            b = tl.load(b_ptr + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                        mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn,
                 acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _rmsnorm_fwd_kernel(x_ptr, g_ptr, y_ptr, rstd_ptr, K, eps,
                            BK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BK)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x) / K + eps)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + row * K + offs, x * rstd * g, mask=mask)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _rmsnorm_bwd_kernel(x_ptr, g_ptr, gy_ptr, rstd_ptr, dx_ptr, dg_ptr, M, K,
                            ROWS: tl.constexpr, BK: tl.constexpr):
        # Each program owns ROWS rows: local dg accumulation, one atomic per k.
        row0 = tl.program_id(0) * ROWS
        offs = tl.arange(0, BK)
        mask = offs < K
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        dg_acc = tl.zeros((BK,), dtype=tl.float32)
        for i in tl.static_range(ROWS):
            row = row0 + i
            rmask = mask & (row < M)
            x = tl.load(x_ptr + row * K + offs, mask=rmask, other=0.0)
            gy = tl.load(gy_ptr + row * K + offs, mask=rmask, other=0.0)
            rstd = tl.load(rstd_ptr + tl.minimum(row, M - 1))
            gyg = gy * g
            dot = tl.sum(gyg * x)
            dx = rstd * gyg - (rstd * rstd * rstd / K) * dot * x
            tl.store(dx_ptr + row * K + offs, dx, mask=rmask)
            dg_acc += gy * x * rstd
        tl.atomic_add(dg_ptr + offs, dg_acc, mask=mask)

    @triton.jit
    def _silu_gate_fwd_kernel(h_ptr, out_ptr, M, H, BH: tl.constexpr):
        row = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs = pid_h * BH + tl.arange(0, BH)
        mask = offs < H
        a = tl.load(h_ptr + row * 2 * H + offs, mask=mask, other=0.0)
        b = tl.load(h_ptr + row * 2 * H + H + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row * H + offs, (a / (1.0 + tl.exp(-a))) * b, mask=mask)

    @triton.jit
    def _silu_gate_bwd_kernel(h_ptr, gy_ptr, gh_ptr, M, H, BH: tl.constexpr):
        row = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs = pid_h * BH + tl.arange(0, BH)
        mask = offs < H
        a = tl.load(h_ptr + row * 2 * H + offs, mask=mask, other=0.0)
        b = tl.load(h_ptr + row * 2 * H + H + offs, mask=mask, other=0.0)
        gy = tl.load(gy_ptr + row * H + offs, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-a))
        silu = a * sig
        tl.store(gh_ptr + row * 2 * H + offs,
                 gy * b * sig * (1.0 + a * (1.0 - sig)), mask=mask)
        tl.store(gh_ptr + row * 2 * H + H + offs, gy * silu, mask=mask)

    @triton.jit
    def _gate_res_fwd_kernel(x_ptr, gl_ptr, f_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        gl = tl.load(gl_ptr + offs, mask=mask, other=0.0)
        f = tl.load(f_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x + f / (1.0 + tl.exp(-gl)) * 1.0, mask=mask)

    @triton.jit
    def _gate_res_bwd_kernel(gl_ptr, f_ptr, gy_ptr, dgl_ptr, df_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        gl = tl.load(gl_ptr + offs, mask=mask, other=0.0)
        f = tl.load(f_ptr + offs, mask=mask, other=0.0)
        gy = tl.load(gy_ptr + offs, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-gl))
        tl.store(dgl_ptr + offs, gy * f * sig * (1.0 - sig), mask=mask)
        tl.store(df_ptr + offs, gy * sig, mask=mask)


# Per-(K, N) tile configs from the RX 9070 XT sweep; default is the broad winner.
_CFG_DEFAULT = (64, 64, 16, 4, 4)
_CFG_TABLE = {
    (160, 320): (64, 64, 32, 4, 4),
    (160, 640): (64, 64, 16, 4, 4),
    (320, 160): (64, 64, 16, 4, 4),
    (160, 160): (64, 64, 16, 4, 4),
    (96, 160): (64, 64, 16, 4, 4),
    (640, 160): (64, 64, 16, 4, 4),
    (260, 160): (32, 128, 32, 4, 4),
}


def _tgemm(a: Tensor, b_ptr_tensor: Tensor, M, N, K, sbk, sbn, out=None) -> Tensor:
    c = out if out is not None else a.new_empty(M, N)
    BM, BN, BK, G, W = _CFG_TABLE.get((K, N), _CFG_DEFAULT)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_kernel[grid](
        a, b_ptr_tensor, c, M, N, K,
        a.stride(0), a.stride(1), sbk, sbn, c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, GROUP=G, num_warps=W,
    )
    return c


def triton_ok() -> bool:
    return _HAS_TRITON and torch.cuda.is_available()


class _FastLinearFn(torch.autograd.Function):
    """y = x @ W^T with Triton GEMMs; dW deferred to the batched flush."""

    @staticmethod
    def forward(ctx, x, weight):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, K = x2.shape
        N = weight.shape[0]
        y = _tgemm(x2, weight, M, N, K, weight.stride(1), weight.stride(0))
        ctx.save_for_backward(x2, weight)
        ctx.in_shape = shape
        return y.view(*shape[:-1], N)

    @staticmethod
    def backward(ctx, gy):
        x2, weight = ctx.saved_tensors
        gy2 = gy.reshape(-1, gy.shape[-1]).contiguous()
        M, N = gy2.shape
        K = weight.shape[1]
        dx = _tgemm(gy2, weight, M, K, N, weight.stride(0), weight.stride(1))
        _STASH.append((weight, x2, gy2))
        return dx.view(ctx.in_shape), None


class _FastLinearFullFn(torch.autograd.Function):
    """Same as _FastLinearFn but computes dW inline (for non-leaf weights,
    e.g. torch.cat of parameters, where deferred accumulation cannot apply)."""

    @staticmethod
    def forward(ctx, x, weight):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, K = x2.shape
        N = weight.shape[0]
        y = _tgemm(x2, weight, M, N, K, weight.stride(1), weight.stride(0))
        ctx.save_for_backward(x2, weight)
        ctx.in_shape = shape
        return y.view(*shape[:-1], N)

    @staticmethod
    def backward(ctx, gy):
        x2, weight = ctx.saved_tensors
        gy2 = gy.reshape(-1, gy.shape[-1]).contiguous()
        M, N = gy2.shape
        K = weight.shape[1]
        dx = _tgemm(gy2, weight, M, K, N, weight.stride(0), weight.stride(1))
        gyt = gy2.t()  # [N, M] strided view; kernel takes explicit strides
        dw = weight.new_empty(N, K)
        BM, BN, BK, G, W = _CFG_DEFAULT
        grid = (triton.cdiv(N, BM) * triton.cdiv(K, BN),)
        _gemm_kernel[grid](
            gyt, x2, dw, N, K, M,
            gyt.stride(0), gyt.stride(1), x2.stride(0), x2.stride(1),
            dw.stride(0), dw.stride(1),
            BM=BM, BN=BN, BK=BK, GROUP=G, num_warps=W,
        )
        return dx.view(ctx.in_shape), dw


class _FastLinearPackedFn(torch.autograd.Function):
    """One GEMM over vertically-stacked weights; dW deferred and row-split."""

    @staticmethod
    def forward(ctx, x, w_cat, *weights):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, K = x2.shape
        N = w_cat.shape[0]
        y = _tgemm(x2, w_cat, M, N, K, w_cat.stride(1), w_cat.stride(0))
        ctx.save_for_backward(x2, w_cat)
        ctx.weights = weights
        ctx.in_shape = shape
        return y.view(*shape[:-1], N)

    @staticmethod
    def backward(ctx, gy):
        x2, w_cat = ctx.saved_tensors
        gy2 = gy.reshape(-1, gy.shape[-1]).contiguous()
        M, N = gy2.shape
        K = w_cat.shape[1]
        dx = _tgemm(gy2, w_cat, M, K, N, w_cat.stride(0), w_cat.stride(1))
        _STASH.append((list(ctx.weights), x2, gy2))
        return (dx.view(ctx.in_shape), None) + (None,) * len(ctx.weights)


def fast_linear_packed(x: Tensor, weights) -> Tensor:
    """Single GEMM over torch.cat(weights); per-weight grads via deferred flush."""
    w_cat = torch.cat([w for w in weights], dim=0)
    if triton_ok() and x.is_cuda and x.dtype == torch.float32:
        return _FastLinearPackedFn.apply(x, w_cat, *weights)
    return torch.nn.functional.linear(x, w_cat)


def fast_linear(x: Tensor, weight: Tensor) -> Tensor:
    if triton_ok() and x.is_cuda and x.dtype == torch.float32:
        if weight.is_leaf:
            return _FastLinearFn.apply(x, weight)
        return _FastLinearFullFn.apply(x, weight)
    return torch.nn.functional.linear(x, weight)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, K = x2.shape
        y = torch.empty_like(x2)
        rstd = x2.new_empty(M)
        BK = triton.next_power_of_2(K)
        _rmsnorm_fwd_kernel[(M,)](x2, g, y, rstd, K, eps, BK=BK)
        ctx.save_for_backward(x2, g, rstd)
        ctx.in_shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, gy):
        x2, g, rstd = ctx.saved_tensors
        gy2 = gy.reshape(-1, gy.shape[-1]).contiguous()
        M, K = x2.shape
        dx = torch.empty_like(x2)
        dg = torch.zeros_like(g)
        BK = triton.next_power_of_2(K)
        ROWS = 32
        _rmsnorm_bwd_kernel[(triton.cdiv(M, ROWS),)](
            x2, g, gy2, rstd, dx, dg, M, K, ROWS=ROWS, BK=BK)
        return dx.view(ctx.in_shape), dg, None


USE_TRITON_GLUE = False  # rmsnorm/silu/gate kernels lose to aten on this stack; rebuild pending


def rms_norm(x: Tensor, g: Tensor, eps: float) -> Tensor:
    if USE_TRITON_GLUE and triton_ok() and x.is_cuda and x.dtype == torch.float32:
        return _RMSNormFn.apply(x, g, eps)
    return torch.nn.functional.rms_norm(x, (g.numel(),), g, eps)


class _SiluGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h):
        shape = h.shape
        h2 = h.reshape(-1, shape[-1]).contiguous()
        M, HH = h2.shape
        H = HH // 2
        out = h2.new_empty(M, H)
        BH = 128
        _silu_gate_fwd_kernel[(M, triton.cdiv(H, BH))](h2, out, M, H, BH=BH)
        ctx.save_for_backward(h2)
        ctx.in_shape = shape
        return out.view(*shape[:-1], H)

    @staticmethod
    def backward(ctx, gy):
        (h2,) = ctx.saved_tensors
        M, HH = h2.shape
        H = HH // 2
        gy2 = gy.reshape(M, H).contiguous()
        gh = torch.empty_like(h2)
        BH = 128
        _silu_gate_bwd_kernel[(M, triton.cdiv(H, BH))](h2, gy2, gh, M, H, BH=BH)
        return gh.view(ctx.in_shape)


def silu_gate(h: Tensor) -> Tensor:
    """silu(h[..., :H]) * h[..., H:] fused."""
    if USE_TRITON_GLUE and triton_ok() and h.is_cuda and h.dtype == torch.float32:
        return _SiluGateFn.apply(h)
    a, b = h.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


class _GateResFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gl, f):
        xc, glc, fc = x.contiguous(), gl.contiguous(), f.contiguous()
        out = torch.empty_like(xc)
        n = xc.numel()
        _gate_res_fwd_kernel[(triton.cdiv(n, 1024),)](xc, glc, fc, out, n, BLOCK=1024)
        ctx.save_for_backward(glc, fc)
        return out

    @staticmethod
    def backward(ctx, gy):
        gl, f = ctx.saved_tensors
        gyc = gy.contiguous()
        dgl = torch.empty_like(gl)
        df = torch.empty_like(f)
        n = gl.numel()
        _gate_res_bwd_kernel[(triton.cdiv(n, 1024),)](gl, f, gyc, dgl, df, n, BLOCK=1024)
        return gyc, dgl, df


def gate_residual(x: Tensor, gate_lin: Tensor, fused: Tensor) -> Tensor:
    """x + sigmoid(gate_lin) * fused, fused into one kernel each direction."""
    if USE_TRITON_GLUE and triton_ok() and x.is_cuda and x.dtype == torch.float32:
        return _GateResFn.apply(x, gate_lin, fused)
    return x + torch.sigmoid(gate_lin) * fused
