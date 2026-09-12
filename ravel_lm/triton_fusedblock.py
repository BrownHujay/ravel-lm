"""Coarse-grained fused sublayers for CUDA/ROCm: one autograd.Function each.

At B=1 on Windows ROCm, every Python autograd.Function crossing costs ~40us,
so modular kernel wrappers pay a ~3ms/step interpreter tax. These Functions
run the same Triton kernels internally but cross Python once per sublayer.
Backward passes are hand-composed from the modular path's formulas; weight
grads ride the deferred batched flush (packed weights are row-split there).
"""
from __future__ import annotations

import torch

from .deferred import _STASH

try:
    import triton

    from .triton_block import (
        _gate_res_bwd_kernel,
        _gate_res_fwd_kernel,
        _rmsnorm_bwd_kernel,
        _rmsnorm_fwd_kernel,
        _silu_gate_bwd_kernel,
        _silu_gate_fwd_kernel,
        _tgemm,
    )
    from .triton_latest1 import (
        _scatter_grad_kernel,
        _search_gather_kernel,
        _write_keys_kernel,
    )
    from .triton_local import _lg_bwd_g_kernel, _lg_bwd_u_kernel, _lg_fwd_kernel

    _HAS_KERNELS = True
except ImportError:  # pragma: no cover
    _HAS_KERNELS = False

Tensor = torch.Tensor


def _rms_fwd(x2, g, eps):
    M, K = x2.shape
    y = torch.empty_like(x2)
    rstd = x2.new_empty(M)
    _rmsnorm_fwd_kernel[(M,)](x2, g, y, rstd, K, eps, BK=triton.next_power_of_2(K))
    return y, rstd


def _rms_bwd_into(x2, g, gy2, rstd, dg):
    M, K = x2.shape
    dx = torch.empty_like(x2)
    ROWS = 32
    _rmsnorm_bwd_kernel[(triton.cdiv(M, ROWS),)](
        x2, g, gy2, rstd, dx, dg, M, K, ROWS=ROWS, BK=triton.next_power_of_2(K))
    return dx


def _lin_fwd(x2, w):
    M, K = x2.shape
    return _tgemm(x2, w, M, w.shape[0], K, w.stride(1), w.stride(0))


def _lin_dx(gy2, w):
    M, N = gy2.shape
    return _tgemm(gy2, w, M, w.shape[1], N, w.stride(0), w.stride(1))


class _FusedFFNFn(torch.autograd.Function):
    """x + w3(silu_gate(w12(rmsnorm(x)))) — one Python crossing."""

    @staticmethod
    def forward(ctx, x, norm_g, w12, w3, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        xn, rstd = _rms_fwd(x2, norm_g, eps)
        h = _lin_fwd(xn, w12)
        M, HH = h.shape
        H = HH // 2
        y = h.new_empty(M, H)
        _silu_gate_fwd_kernel[(M, triton.cdiv(H, 128))](h, y, M, H, BH=128)
        out = _lin_fwd(y, w3)
        out += x2
        ctx.save_for_backward(x2, norm_g, w12, w3, xn, rstd, h, y)
        ctx.shape = shape
        return out.view(shape)

    @staticmethod
    def backward(ctx, gy):
        x2, norm_g, w12, w3, xn, rstd, h, y = ctx.saved_tensors
        gy2 = gy.reshape(-1, gy.shape[-1]).contiguous()
        M, HH = h.shape
        H = HH // 2
        dy = _lin_dx(gy2, w3)
        _STASH.append((w3, y, gy2))
        dh = torch.empty_like(h)
        _silu_gate_bwd_kernel[(M, triton.cdiv(H, 128))](h, dy, dh, M, H, BH=128)
        dxn = _lin_dx(dh, w12)
        _STASH.append((w12, xn, dh))
        dg = torch.zeros_like(norm_g)
        dx = _rms_bwd_into(x2, norm_g, dxn, rstd, dg)
        dx += gy2
        return dx.view(ctx.shape), dg, None, None, None


class _FusedMixerFn(torch.autograd.Function):
    """x + out_proj(local_gate(in_proj(rmsnorm(x)))) — one Python crossing."""

    @staticmethod
    def forward(ctx, x, norm_g, in_w, conv_w, conv_b, out_w, eps):
        shape = x.shape
        B, T = shape[0], shape[1]
        x2 = x.reshape(-1, shape[-1]).contiguous()
        xn, rstd = _rms_fwd(x2, norm_g, eps)
        uv = _lin_fwd(xn, in_w)
        D = uv.shape[1] // 2
        uv3 = uv.view(B, T, 2 * D)
        yb = uv.new_empty(B, T, D)
        conv = torch.empty_like(yb)
        _lg_fwd_kernel[(B * T, triton.cdiv(D, 64))](
            uv3, conv_w, conv_b, yb, conv, T, D, BLOCK_D=64)
        y2 = yb.view(B * T, D)
        out = _lin_fwd(y2, out_w)
        out += x2
        ctx.save_for_backward(x2, norm_g, in_w, conv_w, conv_b, out_w, xn, rstd, uv, conv, y2)
        ctx.shape = shape
        ctx.btd = (B, T, D)
        return out.view(shape)

    @staticmethod
    def backward(ctx, gy):
        x2, norm_g, in_w, conv_w, conv_b, out_w, xn, rstd, uv, conv, y2 = ctx.saved_tensors
        B, T, D = ctx.btd
        gy2 = gy.reshape(-1, gy.shape[-1]).contiguous()
        dy = _lin_dx(gy2, out_w)
        _STASH.append((out_w, y2, gy2))
        uv3 = uv.view(B, T, 2 * D)
        guv = torch.empty_like(uv3)
        g = torch.empty(B, T, D, device=uv.device, dtype=uv.dtype)
        dw = torch.zeros_like(conv_w)
        db = conv_w.new_zeros(D)
        _lg_bwd_g_kernel[(B * T, triton.cdiv(D, 64))](
            uv3, conv.view(B, T, D), dy.view(B, T, D), g, guv, T, D, BLOCK_D=64)
        _lg_bwd_u_kernel[(triton.cdiv(T, 64), triton.cdiv(D, 64), B)](
            uv3, conv_w, g, guv, dw, db, T, D, BLOCK_T=64, BLOCK_D=64)
        guv2 = guv.view(B * T, 2 * D)
        dxn = _lin_dx(guv2, in_w)
        _STASH.append((in_w, xn, guv2))
        dg = torch.zeros_like(norm_g)
        dx = _rms_bwd_into(x2, norm_g, dxn, rstd, dg)
        dx += gy2
        return dx.view(ctx.shape), dg, None, dw, db, None, None


class _FusedMemoryFn(torch.autograd.Function):
    """rmsnorm -> packed projections -> hard addressing (detached) -> exact
    latest-1 lookup -> fuse -> gated residual, one Python crossing."""

    @staticmethod
    def forward(ctx, x, norm_g, pay_w, gate_w, gate_b, fuse_w, waddr_w, raddr_w,
                literal_addr, multipliers, eps, A, C, P, n_cb, cb_size):
        shape = x.shape
        B, T = shape[0], shape[1]
        M = B * T
        x2 = x.reshape(-1, shape[-1]).contiguous()
        xn, rstd = _rms_fwd(x2, norm_g, eps)
        has_addr = waddr_w is not None
        weights = [pay_w, gate_w] + ([waddr_w, raddr_w] if has_addr else [])
        w_cat = torch.cat(weights, dim=0)
        packed = _lin_fwd(xn, w_cat)
        n_pay, n_gate = pay_w.shape[0], gate_w.shape[0]
        gate_lin = packed[:, n_pay:n_pay + n_gate].contiguous()
        gate_lin += gate_b
        payload = packed[:, :n_pay].contiguous()

        if has_addr:
            logits = packed[:, n_pay + n_gate:].detach().view(M, 2, -1, n_cb, cb_size)
            codes = logits.argmax(dim=-1)
            learned = (codes * multipliers.view(1, 1, 1, n_cb)).sum(dim=-1).remainder(A)
            lit = literal_addr.view(M, -1)
            write_addr = torch.cat([lit, learned[:, 0]], dim=1)
            read_addr = torch.cat([lit, learned[:, 1]], dim=1)
        else:
            write_addr = read_addr = literal_addr.view(M, -1)

        N = M * C
        device = x.device
        wa = write_addr.reshape(-1).to(torch.int32)
        keys = torch.empty(N, device=device, dtype=torch.int32)
        _write_keys_kernel[(triton.cdiv(N, 1024),)](wa, keys, N, T, C, A, BLOCK=1024)
        sorted_keys, order = torch.sort(keys)
        order = order.to(torch.int32)
        ra = read_addr.reshape(-1).to(torch.int32).contiguous()
        vals = payload.new_empty(N, P)
        src = torch.empty(N, device=device, dtype=torch.int32)
        msk = torch.empty(N, device=device, dtype=torch.int8)
        BD = max(16, triton.next_power_of_2(P))
        _search_gather_kernel[(N,)](
            sorted_keys, order, ra, payload.view(N, P), vals, src, msk,
            N, P, T + 1, T, C, A, BLOCK_D=BD)
        read_flat = vals.view(M, C * P)
        fused = _lin_fwd(read_flat, fuse_w)
        out = torch.empty_like(x2)
        n = x2.numel()
        _gate_res_fwd_kernel[(triton.cdiv(n, 1024),)](x2, gate_lin, fused, out, n, BLOCK=1024)
        ctx.save_for_backward(x2, norm_g, fuse_w, gate_lin, read_flat, fused, src, xn, rstd, w_cat)
        ctx.weights = weights
        ctx.meta = (shape, B, T, C, P, n_pay, n_gate, w_cat.shape[0])
        return out.view(shape)

    @staticmethod
    def backward(ctx, gy):
        (x2, norm_g, fuse_w, gate_lin, read_flat, fused, src, xn, rstd, w_cat) = ctx.saved_tensors
        shape, B, T, C, P, n_pay, n_gate, n_total = ctx.meta
        M = B * T
        gy2 = gy.reshape(-1, gy.shape[-1]).contiguous()
        n = gy2.numel()
        dgl = torch.empty_like(gate_lin)
        dfused = torch.empty_like(fused)
        _gate_res_bwd_kernel[(triton.cdiv(n, 1024),)](
            gate_lin, fused, gy2, dgl, dfused, n, BLOCK=1024)
        dread = _lin_dx(dfused, fuse_w)
        _STASH.append((fuse_w, read_flat, dfused))
        N = M * C
        dpayload = dread.new_zeros(N, P)
        BD = max(16, triton.next_power_of_2(P))
        _scatter_grad_kernel[(N,)](src, dread.contiguous(), dpayload, N, P, BLOCK_D=BD)
        addr_cols = n_total - n_pay - n_gate
        pieces = [dpayload.view(M, n_pay), dgl]
        if addr_cols:
            pieces.append(dgl.new_zeros(M, addr_cols))
        dpacked = torch.cat(pieces, dim=1)
        dxn = _lin_dx(dpacked, w_cat)
        _STASH.append((list(ctx.weights), xn, dpacked))
        dgate_b = dgl.sum(dim=0)
        dg = torch.zeros_like(norm_g)
        dx = _rms_bwd_into(x2, norm_g, dxn, rstd, dg)
        dx += gy2
        return (dx.view(shape), dg, None, None, dgate_b, None, None, None,
                None, None, None, None, None, None, None, None)


def fused_ok(cfg_conv_kernel: int, d_model: int) -> bool:
    return (
        _HAS_KERNELS
        and torch.cuda.is_available()
        and cfg_conv_kernel == 7
        and d_model % 4 == 0
    )


def fused_memory_ok(cfg) -> bool:
    return (
        _HAS_KERNELS
        and torch.cuda.is_available()
        and cfg.last_k == 1
        and not cfg.use_sum_read
        and cfg.dropout == 0.0
    )


def fused_ffn(x, norm_g, w12, w3, eps):
    return _FusedFFNFn.apply(x, norm_g, w12, w3, eps)


def fused_mixer(x, norm_g, in_w, conv_w, conv_b, out_w, eps):
    return _FusedMixerFn.apply(x, norm_g, in_w, conv_w, conv_b, out_w, eps)


def fused_memory(x, layer, literal_addr):
    cfg = layer.cfg
    waddr = layer.write_addr.proj.weight if layer.write_addr.proj is not None else None
    raddr = layer.read_addr.proj.weight if layer.read_addr.proj is not None else None
    return _FusedMemoryFn.apply(
        x, layer.norm.weight, layer.payload.weight, layer.gate.weight,
        layer.gate.bias, layer.fuse.weight, waddr, raddr,
        literal_addr, layer.write_addr.multipliers,
        layer.norm.eps, cfg.address_space, cfg.n_memory_heads, cfg.payload_dim,
        cfg.n_codebooks, cfg.codebook_size,
    )
