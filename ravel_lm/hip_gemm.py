"""Register-blocked fp32 GEMM for RDNA4 via hiprtc.

C[M,N] = A[M,K] @ B[K,N], all fp32, arbitrary A/B strides so the one kernel
serves forward (B = W^T), input-grad (B = W), and weight-grad (A = gy^T) paths.
RDNA4 has no fp32 matrix cores, so this is LDS-tiled + register-blocked SIMT.
"""
from __future__ import annotations

import ctypes
import functools

import torch

from .hip_rtc import Kernel, available, compile_kernel, i32, ptr

_SRC = r"""
extern "C" __global__ __launch_bounds__({THREADS}) void gemm_f32(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K,
    int sam, int sak, int sbk, int sbn, int scm, int scn)
{{
    const int BM = {BM}, BN = {BN}, BK = {BK}, TM = {TM}, TN = {TN};
    __shared__ float As[BK][BM];   // A tile, transposed for contiguous inner reads
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;                 // 0 .. (BN/TN * BM/TM - 1)
    const int threadsN = BN / TN;               // threads spanning N
    const int trow = tx / threadsN;             // which TM-row group
    const int tcol = tx % threadsN;             // which TN-col group
    const int tg_m = blockIdx.y * BM;
    const int tg_n = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int THREADS = {THREADS};
    for (int k0 = 0; k0 < K; k0 += BK) {{
        // Cooperative load of A[tg_m .. +BM][k0 .. +BK] into As[k][m].
        for (int idx = tx; idx < BM * BK; idx += THREADS) {{
            int m = idx / BK, k = idx % BK;
            int gm = tg_m + m, gk = k0 + k;
            As[k][m] = (gm < M && gk < K) ? A[gm * sam + gk * sak] : 0.0f;
        }}
        // Cooperative load of B[k0 .. +BK][tg_n .. +BN] into Bs[k][n].
        for (int idx = tx; idx < BK * BN; idx += THREADS) {{
            int k = idx / BN, n = idx % BN;
            int gk = k0 + k, gn = tg_n + n;
            Bs[k][n] = (gk < K && gn < N) ? B[gk * sbk + gn * sbn] : 0.0f;
        }}
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {{
            float ar[TM], br[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][trow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) br[j] = Bs[k][tcol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += ar[i] * br[j];
        }}
        __syncthreads();
    }}

    #pragma unroll
    for (int i = 0; i < TM; ++i) {{
        int gm = tg_m + trow * TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {{
            int gn = tg_n + tcol * TN + j;
            if (gn < N) C[gm * scm + gn * scn] = acc[i][j];
        }}
    }}
}}
"""


@functools.lru_cache(maxsize=64)
def _kernel(BM, BN, BK, TM, TN) -> Kernel:
    threads = (BM // TM) * (BN // TN)
    src = _SRC.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN, THREADS=threads)
    return compile_kernel(src, "gemm_f32", options=())


@functools.lru_cache(maxsize=64)
def _kernel_acc(BM, BN, BK, TM, TN) -> Kernel:
    threads = (BM // TM) * (BN // TN)
    # C += A@B; each output element is owned by one thread, so += is race-free.
    src = _SRC.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN, THREADS=threads)
    src = src.replace("void gemm_f32(", "void gemm_f32_acc(")
    src = src.replace("C[gm * scm + gn * scn] = acc[i][j];",
                      "C[gm * scm + gn * scn] += acc[i][j];")
    return compile_kernel(src, "gemm_f32_acc", options=())


# Per-(K, N) tile config; tuned on RX 9070 XT (filled by autotune, safe default).
_CFG_DEFAULT = (64, 64, 8, 4, 4)
_CFG_TABLE = {
    (160, 320): (64, 128, 16, 8, 4),
    (160, 640): (128, 64, 8, 8, 4),
    (320, 160): (32, 64, 16, 4, 4),
    (160, 160): (64, 64, 8, 4, 4),
    (96, 160): (64, 64, 8, 4, 4),
    (640, 160): (64, 64, 8, 4, 4),
    (260, 160): (64, 128, 8, 4, 4),
}


def set_config_table(table: dict):
    _CFG_TABLE.clear()
    _CFG_TABLE.update(table)


def gemm(a: torch.Tensor, b: torch.Tensor, M, N, K, sbk, sbn, out=None) -> torch.Tensor:
    """C[M,N] = a[M,K] @ b (b accessed via strides sbk over K, sbn over N)."""
    c = out if out is not None else a.new_empty(M, N)
    BM, BN, BK, TM, TN = _CFG_TABLE.get((K, N), _CFG_DEFAULT)
    ker = _kernel(BM, BN, BK, TM, TN)
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    threads = (BM // TM) * (BN // TN)
    ker.launch(
        grid, (threads, 1, 1),
        [ptr(a), ptr(b), ptr(c), i32(M), i32(N), i32(K),
         i32(a.stride(0)), i32(a.stride(1)), i32(sbk), i32(sbn),
         i32(c.stride(0)), i32(c.stride(1))],
    )
    return c


def gemm_full(a, b, M, N, K, sam, sak, sbk, sbn, out, accumulate=False):
    """C[M,N] = a @ b with fully explicit strides (no contiguity copies).

    Used for dW = gy^T @ x where A is accessed transposed. ``accumulate`` adds
    into ``out`` instead of overwriting (graph-safe grad accumulation)."""
    scm, scn = out.stride(0), out.stride(1)
    BM, BN, BK, TM, TN = _CFG_TABLE.get((K, N), _CFG_DEFAULT)
    ker = _kernel_acc(BM, BN, BK, TM, TN) if accumulate else _kernel(BM, BN, BK, TM, TN)
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    threads = (BM // TM) * (BN // TN)
    ker.launch(
        grid, (threads, 1, 1),
        [ptr(a), ptr(b), ptr(out), i32(M), i32(N), i32(K),
         i32(sam), i32(sak), i32(sbk), i32(sbn), i32(scm), i32(scn)],
    )
    return out


def gemm_available() -> bool:
    return available()
