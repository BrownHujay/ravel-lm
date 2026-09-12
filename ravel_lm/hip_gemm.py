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


# Vectorized NT GEMM: C[M,N] = A[M,K] @ B[N,K]^T, A and B both row-major [_,K]
# with K contiguous (the nn.Linear forward layout). float4 loads along K.
_SRC_NT = r"""
typedef float float4v __attribute__((ext_vector_type(4)));
extern "C" __global__ __launch_bounds__({THREADS}) void gemm_nt(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int scm)
{{
    const int BM={BM}, BN={BN}, BK={BK}, TM={TM}, TN={TN}, THREADS={THREADS};
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    const int tx = threadIdx.x;
    const int threadsN = BN / TN;
    const int trow = tx / threadsN;
    const int tcol = tx % threadsN;
    const int tg_m = blockIdx.y * BM;
    const int tg_n = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i=0;i<TM;++i) for (int j=0;j<TN;++j) acc[i][j]=0.0f;

    const int K4 = K >> 2;
    for (int k0 = 0; k0 < K; k0 += BK) {{
        // load A[BM][BK] via float4 along K, store transposed As[k][m]
        for (int idx = tx; idx < BM * (BK/4); idx += THREADS) {{
            int m = idx / (BK/4), k4 = idx % (BK/4);
            int gm = tg_m + m;
            float4v v = (gm < M) ? ((const float4v*)A)[gm*K4 + (k0/4) + k4] : float4v{{0,0,0,0}};
            As[k4*4+0][m]=v.x; As[k4*4+1][m]=v.y; As[k4*4+2][m]=v.z; As[k4*4+3][m]=v.w;
        }}
        for (int idx = tx; idx < BN * (BK/4); idx += THREADS) {{
            int n = idx / (BK/4), k4 = idx % (BK/4);
            int gn = tg_n + n;
            float4v v = (gn < N) ? ((const float4v*)B)[gn*K4 + (k0/4) + k4] : float4v{{0,0,0,0}};
            Bs[k4*4+0][n]=v.x; Bs[k4*4+1][n]=v.y; Bs[k4*4+2][n]=v.z; Bs[k4*4+3][n]=v.w;
        }}
        __syncthreads();
        #pragma unroll
        for (int k=0;k<BK;++k) {{
            float ar[TM], br[TN];
            #pragma unroll
            for (int i=0;i<TM;++i) ar[i]=As[k][trow*TM+i];
            #pragma unroll
            for (int j=0;j<TN;++j) br[j]=Bs[k][tcol*TN+j];
            #pragma unroll
            for (int i=0;i<TM;++i) for (int j=0;j<TN;++j) acc[i][j]+=ar[i]*br[j];
        }}
        __syncthreads();
    }}
    #pragma unroll
    for (int i=0;i<TM;++i) {{
        int gm = tg_m + trow*TM + i; if (gm>=M) continue;
        int gn0 = tg_n + tcol*TN;
        if ((TN%4)==0 && gn0+TN<=N && (gn0&3)==0) {{
            #pragma unroll
            for (int j=0;j<TN;j+=4) {{
                float4v o = {{acc[i][j],acc[i][j+1],acc[i][j+2],acc[i][j+3]}};
                *((float4v*)(C + gm*scm + gn0 + j)) = o;
            }}
        }} else {{
            #pragma unroll
            for (int j=0;j<TN;++j) {{ int gn=gn0+j; if (gn<N) C[gm*scm+gn]=acc[i][j]; }}
        }}
    }}
}}
"""




# Vectorized NN GEMM: C[M,Kc] = A[M,Nc] @ B[Nc,Kc], A row-major (Nc contiguous),
# B row-major (Kc contiguous). Used for dX = gy @ W. float4 on both loads.
_SRC_NN = r"""
typedef float float4v __attribute__((ext_vector_type(4)));
extern "C" __global__ __launch_bounds__({THREADS}) void gemm_nn(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int Kc, int Nc, int scm)
{{
    const int BM={BM}, BN={BN}, BK={BK}, TM={TM}, TN={TN}, THREADS={THREADS};
    __shared__ float As[BK][BM];   // As[nc][m]
    __shared__ float Bs[BK][BN];   // Bs[nc][k]
    const int tx = threadIdx.x;
    const int threadsN = BN / TN;
    const int trow = tx / threadsN, tcol = tx % threadsN;
    const int tg_m = blockIdx.y * BM, tg_k = blockIdx.x * BN;
    float acc[TM][TN];
    #pragma unroll
    for (int i=0;i<TM;++i) for (int j=0;j<TN;++j) acc[i][j]=0.0f;
    const int Nc4 = Nc >> 2;
    for (int n0=0;n0<Nc;n0+=BK) {{
        // A[m][n0..+BK] float4 along Nc -> As[nc][m]
        for (int idx=tx; idx<BM*(BK/4); idx+=THREADS) {{
            int m=idx/(BK/4), n4=idx%(BK/4); int gm=tg_m+m;
            float4v v=(gm<M)?((const float4v*)A)[gm*Nc4+(n0/4)+n4]:float4v{{0,0,0,0}};
            As[n4*4+0][m]=v.x; As[n4*4+1][m]=v.y; As[n4*4+2][m]=v.z; As[n4*4+3][m]=v.w;
        }}
        // B[n0..+BK][tg_k..+BN] : Kc contiguous; float4 along k -> Bs[nc][k]
        for (int idx=tx; idx<BK*(BN/4); idx+=THREADS) {{
            int nc=idx/(BN/4), k4=idx%(BN/4); int gnc=n0+nc; int gk=tg_k+k4*4;
            float4v v=(gnc<Nc && gk<Kc) ? *((const float4v*)(B + gnc*Kc + gk)) : float4v{{0,0,0,0}};
            Bs[nc][k4*4+0]=v.x; Bs[nc][k4*4+1]=v.y; Bs[nc][k4*4+2]=v.z; Bs[nc][k4*4+3]=v.w;
        }}
        __syncthreads();
        #pragma unroll
        for (int nc=0;nc<BK;++nc) {{
            float ar[TM], br[TN];
            #pragma unroll
            for (int i=0;i<TM;++i) ar[i]=As[nc][trow*TM+i];
            #pragma unroll
            for (int j=0;j<TN;++j) br[j]=Bs[nc][tcol*TN+j];
            #pragma unroll
            for (int i=0;i<TM;++i) for (int j=0;j<TN;++j) acc[i][j]+=ar[i]*br[j];
        }}
        __syncthreads();
    }}
    #pragma unroll
    for (int i=0;i<TM;++i) {{
        int gm=tg_m+trow*TM+i; if(gm>=M) continue;
        int gk0=tg_k+tcol*TN;
        if ((TN%4)==0 && gk0+TN<=Kc && (gk0&3)==0) {{
            #pragma unroll
            for (int j=0;j<TN;j+=4) {{
                float4v o = {{acc[i][j],acc[i][j+1],acc[i][j+2],acc[i][j+3]}};
                *((float4v*)(C + gm*scm + gk0 + j)) = o;
            }}
        }} else {{
            #pragma unroll
            for (int j=0;j<TN;++j) {{ int gk=gk0+j; if(gk<Kc) C[gm*scm+gk]=acc[i][j]; }}
        }}
    }}
}}
"""



# Split-K dW GEMM: C[Nout,Kout] += A[M,Nout]^T @ B[M,Kout], contraction M split
# across blockIdx.z with atomic accumulation. A=gy (M,Nout), B=x (M,Kout), both
# row-major contiguous. Fixes low occupancy of tiny-output/large-K dW GEMMs.
_SRC_DW = r"""
typedef float float4v __attribute__((ext_vector_type(4)));
extern "C" __global__ __launch_bounds__({THREADS}) void gemm_dw(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int Nout, int Kout, int nsplit)
{{
    const int BM={BM}, BN={BN}, BK={BK}, TM={TM}, TN={TN}, THREADS={THREADS};
    __shared__ float As[BK][BM];   // As[m_local][nout]
    __shared__ float Bs[BK][BN];   // Bs[m_local][kout]
    const int tx=threadIdx.x;
    const int threadsN=BN/TN; const int trow=tx/threadsN, tcol=tx%threadsN;
    const int tg_n=blockIdx.y*BM;   // over Nout
    const int tg_k=blockIdx.x*BN;   // over Kout
    const int zc=blockIdx.z;
    int chunk=(M+nsplit-1)/nsplit; int m0=zc*chunk; int m1=m0+chunk; if(m1>M)m1=M;
    float acc[TM][TN];
    #pragma unroll
    for(int i=0;i<TM;++i) for(int j=0;j<TN;++j) acc[i][j]=0.0f;
    for(int mb=m0; mb<m1; mb+=BK) {{
        for(int idx=tx; idx<BK*BM; idx+=THREADS){{
            int ml=idx/BM, nn=idx%BM; int gm=mb+ml, gn=tg_n+nn;
            As[ml][nn]=(gm<m1 && gn<Nout)?A[gm*Nout+gn]:0.0f;
        }}
        for(int idx=tx; idx<BK*BN; idx+=THREADS){{
            int ml=idx/BN, kk=idx%BN; int gm=mb+ml, gk=tg_k+kk;
            Bs[ml][kk]=(gm<m1 && gk<Kout)?B[gm*Kout+gk]:0.0f;
        }}
        __syncthreads();
        #pragma unroll
        for(int ml=0;ml<BK;++ml){{
            float ar[TM], br[TN];
            #pragma unroll
            for(int i=0;i<TM;++i) ar[i]=As[ml][trow*TM+i];
            #pragma unroll
            for(int j=0;j<TN;++j) br[j]=Bs[ml][tcol*TN+j];
            #pragma unroll
            for(int i=0;i<TM;++i) for(int j=0;j<TN;++j) acc[i][j]+=ar[i]*br[j];
        }}
        __syncthreads();
    }}
    #pragma unroll
    for(int i=0;i<TM;++i){{
        int gn=tg_n+trow*TM+i; if(gn>=Nout) continue;
        #pragma unroll
        for(int j=0;j<TN;++j){{
            int gk=tg_k+tcol*TN+j;
            if(gk<Kout) atomicAdd(&C[gn*Kout+gk], acc[i][j]);
        }}
    }}
}}
"""



# Double-buffered NT GEMM: prefetch next K-tile into registers during compute,
# hiding global-load latency. Same math/result as gemm_nt.
_SRC_NT_DB = r"""
typedef float float4v __attribute__((ext_vector_type(4)));
extern "C" __global__ __launch_bounds__({THREADS}) void gemm_nt_db(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int scm)
{{
    const int BM={BM}, BN={BN}, BK={BK}, TM={TM}, TN={TN}, THREADS={THREADS};
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];
    const int tx=threadIdx.x; const int threadsN=BN/TN;
    const int trow=tx/threadsN, tcol=tx%threadsN;
    const int tg_m=blockIdx.y*BM, tg_n=blockIdx.x*BN;
    const int K4=K>>2;
    const int na=BM*(BK/4), nb=BN*(BK/4);
    float acc[TM][TN];
    #pragma unroll
    for(int i=0;i<TM;++i) for(int j=0;j<TN;++j) acc[i][j]=0.0f;
    // preload tile 0
    for(int idx=tx; idx<na; idx+=THREADS){{
        int m=idx/(BK/4), k4=idx%(BK/4); int gm=tg_m+m;
        float4v v=(gm<M)?((const float4v*)A)[gm*K4+k4]:float4v{{0,0,0,0}};
        As[0][k4*4+0][m]=v.x;As[0][k4*4+1][m]=v.y;As[0][k4*4+2][m]=v.z;As[0][k4*4+3][m]=v.w;
    }}
    for(int idx=tx; idx<nb; idx+=THREADS){{
        int n=idx/(BK/4), k4=idx%(BK/4); int gn=tg_n+n;
        float4v v=(gn<N)?((const float4v*)B)[gn*K4+k4]:float4v{{0,0,0,0}};
        Bs[0][k4*4+0][n]=v.x;Bs[0][k4*4+1][n]=v.y;Bs[0][k4*4+2][n]=v.z;Bs[0][k4*4+3][n]=v.w;
    }}
    __syncthreads();
    int cur=0;
    const int ntiles=(K+BK-1)/BK;
    for(int t=0;t<ntiles;++t){{
        int nxt=1-cur;
        float4v ra[ (na+THREADS-1)/THREADS ];
        float4v rb[ (nb+THREADS-1)/THREADS ];
        int cnt_a=0, cnt_b=0;
        if(t+1<ntiles){{
            int k0n=(t+1)*BK;
            for(int idx=tx; idx<na; idx+=THREADS){{
                int m=idx/(BK/4), k4=idx%(BK/4); int gm=tg_m+m;
                ra[cnt_a++]=(gm<M)?((const float4v*)A)[gm*K4+(k0n/4)+k4]:float4v{{0,0,0,0}};
            }}
            for(int idx=tx; idx<nb; idx+=THREADS){{
                int n=idx/(BK/4), k4=idx%(BK/4); int gn=tg_n+n;
                rb[cnt_b++]=(gn<N)?((const float4v*)B)[gn*K4+(k0n/4)+k4]:float4v{{0,0,0,0}};
            }}
        }}
        #pragma unroll
        for(int k=0;k<BK;++k){{
            float ar[TM], br[TN];
            #pragma unroll
            for(int i=0;i<TM;++i) ar[i]=As[cur][k][trow*TM+i];
            #pragma unroll
            for(int j=0;j<TN;++j) br[j]=Bs[cur][k][tcol*TN+j];
            #pragma unroll
            for(int i=0;i<TM;++i) for(int j=0;j<TN;++j) acc[i][j]+=ar[i]*br[j];
        }}
        if(t+1<ntiles){{
            cnt_a=0;
            for(int idx=tx; idx<na; idx+=THREADS){{
                int m=idx/(BK/4), k4=idx%(BK/4);
                float4v v=ra[cnt_a++];
                As[nxt][k4*4+0][m]=v.x;As[nxt][k4*4+1][m]=v.y;As[nxt][k4*4+2][m]=v.z;As[nxt][k4*4+3][m]=v.w;
            }}
            cnt_b=0;
            for(int idx=tx; idx<nb; idx+=THREADS){{
                int n=idx/(BK/4), k4=idx%(BK/4);
                float4v v=rb[cnt_b++];
                Bs[nxt][k4*4+0][n]=v.x;Bs[nxt][k4*4+1][n]=v.y;Bs[nxt][k4*4+2][n]=v.z;Bs[nxt][k4*4+3][n]=v.w;
            }}
            __syncthreads();
            cur=nxt;
        }}
    }}
    #pragma unroll
    for(int i=0;i<TM;++i){{
        int gm=tg_m+trow*TM+i; if(gm>=M) continue;
        #pragma unroll
        for(int j=0;j<TN;++j){{ int gn=tg_n+tcol*TN+j; if(gn<N) C[gm*scm+gn]=acc[i][j]; }}
    }}
}}
"""


@functools.lru_cache(maxsize=64)
def _kernel(BM, BN, BK, TM, TN) -> Kernel:
    threads = (BM // TM) * (BN // TN)
    src = _SRC.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN, THREADS=threads)
    return compile_kernel(src, "gemm_f32", options=())


@functools.lru_cache(maxsize=64)
def _kernel_nt(BM, BN, BK, TM, TN) -> Kernel:
    threads = (BM // TM) * (BN // TN)
    src = _SRC_NT.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN, THREADS=threads)
    return compile_kernel(src, "gemm_nt", options=())


@functools.lru_cache(maxsize=64)
def _kernel_nt_db(BM, BN, BK, TM, TN) -> Kernel:
    threads = (BM // TM) * (BN // TN)
    src = _SRC_NT_DB.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN, THREADS=threads)
    return compile_kernel(src, "gemm_nt_db", options=())


@functools.lru_cache(maxsize=64)
def _kernel_nn(BM, BN, BK, TM, TN) -> Kernel:
    threads = (BM // TM) * (BN // TN)
    src = _SRC_NN.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN, THREADS=threads)
    return compile_kernel(src, "gemm_nn", options=())


@functools.lru_cache(maxsize=64)
def _kernel_dw(BM, BN, BK, TM, TN) -> Kernel:
    threads = (BM // TM) * (BN // TN)
    src = _SRC_DW.format(BM=BM, BN=BN, BK=BK, TM=TM, TN=TN, THREADS=threads)
    return compile_kernel(src, "gemm_dw", options=())


_DW_NSPLIT = 16
_CFG_DW = (32, 32, 16, 4, 4)


def gemm_dw_accum(gy, x, Nout, Kout, M, grad):
    """grad[Nout,Kout] += gy[M,Nout]^T @ x[M,Kout] via split-K atomics."""
    BM, BN, BK, TM, TN = _CFG_DW
    ker = _kernel_dw(BM, BN, BK, TM, TN)
    grid = ((Kout + BN - 1) // BN, (Nout + BM - 1) // BM, _DW_NSPLIT)
    threads = (BM // TM) * (BN // TN)
    ker.launch(grid, (threads, 1, 1),
               [ptr(gy), ptr(x), ptr(grad), i32(M), i32(Nout), i32(Kout), i32(_DW_NSPLIT)])


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


_CFG_NT_DEFAULT = (64, 64, 16, 4, 4)
_CFG_NT_TABLE = {
    (160, 320): (64, 32, 8, 4, 8),
    (160, 640): (128, 64, 8, 8, 8),
    (320, 160): (32, 32, 8, 4, 4),
    (160, 160): (64, 32, 8, 4, 4),
    (96, 160): (32, 32, 8, 4, 4),
    (640, 160): (64, 32, 8, 4, 4),
    (160, 96): (64, 32, 8, 4, 4),
}


def gemm_nt(a: torch.Tensor, b: torch.Tensor, M, N, K, out=None) -> torch.Tensor:
    """C[M,N] = a[M,K] @ b[N,K]^T; a,b row-major [_,K] contiguous. float4 along K."""
    if K % 4 != 0 or not a.is_contiguous() or not b.is_contiguous():
        return gemm(a.contiguous(), b, M, N, K, b.stride(1), b.stride(0), out=out)
    c = out if out is not None else a.new_empty(M, N)
    BM, BN, BK, TM, TN = _CFG_NT_TABLE.get((K, N), _CFG_NT_DEFAULT)
    if BK % 4 != 0:
        BK = (BK // 4) * 4 or 4
    ker = _kernel_nt(BM, BN, BK, TM, TN)
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    threads = (BM // TM) * (BN // TN)
    ker.launch(grid, (threads, 1, 1),
               [ptr(a), ptr(b), ptr(c), i32(M), i32(N), i32(K), i32(c.stride(0))])
    return c


def set_nt_config_table(table: dict):
    _CFG_NT_TABLE.clear()
    _CFG_NT_TABLE.update(table)


_CFG_NN_DEFAULT = (64, 64, 16, 4, 4)
_CFG_NN_TABLE = {
    (320, 160): (64, 64, 8, 8, 4),
    (640, 160): (64, 32, 8, 4, 4),
    (160, 160): (64, 32, 8, 4, 4),
    (160, 320): (64, 64, 8, 8, 4),
    (160, 640): (64, 128, 8, 8, 8),
}


def gemm_nn(a, b, M, Kc, Nc, out=None):
    """C[M,Kc] = a[M,Nc] @ b[Nc,Kc]; a,b row-major contiguous. dX path."""
    if Nc % 4 != 0 or Kc % 4 != 0 or not a.is_contiguous() or not b.is_contiguous():
        return gemm(a.contiguous(), b, M, Kc, Nc, b.stride(0), b.stride(1), out=out)
    c = out if out is not None else a.new_empty(M, Kc)
    BM, BN, BK, TM, TN = _CFG_NN_TABLE.get((Nc, Kc), _CFG_NN_DEFAULT)
    if BK % 4: BK = (BK // 4) * 4 or 4
    if BN % 4: BN = (BN // 4) * 4 or 4
    ker = _kernel_nn(BM, BN, BK, TM, TN)
    grid = ((Kc + BN - 1) // BN, (M + BM - 1) // BM, 1)
    threads = (BM // TM) * (BN // TN)
    ker.launch(grid, (threads, 1, 1),
               [ptr(a), ptr(b), ptr(c), i32(M), i32(Kc), i32(Nc), i32(c.stride(0))])
    return c


def set_nn_config_table(table: dict):
    _CFG_NN_TABLE.clear()
    _CFG_NN_TABLE.update(table)


def gemm_available() -> bool:
    return available()
