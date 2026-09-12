// Experimental 32x32 FP32 matrix tile; native MPS GEMM was faster.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline short2 matrix_coord(ushort lane) {
    short q = lane / 4;
    return short2((q & 2) * 2 + (lane % 2) * 2, (q & 4) + ((lane / 2) % 4));
}

kernel void matmul32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& as0 [[buffer(6)]],
    constant uint& as1 [[buffer(7)]],
    constant uint& bs0 [[buffer(8)]],
    constant uint& bs1 [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float at[32 * 32];
    threadgroup float bt[32 * 32];
    uint m0 = group.y * 32;
    uint n0 = group.x * 32;
    short2 lc = matrix_coord(lane);
    uint r = (sg / 2) * 16 + lc.y;
    uint c = (sg % 2) * 16 + lc.x;
    simdgroup_float8x8 acc[4];
    for (uint j = 0; j < 4; ++j) {
        acc[j].thread_elements()[0] = 0;
        acc[j].thread_elements()[1] = 0;
    }
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = tid; i < 1024; i += 128) {
            uint rr = i / 32;
            uint cc = i % 32;
            at[i] = (m0 + rr < M && k0 + cc < K)
                ? a[(m0 + rr) * as0 + (k0 + cc) * as1] : 0;
            bt[i] = (k0 + rr < K && n0 + cc < N)
                ? b[(k0 + rr) * bs0 + (n0 + cc) * bs1] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 av[2];
            simdgroup_float8x8 bv[2];
            for (uint j = 0; j < 2; ++j) {
                av[j].thread_elements()[0] = at[(r + j * 8) * 32 + k + lc.x];
                av[j].thread_elements()[1] = at[(r + j * 8) * 32 + k + lc.x + 1];
                bv[j].thread_elements()[0] = bt[(k + lc.y) * 32 + c + j * 8];
                bv[j].thread_elements()[1] = bt[(k + lc.y) * 32 + c + j * 8 + 1];
            }
            for (uint i = 0; i < 2; ++i) {
                for (uint j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(acc[i * 2 + j], av[i], bv[j], acc[i * 2 + j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i) {
        for (uint j = 0; j < 2; ++j) {
            uint rr = m0 + r + i * 8;
            uint cc = n0 + c + j * 8;
            if (rr < M && cc < N) out[rr * N + cc] = acc[i * 2 + j].thread_elements()[0];
            if (rr < M && cc + 1 < N) out[rr * N + cc + 1] = acc[i * 2 + j].thread_elements()[1];
        }
    }
}
