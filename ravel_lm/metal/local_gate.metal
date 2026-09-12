#include <metal_stdlib>
using namespace metal;

kernel void local_gate_forward(
    device const float* uv [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* conv [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    uint d = gid % D;
    uint t = (gid / D) % T;
    uint b = gid / (D * T);
    float z = bias[d];
    for (uint k = 0; k < K; ++k) {
        int s = int(t) + int(k) - int(K) + 1;
        if (s >= 0) z += uv[(b * T + uint(s)) * 2 * D + d] * weight[d * K + k];
    }
    conv[gid] = z;
    out[gid] = (z / (1.0f + exp(-z))) * uv[(b * T + t) * 2 * D + D + d];
}

// Vectorized forward: one thread per (b, t, d4) computing four channels with
// float4 loads. Accumulation order over k and the silu expression match the
// scalar kernel exactly, so outputs are bit-identical.
kernel void local_gate_forward_v4(
    device const float4* uv [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float4* bias [[buffer(2)]],
    device float4* out [[buffer(3)]],
    device float4* conv [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint D4 = D / 4;
    uint d4 = gid % D4;
    uint t = (gid / D4) % T;
    uint b = gid / (D4 * T);
    uint d = d4 * 4;
    float4 z = bias[d4];
    for (uint k = 0; k < 7; ++k) {
        int s = int(t) + int(k) - 6;
        if (s >= 0) {
            float4 u4 = uv[(b * T + uint(s)) * (D / 2) + d4];
            z.x += u4.x * weight[(d + 0) * 7 + k];
            z.y += u4.y * weight[(d + 1) * 7 + k];
            z.z += u4.z * weight[(d + 2) * 7 + k];
            z.w += u4.w * weight[(d + 3) * 7 + k];
        }
    }
    float4 v4 = uv[(b * T + t) * (D / 2) + D4 + d4];
    conv[gid] = z;
    out[gid] = (z / (1.0f + exp(-z))) * v4;
}

inline float conv_grad_g(float z, float gyv, float v) {
    float s = 1.0f / (1.0f + exp(-z));
    return gyv * v * s * (1.0f + z * (1.0f - s));
}

// Backward with threadgroup-staged conv_grad and u tiles: each tile row's
// conv_grad (and its exp) is computed once and reused, instead of up to eight
// redundant recomputations with device-memory reads. Per-element arithmetic
// order matches the previous kernel, so gradients are bit-identical.
kernel void local_gate_backward7(
    device const float* uv [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* conv [[buffer(2)]],
    device const float* gy [[buffer(3)]],
    device float* guv [[buffer(4)]],
    device float* partial [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[8 * 8 * 32];
    threadgroup float g_tile[38 * 32];   // conv_grad for rows t0 .. t0+37
    threadgroup float u_tile[38 * 32];   // u for rows t0-6 .. t0+31
    const uint t0 = group.y * 32;
    const uint b = group.z;
    const uint d_base = group.x * 32;

    // Stage g and u tiles; emit the v-half gradient for owned rows while the
    // inputs are already loaded.
    for (uint idx = tid; idx < 38 * 32; idx += 256) {
        uint r = idx / 32;
        uint c = idx % 32;
        uint d = d_base + c;
        {
            uint t = t0 + r;
            float g = 0.0f;
            if (d < D && t < T) {
                uint row = b * T + t;
                uint pos = row * D + d;
                float z = conv[pos];
                float gyv = gy[pos];
                float v = uv[row * 2 * D + D + d];
                g = conv_grad_g(z, gyv, v);
                if (r < 32) {
                    guv[row * 2 * D + D + d] = gyv * z / (1.0f + exp(-z));
                }
            }
            g_tile[idx] = g;
        }
        {
            int s = int(t0) + int(r) - 6;
            float u = 0.0f;
            if (d < D && s >= 0 && uint(s) < T) u = uv[(b * T + uint(s)) * 2 * D + d];
            u_tile[idx] = u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint d = d_base + lane;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint j = 0; j < 4; ++j) {
        uint r = uint(sg) + j * 8;
        uint t = t0 + r;
        if (d < D && t < T) {
            float g = g_tile[r * 32 + lane];
            acc[7] += g;
            float gin = 0;
            for (uint k = 0; k < 7; ++k) {
                // write source s = t + k - 6 -> u_tile row (r + k)
                acc[k] += g * u_tile[(r + k) * 32 + lane];
                // future = t + 6 - k -> g_tile row (r + 6 - k); zero past T
                gin += g_tile[(r + 6 - k) * 32 + lane] * weight[d * 7 + k];
            }
            guv[(b * T + t) * 2 * D + d] = gin;
        }
    }
    for (uint k = 0; k < 8; ++k) sums[(sg * 8 + k) * 32 + lane] = acc[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && d < D) {
        uint tile = b * ((T + 31) / 32) + group.y;
        for (uint k = 0; k < 8; ++k) {
            float total = 0;
            for (uint s = 0; s < 8; ++s) total += sums[(s * 8 + k) * 32 + lane];
            partial[(tile * D + d) * 8 + k] = total;
        }
    }
}
