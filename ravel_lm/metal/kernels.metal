#include <metal_stdlib>
using namespace metal;

struct RavelLatest1Params {
    uint B;
    uint T;
    uint C;
    uint D;
    uint address_space;
    uint has_write_mask;
    uint total_threads;
};

struct AttentionParams {
    uint B;
    uint H;
    uint T;
    uint D;
    uint threadgroup_size;
    float scale;
};

kernel void ravel_latest1_sweep_f32(
    device const int* write_addr [[buffer(0)]],
    device const float* payload [[buffer(1)]],
    device const int* read_addr [[buffer(2)]],
    device const uchar* write_mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    device uchar* out_mask [[buffer(5)]],
    constant RavelLatest1Params& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.total_threads) {
        return;
    }

    uint d = gid % p.D;
    uint tmp = gid / p.D;
    uint a = tmp % p.address_space;
    tmp /= p.address_space;
    uint c = tmp % p.C;
    uint b = tmp / p.C;
    float latest = 0.0f;
    bool filled = false;

    for (uint t = 0; t < p.T; ++t) {
        uint addr_idx = (b * p.T + t) * p.C + c;
        uint value_idx = addr_idx * p.D + d;
        if ((uint)read_addr[addr_idx] == a) {
            out[value_idx] = filled ? latest : 0.0f;
            if (d == 0) {
                out_mask[addr_idx] = filled ? 1 : 0;
            }
        }
        bool writes = (uint)write_addr[addr_idx] == a;
        if (p.has_write_mask != 0) {
            writes = writes && write_mask[b * p.T + t] != 0;
        }
        if (writes) {
            latest = payload[value_idx];
            filled = true;
        }
    }
}

kernel void ravel_latest1_f32(
    device const int* write_addr [[buffer(0)]],
    device const float* payload [[buffer(1)]],
    device const int* read_addr [[buffer(2)]],
    device const uchar* write_mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    device uchar* out_mask [[buffer(5)]],
    device float* latest_by_address [[buffer(6)]],
    constant RavelLatest1Params& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.total_threads) {
        return;
    }

    const uint d = gid % p.D;
    const uint stream = gid / p.D;
    const uint c = stream % p.C;
    const uint b = stream / p.C;
    const uint state_base = stream * p.address_space * p.D;

    for (uint t = 0; t < p.T; ++t) {
        const uint addr_idx = (b * p.T + t) * p.C + c;
        const uint value_idx = addr_idx * p.D + d;
        const uint read_state_idx = state_base + (uint)read_addr[addr_idx] * p.D + d;
        const float latest = latest_by_address[read_state_idx];
        const bool filled = !isnan(latest);

        out[value_idx] = filled ? latest : 0.0f;
        if (d == 0) {
            out_mask[addr_idx] = filled ? 1 : 0;
        }

        bool writes = true;
        if (p.has_write_mask != 0) {
            writes = writes && (write_mask[b * p.T + t] != 0);
        }
        if (writes) {
            const uint write_state_idx = state_base + (uint)write_addr[addr_idx] * p.D + d;
            latest_by_address[write_state_idx] = payload[value_idx];
        }
    }
}

kernel void ravel_latest1_f32x4(
    device const int* write_addr [[buffer(0)]],
    device const float4* payload [[buffer(1)]],
    device const int* read_addr [[buffer(2)]],
    device const uchar* write_mask [[buffer(3)]],
    device float4* out [[buffer(4)]],
    device uchar* out_mask [[buffer(5)]],
    device float4* latest_by_address [[buffer(6)]],
    constant RavelLatest1Params& p [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.total_threads) {
        return;
    }

    const uint D4 = p.D >> 2;
    const uint d4 = gid % D4;
    const uint stream = gid / D4;
    const uint c = stream % p.C;
    const uint b = stream / p.C;
    const uint state_base = stream * p.address_space * D4;

    for (uint t = 0; t < p.T; ++t) {
        const uint addr_idx = (b * p.T + t) * p.C + c;
        const uint value_idx = addr_idx * D4 + d4;
        const uint read_state_idx = state_base + (uint)read_addr[addr_idx] * D4 + d4;
        const float4 latest = latest_by_address[read_state_idx];
        const bool filled = !isnan(latest.x);

        out[value_idx] = filled ? latest : float4(0.0f);
        if (d4 == 0) {
            out_mask[addr_idx] = filled ? 1 : 0;
        }

        bool writes = true;
        if (p.has_write_mask != 0) {
            writes = write_mask[b * p.T + t] != 0;
        }
        if (writes) {
            const uint write_state_idx = state_base + (uint)write_addr[addr_idx] * D4 + d4;
            latest_by_address[write_state_idx] = payload[value_idx];
        }
    }
}

kernel void causal_softmax_attention_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint MAX_THREADS = 256;
    constexpr uint MAX_T = 2048;
    threadgroup float scratch[MAX_THREADS];
    threadgroup float probs[MAX_T];

    uint row = tg_pos.x;
    uint t = row % p.T;
    uint h = (row / p.T) % p.H;
    uint b = row / (p.T * p.H);
    uint nt = p.threadgroup_size;

    uint row_base = ((b * p.H + h) * p.T + t) * p.D;
    uint bh_base = (b * p.H + h) * p.T * p.D;

    for (uint s = 0; s <= t; ++s) {
        float partial = 0.0f;
        uint key_base = bh_base + s * p.D;
        for (uint d = tid; d < p.D; d += nt) {
            partial += q[row_base + d] * k[key_base + d];
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = nt >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            probs[s] = scratch[0] * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float row_max = probs[0];
        for (uint s = 1; s <= t; ++s) {
            row_max = max(row_max, probs[s]);
        }

        float denom = 0.0f;
        for (uint s = 0; s <= t; ++s) {
            float e = exp(probs[s] - row_max);
            probs[s] = e;
            denom += e;
        }
        float inv_denom = 1.0f / denom;
        for (uint s = 0; s <= t; ++s) {
            probs[s] *= inv_denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < p.D; d += nt) {
        float acc = 0.0f;
        for (uint s = 0; s <= t; ++s) {
            uint value_base = bh_base + s * p.D;
            acc += probs[s] * v[value_base + d];
        }
        out[row_base + d] = acc;
    }
}
