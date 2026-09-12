// Experimental full FFN training shaders; not selected by the model.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint D = 160;
constant uint H = 320;
constant uint BM = 8;

inline short2 ffn_coord(ushort lane) {
    short q = lane / 4;
    return short2((q & 2) * 2 + (lane % 2) * 2, (q & 4) + ((lane / 2) % 4));
}

inline float ffn_silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void ffn_register_forward(
    device const float* x [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* w12 [[buffer(2)]],
    device const float* w3 [[buffer(3)]],
    device float* out [[buffer(4)]],
    device float* xn_out [[buffer(5)]],
    device float* preactivation [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * 8;
    short2 lc = ffn_coord(lane);
    uint row = lc.y;
    uint col = lc.x;
    float inv_rms[8];
    for (uint r = 0; r < 8; ++r) {
        float ss = 0;
        for (uint d = lane; d < D; d += 32) {
            float v = x[(row0 + r) * D + d];
            ss += v * v;
        }
        inv_rms[r] = rsqrt(simd_sum(ss) / float(D) + 1e-6f);
    }
    simdgroup_float8x8 inputs[D / 8];
    simdgroup_float8x8 outputs[D / 8];
    for (uint kt = 0; kt < D / 8; ++kt) {
        uint dc = kt * 8 + col;
        uint offset = (row0 + row) * D + dc;
        float v0 = x[offset] * norm_weight[dc] * inv_rms[row];
        float v1 = x[offset + 1] * norm_weight[dc + 1] * inv_rms[row];
        inputs[kt].thread_elements()[0] = v0;
        inputs[kt].thread_elements()[1] = v1;
        xn_out[offset] = v0;
        xn_out[offset + 1] = v1;
        outputs[kt].thread_elements()[0] = 0;
        outputs[kt].thread_elements()[1] = 0;
    }
    for (uint ht = 0; ht < H / 8; ++ht) {
        simdgroup_float8x8 a;
        simdgroup_float8x8 b;
        a.thread_elements()[0] = a.thread_elements()[1] = 0;
        b.thread_elements()[0] = b.thread_elements()[1] = 0;
        for (uint kt = 0; kt < D / 8; ++kt) {
            uint kr = kt * 8 + row;
            uint hc = ht * 8 + col;
            simdgroup_float8x8 wa;
            simdgroup_float8x8 wb;
            wa.thread_elements()[0] = w12[hc * D + kr];
            wa.thread_elements()[1] = w12[(hc + 1) * D + kr];
            wb.thread_elements()[0] = w12[(hc + H) * D + kr];
            wb.thread_elements()[1] = w12[(hc + H + 1) * D + kr];
            simdgroup_multiply_accumulate(a, inputs[kt], wa, a);
            simdgroup_multiply_accumulate(b, inputs[kt], wb, b);
        }
        uint offset = (row0 + row) * 2 * H + ht * 8 + col;
        preactivation[offset] = a.thread_elements()[0];
        preactivation[offset + 1] = a.thread_elements()[1];
        preactivation[offset + H] = b.thread_elements()[0];
        preactivation[offset + H + 1] = b.thread_elements()[1];
        simdgroup_float8x8 hidden;
        hidden.thread_elements()[0] = ffn_silu(a.thread_elements()[0]) * b.thread_elements()[0];
        hidden.thread_elements()[1] = ffn_silu(a.thread_elements()[1]) * b.thread_elements()[1];
        for (uint dt = 0; dt < D / 8; ++dt) {
            simdgroup_float8x8 weight;
            uint dc = dt * 8 + col;
            uint hr = ht * 8 + row;
            weight.thread_elements()[0] = w3[dc * H + hr];
            weight.thread_elements()[1] = w3[(dc + 1) * H + hr];
            simdgroup_multiply_accumulate(outputs[dt], hidden, weight, outputs[dt]);
        }
    }
    for (uint dt = 0; dt < D / 8; ++dt) {
        uint offset = (row0 + row) * D + dt * 8 + col;
        out[offset] = x[offset] + outputs[dt].thread_elements()[0];
        out[offset + 1] = x[offset + 1] + outputs[dt].thread_elements()[1];
    }
}

kernel void ffn_forward_160x320_f32(
    device const float* x [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* w12 [[buffer(2)]],
    device const float* w3 [[buffer(3)]],
    device float* out [[buffer(4)]],
    device float* xn_out [[buffer(5)]],
    device float* preactivation [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xn[BM * D];
    threadgroup float hidden[BM * H];
    uint row0 = group * BM;
    float sum_sq = 0.0f;
    for (uint d = lane; d < D; d += 32) {
        float value = x[(row0 + simd_group) * D + d];
        sum_sq += value * value;
    }
    float inv_rms = rsqrt(simd_sum(sum_sq) / float(D) + 1e-6f);
    for (uint d = lane; d < D; d += 32) {
        uint offset = (row0 + simd_group) * D + d;
        float value = x[offset] * norm_weight[d] * inv_rms;
        xn[simd_group * D + d] = value;
        xn_out[offset] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    short2 lc = ffn_coord(lane);
    uint row = lc.y;
    uint col = lc.x;
    for (uint nt = simd_group; nt < H / 8; nt += 8) {
        simdgroup_float8x8 gate_acc;
        simdgroup_float8x8 value_acc;
        gate_acc.thread_elements()[0] = gate_acc.thread_elements()[1] = 0.0f;
        value_acc.thread_elements()[0] = value_acc.thread_elements()[1] = 0.0f;
        for (uint kt = 0; kt < D / 8; ++kt) {
            simdgroup_float8x8 input_tile;
            simdgroup_float8x8 gate_tile;
            simdgroup_float8x8 value_tile;
            uint kr = kt * 8 + row;
            uint kc = kt * 8 + col;
            input_tile.thread_elements()[0] = xn[row * D + kc];
            input_tile.thread_elements()[1] = xn[row * D + kc + 1];
            uint nr = nt * 8 + col;
            gate_tile.thread_elements()[0] = w12[nr * D + kr];
            gate_tile.thread_elements()[1] = w12[(nr + 1) * D + kr];
            value_tile.thread_elements()[0] = w12[(H + nr) * D + kr];
            value_tile.thread_elements()[1] = w12[(H + nr + 1) * D + kr];
            simdgroup_multiply_accumulate(gate_acc, input_tile, gate_tile, gate_acc);
            simdgroup_multiply_accumulate(value_acc, input_tile, value_tile, value_acc);
        }
        uint hc = nt * 8 + col;
        uint global = (row0 + row) * (2 * H) + hc;
        float a0 = gate_acc.thread_elements()[0];
        float a1 = gate_acc.thread_elements()[1];
        float b0 = value_acc.thread_elements()[0];
        float b1 = value_acc.thread_elements()[1];
        preactivation[global] = a0;
        preactivation[global + 1] = a1;
        preactivation[global + H] = b0;
        preactivation[global + H + 1] = b1;
        hidden[row * H + hc] = ffn_silu(a0) * b0;
        hidden[row * H + hc + 1] = ffn_silu(a1) * b1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint nt = simd_group; nt < D / 8; nt += 8) {
        simdgroup_float8x8 acc;
        acc.thread_elements()[0] = acc.thread_elements()[1] = 0.0f;
        for (uint kt = 0; kt < H / 8; ++kt) {
            simdgroup_float8x8 input_tile;
            simdgroup_float8x8 weight_tile;
            uint kr = kt * 8 + row;
            uint kc = kt * 8 + col;
            input_tile.thread_elements()[0] = hidden[row * H + kc];
            input_tile.thread_elements()[1] = hidden[row * H + kc + 1];
            uint nr = nt * 8 + col;
            weight_tile.thread_elements()[0] = w3[nr * H + kr];
            weight_tile.thread_elements()[1] = w3[(nr + 1) * H + kr];
            simdgroup_multiply_accumulate(acc, input_tile, weight_tile, acc);
        }
        uint dc = nt * 8 + col;
        uint offset = (row0 + row) * D + dc;
        out[offset] = x[offset] + acc.thread_elements()[0];
        out[offset + 1] = x[offset + 1] + acc.thread_elements()[1];
    }
}

kernel void ffn_backward_data_160x320_f32(
    device const float* x [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* w12 [[buffer(2)]],
    device const float* w3 [[buffer(3)]],
    device const float* preactivation [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device float* grad_x [[buffer(6)]],
    device float* grad_norm_weight [[buffer(7)]],
    device float* grad_ab_out [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float grad_ab[BM * 2 * H];
    threadgroup float grad_xn[BM * D];
    uint row0 = group * BM;
    short2 lc = ffn_coord(lane);
    uint row = lc.y;
    uint col = lc.x;

    for (uint nt = simd_group; nt < H / 8; nt += 8) {
        simdgroup_float8x8 acc;
        acc.thread_elements()[0] = acc.thread_elements()[1] = 0.0f;
        for (uint kt = 0; kt < D / 8; ++kt) {
            simdgroup_float8x8 output_tile;
            simdgroup_float8x8 weight_tile;
            uint kc = kt * 8 + col;
            output_tile.thread_elements()[0] = grad_out[(row0 + row) * D + kc];
            output_tile.thread_elements()[1] = grad_out[(row0 + row) * D + kc + 1];
            uint kr = kt * 8 + row;
            uint nr = nt * 8 + col;
            weight_tile.thread_elements()[0] = w3[kr * H + nr];
            weight_tile.thread_elements()[1] = w3[kr * H + nr + 1];
            simdgroup_multiply_accumulate(acc, output_tile, weight_tile, acc);
        }
        uint hc = nt * 8 + col;
        uint global = (row0 + row) * (2 * H) + hc;
        float a0 = preactivation[global];
        float a1 = preactivation[global + 1];
        float b0 = preactivation[global + H];
        float b1 = preactivation[global + H + 1];
        float s0 = 1.0f / (1.0f + exp(-a0));
        float s1 = 1.0f / (1.0f + exp(-a1));
        float gh0 = acc.thread_elements()[0];
        float gh1 = acc.thread_elements()[1];
        float ga0 = gh0 * b0 * s0 * (1.0f + a0 * (1.0f - s0));
        float ga1 = gh1 * b1 * s1 * (1.0f + a1 * (1.0f - s1));
        float gb0 = gh0 * a0 * s0;
        float gb1 = gh1 * a1 * s1;
        grad_ab[row * (2 * H) + hc] = ga0;
        grad_ab[row * (2 * H) + hc + 1] = ga1;
        grad_ab[row * (2 * H) + H + hc] = gb0;
        grad_ab[row * (2 * H) + H + hc + 1] = gb1;
        grad_ab_out[global] = ga0;
        grad_ab_out[global + 1] = ga1;
        grad_ab_out[global + H] = gb0;
        grad_ab_out[global + H + 1] = gb1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint nt = simd_group; nt < D / 8; nt += 8) {
        simdgroup_float8x8 acc;
        acc.thread_elements()[0] = acc.thread_elements()[1] = 0.0f;
        for (uint kt = 0; kt < (2 * H) / 8; ++kt) {
            simdgroup_float8x8 input_tile;
            simdgroup_float8x8 weight_tile;
            uint kc = kt * 8 + col;
            input_tile.thread_elements()[0] = grad_ab[row * (2 * H) + kc];
            input_tile.thread_elements()[1] = grad_ab[row * (2 * H) + kc + 1];
            uint kr = kt * 8 + row;
            uint nr = nt * 8 + col;
            weight_tile.thread_elements()[0] = w12[kr * D + nr];
            weight_tile.thread_elements()[1] = w12[kr * D + nr + 1];
            simdgroup_multiply_accumulate(acc, input_tile, weight_tile, acc);
        }
        uint dc = nt * 8 + col;
        grad_xn[row * D + dc] = acc.thread_elements()[0];
        grad_xn[row * D + dc + 1] = acc.thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_sq = 0.0f;
    float correction = 0.0f;
    for (uint d = lane; d < D; d += 32) {
        uint offset = (row0 + simd_group) * D + d;
        float xv = x[offset];
        float gxnv = grad_xn[simd_group * D + d];
        sum_sq += xv * xv;
        correction += gxnv * norm_weight[d] * xv;
    }
    float inv_rms = rsqrt(simd_sum(sum_sq) / float(D) + 1e-6f);
    correction = simd_sum(correction);
    for (uint d = lane; d < D; d += 32) {
        uint offset = (row0 + simd_group) * D + d;
        float xv = x[offset];
        float gxnv = grad_xn[simd_group * D + d];
        float weighted = gxnv * norm_weight[d];
        grad_x[offset] = grad_out[offset] + weighted * inv_rms
            - xv * inv_rms * inv_rms * inv_rms * correction / float(D);
        grad_xn[simd_group * D + d] = gxnv * xv * inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint d = simd_group * 32 + lane;
    if (d < D) {
        float total = 0;
        for (uint r = 0; r < BM; ++r) total += grad_xn[r * D + d];
        grad_norm_weight[group * D + d] = total;
    }
}

kernel void ffn_grad_w3_160x320_f32(
    device const float* preactivation [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_w3 [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint tile = group * 8 + simd_group;
    if (tile >= (D / 8) * (H / 8)) return;
    uint d_tile = tile / (H / 8);
    uint h_tile = tile % (H / 8);
    short2 lc = ffn_coord(lane);
    uint row = lc.y;
    uint col = lc.x;
    simdgroup_float8x8 acc;
    acc.thread_elements()[0] = acc.thread_elements()[1] = 0.0f;
    for (uint nt = 0; nt < rows / 8; ++nt) {
        simdgroup_float8x8 output_tile;
        simdgroup_float8x8 hidden_tile;
        uint n_col = nt * 8 + col;
        output_tile.thread_elements()[0] = grad_out[n_col * D + d_tile * 8 + row];
        output_tile.thread_elements()[1] = grad_out[(n_col + 1) * D + d_tile * 8 + row];
        uint n_row = nt * 8 + row;
        uint hc = h_tile * 8 + col;
        uint p0 = n_row * (2 * H) + hc;
        uint p1 = (n_row + 1) * (2 * H) + hc;
        hidden_tile.thread_elements()[0] =
            ffn_silu(preactivation[p0]) * preactivation[p0 + H];
        hidden_tile.thread_elements()[1] =
            ffn_silu(preactivation[p0 + 1]) * preactivation[p0 + H + 1];
        simdgroup_multiply_accumulate(acc, output_tile, hidden_tile, acc);
    }
    uint offset = (d_tile * 8 + row) * H + h_tile * 8 + col;
    grad_w3[offset] = acc.thread_elements()[0];
    grad_w3[offset + 1] = acc.thread_elements()[1];
}

kernel void ffn_grad_w12_160x320_f32(
    device const float* grad_ab [[buffer(0)]],
    device const float* xn [[buffer(1)]],
    device float* grad_w12 [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint tile = group * 8 + simd_group;
    if (tile >= ((2 * H) / 8) * (D / 8)) return;
    uint o_tile = tile / (D / 8);
    uint d_tile = tile % (D / 8);
    short2 lc = ffn_coord(lane);
    uint row = lc.y;
    uint col = lc.x;
    simdgroup_float8x8 acc;
    acc.thread_elements()[0] = acc.thread_elements()[1] = 0.0f;
    for (uint nt = 0; nt < rows / 8; ++nt) {
        simdgroup_float8x8 grad_tile;
        simdgroup_float8x8 input_tile;
        uint n0 = nt * 8 + col;
        uint oc = o_tile * 8 + row;
        grad_tile.thread_elements()[0] = grad_ab[n0 * (2 * H) + oc];
        grad_tile.thread_elements()[1] = grad_ab[(n0 + 1) * (2 * H) + oc];
        uint n_row = nt * 8 + row;
        uint dc = d_tile * 8 + col;
        input_tile.thread_elements()[0] = xn[n_row * D + dc];
        input_tile.thread_elements()[1] = xn[n_row * D + dc + 1];
        simdgroup_multiply_accumulate(acc, grad_tile, input_tile, acc);
    }
    uint offset = (o_tile * 8 + row) * D + d_tile * 8 + col;
    grad_w12[offset] = acc.thread_elements()[0];
    grad_w12[offset + 1] = acc.thread_elements()[1];
}
