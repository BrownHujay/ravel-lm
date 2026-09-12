// Experimental predecessor implementations; not selected by the model.
#include <metal_stdlib>
using namespace metal;

kernel void simd_latest(
    device const long* write [[buffer(0)]],
    device const long* read [[buffer(1)]],
    device const float* payload [[buffer(2)]],
    device float* out [[buffer(3)]],
    device int* sources [[buffer(4)]],
    device bool* valid [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant uint& C [[buffer(7)]],
    constant uint& A [[buffer(8)]],
    constant uint& P [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    int t = int((row / C) % T);
    uint b = row / (C * T);
    uint c = row % C;
    long target = (read[row] % long(A) + long(A)) % long(A);
    int source = -1;
    for (int end = t - 1; end >= 0; end -= 32) {
        int s = end - int(lane);
        int candidate = -1;
        if (s >= 0) {
            uint r = (b * T + uint(s)) * C + c;
            long address = (write[r] % long(A) + long(A)) % long(A);
            if (address == target) candidate = int(r);
        }
        source = simd_max(candidate);
        if (source >= 0) break;
    }
    if (lane == 0) {
        sources[row] = source;
        valid[row] = source >= 0;
    }
    for (uint d = lane; d < P; d += 32) {
        out[row * P + d] = source >= 0 ? payload[uint(source) * P + d] : 0.0f;
    }
}

// Two levels cover up to 1024 time words (32768 tokens).
kernel void bitmap_build(
    device const long* write [[buffer(0)]],
    device atomic_uint* bits [[buffer(1)]],
    device atomic_uint* summary [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& A [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    uint t = (row / C) % T;
    uint b = row / (C * T);
    uint c = row % C;
    uint a = uint((write[row] % long(A) + long(A)) % long(A));
    uint stream = (b * C + c) * A + a;
    uint words = (T + 31) / 32;
    uint summaries = (words + 31) / 32;
    uint word = t / 32;
    atomic_fetch_or_explicit(bits + stream * words + word, 1u << (t % 32), memory_order_relaxed);
    atomic_fetch_or_explicit(summary + stream * summaries + word / 32, 1u << (word % 32), memory_order_relaxed);
}

kernel void bitmap_read(
    device const long* read [[buffer(0)]],
    device const uint* bits [[buffer(1)]],
    device const uint* summary [[buffer(2)]],
    device const float* payload [[buffer(3)]],
    device float* out [[buffer(4)]],
    device int* sources [[buffer(5)]],
    device bool* valid [[buffer(6)]],
    constant uint& T [[buffer(7)]],
    constant uint& C [[buffer(8)]],
    constant uint& A [[buffer(9)]],
    constant uint& P [[buffer(10)]],
    uint gid [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    int source = -1;
    if (lane == 0) {
        uint t = (row / C) % T;
        uint b = row / (C * T);
        uint c = row % C;
        uint a = uint((read[row] % long(A) + long(A)) % long(A));
        uint stream = (b * C + c) * A + a;
        uint words = (T + 31) / 32;
        uint summaries = (words + 31) / 32;
        uint word = t / 32;
        uint mask = (1u << (t % 32)) - 1u;
        uint hits = bits[stream * words + word] & mask;
        if (hits == 0 && word > 0) {
            int sw = int(word / 32);
            uint smask = (1u << (word % 32)) - 1u;
            uint occupied = summary[stream * summaries + sw] & smask;
            while (occupied == 0 && sw > 0) {
                --sw;
                occupied = summary[stream * summaries + sw];
            }
            if (occupied != 0) {
                word = uint(sw) * 32 + 31 - clz(occupied);
                hits = bits[stream * words + word];
            }
        }
        if (hits != 0) {
            uint s = word * 32 + 31 - clz(hits);
            source = int((b * T + s) * C + c);
        }
        sources[row] = source;
        valid[row] = source >= 0;
    }
    source = simd_broadcast(source, 0);
    for (uint d = lane; d < P; d += 32) {
        out[row * P + d] = source >= 0 ? payload[uint(source) * P + d] : 0.0f;
    }
}

kernel void bitmap_backward(
    device const int* sources [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic_float* grad_payload [[buffer(2)]],
    constant uint& P [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint row = gid / P;
    int source = sources[row];
    if (source >= 0) {
        atomic_fetch_add_explicit(grad_payload + uint(source) * P + gid % P,
                                 grad_out[gid], memory_order_relaxed);
    }
}
