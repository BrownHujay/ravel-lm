# GPU/kernel notes

This project is written in pure PyTorch, but the operator decomposition is intentionally GPU-friendly.

## Training/prefill path

`causal_last_k_lookup` maps to these bulk kernels:

```text
1. address/payload projection: GEMM
2. composite key creation: vectorized elementwise kernel
3. sort keys: torch sort backend; replaceable by radix-sort/CUB custom op
4. searchsorted: vectorized binary search kernel
5. map selected sorted keys back to source indices and gather payloads once
6. fuse read payloads: GEMM
```

The expensive neural pieces are dense matrix multiplications. The irregular piece is lookup, but it is batched over all tokens and heads, not scalar Python work.

On MPS, the training scripts compile the complete fixed-shape model with the
Inductor backend by default. Full-graph compilation fuses the elementwise portions
of RMSNorm, SwiGLU, residual updates, address construction, and memory lookup around
the backend GEMM/sort operations. It is disabled explicitly with
`--no-compile-models` in `training_suite.py` or `--no-compile` in
`bench_ravel_train_step.py`.

The same MPS path compiles global L2 gradient clipping as one fixed-parameter
operation and uses PyTorch's fused AdamW implementation. Both retain float32 model,
gradient, and optimizer-state tensors; this removes optimizer dispatch overhead and
does not change the update rule.

The lookup keeps only sorted integer keys and source indices. Payload vectors are
not permuted into a second complete tape before selection; selected source indices
gather directly from the original payload tensor. This preserves exact latest-k
semantics and removes one dense payload gather plus its backward scatter.

Composite event keys use int32 whenever the complete `(batch, head, address,
time)` key range fits, which covers the included models and training shapes. Larger
ranges automatically retain the int64 path. This halves key sort/search bandwidth
without narrowing addresses or changing lookup semantics.

## Decode path

`RavelLayerCache` uses dense tensors:

```text
values: [B, C, A, D_v]
filled: [B, C, A]
```

Reads:

```text
torch.gather(values, address_index)
```

Writes:

```text
values.scatter_(address_index, payload)
```

This is simple to batch across users because each generated token produces a small fixed number of probes. For production, probes from many users should be grouped by memory layer/address page before gather to improve locality.

## Custom kernel upgrade path

The cleanest future speed path is:

```text
radix_sort_u64_keys
segmented_last_k_scan
segmented_sum_scan
gather_fuse_payloads
```

The PyTorch implementation keeps the semantics obvious and testable. A CUDA extension can replace only `ravel_lm/ravel_memory.py` while preserving the public model API.

## Raw Metal kernels

The repo also includes a direct Metal backend in:

```text
ravel_lm/metal/kernels.metal
ravel_lm/metal_kernels.py
scripts/bench_metal_kernels.py
```

This path uses `Metal.framework` through PyObjC. It does not use PyTorch MPS.

Implemented kernels:

```text
ravel_latest1_f32
causal_softmax_attention_f32
```

`ravel_latest1_f32` implements strict-causal latest-record lookup for the RAVEL operator with `k=1`, float32 payloads, optional write masking, and separate write/read addresses. It scans each observed event stream once and indexes a latest-value state table directly, giving `O(B * C * D * T)` work instead of sweeping all `A` possible addresses. `ravel_latest1_f32x4` vectorizes the common payload-dimension-multiple-of-four case; the scalar kernel remains the fallback.

`ravel_latest1_sweep_f32` preserves the original `O(B * C * A * D * T)` implementation for same-process regression benchmarks only. It is never selected by the default `auto` path.

`causal_softmax_attention_f32` implements standard causal scaled dot-product softmax attention for float32 `[B, H, T, D]` tensors. It uses one threadgroup per query row, parallel dot-product reductions over `D`, a stable max/subtract softmax, and then weighted value accumulation.

The benchmark script reports command-buffer GPU time and wall time. Wall time includes Python, Metal buffer setup, and readback. Correctness is checked against the PyTorch reference operators in the same run.
