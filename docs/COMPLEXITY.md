# Complexity notes

Let:

- `B` = batch size
- `T` = sequence length
- `C` = memory heads per layer
- `D_v` = payload dimension
- `D` = model width
- `L` = number of layers
- `A` = address space

## Training / prefill

For each RAVEL memory layer, the exact latest-k implementation operates over `B*T*C` write records.

Main operations:

```text
address projection:      O(B*T*D*address_projector_width)
payload projection:      O(B*T*D*C*D_v)
sort/search/gather:      O(B*T*C log(B*T*C)) with torch.sort
read fusion GEMM:        O(B*T*C*D_v*D)
```

With a radix-sort implementation on GPU, the event sort can be treated as fixed-pass over fixed-width keys, making it effectively linear in record count. This repo uses portable PyTorch `torch.sort` so it runs without custom CUDA.

## Inference with latest cache

For one generated token per layer:

```text
local conv step:         O(D * kernel_size)
address projection:      O(D * address_projector_width)
cache read/write:        O(C * D_v)
fusion projection:       O(C * D_v * D)
MLP:                     O(D * hidden_dim)
```

There is no term proportional to total context length for latest-address reads.

## Storage

Training/prefill stores normal activations plus the temporary event-key tensors. Decode storage per RAVEL layer is:

```text
B * C * A * D_v values + B * C * A filled-bits
```

This is independent of generated context length for latest-record memory. If exact historical last-k or arbitrary-position memory is required at serving time, keep the full event tape, which is O(n) in the number of written records.

## FLOP intuition

Dense attention core per layer is roughly:

```text
QK^T + AV ≈ 4 * T^2 * D FLOPs
```

RAVEL read fusion per memory layer is roughly:

```text
2 * T * C * D_v * D FLOPs
```

The latter is linear in sequence length for fixed `C`, `D_v`, and `D`.
