# RAVEL design

RAVEL stands for **Routed Algebraic Value-Event Lattice**. This project implements the architecture as a causal language model that uses exact addressed event-memory instead of softmax attention.

## Block layout

Each `RavelBlock` in `ravel_lm/model.py` applies:

```text
RMSNorm → gated causal depthwise convolution → residual
RAVEL exact memory read/write layer → residual
RMSNorm → SwiGLU feed-forward → residual
```

There is no `QK^T` attention matrix. Local mixing is handled by a causal depthwise convolution. Long-range recurrence is handled by exact addressed memory.

## Event-memory operator

For every sequence position `t`, every memory head writes:

```text
(batch, head, address, time=t, payload)
```

A read at position `t` asks for the previous record with the same `(batch, head, address)` and time `< t`. In training/prefill, `ravel_lm/ravel_memory.py` creates composite integer keys:

```text
key = ((batch_head_address) * (T + 1)) + time
```

Then it uses:

```text
sort(keys) → searchsorted(read_key) → gather(payload)
```

This gives exact causal latest-k lookup for the model-defined address lattice. The file also includes a segmented prefix-sum read for algebraic-sum memory experiments.

## Address families

`RavelMemoryLayer` combines literal and learned address heads.

Literal heads are deterministic:

```text
head 0: token ID
head 1: rolling bigram hash(prev_token, token)
head 2: rolling trigram hash(prev2, prev1, token), if enabled
head 3+: token/position mixed hashes
```

When `address_space >= vocab_size`, the token-ID literal head has exact token-ID addressability.

Learned heads use hard product-code addresses:

```text
hidden → codebook logits → argmax code per codebook → integer address
```

This is intentionally discrete routing, not similarity attention. The current code keeps this simple and PyTorch-native. The payload and fusion paths are fully differentiable; hard address choices are non-smooth by design.

## Decode cache

For incremental generation, `RavelLayerCache` stores latest values in a tensor:

```text
[batch, memory_heads, address_space, payload_dim]
```

Reads are `torch.gather`; writes are `scatter_`. This gives O(1) latest-address reads per generated token for the latest-record monoid. The training/prefill path still supports exact last-k over the full sequence.

## Why no attention fallback?

The goal of this repo is to exercise the different math tendency: exact addressed event-memory. So the included model avoids local attention and global sparse attention. The local mixer is convolutional, and the global path is address lookup + algebraic fusion.
