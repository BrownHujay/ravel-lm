# RAVEL-LM

RAVEL-LM is a runnable PyTorch project for a non-attention causal language model built around **RAVEL: Routed Algebraic Value-Event Lattice**. Instead of forming a dense `QK^T` matrix, tokens write compact typed payloads to exact address streams, and later tokens read previous matching records through sort/search/gather event-memory primitives.

The repo includes:

- `ravel_lm/model.py`: causal LM with local non-attention mixer + exact RAVEL memory layers.
- `ravel_lm/ravel_memory.py`: exact causal latest-k and segmented-sum event-tape primitives.
- `ravel_lm/tokenizers/`: reversible byte tokenizer and dependency-free byte-level BPE tokenizer.
- `ravel_lm/data/`: TinyStories/local text streaming and packed-token datasets.
- `configs/`: byte and BPE model configs from roughly 200k to 5M parameters.
- `scripts/`: BPE training, TinyStories preparation, parameter counting, smoke training.
- `tests/`: tokenizer, memory, model, generation, and incremental-cache tests.

## Install

```bash
pip install -e .[dev,data]
```

For offline smoke tests, only PyTorch, NumPy, and pytest are needed. Hugging Face `datasets` is only required when streaming TinyStories directly.

## Verify

```bash
pytest
python scripts/count_params.py
python scripts/smoke_train.py
```

## TinyStories byte training

```bash
python -m ravel_lm.train \
  --config configs/byte/ravel_200k_byte.json \
  --tokenizer byte \
  --max-records 10000 \
  --out-dir runs/ravel_200k_byte
```

This streams `roneneldan/TinyStories` through Hugging Face `datasets` unless `--local-text` or `--toy` is passed.

## Train a BPE tokenizer

```bash
python scripts/train_bpe.py \
  --out tokenizers/tinystories_bpe1024.json \
  --vocab-size 1024 \
  --max-texts 20000
```

Then train a BPE config:

```bash
python -m ravel_lm.train \
  --config configs/bpe/ravel_1m_bpe1k.json \
  --tokenizer tokenizers/tinystories_bpe1024.json \
  --max-records 10000 \
  --out-dir runs/ravel_1m_bpe
```

## Prepare a flat token file

```bash
python scripts/prepare_tinystories.py \
  --out data/tinystories_byte_train.bin \
  --tokenizer byte \
  --max-records 100000

python -m ravel_lm.train \
  --config configs/byte/ravel_500k_byte.json \
  --tokenizer byte \
  --token-bin data/tinystories_byte_train.bin \
  --out-dir runs/ravel_500k_byte
```

## Generate

```bash
python -m ravel_lm.generate \
  --config runs/ravel_200k_byte/config.json \
  --checkpoint runs/ravel_200k_byte/final.pt \
  --tokenizer runs/ravel_200k_byte/tokenizer.json \
  --prompt "Once upon a time" \
  --max-new-tokens 80
```

## Design notes

RAVEL is exact for the memory operator it defines. It is not exact softmax attention and does not try to approximate `QK^T`. Literal address heads provide exact token-ID recurrence when the address space is at least the tokenizer vocab. Learned product-code heads provide discrete learned routing. Training/prefill uses sort + search + gather over a virtual event tape; decode uses a tensorized latest-value cache with gather/scatter.

See `docs/DESIGN.md`, `docs/COMPLEXITY.md`, and `docs/KERNELS.md` for details.
