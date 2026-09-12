# RAVEL-LM build checklist

Every item below is checked after implementation. Evidence paths point to the exact files or verification logs to inspect.

- [x] Full PyTorch project folder created. Evidence: `pyproject.toml`, `requirements.txt`, `ravel_lm/__init__.py`, `README.md`.
- [x] RAVEL exact addressed event-memory operator implemented. Evidence: `ravel_lm/ravel_memory.py` (`causal_last_k_lookup`, `causal_sum_lookup`, `RavelLayerCache`).
- [x] Non-attention causal LM implemented around RAVEL memory. Evidence: `ravel_lm/model.py` (`RavelLM`, `RavelBlock`, `RavelMemoryLayer`, `LocalMixer`).
- [x] GPU-friendly/batchable primitive decomposition documented. Evidence: `docs/KERNELS.md`.
- [x] Complexity and FLOP intuition documented. Evidence: `docs/COMPLEXITY.md`, `ravel_lm/bench.py`, `docs/VERIFICATION.md`.
- [x] Byte tokenizer implemented. Evidence: `ravel_lm/tokenizers/byte_tokenizer.py`, `tests/test_tokenizers.py`.
- [x] Dependency-free byte-level BPE tokenizer implemented. Evidence: `ravel_lm/tokenizers/bpe_tokenizer.py`, `scripts/train_bpe.py`, `tests/test_tokenizers.py`.
- [x] TinyStories loader implemented. Evidence: `ravel_lm/data/tinystories.py`, `ravel_lm/data/packed_dataset.py`, `scripts/prepare_tinystories.py`, `ravel_lm/train.py`.
- [x] Local text and offline toy data fallback implemented. Evidence: `ravel_lm/data/tinystories.py` (`iter_local_text`, `toy_stories`), `scripts/smoke_train.py`.
- [x] Training script implemented. Evidence: `ravel_lm/train.py`.
- [x] Generation script implemented. Evidence: `ravel_lm/generate.py`, `docs/VERIFICATION.md` generation smoke test.
- [x] Parameter-count configs from ~200k to ~5M implemented. Evidence: `configs/byte/ravel_200k_byte.json`, `configs/byte/ravel_500k_byte.json`, `configs/byte/ravel_1m_byte.json`, `configs/byte/ravel_3m_byte.json`, `configs/byte/ravel_5m_byte.json`, `docs/PARAM_COUNTS.md`.
- [x] BPE model configs implemented. Evidence: `configs/bpe/ravel_300k_bpe1k.json`, `configs/bpe/ravel_1m_bpe1k.json`, `configs/bpe/ravel_5m_bpe2k.json`.
- [x] Unit tests implemented for tokenizers, event memory, model forward/backward, incremental cache, and generation. Evidence: `tests/`.
- [x] Unit tests passed in the container. Evidence: `docs/VERIFICATION.md`.
- [x] Smoke training passed in the container. Evidence: `docs/VERIFICATION.md`, `runs/smoke/final.pt`.
- [x] Exact memory bench passed in the container. Evidence: `docs/VERIFICATION.md` (`exact_latest_check=True`).
- [x] Parameter-count evidence generated. Evidence: `docs/PARAM_COUNTS.md`.

## Important honesty notes

- This project implements exact training for the **RAVEL operator**, not exact softmax attention.
- The included PyTorch event-memory path is portable and testable. A production CUDA extension would replace the sort/search/gather internals in `ravel_lm/ravel_memory.py` for higher throughput.
- The container has no CUDA and no internet dataset access, so verification used offline toy data. TinyStories streaming is implemented through Hugging Face `datasets` and can run wherever that package and dataset access are available.
