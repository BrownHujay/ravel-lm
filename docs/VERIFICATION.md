# Verification log

Generated from commands run in the container before zipping.

## Unit tests

Command:

```bash
pytest -q
```

Output:

```text
........                                                                 [100%]
```

Evidence files:

- `tests/test_tokenizers.py`
- `tests/test_memory.py`
- `tests/test_model.py`

## Parameter counts

Command:

```bash
python scripts/count_params.py --markdown docs/PARAM_COUNTS.md
```

Output:

```text
config                               name                 vocab d_model layers       params
configs/bpe/ravel_1m_bpe1k.json      ravel_1m_bpe1k        1024     120      5    1,021,080
configs/bpe/ravel_300k_bpe1k.json    ravel_300k_bpe1k      1024      68      3      260,508
configs/bpe/ravel_5m_bpe2k.json      ravel_5m_bpe2k        2048     192     11    5,415,360
configs/byte/ravel_1m_byte.json      ravel_1m_byte          260     128      5    1,042,560
configs/byte/ravel_200k_byte.json    ravel_200k_byte        260      68      3      208,556
configs/byte/ravel_3m_byte.json      ravel_3m_byte          260     160     10    3,112,480
configs/byte/ravel_500k_byte.json    ravel_500k_byte        260      80      6      506,960
configs/byte/ravel_5m_byte.json      ravel_5m_byte          260     192     11    5,072,064
configs/byte/ravel_smoke_byte.json   ravel_smoke_byte       260      32      2       32,416
```

## Smoke training

Command:

```bash
python scripts/smoke_train.py
```

Output:

```text
model parameters: 32,416
step 00000 loss 5.5155 ema 5.5155 lr 1.00e-03
step 00001 loss 5.4899 ema 5.5142 lr 1.00e-03
saved final checkpoint to /mnt/data/ravel_lm_project/runs/smoke/final.pt
```

The smoke config is intentionally tiny so it runs on CPU. The real size ladder remains in `configs/byte/` and `configs/bpe/`.

## Generation smoke test

Command:

```bash
python -m ravel_lm.generate --config runs/smoke/config.json --checkpoint runs/smoke/final.pt --tokenizer runs/smoke/tokenizer.json --prompt "Once" --max-new-tokens 5 --temperature 0 --top-k 10
```

Output:

```text
Onceeeeee
```

This only proves the checkpoint/tokenizer/generation path executes; the model is not meaningfully trained after two toy steps.

## Exact memory/flop bench

Command:

```bash
python -m ravel_lm.bench --n 256 --heads 3 --payload-dim 16 --address-space 512
```

Output:

```text
exact_latest_check=True
lookup_time_seconds=0.208929 device=cpu
dense_attention_core_flops_n=256: 33,554,432
ravel_fuse_flops_n=256: 3,145,728
```
