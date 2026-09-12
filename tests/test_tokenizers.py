from pathlib import Path

from ravel_lm.tokenizers import ByteTokenizer, BytePairTokenizer
from ravel_lm.data import toy_stories


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "Hello, tiny cat! 🐱"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id
    assert tok.decode(ids) == text
    assert tok.vocab_size == 260


def test_bpe_train_save_load_roundtrip(tmp_path: Path):
    tok = BytePairTokenizer.train(toy_stories(), vocab_size=300, min_pair_count=2)
    text = "The little cat had a tiny cake."
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert max(ids) < tok.vocab_size
    assert tok.decode(ids) == text
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BytePairTokenizer.load(path)
    assert tok2.encode(text, add_bos=True, add_eos=True) == ids
    assert tok2.decode(ids) == text
