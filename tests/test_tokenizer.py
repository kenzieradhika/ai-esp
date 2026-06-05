"""The shipped tokenizer must satisfy the layout the firmware's bpe.h assumes:
id 0 = <|endoftext|>, ids 1..256 = byte tokens, merge rank r -> id 257+r. If
this drifts, src/gen_assets.py asserts and the device tokenizer breaks.
"""

import json
import os

import pytest

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, "data", "bpe_id32768.json")


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_file(TOK)


def test_shipped_tokenizer_loads(tok):
    assert tok.get_vocab_size() == 32768
    assert tok.token_to_id("<|endoftext|>") == 0


def test_merge_rank_matches_id():
    raw = json.load(open(TOK, encoding="utf-8"))
    merges = raw["model"]["merges"]
    vocab = raw["model"]["vocab"]
    assert len(merges) == 32768 - 257
    for r, (a, b) in enumerate(merges):
        assert vocab[a + b] == 257 + r


def test_byte_tokens_occupy_ids_1_to_256():
    from gen_assets import bytes_to_unicode

    raw = json.load(open(TOK, encoding="utf-8"))
    vocab = raw["model"]["vocab"]
    b2u = bytes_to_unicode()
    ids = sorted(vocab[b2u[b]] for b in range(256))
    assert ids == list(range(1, 257))
