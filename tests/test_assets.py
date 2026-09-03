"""The firmware's `vocab.h` and `bpe.h` are generated from the shipped tokenizer
and committed. If the tokenizer is ever retrained without regenerating them, the
device tokenizes with one vocabulary and detokenizes with another -- every reply
comes out as mojibake, with nothing in the Python tests failing. Regenerate into
a temp dir and diff against the committed headers to catch that drift.
"""

import os
import sys

import gen_assets

COMMITTED_VOCAB = os.path.join(gen_assets.HERE, "..", "firmware", "esp32_llm", "vocab.h")
COMMITTED_BPE = os.path.join(gen_assets.HERE, "..", "firmware", "esp32_llm", "bpe.h")


def _regenerate(tmp_path, monkeypatch):
    out_vocab = tmp_path / "vocab.h"
    out_bpe = tmp_path / "bpe.h"
    monkeypatch.setattr(gen_assets, "OUT", str(out_vocab))
    monkeypatch.setattr(gen_assets, "OUT_BPE", str(out_bpe))
    monkeypatch.setattr(sys, "argv", ["gen_assets.py", "--tok", gen_assets.TOK])
    gen_assets.main()
    return out_vocab, out_bpe


def test_committed_vocab_header_matches_the_tokenizer(tmp_path, monkeypatch):
    out, _ = _regenerate(tmp_path, monkeypatch)
    assert out.read_bytes() == open(COMMITTED_VOCAB, "rb").read()


def test_committed_bpe_header_matches_the_tokenizer(tmp_path, monkeypatch):
    _, out = _regenerate(tmp_path, monkeypatch)
    assert out.read_bytes() == open(COMMITTED_BPE, "rb").read()
