"""download() integrity guard.

A dropped connection ends the chunk loop early. Left unchecked, the next stage
trains a tokenizer and token bins on a fraction of the corpus, and every number
downstream is quietly wrong with no error to point at it.
"""

import prepare
import pytest


class FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_download_rejects_a_truncated_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "RAW", str(tmp_path / "raw.txt"))
    monkeypatch.setattr(prepare, "SLICE_BYTES", 1000)
    # Server closes after 100 of 1000 bytes.
    monkeypatch.setattr(prepare.requests, "get",
                        lambda *a, **k: FakeResponse([b"x" * 100]))
    with pytest.raises(RuntimeError, match="rerun"):
        prepare.download()


def test_download_accepts_a_full_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "RAW", str(tmp_path / "raw.txt"))
    monkeypatch.setattr(prepare, "SLICE_BYTES", 1000)
    monkeypatch.setattr(prepare.requests, "get",
                        lambda *a, **k: FakeResponse([b"x" * 1200]))
    prepare.download()
    assert (tmp_path / "raw.txt").stat().st_size == 1200
