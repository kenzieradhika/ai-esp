"""Exporter guards.

`src/export.py` writes the flash artifact and the golden logits the C port is
verified against. It hard-codes the `ple` tensor plan, so exporting any other
arm would KeyError or silently drop a tensor -- both produce a golden that
cannot describe the flashed model. The guard must fire before any file write.
"""

import sys

import pytest
import torch

import export
from model import Config, TinyLM


def _checkpoint(tmp_path, arm):
    cfg = Config(arm=arm, vocab_size=64, d_model=32, n_layers=1, n_heads=4,
                 ffn_hidden=64, seq_len=16, ple_dim=16)
    m = TinyLM(cfg)
    torch.save({"cfg": m.cfg.__dict__, "state": m.state_dict()},
               tmp_path / "run.pt")


@pytest.mark.parametrize("arm", ["baseline", "fatembed", "ple_notable", "bigcore"])
def test_export_rejects_non_ple_arm(tmp_path, monkeypatch, arm):
    _checkpoint(tmp_path, arm)
    monkeypatch.setattr(export, "RUNS", str(tmp_path))
    monkeypatch.setattr(export, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(sys, "argv", ["export.py", "run"])
    with pytest.raises(SystemExit):
        export.main()
    # Nothing written: the guard fires before the binary/golden are opened.
    assert not (tmp_path / "out" / "model.bin").exists()


def test_export_rejects_prompt_ids_past_the_vocab(tmp_path, monkeypatch):
    # The golden prompt is a fixed id list; a checkpoint whose vocab is smaller
    # than the largest id would index the embedding out of range. Fail loudly.
    _checkpoint(tmp_path, "ple")
    monkeypatch.setattr(export, "RUNS", str(tmp_path))
    monkeypatch.setattr(export, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(sys, "argv", ["export.py", "run"])
    with pytest.raises(SystemExit, match="vocab_size"):
        export.main()
    assert not (tmp_path / "out" / "model.bin").exists()


def test_export_names_a_missing_checkpoint(tmp_path, monkeypatch):
    # torch.load on a missing path raises FileNotFoundError pointing at a temp
    # path; the message must name the checkpoint under runs/ instead.
    monkeypatch.setattr(export, "RUNS", str(tmp_path))
    monkeypatch.setattr(export, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(sys, "argv", ["export.py", "absent"])
    with pytest.raises(SystemExit, match="absent.pt"):
        export.main()


def test_export_names_a_foreign_checkpoint(tmp_path, monkeypatch):
    # A .pt that is not one of our checkpoints (no cfg/state) would KeyError at
    # the config line; name the file and the missing keys instead.
    torch.save({"something": 1}, tmp_path / "run.pt")
    monkeypatch.setattr(export, "RUNS", str(tmp_path))
    monkeypatch.setattr(export, "OUT", str(tmp_path / "out"))
    monkeypatch.setattr(sys, "argv", ["export.py", "run"])
    with pytest.raises(SystemExit, match="not a checkpoint"):
        export.main()
