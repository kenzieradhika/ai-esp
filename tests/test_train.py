"""LR schedule invariants. `lr_at` is what every ablation arm trains under; if
the warmup/cosine shape drifts, runs stop being comparable and RESULTS.md's
loss curves no longer describe the committed code.
"""

import math

from train import lr_at


def test_warmup_ramps_from_zero():
    # Step 0 must be the smallest LR (peak/warmup), not peak.
    assert lr_at(0, 100, 1e-3, 10) == 1e-3 / 10
    assert lr_at(0, 100, 1e-3, 10) < lr_at(5, 100, 1e-3, 10) < lr_at(10, 100, 1e-3, 10)


def test_peak_reached_at_end_of_warmup():
    assert math.isclose(lr_at(10, 100, 1e-3, 10), 1e-3, rel_tol=1e-9)


def test_cosine_decays_to_floor():
    # After warmup the schedule is 0.1*peak + 0.9*peak*cos-shaped; the final
    # step lands just above the 0.1*peak floor.
    assert math.isclose(lr_at(99, 100, 1e-3, 10), 1e-3 * 0.1, rel_tol=0.01)


def test_lr_is_monotone_non_increasing_after_peak():
    vals = [lr_at(s, 200, 1e-3, 20) for s in range(20, 200, 7)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:], strict=False))


def test_batcher_shapes_and_alignment(tmp_path, monkeypatch):
    # Batcher reads a uint16 memmap and returns (x, y) where y is x shifted by
    # one token: the training target. If the shift drifts, every arm trains on
    # the wrong objective and RESULTS.md becomes meaningless.
    import numpy as np
    import torch

    import train

    data = np.arange(1000, dtype=np.uint16)
    path = tmp_path / "train.bin"
    data.tofile(path)
    monkeypatch.setattr(train, "DATA", str(tmp_path))

    b = train.Batcher("train", batch_size=4, seq_len=8, device="cpu")
    x, y = b()
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])  # y is x shifted one to the right


def test_batcher_rejects_a_bin_shorter_than_seq_len(tmp_path, monkeypatch):
    # A too-short bin (wrong suffix, or a truncated download) makes the offset
    # draw empty; raise a message naming the bin instead of numpy's "low >= high".
    import numpy as np
    import pytest

    import train

    np.arange(4, dtype=np.uint16).tofile(tmp_path / "train.bin")
    monkeypatch.setattr(train, "DATA", str(tmp_path))
    with pytest.raises(ValueError, match="train.bin"):
        train.Batcher("train", batch_size=4, seq_len=8, device="cpu")


def test_val_batcher_is_deterministic(tmp_path, monkeypatch):
    # The val batcher seeds with a fixed rng so every arm is scored on identical
    # batches; two fresh val batchers must produce the same draw.
    import numpy as np

    import train

    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "val.bin")
    monkeypatch.setattr(train, "DATA", str(tmp_path))

    a = train.Batcher("val", 4, 8, "cpu")()
    b = train.Batcher("val", 4, 8, "cpu")()
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
