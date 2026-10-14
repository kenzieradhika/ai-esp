"""Quantization invariants. The deploy model is stored as group-wise int4 with
fp16 scales; if the Python simulator (src/quantize.py) and the exporter
(src/export.py) disagree, the golden logits no longer describe the flashed
artifact and the host verify check stops meaning anything.
"""

import torch

from export import GROUP, quant_pack
from quantize import quantize_groupwise


def test_quantize_groupwise_preserves_shape_and_dtype():
    w = torch.randn(8, 130)
    dq = quantize_groupwise(w, bits=4, group=64)
    assert dq.shape == w.shape
    assert dq.dtype == w.dtype


def test_quantize_groupwise_error_is_bounded():
    # A group of size `g` has values in [-qmax*scale, qmax*scale]; the per-element
    # error is at most half a step = scale/2 = max|group|/(2*qmax).
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    dq = quantize_groupwise(w, bits=4, group=64)
    qmax = 7
    step = w.abs().amax(dim=1, keepdim=True) / qmax
    assert (w - dq).abs().max() <= (step / 2).max() + 1e-6


def test_quantize_groupwise_zeros_stay_zero():
    w = torch.zeros(4, 32)
    assert torch.count_nonzero(quantize_groupwise(w, group=32)) == 0


def test_quant_pack_ragged_row_bytes():
    w = torch.randn(5, 100)  # 100 cols -> 50 bytes, 1 group at group=128
    packed, scales, dq = quant_pack(w, group=GROUP)
    assert packed.size == 5 * 50
    assert scales.size == 5 * 1
    assert dq.shape == w.shape


def test_quant_pack_odd_cols():
    w = torch.randn(3, 7)  # odd width: last nibble padded
    packed, scales, dq = quant_pack(w, group=4)
    assert packed.size == 3 * 4  # ceil(7/2) = 4
    assert dq.shape == w.shape


def test_quant_pack_error_is_bounded_by_group_scale():
    torch.manual_seed(0)
    w = torch.randn(6, 128)
    _, _, dq = quant_pack(w, group=128)
    step = w.abs().amax(dim=1, keepdim=True) / 7
    assert (w - dq).abs().max() <= (step / 2).max() + 1e-4


def test_val_loss_names_a_too_short_val_bin(tmp_path, monkeypatch):
    # rng.integers(0, len-sl-1) needs a positive bound; a truncated val bin
    # otherwise dies in numpy with an opaque "low >= high".
    import numpy as np
    import pytest

    import quantize

    np.arange(8, dtype=np.uint16).tofile(tmp_path / "val.bin")
    monkeypatch.setattr(quantize, "DATA", str(tmp_path))
    with pytest.raises(ValueError, match="val.bin"):
        quantize.val_loss(None, 4096, "cpu", iters=1, bs=1, sl=16)


def test_quantize_groupwise_rejects_sub_byte_width():
    # bits=0 makes qmax=0 and the scale max|w|/0 -> every weight inf/nan. Fail
    # with a clear message instead of returning a poisoned tensor.
    import pytest

    with pytest.raises(ValueError, match="bits"):
        quantize_groupwise(torch.randn(4, 32), bits=0, group=32)


def test_simulator_and_exporter_agree():
    # Same math, two implementations: the group-64 simulator and the group-64
    # exporter must reconstruct the same weights (fp16 scales differ by design in
    # export.py, so compare at group=64 with fp16 rounding enabled in both).
    torch.manual_seed(1)
    w = torch.randn(4, 128)
    sim = quantize_groupwise(w, bits=4, group=128, fp16_scales=True)
    _, _, exp = quant_pack(w, group=128)
    assert torch.allclose(sim, exp, atol=1e-4)
