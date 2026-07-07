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
