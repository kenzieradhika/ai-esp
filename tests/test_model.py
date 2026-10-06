"""Model/architecture invariants that the ablation results depend on.

These guard the *accounting* and *plumbing* the whole PLE-vs-baseline
comparison rests on: if core-budget matching or the table/stream split silently
drifts, every number in RESULTS.md becomes incomparable. Fast, CPU-only.
"""

import math

import pytest
import torch

from model import Config, TinyLM, apply_rope, build_rope, load_checkpoint, make_model

SMALL = dict(vocab_size=256, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64,
             seq_len=32, ple_dim=16)


def cfg(arm, **over):
    return Config(**{**SMALL, "arm": arm, **over})


@pytest.mark.parametrize("arm", ["baseline", "ple", "ple_notable", "fatembed"])
def test_forward_shapes(arm):
    m = TinyLM(cfg(arm))
    x = torch.randint(0, SMALL["vocab_size"], (2, SMALL["seq_len"]))
    logits, loss = m(x)
    assert logits.shape == (2, SMALL["seq_len"], SMALL["vocab_size"])
    assert loss is None


def test_forward_loss_finite():
    m = TinyLM(cfg("ple"))
    x = torch.randint(0, SMALL["vocab_size"], (2, SMALL["seq_len"]))
    logits, loss = m(x, x)
    assert loss is not None and math.isfinite(loss.item())


def test_param_budget_sums_to_total():
    for arm in ["baseline", "ple", "ple_notable", "fatembed"]:
        b = TinyLM(cfg(arm)).param_budget()
        assert b["core"] + b["stream"] + b["table"] == b["total"]
        assert b["core"] > 0 and b["stream"] > 0


def test_head_excluded_from_core():
    # The three-tier split exists so a large vocab is "slow, not unaffordable":
    # growing the vocab must grow `stream`, not `core`.
    small = TinyLM(cfg("baseline", vocab_size=256)).param_budget()
    big = TinyLM(cfg("baseline", vocab_size=4096)).param_budget()
    assert big["core"] == small["core"]
    assert big["stream"] > small["stream"]


def test_fatembed_table_matches_ple_table_size():
    # `fatembed` is only a fair comparison if its table has the same param count.
    p = TinyLM(cfg("ple")).param_budget()
    f = TinyLM(cfg("fatembed")).param_budget()
    assert p["table"] == f["table"]


def test_ple_per_layer_branch_starts_as_noop():
    # ple_norm gain is zeroed at init so every arm starts from the same function.
    m = TinyLM(cfg("ple"))
    for blk in m.blocks:
        assert torch.count_nonzero(blk.ple_norm.weight) == 0


def test_make_model_matches_core_budget():
    target = 200_000
    for arm in ["baseline", "ple", "ple_notable", "fatembed"]:
        m = make_model(arm, target, cfg(arm), verbose=False)
        assert m.param_budget()["core"] <= target


def test_make_model_bigcore_tiny_target_does_not_crash():
    m = make_model("bigcore", 0, cfg("baseline"), verbose=False)
    assert m.cfg.d_model >= SMALL["d_model"]


def test_rope_is_rotation_preserving():
    cos, sin = build_rope(8, 8, 10000.0, "cpu")
    x = torch.randn(1, 1, 8, 8)
    y = apply_rope(x, cos, sin)
    # RoPE is a rotation: it preserves the per-head norm.
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_generate_top_k_zero_disables_the_cut():
    # sample.py --top-k 0 means "no cut"; generate must not call topk(0).
    torch.manual_seed(0)
    m = TinyLM(cfg("ple")).eval()
    ids = torch.randint(0, SMALL["vocab_size"], (1, 4))
    out = m.generate(ids, 3, temperature=0.8, top_k=0)
    assert out.shape == (1, 7)


def test_generate_negative_top_k_is_treated_as_no_cut():
    # A negative --top-k is truthy; without the > 0 guard, torch.topk(k<0) raises.
    torch.manual_seed(0)
    m = TinyLM(cfg("ple")).eval()
    ids = torch.randint(0, SMALL["vocab_size"], (1, 4))
    out = m.generate(ids, 3, temperature=0.8, top_k=-5)
    assert out.shape == (1, 7)


def test_generate_temperature_zero_is_greedy():
    torch.manual_seed(0)
    m = TinyLM(cfg("ple")).eval()
    ids = torch.randint(0, SMALL["vocab_size"], (1, 4))
    out = m.generate(ids, 3, temperature=0.0)
    assert out.shape == (1, 7)


def test_load_checkpoint_names_a_missing_key(tmp_path):
    # A truncated or foreign .pt gives a bare KeyError('cfg'); the message must
    # point at the file so the user knows which checkpoint is broken.
    path = tmp_path / "run.pt"
    torch.save({"state": {}}, path)
    with pytest.raises(KeyError, match="run.pt"):
        load_checkpoint(str(path), "cpu")


def test_load_checkpoint_roundtrips(tmp_path):
    # The checkpoint format every consumer reads: {"cfg", "state"} -> eval model
    # with identical weights. A shape/name drift here silently loads garbage.
    m = TinyLM(cfg("ple"))
    path = tmp_path / "run.pt"
    torch.save({"cfg": m.cfg.__dict__, "state": m.state_dict()}, path)
    got = load_checkpoint(str(path), "cpu")
    assert got.cfg.arm == "ple"
    assert not got.training
    for (n, a), (n2, b) in zip(m.state_dict().items(), got.state_dict().items(),
                              strict=True):
        assert n == n2 and torch.equal(a, b)
