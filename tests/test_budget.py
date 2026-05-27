"""Budget-model agreement: src/budget.py reports the same tier sizes that
model.py's param_budget() computes. They are deliberately separate files (see
budget.py's docstring), so this test is the only thing keeping them in sync.
"""

from budget import tiers
from model import Config, TinyLM

VOCAB, D, L, P, F, H = 4096, 128, 6, 64, 187, 4


def _cfg(arm):
    return Config(arm=arm, vocab_size=VOCAB, d_model=D, n_layers=L, n_heads=H,
                  ffn_hidden=F, ple_dim=P)


def test_stream_and_table_match_model():
    t = tiers(VOCAB, D, L, P, F, n_heads=H)
    b = TinyLM(_cfg("ple")).param_budget()
    assert t["stream"] == b["stream"]
    assert t["table"] == b["table"]


def test_core_close_to_model():
    # budget.py is an analytic estimate (it lumps qkv+proj as 4*d*d and omits
    # tied-weight nuances); assert same order, not exact equality.
    t = tiers(VOCAB, D, L, P, F, n_heads=H)
    b = TinyLM(_cfg("ple")).param_budget()
    assert abs(t["core"] - b["core"]) / b["core"] < 0.05
