"""Config/derived-property and generation invariants.

The architecture switch lives entirely on `Config.arm`, so these properties are
what every other module keys off. A silent change here (e.g. table_width no
longer matching the PLE table, or `uses_per_layer` losing an arm) would quietly
invalidate the ablation without failing any forward pass.
"""

import pytest
import torch

from model import Config, TinyLM


def test_default_config_is_baseline():
    c = Config()
    assert c.arm == "baseline"
    assert not c.uses_per_layer


def test_head_dim_divides_d_model():
    c = Config(d_model=96, n_heads=4)
    assert c.head_dim == 24
    assert c.d_model == c.head_dim * c.n_heads


def test_uses_per_layer_only_for_ple_arms():
    assert Config(arm="ple").uses_per_layer
    assert Config(arm="ple_notable").uses_per_layer
    assert not Config(arm="baseline").uses_per_layer
    assert not Config(arm="fatembed").uses_per_layer


def test_table_width_matches_ple_table_params():
    # fatembed's wide embedding must carry exactly the PLE table's param count.
    c = Config(n_layers=6, ple_dim=64)
    assert c.table_width == 6 * 64


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        Config(d_model=96, n_heads=5)


def test_config_rejects_odd_head_dim():
    # head_dim must be even for split-half RoPE; d_model=96, n_heads=32 -> 3.
    with pytest.raises(ValueError):
        Config(d_model=96, n_heads=32)


def test_bigcore_arm_has_no_table_params():
    c = Config(arm="bigcore", vocab_size=256, d_model=32, n_layers=2,
               n_heads=4, ffn_hidden=64, ple_dim=16)
    assert TinyLM(c).param_budget()["table"] == 0


def test_generate_returns_prompt_plus_new_tokens():
    c = Config(arm="baseline", vocab_size=64, d_model=32, n_layers=1,
               n_heads=4, ffn_hidden=64, seq_len=32, ple_dim=16)
    m = TinyLM(c).eval()
    torch.manual_seed(0)
    prompt = torch.randint(0, 64, (1, 5))
    out = m.generate(prompt, max_new_tokens=7, temperature=1.0, top_k=0)
    assert out.shape == (1, 12)
    # The prompt is preserved verbatim; only new tokens are appended.
    assert torch.equal(out[:, :5], prompt)


def test_generate_clips_context_to_seq_len():
    # With a long prompt, generation must not exceed the trained window: the
    # model slices to the last seq_len tokens each step instead of erroring.
    c = Config(arm="baseline", vocab_size=64, d_model=32, n_layers=1,
               n_heads=4, ffn_hidden=64, seq_len=8, ple_dim=16)
    m = TinyLM(c).eval()
    torch.manual_seed(0)
    prompt = torch.randint(0, 64, (1, 20))
    out = m.generate(prompt, max_new_tokens=3, temperature=1.0, top_k=0)
    assert out.shape == (1, 23)
