"""`analyze.py` turns the ablation run JSONs into the table that answers the
PLE question. The failure mode worth guarding is silent: a run whose `arm` is
not in `ORDER` is dropped from the summary without a word, so the numbers can
look complete while missing a whole arm.
"""

import json

import pytest

import analyze


def _run(arm, final_val, core=1_000_000, table=0):
    return {"arm": arm, "final_val": final_val, "params": {"core": core, "table": table}}


def _write_runs(tmp_path, monkeypatch, runs):
    for i, r in enumerate(runs):
        (tmp_path / f"run{i}.json").write_text(json.dumps(r))
    monkeypatch.setattr(analyze, "RUNS", str(tmp_path))


def test_load_groups_runs_by_arm(tmp_path, monkeypatch):
    _write_runs(tmp_path, monkeypatch, [_run("ple", 3.0), _run("ple", 3.2), _run("baseline", 3.5)])
    by_arm = analyze.load()
    assert sorted(by_arm) == ["baseline", "ple"]
    assert [r["final_val"] for r in by_arm["ple"]] == [3.0, 3.2]


def test_mean_of_empty_is_not_called_but_mean_is_plain():
    assert analyze.mean([1, 2, 3]) == 2.0


def test_main_reports_mean_loss_and_run_count(tmp_path, monkeypatch, capsys):
    _write_runs(tmp_path, monkeypatch, [_run("ple", 3.0), _run("ple", 3.2)])
    analyze.main()
    out = capsys.readouterr().out
    assert "3.1000" in out          # mean of 3.0 and 3.2
    assert "+/-0.1000" in out       # half the range across the two seeds


def test_main_warns_about_unknown_arms_instead_of_dropping_them(tmp_path, monkeypatch, capsys):
    _write_runs(tmp_path, monkeypatch, [_run("ple", 3.0), _run("mystery", 1.0)])
    analyze.main()
    out = capsys.readouterr().out
    assert "mystery" in out
    assert "unknown arm" in out


def test_load_names_the_file_when_a_run_has_no_arm(tmp_path, monkeypatch):
    # A killed training run leaves a partial JSON; the bare KeyError('arm') does
    # not say which of the run files is broken.
    (tmp_path / "run0.json").write_text(json.dumps({"final_val": 3.0}))
    monkeypatch.setattr(analyze, "RUNS", str(tmp_path))
    with pytest.raises(KeyError, match="run0.json"):
        analyze.load()


def test_main_says_so_when_there_are_no_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(analyze, "RUNS", str(tmp_path))
    analyze.main()
    assert "no runs found" in capsys.readouterr().out


def test_ple_recovery_line_needs_all_three_arms(tmp_path, monkeypatch, capsys):
    # baseline 3.5, bigcore 3.0 -> gap 0.5; ple 3.2 -> got 0.3 -> 60%.
    _write_runs(tmp_path, monkeypatch,
                [_run("baseline", 3.5), _run("bigcore", 3.0), _run("ple", 3.2)])
    analyze.main()
    assert "recovers 60%" in capsys.readouterr().out
