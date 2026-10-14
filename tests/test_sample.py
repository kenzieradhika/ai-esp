"""sample.py CLI guards. These fire before any checkpoint is loaded, so a dummy
--run is enough."""

import sys

import pytest

import sample


def test_main_rejects_zero_samples(monkeypatch):
    # range(0) prints the header and exits 0 with no output, which reads as a
    # broken model rather than a bad flag. Must be a usage error up front.
    monkeypatch.setattr(sys, "argv", ["sample.py", "--run", "nope.pt", "--samples", "0"])
    with pytest.raises(SystemExit):
        sample.main()
