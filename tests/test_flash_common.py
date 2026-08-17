"""`firmware/_flash_common.latest_build` picks the newest Arduino build dir that
actually contains a compiled app binary. Both flash scripts default to it, so a
regression (returning a dir without the .bin, or the wrong dir) makes flashing
fail with a confusing "Missing:" message instead of a clear cause.
"""

import os

import _flash_common


def _mkbuild(root, name, with_bin):
    d = root / name
    d.mkdir()
    if with_bin:
        (d / "esp32_llm.ino.bin").write_bytes(b"x")
    return d


def test_picks_newest_build_with_a_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(_flash_common, "APPDATA", str(tmp_path))
    tmp = tmp_path / "Temp"
    tmp.mkdir()
    _mkbuild(tmp, "arduino_build_100", with_bin=True)
    newest = _mkbuild(tmp, "arduino_build_200", with_bin=True)
    os.utime(tmp / "arduino_build_100", (1, 1))  # make ordering unambiguous
    os.utime(newest, (2, 2))
    assert _flash_common.latest_build() == str(newest)


def test_skips_newest_dir_without_a_binary(tmp_path, monkeypatch):
    # A newer build dir that failed to produce the app must not shadow an older
    # usable one -- otherwise the flash silently targets nothing.
    monkeypatch.setattr(_flash_common, "APPDATA", str(tmp_path))
    tmp = tmp_path / "Temp"
    tmp.mkdir()
    usable = _mkbuild(tmp, "arduino_build_100", with_bin=True)
    empty = _mkbuild(tmp, "arduino_build_200", with_bin=False)
    os.utime(usable, (1, 1))
    os.utime(empty, (2, 2))
    assert _flash_common.latest_build() == str(usable)


def test_returns_none_when_no_builds(tmp_path, monkeypatch):
    monkeypatch.setattr(_flash_common, "APPDATA", str(tmp_path))
    assert _flash_common.latest_build() is None
