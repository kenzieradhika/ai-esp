"""flash_windows.py probe loop.

The probe is run against a chip that is still booting while the user holds BOOT,
so a hanging `esptool chip-id` is the normal case, not an error. A raised
TimeoutExpired must be retried, not bubble out as a traceback.
"""

import subprocess
import sys

import flash_windows


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_probe_timeout_is_retried(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    for f in ("esp32_llm.ino.bootloader.bin", "esp32_llm.ino.bin", "boot_app0.bin"):
        (build / f).write_bytes(b"x")

    monkeypatch.setattr(flash_windows, "ESPTOOL", sys.executable)
    monkeypatch.setattr(flash_windows, "GEN_PART", str(tmp_path / "gen.py"))
    (tmp_path / "gen.py").write_text("")
    monkeypatch.setattr(sys, "argv", ["flash_windows.py", "COM9", str(build)])

    probes = {"n": 0}

    def fake_run(cmd, **kwargs):
        if "chip-id" in cmd:
            probes["n"] += 1
            if probes["n"] == 1:
                raise subprocess.TimeoutExpired(cmd, 20)
            return Result(stdout="Chip is ESP32-S3")
        return Result()

    monkeypatch.setattr(flash_windows.subprocess, "run", fake_run)
    assert flash_windows.main() == 0
    assert probes["n"] == 2
