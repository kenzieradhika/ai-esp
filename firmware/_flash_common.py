"""Shared bits for the Windows flashing helpers.

Both `flash_windows.py` and `flash_download_loop.py` locate the Arduino build
output, the bundled esptool, and the partition-table generator. Keeping those in
one place means a version bump or a different Arduino install is fixed once.
"""

import glob
import os

APPDATA = os.environ.get("LOCALAPPDATA", "")
ESPTOOL = os.path.join(
    APPDATA, "Arduino15", "packages", "esp32", "tools", "esptool_py", "5.3.1", "esptool.exe"
)
GEN_PART = os.path.join(
    APPDATA, "Arduino15", "packages", "esp32", "hardware", "esp32", "3.3.11",
    "tools", "gen_esp32part.py",
)


def latest_build():
    """Newest arduino_build_* temp dir that holds a compiled app binary."""
    builds = sorted(glob.glob(os.path.join(APPDATA, "Temp", "arduino_build_*")),
                    key=os.path.getmtime, reverse=True)
    for b in builds:
        if os.path.exists(os.path.join(b, "esp32_llm.ino.bin")):
            return b
    return None
