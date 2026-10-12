"""Shared bits for the serial helpers that talk to the ESP32-S3.

`flash_windows.py` / `flash_download_loop.py` locate the Arduino build output, the
bundled esptool, and the partition-table generator. `bootlog.py` /
`reset_download.py` pulse EN and capture the boot log. Keeping both in one place
means a version bump or a different Arduino install is fixed once.
"""

import glob
import os
import time

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


def pulse_en_and_read(port, settle_s=0.25, idle_s=1.0, max_s=6.0):
    """Reset the chip via RTS (EN) and return the boot log it prints.

    RTS is wired to EN on this board, so we can reset from software. The log ends
    once the chip has been quiet for `idle_s`, or after `max_s` total.
    """
    import serial  # lazy: keeps this module importable without pyserial installed

    ser = serial.Serial(port, 115200, timeout=0.2)
    try:
        ser.setDTR(False)
        ser.setRTS(True)   # EN low -> chip in reset
        time.sleep(settle_s)
        ser.setRTS(False)  # release EN -> boot

        log = b""
        deadline = time.time() + max_s
        while time.time() < deadline:
            chunk = ser.read(4096)
            if chunk:
                log += chunk
                deadline = time.time() + idle_s
        return log
    finally:
        # Close even if the port dies mid-reset (unplugged board), so the handle
        # is not leaked and a retry can reopen it.
        ser.close()
