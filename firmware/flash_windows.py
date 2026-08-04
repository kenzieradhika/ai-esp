"""Flash the ESP32-S3 sketch on Windows when DTR auto-reset is broken.

This board's USB-Serial/JTAG does not forward DTR to GPIO0, so esptool's
automatic reset never holds the chip in download mode. The reliable path is
the physical BOOT button: this script loops `--before no-reset` connections
until the chip answers, then flashes bootloader + partitions + app.

Usage:
    python firmware/flash_windows.py [COM port] [arduino build folder]

Press-and-hold BOOT, tap RST/EN, release BOOT while the script is waiting.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

APPDATA = os.environ.get("LOCALAPPDATA", "")
ESPTOOL = os.path.join(
    APPDATA, "Arduino15", "packages", "esp32", "tools", "esptool_py", "5.3.1", "esptool.exe"
)
GEN_PART = os.path.join(
    APPDATA, "Arduino15", "packages", "esp32", "hardware", "esp32", "3.3.11",
    "tools", "gen_esp32part.py",
)


def latest_build():
    builds = sorted(glob.glob(os.path.join(APPDATA, "Temp", "arduino_build_*")),
                    key=os.path.getmtime, reverse=True)
    for b in builds:
        app = os.path.join(b, "esp32_llm.ino.bin")
        if os.path.exists(app):
            return b
    return None


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM6"
    build = sys.argv[2] if len(sys.argv) > 2 else latest_build()
    if not build:
        print("No arduino build found. Compile the sketch in Arduino IDE first.")
        return 1
    if not os.path.exists(ESPTOOL):
        print("esptool not found:", ESPTOOL)
        return 1

    bootloader = os.path.join(build, "esp32_llm.ino.bootloader.bin")
    app = os.path.join(build, "esp32_llm.ino.bin")
    boot_app0 = os.path.join(build, "boot_app0.bin")
    for f in (bootloader, app, boot_app0):
        if not os.path.exists(f):
            print("Missing:", f)
            return 1

    # Rebuild the project's own partition table (custom "model" partition).
    partitions = os.path.join(HERE, "..", "firmware", "model", "partitions.bin")
    subprocess.run([sys.executable, GEN_PART,
                    os.path.join(HERE, "esp32_llm", "partitions.csv"), partitions],
                   check=True)

    base = [ESPTOOL, "--chip", "esp32s3", "--port", port, "--baud", "921600",
            "--before", "no-reset", "--after", "hard-reset"]
    probe = base + ["--no-stub", "chip-id"]

    print("=" * 62)
    print("  Menunggu chip dalam download mode...")
    print("  [Prasyarat: GPIO46 dijumper ke GND]")
    print("  >>> TAHAN tombol BOOT, tekan-lepas RST/EN, LEPAS BOOT <<<")
    print("  (cukup sekali -- chip menunggu di download mode sampai konek)")
    print("=" * 62)

    connected = False
    deadline = time.time() + 120
    while time.time() < deadline:
        r = subprocess.run(probe, capture_output=True, text=True, timeout=20)
        out = r.stdout + r.stderr
        if "Chip is ESP32-S3" in out:
            connected = True
            break
        sys.stdout.write("."); sys.stdout.flush()
        time.sleep(2)
    print()

    if not connected:
        print("Gagal terhubung dalam 120 detik. Periksa kabel dan coba lagi.")
        return 1

    flash = base + ["write-flash", "-z", "--flash-mode", "keep",
                    "--flash-freq", "keep", "--flash-size", "keep",
                    "0x0", bootloader,
                    "0x8000", partitions,
                    "0xe000", boot_app0,
                    "0x10000", app]
    r = subprocess.run(flash, capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)
    if r.returncode == 0:
        print("FLASH OK. Model partition kosong (model.bin masih stub) -- "
              "jalankan src/train.py + src/export.py dulu untuk model asli.")
        return 0
    print("Flash gagal (kode %d)." % r.returncode)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
