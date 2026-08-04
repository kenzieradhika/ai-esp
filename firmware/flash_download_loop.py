"""Flash the ESP32-S3 by looping EN-reset while the user holds BOOT.

This board cannot auto-enter download mode (no DTR->GPIO0, GPIO46 needs a
jumper to GND). Instead of asking the user to time a BOOT+RST gesture, we
pulse EN via RTS repeatedly; the user just KEEPS BOOT held for the whole
run. As soon as one pulse lands while BOOT is held (GPIO0 low + GPIO46 low
via jumper) the chip boots into DOWNLOAD_BOOT and stays there; the next
esptool probe connects and we flash.

Usage:
    python firmware/flash_download_loop.py [COM port] [arduino build folder]
"""

import glob
import os
import subprocess
import sys
import time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
APPDATA = os.environ.get("LOCALAPPDATA", "")
ESPTOOL = os.path.join(
    APPDATA, "Arduino15", "packages", "esp32", "tools", "esptool_py", "5.3.1", "esptool.exe"
)
GEN_PART = os.path.join(
    APPDATA, "Arduino15", "packages", "esp32", "hardware", "esp32", "3.3.11",
    "tools", "gen_esp32part.py",
)
TIMEOUT_S = 600


def latest_build():
    builds = sorted(glob.glob(os.path.join(APPDATA, "Temp", "arduino_build_*")),
                    key=os.path.getmtime, reverse=True)
    for b in builds:
        app = os.path.join(b, "esp32_llm.ino.bin")
        if os.path.exists(app):
            return b
    return None


def pulse_en(port):
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.25)
        ser.setRTS(False)
        ser.close()
        return True
    except Exception as e:
        print("EN pulse gagal:", e)
        return False


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM6"
    build = sys.argv[2] if len(sys.argv) > 2 else latest_build()
    if not build:
        print("No arduino build found. Compile the sketch in Arduino IDE first.")
        return 1
    for f in ("esp32_llm.ino.bootloader.bin", "esp32_llm.ino.bin", "boot_app0.bin"):
        if not os.path.exists(os.path.join(build, f)):
            print("Missing:", os.path.join(build, f))
            return 1

    partitions = os.path.join(HERE, "model", "partitions.bin")
    subprocess.run([sys.executable, GEN_PART,
                    os.path.join(HERE, "esp32_llm", "partitions.csv"), partitions],
                   check=True)

    base = [ESPTOOL, "--chip", "esp32s3", "--port", port, "--baud", "921600",
            "--before", "no-reset", "--after", "hard-reset"]
    probe = base + ["--no-stub", "chip-id"]

    print("=" * 62)
    print("  CARI GPIO46 ASLI + FLASH otomatis")
    print("  [Prasyarat: pin '46' header TERBUKTI tidak tersambung ke chip]")
    print("  >>> TAHAN tombol BOOT terus <<<")
    print("  >>> JALAN DENGAN KAWAT: sentuh satu pin ke GND ~5 detik, ")
    print("      lalu pindah ke pin berikutnya, sampai muncul 'Terkoneksi' <<<")
    print("  (boot:0x3 muncul saat kawat menyentuh GPIO46 sejati;")
    print("   begitu konek, LEPAS semua, flash berjalan otomatis)")
    print("  JANGAN hubungkan 3V3/5V/EN/GND ke GND!")
    print("=" * 62)

    connected = False
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        if not pulse_en(port):
            time.sleep(1)
            continue
        try:
            r = subprocess.run(probe, capture_output=True, text=True,
                               timeout=8)
        except subprocess.TimeoutExpired:
            sys.stdout.write("T"); sys.stdout.flush()
            continue
        out = r.stdout + r.stderr
        if "Chip is ESP32-S3" in out:
            connected = True
            break
        sys.stdout.write("."); sys.stdout.flush()
        time.sleep(1)
    print()

    if not connected:
        print("Gagal dalam %ds. Pastikan jumper GPIO46->GND terpasang "
              "dan BOOT ditahan." % TIMEOUT_S)
        return 1

    print("Terkoneksi! Flashing (jangan lepas BOOT)...")
    flash = base + ["write-flash", "-z", "--flash-mode", "keep",
                    "--flash-freq", "keep", "--flash-size", "keep",
                    "0x0", os.path.join(build, "esp32_llm.ino.bootloader.bin"),
                    "0x8000", partitions,
                    "0xe000", os.path.join(build, "boot_app0.bin"),
                    "0x10000", os.path.join(build, "esp32_llm.ino.bin")]
    r = subprocess.run(flash, capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)
    if r.returncode == 0:
        print("FLASH OK. Sekarang LEPAS tombol BOOT, lalu tekan-lepas RST "
              "sekali untuk boot app. (Jumper GPIO46 bisa tetap terpasang.)")
        return 0
    print("Flash gagal (kode %d)." % r.returncode)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
