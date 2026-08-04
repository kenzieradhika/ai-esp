"""Enter download mode by pulsing EN via RTS while the user holds BOOT.

RTS is wired to EN on this board, so we can reset the chip from software.
With GPIO46 jumpered to GND and BOOT held (GPIO0 low), the chip boots into
DOWNLOAD_BOOT (boot:0x3) and waits forever for esptool.

Usage:
    python firmware/reset_download.py [COM port]
"""

import serial
import sys
import time

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM6"

print(">>> TAHAN tombol BOOT sekarang, jangan lepas sampai saya bilang <<<")
time.sleep(3)

ser = serial.Serial(PORT, 115200, timeout=0.2)
ser.setDTR(False)
ser.setRTS(True)   # EN low -> chip in reset
time.sleep(0.3)
ser.setRTS(False)  # release EN -> boot with GPIO0 low (BOOT held), GPIO46 low (jumper)

log = b""
deadline = time.time() + 8
while time.time() < deadline:
    chunk = ser.read(4096)
    if chunk:
        log += chunk
        deadline = time.time() + 1.5

ser.close()
text = log.decode("utf-8", "replace")
print("--- boot log ---")
print(text or "(kosong)")
if "boot:0x3" in text:
    print("RESULT: DOWNLOAD_BOOT (0x3) tercapai. Chip menunggu esptool -- LANJUT.")
elif "boot:0x4" in text:
    print("RESULT: SPI_FLASH_BOOT (0x4) -- GPIO46 masih high? Cek kontak jumper.")
elif "boot:0xc" in text or "boot:0x" in text:
    print("RESULT: normal boot -- GPIO0 tidak low saat reset (BOOT tidak ditahan?).")
else:
    print("RESULT: tidak ada log terbaca -- cek port/kabel.")
