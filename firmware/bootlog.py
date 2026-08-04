"""Pulse EN via RTS and capture the boot log. Tells us the chip's state.

Usage:
    python firmware/bootlog.py [COM port] [hold_seconds]
"""

import serial
import sys
import time

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM6"

ser = serial.Serial(PORT, 115200, timeout=0.2)
ser.setDTR(False)
ser.setRTS(True)   # reset asserted
time.sleep(0.25)
ser.setRTS(False)  # release -> boot

log = b""
deadline = time.time() + 6
while time.time() < deadline:
    chunk = ser.read(4096)
    if chunk:
        log += chunk
        deadline = time.time() + 1.0

ser.close()
text = log.decode("utf-8", "replace")
print("--- boot log ---")
print(text or "(kosong)")
