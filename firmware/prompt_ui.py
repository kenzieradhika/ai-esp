"""CMD interface to prompt the ESP32-S3 PLE TinyLM over USB serial.

The firmware prints "prompt> " and waits for one line. This tool watches the
serial stream, prompts you for a line, sends it, and shows the replies (which
are also drawn on the OLED). After each reply the firmware opens a new prompt
window and the cycle repeats.

On connect it sends an empty line: if the ESP is already parked at "prompt> "
from before we attached (the marker scrolls past us), the empty line forces a
fresh "prompt> " into the stream so the interface always syncs immediately.

Usage:
    python firmware/prompt_ui.py [COM port]

Port is auto-detected (Espressif USB device) when omitted. Ctrl+C to quit.
"""

import os
import sys
import threading
import time

import serial
import serial.tools.list_ports

BAUD = 115200
PROMPT_MARK = b"prompt> "
READ_TIMEOUT = 0.25


def find_port():
    env = os.environ.get("ESP32_PORT")
    if env:
        return env
    for p in serial.tools.list_ports.comports():
        if "303A" in p.hwid:  # Espressif (ESP32-S3 native USB)
            return p.device
    for p in serial.tools.list_ports.comports():
        if "usb" in p.device.lower() or "com" in p.device.lower():
            return p.device
    return None


def open_port(name, tries=30):
    for _ in range(tries):
        try:
            return serial.Serial(name, BAUD, timeout=READ_TIMEOUT)
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"tidak bisa membuka port {name}. Cek kabel/board.")


class Reader(threading.Thread):
    """Serial -> stdout, and signals when a prompt window opens."""

    def __init__(self, ser, port_name):
        super().__init__(daemon=True)
        self.ser = ser
        self.port_name = port_name
        self.prompt_open = threading.Event()
        self._tail = b""

    def run(self):
        while True:
            try:
                chunk = self.ser.read(4096)
            except Exception:
                # USB CDC re-enumerates on reset; reopen the port and continue.
                time.sleep(1)
                try:
                    self.ser.close()
                except Exception:
                    pass
                try:
                    self.ser = open_port(self.port_name, tries=30)
                except Exception:
                    continue
                continue
            if not chunk:
                continue
            sys.stdout.write(chunk.decode("utf-8", "replace"))
            sys.stdout.flush()
            self._tail = (self._tail + chunk)[-32:]
            if self._tail.endswith(PROMPT_MARK):
                self.prompt_open.set()


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_port()
    if not port:
        raise SystemExit("ESP32 tidak terdeteksi. Berikan port: python firmware/prompt_ui.py COM4")
    print(f"[prompt-ui] port: {port} @ {BAUD} baud (Ctrl+C untuk keluar)")
    print("[prompt-ui] menyinkronkan dengan ESP32...")

    ser = open_port(port)
    reader = Reader(ser, port)
    reader.start()
    time.sleep(0.5)
    try:
        ser.write(b"\n")  # force a fresh prompt window so we never miss the marker
    except Exception:
        pass

    try:
        while True:
            reader.prompt_open.wait()
            reader.prompt_open.clear()
            try:
                line = input("Anda> ")
            except EOFError:
                break
            if not line.strip():
                continue
            ser.write(line.encode("utf-8") + b"\n")
    except KeyboardInterrupt:
        print("\n[prompt-ui] selesai.")
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
