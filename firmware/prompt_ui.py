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


class PortHolder:
    """Shared mutable reference to the open Serial object. The reader thread
    swaps it when USB re-enumerates; the main thread always writes through it,
    so it never touches a stale, closed port."""

    def __init__(self, ser):
        self.ser = ser


def write_line(holder, data):
    """Write with retry: if the port died (board reset), wait for the reader
    thread to reopen it and then retry against the fresh object."""
    for _ in range(30):
        try:
            holder.ser.write(data)
            return
        except (serial.SerialException, OSError):
            time.sleep(1)
    raise SystemExit("[prompt-ui] port tidak merespons, keluar.")


class Reader(threading.Thread):
    """Serial -> stdout, and signals when a prompt window opens."""

    def __init__(self, holder, port_name):
        super().__init__(daemon=True)
        self.holder = holder
        self.port_name = port_name
        self.prompt_open = threading.Event()
        self._tail = b""

    def run(self):
        while True:
            try:
                chunk = self.holder.ser.read(4096)
            except Exception:
                # USB CDC re-enumerates on reset; reopen the port and continue.
                time.sleep(1)
                try:
                    self.holder.ser.close()
                except Exception:
                    pass
                try:
                    self.holder.ser = open_port(self.port_name, tries=30)
                except Exception:
                    continue
                continue
            if not chunk:
                continue
            try:
                # Write raw bytes: avoids Windows cp1252 UnicodeEncodeError on
                # any UTF-8 (emojis etc.) the model emits.
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
            except Exception:
                pass
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
    holder = PortHolder(ser)
    reader = Reader(holder, port)
    reader.start()
    time.sleep(0.5)
    try:
        write_line(holder, b"\n")  # force a fresh prompt window so we never miss the marker
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
            write_line(holder, line.encode("utf-8") + b"\n")
    except KeyboardInterrupt:
        print("\n[prompt-ui] selesai.")
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
