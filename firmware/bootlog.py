"""Pulse EN via RTS and capture the boot log. Tells us the chip's state.

Usage:
    python firmware/bootlog.py [COM port]
"""

import sys

from _flash_common import pulse_en_and_read


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM6"
    log = pulse_en_and_read(port)
    text = log.decode("utf-8", "replace")
    print("--- boot log ---")
    print(text or "(kosong)")


if __name__ == "__main__":
    main()
