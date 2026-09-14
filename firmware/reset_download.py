"""Enter download mode by pulsing EN via RTS while the user holds BOOT.

RTS is wired to EN on this board, so we can reset the chip from software.
With GPIO46 jumpered to GND and BOOT held (GPIO0 low), the chip boots into
DOWNLOAD_BOOT (boot:0x3) and waits forever for esptool.

Usage:
    python firmware/reset_download.py [COM port]
"""

import sys
import time

from _flash_common import pulse_en_and_read


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM6"

    print(">>> TAHAN tombol BOOT sekarang, jangan lepas sampai saya bilang <<<")
    time.sleep(3)

    log = pulse_en_and_read(port, settle_s=0.3, idle_s=1.5, max_s=8.0)
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


if __name__ == "__main__":
    main()
