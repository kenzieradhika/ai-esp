# ESP32-S3 on-chip inference

This sketch runs the 28.9M-parameter PLE TinyLM on an ESP32-S3 N16R8. The model
lives in the custom `model` flash partition at `0x110000`; the tied
embedding/output head is staged in PSRAM at boot.

## Build and verify

Export the group-128 ragged-int4 model and verify the portable C runtime first:

```bash
cd src
uv run python export.py
cd ..
cc -O3 -o /tmp/esp32-llm-verify firmware/host_verify/verify.c -lm
/tmp/esp32-llm-verify firmware/model/model.bin firmware/model/golden.txt
```

Build the device firmware with Arduino ESP32 core 3.3.10:

```bash
arduino-cli compile \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build \
  firmware/esp32_llm
```

## Flash and run

Replace the port if the board enumerates under a different device name:

```bash
arduino-cli upload \
  -p /dev/cu.usbmodem2101 \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --input-dir /tmp/esp32-llm-build \
  firmware/esp32_llm

esptool.py --chip esp32s3 --port /dev/cu.usbmodem2101 --baud 921600 \
  write_flash 0x110000 firmware/model/model.bin

arduino-cli monitor -p /dev/cu.usbmodem2101 --config baudrate=115200
```

The model payload only needs reflashing after a new export. Firmware-only
changes can be uploaded without rewriting the model partition.

The model used for the measurements below has SHA-256:

```text
21067f5d78113f6c64a8720b05ff7e5c774dab0276797a522f81a6797253d97c
```

Expected boot diagnostics for the current artifact:

```text
model: V=32768 D=96 L=6 H=4 F=66 P=128
head staged in PSRAM: 1.64 MB
PSRAM free after alloc: 5228 KB
```

The current exact baseline is 139.4ms per model step (7.17 tok/s compute-only)
across two 200-token runs. Attached serial runs measured 5.67-6.22 tok/s
including output. The on-device profile is 93.2ms output head, 26.4ms attention,
12.9ms combined PLE input/path, and 6.9ms FFN per token.
