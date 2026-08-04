# ESP32-S3 on-chip inference

This sketch runs the 28.9M-parameter PLE TinyLM on an ESP32-S3 N16R8. The model
lives in the custom `model` flash partition at `0x110000`; the tied
embedding/output head is staged in PSRAM at boot.

The firmware now talks back: it runs an **Indonesian chat model** (format
`user: <you>\nJawaban:`) with a small word memory in flash. See
[Chat and memory](#chat-and-memory) below.

## Windows: serial chat UI

```bat
cd D:\esp32-ai
.venv\Scripts\python firmware\prompt_ui.py          :: auto-detects the port
.venv\Scripts\python firmware\prompt_ui.py COM7     :: or pass it explicitly
```

Type a line and press Enter. The device prints `thinking...` while the model
runs, then the reply. Ctrl+C to exit. Anything you type that is not a command
is fed to the model.

## Chat and memory

Every line you send is matched against these patterns first:

| pattern | effect |
|---|---|
| `story: <topic>` | generates a story, prompting `Ceritakan tentang {name} yang suka {like}. {topic}` (200 tokens, ~37 s) |
| `nama aku <X>` / `panggil aku <X>` / `namaku <X>` / `my name is <X>` | saves your name to NVS (key `\|\|nama\|\|`) |
| `aku suka <Y>` | saves a preference to NVS (key `\|\|suka\|\|`) |
| `<X> artinya <Y>` / `<X> adalah <Y>` / `<X> means <Y>` / `<X>=<Y>` | stores the meaning of word X (key `\|\|X\|\|`) |
| blank or whitespace-only line | ignored (sync line) |
| anything else | sent to the model as `user: <line>\nJawaban:` (80 tokens, ~15 s) |

Memory lives in an NVS `Preferences` namespace (`chatmem`, max 24 entries,
FIFO eviction), so it survives reboots and firmware reflashes. Unfamiliar
words in ordinary lines are picked up and remembered automatically, so the
device picks up vocabulary as you talk to it. Display name:
`Kamu bisa belajar kata baru dari percakapan.`.


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
94cdf8cfdfd4a3bb0bf880414f8ca270e0c82775063d30643fe8a37d6efb7acb
```

Expected boot diagnostics for the current artifact:

```text
model: V=32768 D=96 L=6 H=4 F=66 P=128
head staged int8: 2.53 MB
PSRAM free after alloc: ~5100 KB
```

The current runtime measures 102.9ms per model step (9.72 tok/s compute-only);
attached serial runs measure ~9.5 tok/s including output. On-device profile:
57.6ms output head, 25.6ms attention, 8.5ms PLE path, 6.9ms FFN, 4.4ms input.
The head is staged as int8 with int8 activations (host-validated, val perplexity
delta ~0) and is now PSRAM-bandwidth-bound. The fp32 host golden still matches
PyTorch to 1e-5.
