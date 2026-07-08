---
name: Bug report
about: Something in the model, the C runtime, or the firmware is wrong
labels: bug
---

## What happens

<!-- What did you run, what did you expect, what did you get? -->

## Reproduce

```bash
# exact command(s)
```

## Where

- [ ] `src/` training / quantization / export (Python)
- [ ] `firmware/common/llm.h` or a `firmware/host_verify/` check (C)
- [ ] `firmware/esp32_llm/` sketch (device)
- [ ] `data/` preparation
- [ ] docs / tooling

## Checks

- [ ] `uv run pytest -q` passes (paste the summary line)
- [ ] `cc -O3 -Wall -Wextra -Werror -o /tmp/verify firmware/host_verify/verify.c -lm` builds clean

## Environment

- Python / torch version:
- Board + core version (if device): ESP32-S3 N16R8, arduino-esp32 ...
- OS:
