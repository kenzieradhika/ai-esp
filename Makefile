# Convenience targets. The Python side runs from src/ (flat imports); the C
# side is verified on the host before anything is flashed.

.PHONY: help test lint verify check build clean

help:
	@echo "make test    - run the pytest suite"
	@echo "make lint    - run ruff over the Python sources"
	@echo "make verify  - build + run the host C runtime checks"
	@echo "make check   - lint + test + verify (everything CI runs)"
	@echo "make build   - compile the ESP32-S3 sketch (arduino-cli)"
	@echo "make clean   - remove __pycache__ and test binaries"

test:
	uv run pytest -q

lint:
	uv run ruff check src data firmware tests

# One command for the same gate CI enforces: lint, then the Python suite, then
# the host C checks. Ordered so the cheap failure comes first.
check: lint test verify

verify:
	cc -O3 -Wall -Wextra -Werror -o /tmp/esp-verify firmware/host_verify/verify.c -lm
	cc -O3 -Wall -Wextra -Werror -o /tmp/esp-ppl-fp32 firmware/host_verify/ppl.c -lm
	cc -O3 -Wall -Wextra -Werror -DLLM_INT8_ACT -o /tmp/esp-ppl-int8 firmware/host_verify/ppl.c -lm
	cc -O3 -Wall -Wextra -Werror -DLLM_INT8_ACT -o /tmp/esp-kernels firmware/host_verify/test_kernels.c -lm
	/tmp/esp-kernels

build:
	arduino-cli compile \
	  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
	  --build-property compiler.optimization_flags=-O3 \
	  firmware/esp32_llm

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f /tmp/esp-verify /tmp/esp-kernels /tmp/esp-ppl-fp32 /tmp/esp-ppl-int8
