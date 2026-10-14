# Changelog

All notable changes to this project. Format loosely follows Keep a Changelog.

## [Unreleased]

### Added
- `tests/` pytest suite: model accounting/rope invariants, int4 quantizer and
  exporter agreement, `budget.py` ↔ `model.py` tier consistency.
- CI jobs: host C verifier build (`-Werror`) and the Python test suite.
- English TL;DR and a Development section in the README.
- `CONTRIBUTING.md`.
- Host kernel unit tests (`firmware/host_verify/test_kernels.c`) covering
  `half2float` (subnormals, max finite, NaN), `deq_row`/`matvec_q` across a
  group boundary, and ranged-vs-full matvec.
- A drift test that regenerates `vocab.h`/`bpe.h` and diffs them against the
  committed headers, so a retrained tokenizer cannot silently desync the device.
- `firmware/_flash_common.py`: shared Arduino build lookup, esptool paths, and
  the EN-pulse boot-log capture used by `bootlog.py` / `reset_download.py`.
- `sample.py --top-k`, matching `chat.py`'s decoding knob.
- `make check`: one target running lint, the pytest suite, and the host C
  checks in that order — the same gate CI enforces.

### Fixed
- `make_model("bigcore", ...)` no longer crashes (None config) when even the
  narrowest width exceeds a tight core budget.
- `src/gen_assets.py` defaults to the shipped Indonesian tokenizer
  (`bpe_id32768.json`) and a chat-format demo prompt.
- Unused-function warnings in the C runtime under non-int8 builds; host
  verifiers now compile clean with `-Wall -Wextra`.
- `try_math`: the digit cap is enforced inside the accumulation loop (a long
  digit run overflowed `long long` first), and spaces around the operator are
  accepted.
- `generate()` treats `top_k <= 0` as "no cut" (a negative top-k reached
  `torch.topk`); `sample_token` no longer writes past `samp_logit`.
- `mem_store`/`mem_begin` no-op the NVS writes when the `chatmem` namespace
  fails to open, and teaching a word no longer stores the marker itself.
- `export.py` rejects a golden prompt whose ids exceed the checkpoint's vocab;
  `ppl.c`/`verify.c` reject empty model files and out-of-range val tokens.
- `prepare.py`/`prepare_id.py` never emit an empty train bin on a tiny corpus
  (`arr[:-0]` inverted the split); `train.py` rejects `--eval-every`/`--eval-iters`
  below 1 and `Batcher` names an over-short bin.
- `analyze.py` names the run file when it has no `arm` key; `quantize.py` prints
  `n/a` instead of dividing by a zero fp32 edge.
- `stage_head_int8` refuses a multi-group or over-wide head instead of
  mis-scaling or overrunning `head_actq`.
- `export.py` names a missing or foreign checkpoint (no `cfg`/`state`) instead of
  a bare `FileNotFoundError`/`KeyError`; `sample.py` rejects `--samples < 1`;
  `chat.py`'s REPL stop words are case-insensitive.
- `prepare.py` fails when the corpus download truncates instead of training on a
  partial slice.
- `firmware/_flash_common.py`: `pulse_en_and_read` keeps a separate `max_s` cap,
  so a chip that never idles for `idle_s` no longer extends the EN-pulse reset
  forever.
- `firmware/flash_windows.py` retries a probe `TimeoutExpired` like
  `flash_download_loop.py`, instead of dying mid-BOOT-hold.
- `llm.h`: `llm_load` rejects an odd `head_dim` (split-half RoPE needs even),
  matching `model.py:Config`.
- Firmware stats and `Config` dims no longer divide by zero on empty/degenerate
  input.

### Changed
- `pyproject.toml`: real description, declared `pandas`/`pyarrow`/`pyserial`
  (used by `data/prepare_id.py` and `firmware/prompt_ui.py`), pytest config.
- Stopped tracking `training.log`; `*.log` is ignored.
- `experiments/run_ablation.sh`, `run_seed1.sh`, and `run_ple_dim_sweep.sh` no
  longer `set -e`: a single failed arm logs and the sweep continues, matching
  `overnight.sh`/`deploy_seed1.sh`. Each arm writes its own run JSON, so one
  failure should lose only that arm.
- CI lints and byte-compiles `tests/` and every `firmware/*.py` helper;
  `make verify` also builds the ppl probe, matching CI.
- README repo structure now lists the C verifier and test directories.

### Removed
- Dead imports across `src/`, `firmware/`.

## [0.1.0]

Initial public state: 28.9M-parameter PLE model, Indonesian chat fine-tune,
ESP32-S3 firmware, 4-bit export + host golden verifier. See `RESULTS.md`.
