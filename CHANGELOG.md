# Changelog

All notable changes to this project. Format loosely follows Keep a Changelog.

## [Unreleased]

### Added
- `tests/` pytest suite: model accounting/rope invariants, int4 quantizer and
  exporter agreement, `budget.py` ↔ `model.py` tier consistency.
- CI jobs: host C verifier build (`-Werror`) and the Python test suite.
- English TL;DR and a Development section in the README.
- `CONTRIBUTING.md`.

### Fixed
- `make_model("bigcore", ...)` no longer crashes (None config) when even the
  narrowest width exceeds a tight core budget.
- `src/gen_assets.py` defaults to the shipped Indonesian tokenizer
  (`bpe_id32768.json`) and a chat-format demo prompt.
- Unused-function warnings in the C runtime under non-int8 builds; host
  verifiers now compile clean with `-Wall -Wextra`.

### Changed
- `pyproject.toml`: real description, declared `pandas`/`pyarrow`/`pyserial`
  (used by `data/prepare_id.py` and `firmware/prompt_ui.py`), pytest config.
- Stopped tracking `training.log`; `*.log` is ignored.

### Removed
- Dead imports across `src/`, `firmware/`.

## [0.1.0]

Initial public state: 28.9M-parameter PLE model, Indonesian chat fine-tune,
ESP32-S3 firmware, 4-bit export + host golden verifier. See `RESULTS.md`.
