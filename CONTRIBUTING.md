# Contributing

Thanks for looking. This is a small research repo; the bar is "don't break the
measurements".

## Ground rules

- **Keep the accounting intact.** `model.py:param_budget()` and `budget.py` are
  what make the PLE-vs-baseline comparison fair. Tests in `tests/test_budget.py`
  and `tests/test_model.py` guard this. If you change tier semantics, say so in
  the PR and update `RESULTS.md`.
- **Any change to quantization or the export format must keep the host golden
  passing.** `src/quantize.py` and `src/export.py` have to agree; `tests/test_quantize.py`
  checks that they do.
- **The C runtime matches `src/model.py` op-for-op.** Verify before flashing:
  build `firmware/host_verify/verify.c` and confirm it prints `PASS`.

## Before you push

```bash
make check
```

Runs lint, the pytest suite, and the host C checks in one shot — the same gate
CI enforces. Both must be clean. CI also builds the firmware with arduino-cli.

## Style

- Python: standard library first, no new dependency unless it clearly earns its
  place. Comments explain *why*, not *what*.
- C: single-header, no dynamic allocation in `llm.h` itself (the caller owns
  scratch). Keep host and device builds warning-free.
- Numbers in `RESULTS.md` must be reproducible from a committed script.
