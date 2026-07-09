## What changed

<!-- One paragraph. If it touches the accounting, quantization, or the C
runtime, say so explicitly. -->

## Why

<!-- The problem this solves. Link an issue if there is one. -->

## Verification

- [ ] `uv run pytest -q` passes
- [ ] `cc -O3 -Wall -Wextra -Werror -o /tmp/verify firmware/host_verify/verify.c -lm` builds clean
- [ ] `make verify` prints `kernel tests PASS`
- [ ] If results changed: `RESULTS.md` updated and re-measured from a committed script

## Notes

<!-- Anything a reviewer should know: deliberate simplifications, follow-ups. -->
