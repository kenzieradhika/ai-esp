#!/usr/bin/env bash
# Second seed on the three arms that carry the claims, to size run-to-run noise.
# set +e on purpose: one failed arm must not drop the other two (see overnight.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
for arm in baseline ple fatembed; do
  log "=== $arm (seed 1) ==="
  uv run python src/train.py --arm "$arm" --steps 3000 --target-core 1500000 \
      --seed 1 2>&1 || log "!! FAILED: $arm (seed 1)"
done
