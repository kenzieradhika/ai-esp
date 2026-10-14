#!/usr/bin/env bash
# Core-matched ablation. Every arm gets the same dense/SRAM parameter budget;
# they differ only in how (and whether) extra sparse/flash params are spent.
#
# set +e on purpose: each arm writes its own runs/<name>.json, so a single
# failure must lose only that arm, not the rest of the sweep (matching
# overnight.sh). `set -u`/pipefail stay on.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-3000}
CORE=${CORE:-1500000}
SEED=${SEED:-0}
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

for arm in baseline ple ple_notable fatembed bigcore; do
  log "=== $arm (seed $SEED) ==="
  uv run python src/train.py --arm "$arm" --steps "$STEPS" --target-core "$CORE" \
      --seed "$SEED" 2>&1 || log "!! FAILED: $arm (seed $SEED)"
done
