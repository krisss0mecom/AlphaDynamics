#!/bin/bash
# Reproducible aligned five-domain audit benchmark.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1

DEVICE="${DEVICE:-auto}"
STEPS="${STEPS:-4000}"
BATCH="${BATCH:-512}"
OUT_PREFIX="${OUT_PREFIX:-mdcath_aligned5_results}"

python3 src/mdcath_convert_v3.py \
  --bench_dir mdcath_raw \
  --out_dir mdcath_real_data/mdcath_348K \
  --force

python3 src/mdcath_benchmark.py \
  --data_dir mdcath_real_data/mdcath_348K \
  --out_prefix "$OUT_PREFIX" \
  --steps "$STEPS" \
  --batch "$BATCH" \
  --device "$DEVICE"
