#!/usr/bin/env bash
# C4 Dreamer loop, step 3 (2026-10-16): run the learned policy in SUMO to collect fresh WM training data.
# Usage (on a CPU box, from the code root):  POLICY=<policy ckpt.pt> RUN_ID=<id> bash scripts/c4/loop_collect_v6.sh
# Seeds: unused training seeds of the registered C4_V6_LONGPAIR block (965184-965999), TRAIN only.
# Penetrations stay at the training set (0.05/0.1/0.2/0.4) so 0.15/0.3/0.6 remain unseen for evaluation.
set -euo pipefail
: "${POLICY:?POLICY=<policy checkpoint>}" "${RUN_ID:?RUN_ID}"
SEEDS=${SEEDS:-965184-965383}
WORKERS=${WORKERS:-46}
OUT=${OUT_ROOT:-$HOME/c4/loop}/$RUN_ID
PY=${PY:-$HOME/c4env/bin/python}
export PYTHONPATH=${PYTHONPATH:-$PWD/src:$PWD} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -n 65536 || true
mkdir -p "$OUT"
echo "policy=$POLICY sha256=$(sha256sum "$POLICY" | cut -d' ' -f1) seeds=$SEEDS start=$(date -u +%FT%TZ) code=$(cat BOXOPS_CODE_ID 2>/dev/null || git rev-parse HEAD 2>/dev/null)" > "$OUT/LOOP_RUN.txt"
"$PY" -m traffic_shepherd.c4.collect --policy "$POLICY" --population textbook --seeds "$SEEDS" \
  --case-plan weighted-cycle --penetrations 0.05,0.1,0.2,0.4 --pen-weights 1,1,2,2 \
  --demands 1900,2100,2400,2700 --warmup 300 --horizon 900 --workers "$WORKERS" --no-physics-log \
  --out "$OUT/policy" > "$OUT/collect.log" 2>&1
echo "collect_end=$(date -u +%FT%TZ) exit=$?" >> "$OUT/LOOP_RUN.txt"
