#!/bin/bash
# WM-S2 dense per-driver truth post-processing (CPU only, gpuA box).
#
# Reads the raw heterogeneous-driver runs (/root/autodl-tmp/hetdata_runs) and
# writes a dense per-run slot -> vehicle-id -> truth-params sidecar.  This is an
# OFFLINE post-processing step; it launches no SUMO and no GPU work, and it never
# reads CAL/HOLDOUT/sealed/chunk-5.  Memory rule: one run per worker, arrays
# memmapped; keep --workers low enough that workers * per-run RAM stays < 6 GB
# (a 1200 s run is ~2 MB of slot table).
#
# Usage (from the extracted code tree, e.g. $C/output/wd01/wm_n1):
#   bash wms/build_dense_truth.sh
set -euo pipefail

PY=${PY:-/root/autodl-tmp/gpu1/env/rd_gpu/bin/python}
RUNS=${RUNS:-/root/autodl-tmp/hetdata_runs}
OUT=${OUT:-/root/autodl-tmp/dense_truth_20261007}
WORKERS=${WORKERS:-4}
KMAX=${KMAX:-192}

echo "[dense-truth] runs=$RUNS out=$OUT workers=$WORKERS k_max=$KMAX"
"$PY" -m wms.dense_truth build \
  --runs-root "$RUNS" --out "$OUT" --k-max "$KMAX" --workers "$WORKERS"
"$PY" -m wms.dense_truth verify --out "$OUT"
echo "[dense-truth] done -> $OUT"
