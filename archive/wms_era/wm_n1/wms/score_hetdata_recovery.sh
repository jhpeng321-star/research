#!/bin/bash
# WM-S2 held-out per-driver recovery scoring (CPU only, gpuA box).
#
# Materialises state records from the held-out VAL/TEST corpus packs and scores
# the trained driver-parameter encoder against the dense per-run truth:
#   per-parameter MAE/RMSE/bias + rank correlation + per anchor class.
# No SUMO, no GPU; one state at a time; packs memmapped (mmap_mode='r').
#
# Prereq (run first): bash wms/build_dense_truth.sh (or point SIDECAR at a hetpack
# driver_params_hetdata_{val,test} bundle directory).
#
# Usage (from the extracted code tree, e.g. $C/output/wd01/wm_n1):
#   CKPT=/root/autodl-tmp/wms_s2_20261006/work/full/wm_s.pt \
#   bash wms/score_hetdata_recovery.sh
set -euo pipefail

PY=${PY:-/root/autodl-tmp/gpu1/env/rd_gpu/bin/python}
C=${C:-/root/autodl-tmp/wms_s2_20261006/work/code/output/wd01/wm_n1}
HP3=${HP3:-/root/autodl-tmp}
CKPT=${CKPT:?set CKPT to the trained checkpoint, e.g. .../work/full/wm_s.pt}
SIDECAR=${SIDECAR:-/root/autodl-tmp/dense_truth_20261007}   # dense dir OR bundle dir
OUT=${OUT:-/root/autodl-tmp/wms_s2_20261006/work/recovery}
PHYSICS=${PHYSICS:-}
cd "$C"

for SPLIT in VAL TEST; do
  lc=$(echo "$SPLIT" | tr 'A-Z' 'a-z')
  PACK=${PACK:-$HP3/hetpack3_$SPLIT/pack_hetdata_$lc}
  echo "[recovery] $SPLIT pack=$PACK sidecar=$SIDECAR"
  "$PY" -m wms_eval materialise --pack "$PACK" \
    --out "$OUT/states_$lc.json"
  "$PY" -m wms_eval driver --states "$OUT/states_$lc.json" \
    --ckpt "$CKPT" --sidecar "$SIDECAR" --out "$OUT/$lc" \
    ${PHYSICS:+--physics "$PHYSICS"}
done
echo "[recovery] done -> $OUT"
