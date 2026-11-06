#!/usr/bin/env bash
# WM-D qualification on gpuA (runs ON gpuA). One run = one NEW output dir; never overwrites.
#   bash gpua_qual.sh CODE_DIR DATA_DIR OUT_DIR CHECKPOINT TRAIN_MANIFEST [PHYSICS_TOML] [EXTRA ARGS...]
# CODE_DIR  = WM-E deployed code (identical src/wms) + WM-D tools/wzqual overlay
# DATA_DIR  = tapes/ identity/ labels_snapshot/ moments/
# PHYSICS_TOML (baseline only): training TOML whose build_physics() + flat config replace the
#           checkpoint's missing physics.json (WM2); '-' = use the checkpoint's own physics.json.
# Own PIDs only (RUNNER_PID); nice 10; GPU shared with WM-E (WM-E has priority).
set -euo pipefail
CODE=${1:?CODE}; DATA=${2:?DATA}; OUT=${3:?OUT}; CKPT=${4:?CHECKPOINT}; MAN=${5:?TRAIN_MANIFEST}
TOML=${6:--}; shift 6 || shift $#
PY=${WZQ_PYTHON:-/root/miniconda3/bin/python}
[ -e "$OUT" ] && { echo "REFUSED: $OUT exists"; exit 3; }
mkdir -p "$(dirname "$OUT")"
cd "$CODE"
export PYTHONPATH="$CODE/src:$CODE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WMS_MAP_SPEC="$CODE/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json"
ARGS=(--out "$OUT" --tapes "$DATA/tapes/runs/*/s*/prefix/TAPE.npz" --identity "$DATA/identity/IDENTITY.json"
      --labels-dir "$DATA/labels_snapshot" --moments "$DATA/moments/*.json" --backend wm --device cuda
      --checkpoint "$CKPT" --train-worlds "$MAN" --map-spec "$WMS_MAP_SPEC")
[ "$TOML" != "-" ] && ARGS+=(--physics-toml "$CODE/$TOML")
LOG="${OUT}.log"
setsid nohup nice -n 10 "$PY" -B -m tools.wzqual.run_qual "${ARGS[@]}" "$@" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "${OUT}.pid"
sleep 2
echo "STARTED pid=$PID start_ticks=$(awk '{print $22}' /proc/$PID/stat 2>/dev/null) out=$OUT log=$LOG"
