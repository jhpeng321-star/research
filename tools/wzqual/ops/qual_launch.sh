#!/usr/bin/env bash
# WM-D qualification shard launcher (runs ON gpuA or cpuV). One shard = one NEW output dir.
#   bash qual_launch.sh CODE TAPES_GLOB DATA OUT CKPT MANIFEST DEVICE THREADS TOML [run_qual args...]
# TAPES_GLOB e.g. '/home/ubuntu/wmqual_20261106/runs/*/s*/prefix/TAPE.npz'; DATA has identity/
# (or IDENTITY.json via --identity in args), labels_snapshot/, moments/. TOML '-' = checkpoint physics.
# Own PID only (OUT.pid); nice 10. Never signals other processes.
set -euo pipefail
CODE=${1:?}; TAPES=${2:?}; DATA=${3:?}; OUT=${4:?}; CKPT=${5:?}; MAN=${6:?}; DEV=${7:?}; THR=${8:?}; TOML=${9:?}
shift 9
PY=${WZQ_PYTHON:-python3}
[ -e "$OUT" ] && { echo "REFUSED: $OUT exists"; exit 3; }
mkdir -p "$(dirname "$OUT")"
cd "$CODE"
export PYTHONPATH="$CODE/src:$CODE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR
export WMS_MAP_SPEC="$CODE/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json" WZQ_THREADS=$THR
ARGS=(--out "$OUT" --tapes "$TAPES" --labels-dir "$DATA/labels_snapshot" --moments "$DATA/moments/*.json"
      --backend wm --device "$DEV" --checkpoint "$CKPT" --train-worlds "$MAN" --map-spec "$WMS_MAP_SPEC")
[ "$TOML" != "-" ] && ARGS+=(--physics-toml "$CODE/$TOML")
setsid nohup nice -n 10 "$PY" -B -m tools.wzqual.run_qual "${ARGS[@]}" "$@" >> "${OUT}.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "${OUT}.pid"
sleep 2
echo "STARTED pid=$PID start_ticks=$(awk '{print $22}' /proc/$PID/stat 2>/dev/null) out=$OUT"
