#!/usr/bin/env bash
# WM-D worker: run qualification shards k/N one after another (each its own NEW out dir RUN/sh_k;
# a shard whose WM_QUAL_METRICS.json exists is skipped, a half-done dir is kept and re-run as sh_k_rN).
#   bash shard_loop.sh WORKER_TAG "K1 K2 ..." N CODE TAPES DATA RUN CKPT MANIFEST DEVICE THREADS TOML [run_qual args...]
# Start it detached:  setsid nohup bash shard_loop.sh ... > RUN/WORKER_TAG.log 2>&1 < /dev/null &
# Own PIDs only; nice WZQ_NICE (default 19); WZQ_GPU_FRACTION caps CUDA memory for cuda workers.
set -uo pipefail
TAG=${1:?}; LIST=${2:?}; N=${3:?}; CODE=${4:?}; TAPES=${5:?}; DATA=${6:?}; RUN=${7:?}; CKPT=${8:?}; MAN=${9:?}
DEV=${10:?}; THR=${11:?}; TOML=${12:?}
shift 12
PY=${WZQ_PYTHON:-python3}
mkdir -p "$RUN"
echo "$$" > "$RUN/$TAG.loop.pid"
cd "$CODE"
export PYTHONPATH="$CODE/src:$CODE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR
export WMS_MAP_SPEC="$CODE/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json" WZQ_THREADS=$THR
for k in $LIST; do
  done_dir=$(ls -d "$RUN"/sh_${k} "$RUN"/sh_${k}_r* 2>/dev/null | while read d; do [ -f "$d/WM_QUAL_METRICS.json" ] && echo "$d"; done | head -1)
  if [ -n "$done_dir" ]; then echo "$(date +%FT%T) skip $k (done in $done_dir)"; continue; fi
  OUT="$RUN/sh_$k"; r=1
  while [ -e "$OUT" ]; do OUT="$RUN/sh_${k}_r$r"; r=$((r+1)); done
  ARGS=(--out "$OUT" --tapes "$TAPES" --labels-dir "$DATA/labels_snapshot" --moments "$DATA/moments/*.json"
        --backend wm --device "$DEV" --checkpoint "$CKPT" --train-worlds "$MAN" --map-spec "$WMS_MAP_SPEC"
        --moment-shard "$k/$N" --start-shard "$k/$N")
  [ "$TOML" != "-" ] && ARGS+=(--physics-toml "$CODE/$TOML")
  echo "$(date +%FT%T) start shard $k -> $OUT"
  nice -n "${WZQ_NICE:-19}" "$PY" -B -m tools.wzqual.run_qual "${ARGS[@]}" "$@" > "$OUT.log" 2>&1
  echo "$(date +%FT%T) end shard $k rc=$?"
done
echo "$(date +%FT%T) worker $TAG finished"
