#!/usr/bin/env bash
# WM-D queue worker: claims shards k/N atomically (mkdir RUN/claims/k) from a shared list, so any number
# of workers share the queue without duplicates.  Same arguments as shard_loop.sh plus WAIT_PAT:
#   bash queue_loop.sh TAG "K1 K2 ..." N CODE TAPES DATA RUN CKPT MANIFEST DEVICE THREADS TOML WAIT_PIDS [run_qual args]
# WAIT_PIDS (or '-'): a file of PIDs (e.g. the previous generation's in-flight shards) that must
# exit before this worker starts, to avoid GPU memory overlap.
set -uo pipefail
TAG=${1:?}; LIST=${2:?}; N=${3:?}; CODE=${4:?}; TAPES=${5:?}; DATA=${6:?}; RUN=${7:?}; CKPT=${8:?}; MAN=${9:?}
DEV=${10:?}; THR=${11:?}; TOML=${12:?}; WAIT=${13:?}
shift 13
PY=${WZQ_PYTHON:-python3}
mkdir -p "$RUN/claims"
echo "$$" > "$RUN/$TAG.loop.pid"
if [ "$WAIT" != "-" ]; then          # WAIT = file listing PIDs that must exit first
  while :; do
    alive=0
    for p in $(cat "$WAIT"); do [ -d "/proc/$p" ] && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 30
  done
fi
cd "$CODE"
export PYTHONPATH="$CODE/src:$CODE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR
export WMS_MAP_SPEC="$CODE/assets/research/newmap_candidates/wz3to2/MAP_SPEC.json" WZQ_THREADS=$THR
for k in $LIST; do
  mkdir "$RUN/claims/$k" 2>/dev/null || continue
  OUT="$RUN/sh_$k"; r=1
  while [ -e "$OUT" ]; do OUT="$RUN/sh_${k}_r$r"; r=$((r+1)); done
  ARGS=(--out "$OUT" --tapes "$TAPES" --labels-dir "$DATA/labels_snapshot" --moments "$DATA/moments/*.json"
        --backend wm --device "$DEV" --checkpoint "$CKPT" --train-worlds "$MAN" --map-spec "$WMS_MAP_SPEC"
        --moment-shard "$k/$N" --start-shard "$k/$N")
  [ "$TOML" != "-" ] && ARGS+=(--physics-toml "$CODE/$TOML")
  echo "$(date +%FT%T) $TAG start shard $k -> $OUT"
  nice -n "${WZQ_NICE:-19}" "$PY" -B -m tools.wzqual.run_qual "${ARGS[@]}" "$@" > "$OUT.log" 2>&1
  echo "$(date +%FT%T) $TAG end shard $k rc=$?"
done
echo "$(date +%FT%T) worker $TAG finished"
