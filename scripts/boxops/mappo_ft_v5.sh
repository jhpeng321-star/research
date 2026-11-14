#!/usr/bin/env bash
set -euo pipefail
POPULATION=${POPULATION:-textbook}
REQUIRE_POPULATION=${REQUIRE_POPULATION:-1}
WORKERS=${WORKERS:-46}
UPDATE_THREADS=${UPDATE_THREADS:-44}
MAX_SUMO_SECONDS=${MAX_SUMO_SECONDS:-100000000}
INIT_FROM=${INIT_FROM:-}
INIT_ACCEL_MAX=${INIT_ACCEL_MAX:-2.6}
CONFIG=${CONFIG:-configs/c4_mappo.toml}
SPOT_POLL_SECONDS=${SPOT_POLL_SECONDS:-5}
usage() { echo 'Usage: mappo_ft_v5.sh RUN_ID | --help (fresh run needs INIT_FROM; repeat RUN_ID resumes; STOP_REQUESTED stops safely)'; }
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# == 1 ]] || { usage >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
RUN_ID=$1; valid_id "$RUN_ID"; safe_path "$RUN_ID"
activate_box
safe_path "$CONFIG"; [[ -z $INIT_FROM ]] || safe_path "$INIT_FROM"
ulimit -n 65536
"$PYTHON" "$HERE/ops.py" check-spot "$SPOT_POLL_SECONDS"
OUT=${RUN_ROOT:-$HOME/c4/mappo_v5}/$RUN_ID
begin_run "$OUT" mappo "population=$POPULATION" "require_population=$REQUIRE_POPULATION" \
    "config=$CONFIG" "init_from=$INIT_FROM" "init_accel_max=$INIT_ACCEL_MAX" \
    "workers=$WORKERS" "update_threads=$UPDATE_THREADS" "budget=$MAX_SUMO_SECONDS" \
    "warmup=300" "horizon=600" "episodes_per_iteration=$WORKERS"
[[ ! -f $OUT/TRAIN_DONE ]] || { push_run mappo; exit 0; }
[[ ! -f $OUT/STOP_REQUESTED ]] || die "stop request still present; archive it before resuming: $OUT/STOP_REQUESTED"
"$PYTHON" "$HERE/ops.py" mappo-config "$REPO" "$CONFIG" "$OUT/boxops_config.toml" "$POPULATION" "$REQUIRE_POPULATION"
ARGS=(--config "$OUT/boxops_config.toml" --workers "$WORKERS" --out "$OUT"
      --episodes-per-iteration "$WORKERS" --warmup-s 300 --horizon-s 600
      --update-threads "$UPDATE_THREADS" --max-sumo-seconds "$MAX_SUMO_SECONDS")
if compgen -G "$OUT/checkpoints/ckpt_*.pt" >/dev/null; then
    ARGS+=(--resume)
else
    [[ -n $INIT_FROM && -f $INIT_FROM ]] || die 'a fresh fine-tune needs INIT_FROM pointing to a MAPPO checkpoint/actor'
    ARGS+=(--init-from "$INIT_FROM" --init-accel-max "$INIT_ACCEL_MAX")
fi
"$PYTHON" -m wms.control.mappo.train "${ARGS[@]}" >>"$OUT/logs/train.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" >"$OUT/TRAIN_PID"
"$PYTHON" "$HERE/ops.py" watch-stop "$OUT" "$TRAIN_PID" "$SPOT_POLL_SECONDS" >>"$OUT/logs/spot.log" 2>&1 &
WATCH_PID=$!
# Only the watcher we started is killed. The trainer is asked to checkpoint itself.
request_stop() { touch "$OUT/STOP_REQUESTED"; }
trap request_stop TERM INT HUP
RC=0
while true; do
    if wait "$TRAIN_PID"; then RC=0; break; else RC=$?; fi
    # A trapped signal can interrupt wait while the trainer is still running.
    if ((RC > 128)) && kill -0 "$TRAIN_PID" 2>/dev/null; then continue; fi
    break
done
trap - TERM INT HUP
kill -TERM "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true
if ((RC == 0)); then touch "$OUT/TRAIN_DONE"; fi
RUN_RC=$RC
"$PYTHON" "$HERE/ops.py" end "$OUT" "$RC"
push_run mappo
exit "$RC"
