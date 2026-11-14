#!/usr/bin/env bash
set -euo pipefail
POPULATION=${POPULATION:-textbook}
CONTROLLERS=${CONTROLLERS:-native,damping,pi,maxaccel_lc513}
DEMANDS=${DEMANDS:-1900,2100,2400,2700}
PENETRATIONS=${PENETRATIONS:-0.05,0.1,0.15,0.2,0.3,0.4,0.6}
SEEDS=${SEEDS:-964000-964041}
WORKERS=${WORKERS:-46}
WARMUP=${WARMUP:-300}
HORIZON=${HORIZON:-600}
usage() { echo 'Usage: eval_v5.sh RUN_ID | --help (CONTROLLERS is comma-separated policy names/checkpoint paths; product grid)'; }
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# == 1 ]] || { usage >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
RUN_ID=$1; valid_id "$RUN_ID"; safe_path "$RUN_ID"
activate_box
"$PYTHON" "$HERE/ops.py" require-population "$REPO"
OUT=${RUN_ROOT:-$HOME/c4/eval_v5}/$RUN_ID
begin_run "$OUT" eval "population=$POPULATION" "controllers=$CONTROLLERS" "demands=$DEMANDS" \
    "pens=$PENETRATIONS" "seeds=$SEEDS" "workers=$WORKERS" "warmup=$WARMUP" "horizon=$HORIZON"
IFS=',' read -r -a POLICIES <<<"$CONTROLLERS"
[[ ${#POLICIES[@]} -gt 0 ]] || die 'empty controller list'
INDEX=0
for controller in "${POLICIES[@]}"; do
    [[ -n $controller ]] || die 'empty controller'
    safe_path "$controller"
    # Numeric index avoids basename collisions between checkpoint files.
    name=$(printf 'controller_%03d' "$INDEX")
    phase "$name" "$PYTHON" -m traffic_shepherd.c4.collect --policy "$controller" \
        --population "$POPULATION" --seeds "$SEEDS" --demands "$DEMANDS" --penetrations "$PENETRATIONS" \
        --case-plan product --warmup "$WARMUP" --horizon "$HORIZON" --workers "$WORKERS" \
        --no-physics-log --cav-vtype normal_car --out "$OUT/$name"
    INDEX=$((INDEX + 1))
done
"$PYTHON" "$HERE/ops.py" summary "$OUT"
"$PYTHON" "$HERE/ops.py" end "$OUT" 0
relay_config
REMOTE=$RELAY_DATA/$RUN_ID/eval
if [[ ! -f $OUT/PUSHED ]]; then
    retry_transfer "${SSH[@]}" "$RELAY_HOST" "mkdir -p '$REMOTE'" >>"$OUT/logs/push.log" 2>&1
    # Whitelist: no episode arrays, simulation files, or arbitrary checkpoint files.
    retry_transfer rsync -az --partial --timeout=120 -e "$RSYNC_SSH" \
        --exclude='*.incomplete-*/' --include='*/' --include='summary.json' \
        --include='COLLECT_SUMMARY*' --include='INDEX.json' --include='CODE_ID.txt' \
        --include='RUN_SPEC.json' --include='*.DONE' --exclude='*' \
        "$OUT/" "$RELAY_HOST:$REMOTE/" >>"$OUT/logs/push.log" 2>&1
    touch "$OUT/PUSHED"
    retry_transfer rsync -az --partial --timeout=120 -e "$RSYNC_SSH" "$OUT/PUSHED" \
        "$RELAY_HOST:$REMOTE/PUSHED" >>"$OUT/logs/push.log" 2>&1 || {
        mv "$OUT/PUSHED" "$OUT/PUSH_PENDING"; exit 1;
    }
fi
