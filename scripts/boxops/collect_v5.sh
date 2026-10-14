#!/usr/bin/env bash
set -euo pipefail
POPULATION=${POPULATION:-textbook}
NATIVE_SEEDS=${NATIVE_SEEDS:-963000-963479}
GENRAND_SEEDS=${GENRAND_SEEDS:-963480-963959}
BRANCH_SEEDS=${BRANCH_SEEDS:-963960-963995}
PENETRATIONS=${PENETRATIONS:-0.05,0.1,0.2,0.4}
PEN_WEIGHTS=${PEN_WEIGHTS:-1,1,2,2}
DEMANDS=${DEMANDS:-1900,2100,2400,2700}
WORKERS=${WORKERS:-46}
WARMUP=${WARMUP:-300}
HORIZON=${HORIZON:-900}
BRANCH_CADENCE=${BRANCH_CADENCE:-60}
BRANCH_ARMS=${BRANCH_ARMS:-4}
BRANCH_HORIZON=${BRANCH_HORIZON:-60}
usage() { echo 'Usage: collect_v5.sh RUN_ID | --help (480 native + 480 random + 36 paired base episodes; env overrides at top)'; }
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# == 1 ]] || { usage >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
RUN_ID=$1; valid_id "$RUN_ID"; safe_path "$RUN_ID"
activate_box
"$PYTHON" "$HERE/ops.py" require-population "$REPO"
OUT=${RUN_ROOT:-$HOME/c4/data_v5}/$RUN_ID
begin_run "$OUT" collect "population=$POPULATION" "native_seeds=$NATIVE_SEEDS" \
    "generic_seeds=$GENRAND_SEEDS" "branch_seeds=$BRANCH_SEEDS" "pens=$PENETRATIONS" \
    "weights=$PEN_WEIGHTS" "demands=$DEMANDS" "workers=$WORKERS" "warmup=$WARMUP" \
    "horizon=$HORIZON" "cadence=$BRANCH_CADENCE" "arms=$BRANCH_ARMS" "branch_h=$BRANCH_HORIZON"
ARGS=(--population "$POPULATION" --penetrations "$PENETRATIONS" --pen-weights "$PEN_WEIGHTS"
      --case-plan weighted-cycle --demands "$DEMANDS" --warmup "$WARMUP" --horizon "$HORIZON"
      --workers "$WORKERS" --no-physics-log --gr-mode speed --cav-vtype normal_car)
phase native "$PYTHON" -m traffic_shepherd.c4.collect --policy native --seeds "$NATIVE_SEEDS" "${ARGS[@]}" --out "$OUT/native"
phase generic_random "$PYTHON" -m traffic_shepherd.c4.collect --policy generic_random --seeds "$GENRAND_SEEDS" "${ARGS[@]}" --out "$OUT/generic_random"
# Existing driver has no population option. The adapter forwards it to each collector.
export BOXOPS_REAL_PYTHON=$PYTHON BOXOPS_POPULATION=$POPULATION
phase paired env PYTHON="$HERE/python_bridge.sh" bash "$REPO/scripts/genrand/collect_genrand.sh" \
    "$OUT/paired" '' "$BRANCH_SEEDS" "$WORKERS" --penetrations "$PENETRATIONS" \
    --pen-weights "$PEN_WEIGHTS" --demands "$DEMANDS" --warmup "$WARMUP" --horizon "$HORIZON" \
    --gr-mode speed --branch-cadence "$BRANCH_CADENCE" --branch-arms "$BRANCH_ARMS" \
    --branch-horizon "$BRANCH_HORIZON" --no-physics-log --no-convert
convert_phase native "$OUT/native"
convert_phase generic_random "$OUT/generic_random"
convert_phase paired "$OUT/paired/branch"
push_run
