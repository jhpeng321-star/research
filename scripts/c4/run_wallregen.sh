#!/usr/bin/env bash
# WALLREGEN: regenerate the unseen-wall WM evaluation states in a new SUMO driver population.  LF only.
#
#   bash run_wallregen.sh check        # imports + population support + plan dry run (no SUMO), foreground
#   bash run_wallregen.sh smoke        # 1 world (seed 962199), 182 s branches, 1 worker, + convert, foreground
#   bash run_wallregen.sh run          # detached: plan -> SUMO worlds (WORKERS) -> convert h60 / real / req
#   RESUME=1 RUN_ID=<id> bash run_wallregen.sh run    # continue the same run dir (complete worlds skipped)
#
# Interfaces (see WALLREGEN_NOTES.md):
#   python -m traffic_shepherd.c4.wallregen plan --out P --seeds S --population POP [--branch-s 600]
#   python -m traffic_shepherd.c4.wallregen run --plan P --out RUN --workers N        (resumable)
#   python -m traffic_shepherd.c4.wallregen_states --run RUN --variant h60|real|req --out DIR
# Outputs: $RUN_DIR/out/{plan.json,sim/,states_h60/,states_real/,states_req/}; copy the states_* dirs to
# data_cache/wallc1t_states_h60, data_cache/wallc1t_states_h60_real, data_cache/wallc1t_states_req.
#
# Env: DEMANDS (REQUIRED for run, e.g. DEMANDS=1600,1800; comma-separated veh/h, cycled per
#      penetration triple; or set DEMANDS_VPH in traffic_shepherd/c4/wallregen.py),
#      POPULATION (default textbook; legacy3 = old 3-type population), SEEDS (default 962000-962035 =
#      36 worlds), BRANCH_S (default 600), WORKERS (default usable cores - 2), PLAN_SEED (20261015),
#      RUN_ID, RESUME=1, RESTART=1, FOREGROUND=1.
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/c4_env.sh"
c4_refuse_crlf "$HERE/c4_env.sh" "$HERE/c4_jobwrap.sh"

MODE=${1:-}; [[ -n "$MODE" ]] || c4_die "usage: run_wallregen.sh check|smoke|run"; shift || true
case "$MODE" in check|smoke|run) ;; *) c4_die "mode must be check, smoke or run" ;; esac
c4_require_repo; c4_export_env; cd "$C4_REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1        # one SUMO + one python per worker process
POPULATION=${POPULATION:-textbook}
SEEDS=${SEEDS:-962000-962035}
DEMANDS=${DEMANDS:-}
BRANCH_S=${BRANCH_S:-600}
PLAN_SEED=${PLAN_SEED:-20261015}
CORES=$(c4_usable_cores)
WORKERS=${WORKERS:-$(( CORES > 3 ? CORES - 2 : 1 ))}
command -v sumo >/dev/null 2>&1 || c4_die "sumo not on PATH (set SUMO_HOME; box_setup.sh)"

pipeline() {   # usage: pipeline OUT SEEDS BRANCH_S WORKERS  (plan once; run is resumable; convert after)
  local out=$1 seeds=$2 branch=$3 workers=$4 v
  mkdir -p "$out"
  if [[ ! -f "$out/plan.json" ]]; then
    "$PY" -m traffic_shepherd.c4.wallregen plan --out "$out/plan.json" --seeds "$seeds" \
        ${DEMANDS:+--demands "$DEMANDS"} \
        --population "$POPULATION" --branch-s "$branch" --plan-seed "$PLAN_SEED"
  fi
  set +e
  "$PY" -m traffic_shepherd.c4.wallregen run --plan "$out/plan.json" --out "$out/sim" --workers "$workers"
  local rc=$?
  set -e
  for v in h60 real req; do
    if [[ -e "$out/states_$v" ]]; then mv "$out/states_$v" "$out/states_$v.prev_$(date +%s)"; fi
    "$PY" -m traffic_shepherd.c4.wallregen_states --run "$out/sim" --variant "$v" --out "$out/states_$v"
  done
  "$PY" -m traffic_shepherd.c4.wallregen status --out "$out/sim"
  return $rc
}
export -f pipeline
export PY POPULATION PLAN_SEED DEMANDS

case "$MODE" in
  check)
    c4_prepare_run "wallregen_check"
    FOREGROUND=1 c4_launch wallregen_check "$PY" -c "
import inspect, json
from traffic_shepherd.c4 import wallregen as W, wallregen_states  # noqa: F401
from traffic_shepherd.c4.env import CavSumoEnv
import traffic_shepherd.controller_factory  # noqa: F401  (needs output/controller_verification_v7)
has = 'population' in inspect.signature(CavSumoEnv.__init__).parameters
assert has or '$POPULATION' in ('legacy', 'legacy3'), 'CavSumoEnv has no population argument (c4/popfix missing)'
dem = [int(x) for x in '${DEMANDS:-1800}'.split(',')]
p = W.make_plan(seeds=W.parse_seeds('$SEEDS'), penetrations=W.DEFAULT_PENETRATIONS, demands=dem,
                t_choices=W.DEFAULT_T_CHOICES, plan_seed=$PLAN_SEED, branch_s=$BRANCH_S, population='$POPULATION')
print(json.dumps({'population_arg': has, 'n_worlds': p['n_worlds'], 'workers': $WORKERS}))" ;;
  smoke)
    c4_prepare_run "wallregen_smoke"
    DEMANDS=${DEMANDS:-1800} FOREGROUND=1 c4_launch wallregen_smoke bash -c "pipeline '$OUT_DIR' 962199 182 1" ;;
  run)
    [[ -n "$DEMANDS" ]] || c4_die "set DEMANDS=<veh/h,...> (textbook saturates ~1930 veh/h)"
    c4_require_disk "${MIN_FREE_GB:-10}"
    c4_prepare_run "wallregen"
    c4_launch wallregen bash -c "pipeline '$OUT_DIR' '$SEEDS' '$BRANCH_S' '$WORKERS'" ;;
esac
