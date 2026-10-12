#!/usr/bin/env bash
# Final C4 evaluation campaign on the CPU box (stage e).  LF only.
#
#   bash run_eval.sh check      # traffic_shepherd inspect + SUMO version, foreground, no episodes
#   bash run_eval.sh plan       # job counts and the core-hour estimate, nothing started
#   bash run_eval.sh run        # one detached campaign driver (resumable: RESUME=1 RUN_ID=<same id>)
#   bash run_eval.sh status     # done / failed / pending per controller
#
# Controllers: native, damping, pi, c4_policy, c4_policy_rsu_check_oracle, c4_policy_rsu_check_belief
# (+ wall with INCLUDE_WALL=1).  MAPPO is deferred.  Penetrations 0.02..0.60, demands 2400/2600/2700,
# development seeds from eval_seeds.json (make_eval_seeds.py), paired across controllers.
# Env: SEEDS, POLICY_CKPT, WM_CKPT, BELIEF_CKPT (optional, ignored: the belief variant uses the RSU
#      BeliefTracker), WORKERS (default usable cores - 2), PER_DEMAND=8, CONTROLLERS, PENETRATIONS, DEMANDS,
#      TIMEOUT_S=7200, MIN_PER_EP=6 (policy_check at default settings needs ~25 (4 threads) to ~45 (1 thread) min/episode on CPU,
#      DEPLOY_NOTES.md), C4_EVAL_MODULE (default wms.control.deploy), C4_CLASSIC_HARNESS=evaluate|deploy,
#      C4_DEPLOY_ARGS (extra runner args, e.g. "--horizon 600 --margin 2"), RUN_ID, RESUME=1, FOREGROUND=1.
# NOTE: the C4 runner only admits seeds 4107-4148 (traffic_shepherd.c4.seeds); eval_seeds.json currently
#      lists WM28 fresh-validation seeds, which it refuses (OPS_NOTES section 10).
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/c4_env.sh"
c4_refuse_crlf "$HERE/c4_env.sh" "$HERE/c4_jobwrap.sh"

MODE=${1:-}; [[ -n "$MODE" ]] || c4_die "usage: run_eval.sh check|plan|run|status"; shift
case "$MODE" in check|plan|run|status) ;; *) c4_die "mode must be check, plan, run or status" ;; esac
c4_require_repo; c4_export_env; cd "$C4_REPO"
SEEDS=${SEEDS:-$C4_REPO/runs/active/WM-S-C4-20261013/eval_seeds.json}
[[ -f "$SEEDS" ]] || c4_die "seed file $SEEDS not found (python scripts/c4/make_eval_seeds.py)"
WORKERS=${WORKERS:-$(( C4_CORES > 3 ? C4_CORES - 2 : 1 ))}

ARGS=(--seeds "$SEEDS" --workers "$WORKERS" --python "$PY" --per-demand "${PER_DEMAND:-8}"
      --timeout "${TIMEOUT_S:-7200}" --minutes-per-episode "${MIN_PER_EP:-6}")
[[ -z "${CONTROLLERS:-}" ]] || ARGS+=(--controllers "$CONTROLLERS")
[[ -z "${PENETRATIONS:-}" ]] || ARGS+=(--penetrations "$PENETRATIONS")
[[ -z "${DEMANDS:-}" ]] || ARGS+=(--demands "$DEMANDS")
[[ "${INCLUDE_WALL:-0}" != 1 ]] || ARGS+=(--include-wall)
[[ -z "${POLICY_CKPT:-}" ]] || ARGS+=(--policy-ckpt "$POLICY_CKPT")
[[ -z "${WM_CKPT:-}" ]] || ARGS+=(--wm-ckpt "$WM_CKPT")
[[ -z "${BELIEF_CKPT:-}" ]] || ARGS+=(--belief-ckpt "$BELIEF_CKPT")
[[ -z "${C4_EVAL_MODULE:-}" ]] || ARGS+=(--c4-module "$C4_EVAL_MODULE")
[[ "${RERUN_FAILED:-0}" != 1 ]] || ARGS+=(--rerun-failed)

case "$MODE" in
  check)
    command -v sumo >/dev/null 2>&1 || c4_die "sumo not on PATH (box_setup.sh installs it)"
    VER=$(sumo --version 2>&1 | head -1)
    c4_log "$VER"
    [[ "$VER" == *1.27.1* || "${SKIP_SUMO_CHECK:-0}" == 1 ]] || c4_die "SUMO 1.27.1 required (SKIP_SUMO_CHECK=1 to override)"
    "$PY" -m traffic_shepherd inspect | head -20 ;;
  plan)
    "$PY" "$HERE/eval_campaign.py" plan --out "$C4_RUNS/eval_plan_only" "${ARGS[@]}" ;;
  status)
    : "${RUN_ID:?set RUN_ID=<eval run id>}"
    "$PY" "$HERE/eval_campaign.py" status --out "$C4_RUNS/$RUN_ID/out" "${ARGS[@]}" ;;
  run)
    command -v sumo >/dev/null 2>&1 || c4_die "sumo not on PATH (box_setup.sh installs it)"
    [[ "$(sumo --version 2>&1 | head -1)" == *1.27.1* || "${SKIP_SUMO_CHECK:-0}" == 1 ]] || c4_die "SUMO 1.27.1 required"
    c4_require_disk "${MIN_FREE_GB:-20}"
    c4_prepare_run eval
    c4_launch eval "$PY" "$HERE/eval_campaign.py" run --out "$OUT_DIR" "${ARGS[@]}" ;;
esac
