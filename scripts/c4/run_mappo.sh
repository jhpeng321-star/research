#!/usr/bin/env bash
# C4 MAPPO baseline (fair universal MAPPO, model-free, trained directly in SUMO).  LF only.
# Windows spot instance (48-core): use scripts/c4/run_mappo.ps1 instead.
#
#   bash run_mappo.sh check              # import + config parse only
#   bash run_mappo.sh smoke              # CLI --smoke on the CPU fake env (no SUMO), foreground
#   bash run_mappo.sh sumo-smoke         # real SUMO: --workers 2, 2 iterations, warmup 300 s, horizon 60 s, foreground
#   bash run_mappo.sh run                # detached training run (WORKERS SUMO processes)
#   RESUME=1 RUN_ID=<id> bash run_mappo.sh run   # continue that run in place (trainer resumes from the newest valid out/checkpoints/ckpt_*.pt)
#
# Interface: python -m wms.control.mappo.train --config C --workers N --out DIR [--max-sumo-seconds X]
#            [--iterations N] [--episodes-per-iteration E] [--horizon-s H] [--warmup-s W] [--update-threads T]
#            [--set KEY=VALUE ...] [--smoke]
# Writes out/log.jsonl (config/resume, episode, failure, iter rows), out/RUN_SUMMARY.json, out/status.json,
# out/checkpoints/ckpt_XXXXX.pt (every iteration, last 3 kept) + out/checkpoints/LATEST (pointer with sha256),
# out/policy_latest.pt (+ policy_XXXXX.pt): deploy actor in the c4_dreamer/3 schema, evaluate with
#   python -m wms.control.deploy --controller policy --policy out/policy_latest.pt ...
#
# Env: CONFIG (default configs/c4_mappo.toml), WORKERS (default usable cores - 2), MAX_SUMO_SECONDS,
#      UPDATE_THREADS (default = WORKERS), EXTRA_ARGS, RUN_ID, RESUME=1, RESTART=1, FOREGROUND=1.
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/c4_env.sh"
c4_refuse_crlf "$HERE/c4_env.sh" "$HERE/c4_jobwrap.sh"

MODE=${1:-}; [[ -n "$MODE" ]] || c4_die "usage: run_mappo.sh check|smoke|sumo-smoke|run"; shift || true
case "$MODE" in check|smoke|sumo-smoke|run) ;; *) c4_die "mode must be check, smoke, sumo-smoke or run" ;; esac
c4_require_repo; c4_export_env; cd "$C4_REPO"
CONFIG=${CONFIG:-configs/c4_mappo.toml}
[[ -f "$CONFIG" ]] || c4_die "config $CONFIG not found in $C4_REPO"
CORES=$(c4_usable_cores)
WORKERS=${WORKERS:-$(( CORES > 3 ? CORES - 2 : 1 ))}
UPDATE_THREADS=${UPDATE_THREADS:-$WORKERS}
ARGS=(--config "$CONFIG" --update-threads "$UPDATE_THREADS")
[[ -z "${MAX_SUMO_SECONDS:-}" ]] || ARGS+=(--max-sumo-seconds "$MAX_SUMO_SECONDS")
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})
MOD=wms.control.mappo.train

case "$MODE" in
  check)
    c4_prepare_run "mappo_check"
    FOREGROUND=1 c4_launch mappo_check "$PY" -c "import wms.control.mappo.train as m, wms.control.mappo.config_helpers as h; h.config_from_dict(m.load_config('$CONFIG')); print('import+config ok')" ;;
  smoke)
    c4_prepare_run "mappo_smoke"
    FOREGROUND=1 c4_launch mappo_smoke "$PY" -m "$MOD" "${ARGS[@]}" --workers 1 --out "$OUT_DIR" --smoke ;;
  sumo-smoke)
    c4_require_disk "${MIN_FREE_GB:-5}"
    c4_prepare_run "mappo_sumo_smoke"
    FOREGROUND=1 c4_launch mappo_sumo_smoke "$PY" -m "$MOD" "${ARGS[@]}" --workers 2 --out "$OUT_DIR" \
        --iterations 2 --episodes-per-iteration 2 --warmup-s 300 --horizon-s 60 --set minibatch_frames=32 ;;
  run)
    c4_require_disk "${MIN_FREE_GB:-20}"
    c4_prepare_run "mappo"
    c4_launch mappo "$PY" -m "$MOD" "${ARGS[@]}" --workers "$WORKERS" --out "$OUT_DIR" "${EXTRA[@]}" ;;
esac
