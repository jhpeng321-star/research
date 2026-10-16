#!/usr/bin/env bash
# C4 world-model training on the GPU box (stage a of the plan).  LF only.
#
#   bash run_wm.sh preflight                 # config + data resolution check, foreground, no GPU use
#   bash run_wm.sh cpu-smoke                 # tools/c4_smoke.py: 2 real-data updates on CPU (foreground, no GPU)
#   bash run_wm.sh smoke                     # tiny real-data GPU smoke (foreground); writes the smoke-ok marker
#   bash run_wm.sh run [a.b=c ...]           # the ONE detached full run (needs a passing smoke for this config)
#
# Real interface (src/wms/cli): python -m wms preflight|train --config C [--mode preflight|run] [--set a.b=c]... [--out DIR]
#   device = config runtime.device ("cuda" in configs/c4_wm.toml); DEVICE=cpu overrides it via --set runtime.device.
# Env: CONFIG (default configs/c4_wm.toml), RUN_ID, SMOKE_SET (default curriculum.stage_steps=[4,4,3]: the c4 config has
#      THREE stages, 10/30/60 s), RESTART=1 [RESTART_CKPT=path]  continue a failed attempt in a NEW directory, warm-started
#      OMP (threads), DEVICE, SKIP_SMOKE=1 (not recommended), FOREGROUND=1, GPU_BUSY_MIB, MIN_FREE_GB (default 30).
# Required data/env (c4_env.sh sets the first; the bundle ships the lists and checkpoint):
#   WMS_DATA_CKPT_R5B_BEST  R5b best wm_s_best.pt (model.init_from = "ckpt_r5b_best"); default data_cache/ckpt_r5b/wm_s_best.pt
#   runs/active/c4_wm_lists (registry dataset c4_wm_lists) shipped in the bundle
#   the zero-shot guard file (data.switching.zero_shot_heldout): NOT in the bundle, see OPS_NOTES.md section 11
# Budget (config comment, estimate): stages 10/30/60 s, 600/800/600 steps, batch 16, ~2.5 h on a 4090 D; max_hours cap 5.
# Extra arguments after the mode are forwarded as --set overrides.
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/c4_env.sh"
c4_refuse_crlf "$HERE/c4_env.sh" "$HERE/c4_jobwrap.sh"

MODE=${1:-}; [[ -n "$MODE" ]] || c4_die "usage: run_wm.sh preflight|cpu-smoke|smoke|run [a.b=c ...]"; shift
case "$MODE" in preflight|cpu-smoke|smoke|run) ;; *) c4_die "mode must be preflight, cpu-smoke, smoke or run" ;; esac
c4_require_repo; c4_export_env; cd "$C4_REPO"
CONFIG=${CONFIG:-configs/c4_wm.toml}
[[ -f "$CONFIG" ]] || c4_die "config $CONFIG not found in $C4_REPO"
[[ -f "$WMS_DATA_CKPT_R5B_BEST" ]] || c4_die "R5b checkpoint $WMS_DATA_CKPT_R5B_BEST not found (set WMS_DATA_CKPT_R5B_BEST)"
[[ -d runs/active/c4_wm_lists ]] || c4_die "runs/active/c4_wm_lists missing (shipped by the bundle group c4_wm_lists)"
c4_require_zero_shot_guard
CFG_SHA=$(sha256sum "$CONFIG" | cut -c1-12)
SETS=(--set "runtime.omp=$OMP" --set "runtime.threads=$OMP")
[[ -z "${DEVICE:-}" ]] || SETS+=(--set "runtime.device=\"$DEVICE\"")
for item in "$@"; do SETS+=(--set "$item"); done
c4_require_disk "${MIN_FREE_GB:-30}"

case "$MODE" in
  preflight)
    c4_prepare_run "wm_preflight"
    FOREGROUND=1 c4_launch wm_preflight "$PY" -m wms preflight --config "$CONFIG" --out "$OUT_DIR" "${SETS[@]}" ;;
  cpu-smoke)
    c4_prepare_run "wm_cpusmoke"
    FOREGROUND=1 c4_launch wm_cpusmoke "$PY" tools/c4_smoke.py "$OUT_DIR"
    [[ "$(cat "$RUN_DIR/STATE")" == done ]] || c4_die "cpu smoke failed; see $RUN_DIR/launch.log (kept, not deleted)" ;;
  smoke)
    c4_require_gpu_idle
    c4_prepare_run "wm_smoke"
    SMOKE_SET=${SMOKE_SET:-curriculum.stage_steps=[4,4,3]}
    FOREGROUND=1 c4_launch wm_smoke "$PY" -m wms train --config "$CONFIG" --mode run --out "$OUT_DIR" \
        "${SETS[@]}" --set "$SMOKE_SET"
    [[ "$(cat "$RUN_DIR/STATE")" == done ]] || c4_die "smoke failed; see $RUN_DIR/launch.log (kept, not deleted)"
    [[ -f "$OUT_DIR/RUN_SUMMARY.json" ]] || c4_die "smoke exited 0 but wrote no RUN_SUMMARY.json"
    echo "$RUN_ID" > "$C4_RUNS/SMOKE_OK_wm_$CFG_SHA"
    c4_log "smoke passed for config sha $CFG_SHA; per-stage timings/peak memory: $OUT_DIR/train_log.jsonl and $RUN_DIR/monitor.csv" ;;
  run)
    c4_require_gpu_idle
    if [[ "${SKIP_SMOKE:-0}" != 1 && ! -f "$C4_RUNS/SMOKE_OK_wm_$CFG_SHA" ]]; then
      c4_die "no passing smoke for config sha $CFG_SHA; run: bash run_wm.sh smoke"
    fi
    c4_prepare_run "wm_main"
    if [[ -n "${RESTART_CKPT:-}" ]]; then
      [[ -f "$RESTART_CKPT" ]] || c4_die "RESTART_CKPT $RESTART_CKPT not found"
      export WMS_DATA_CKPT_RESTART=$RESTART_CKPT
      SETS+=(--set 'model.init_from="ckpt_restart"')
    fi
    c4_launch wm_main "$PY" -m wms train --config "$CONFIG" --mode run --out "$OUT_DIR" "${SETS[@]}" ;;
esac
