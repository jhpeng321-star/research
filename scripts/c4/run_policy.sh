#!/usr/bin/env bash
# C4 imagination policy training inside the frozen world model (stage b).  LF only.
#
#   bash run_policy.sh check                 # import + config parse only (CPU, foreground)
#   bash run_policy.sh smoke                 # CLI --smoke: 2 updates, batch 2, H 4, CPU, foreground (needs WM_CKPT)
#   bash run_policy.sh run [N]               # one detached training run; optional N = --max-updates
#
# Real interface (src/wms/control/train_cli.py):
#   python -m wms.control.train_cli --config C --wm-ckpt CKPT [CKPT ...] --device cuda --output-dir DIR
#                                   [--max-updates N] [--init-policy CKPT] [--resume CKPT] [--smoke]
#   Writes policy_best.pt, policy_latest.pt, policy_NNNNNN.pt, updates.jsonl, validation.jsonl, RUN_SUMMARY.json.
#   Config/checkpoint/data/output paths must be inside the repo root or inside C4_ALLOWED_ROOTS
#   (c4_env.sh sets C4_ALLOWED_ROOTS=$C4_RUNS:$C4_HOME, so run directories outside the repo are accepted).
#
# Env: CONFIG (default configs/c4_policy.toml), WM_CKPT (REQUIRED for smoke/run: one or more space-separated frozen
#      wm_s checkpoints = ensemble members; a physics.json must sit next to each), POLICY_INIT (--init-policy),
#      POLICY_RESUME (--resume), DEVICE (default cuda), POLICY_DATA_CONFIG (WM config whose [data] section supplies the
#      TRAIN start states; default configs/c4_wm.toml, see below), RUN_ID, RESTART=1, OMP, FOREGROUND=1.
#
# START-STATE DATA: configs/c4_policy.toml defaults to data.wm_config = configs/r5b.toml, whose windows include the
# CIRCLES-damping and pack_damping runs.  The bundle does not ship those rows (C4 data rule: native + random direct
# actions only), so on the box the policy trainer would read zero-filled rows.  This script therefore writes
# $RUN_DIR/policy_box.toml = {extends the config; data.wm_config = POLICY_DATA_CONFIG} and passes that.  Set
# POLICY_DATA_CONFIG=none to use the config unchanged (only if the full lpwo packs were shipped).
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/c4_env.sh"
c4_refuse_crlf "$HERE/c4_env.sh" "$HERE/c4_jobwrap.sh"

MODE=${1:-}; [[ -n "$MODE" ]] || c4_die "usage: run_policy.sh check|smoke|run [max_updates]"; shift
case "$MODE" in check|smoke|run) ;; *) c4_die "mode must be check, smoke or run" ;; esac
c4_require_repo; c4_export_env; cd "$C4_REPO"
CONFIG=${CONFIG:-configs/c4_policy.toml}
POLICY_MODULE=${POLICY_MODULE:-wms.control.train_cli}
DEVICE=${DEVICE:-cuda}
POLICY_DATA_CONFIG=${POLICY_DATA_CONFIG:-configs/c4_wm.toml}
[[ -f "$CONFIG" ]] || c4_die "config $CONFIG not found in $C4_REPO"
MAX_UPDATES=${1:-${MAX_UPDATES:-}}
c4_require_disk "${MIN_FREE_GB:-20}"

# ---- WM checkpoints (space-separated list), each with its physics.json
WM_ARGS=()
if [[ "$MODE" != check ]]; then
  [[ -n "${WM_CKPT:-}" ]] || c4_die "WM_CKPT is required (frozen WM checkpoint path(s), space-separated)"
  for ck in $WM_CKPT; do
    [[ -f "$ck" ]] || c4_die "WM_CKPT member $ck not found"
    [[ -f "$(dirname "$ck")/physics.json" ]] || c4_die "no physics.json next to $ck (train_cli needs it)"
    WM_ARGS+=("$ck")
  done
fi
EXTRA=()
if [[ -n "${POLICY_INIT:-}" ]]; then [[ -f "$POLICY_INIT" ]] || c4_die "POLICY_INIT $POLICY_INIT not found"; EXTRA+=(--init-policy "$POLICY_INIT"); fi
if [[ -n "${POLICY_RESUME:-}" ]]; then [[ -f "$POLICY_RESUME" ]] || c4_die "POLICY_RESUME $POLICY_RESUME not found"; EXTRA+=(--resume "$POLICY_RESUME"); fi
[[ -z "$MAX_UPDATES" ]] || EXTRA+=(--max-updates "$MAX_UPDATES")

write_box_config() {   # usage: write_box_config  -> sets BOX_CONFIG (inside the run dir, allowed by C4_ALLOWED_ROOTS)
  BOX_CONFIG=$CONFIG
  if [[ "$POLICY_DATA_CONFIG" != none ]]; then
    [[ -f "$POLICY_DATA_CONFIG" ]] || c4_die "POLICY_DATA_CONFIG $POLICY_DATA_CONFIG not found"
    BOX_CONFIG=$RUN_DIR/policy_box.toml
    printf 'extends = "%s"\n\n[data]\nwm_config = "%s"\n' "$C4_REPO/$CONFIG" "$C4_REPO/$POLICY_DATA_CONFIG" > "$BOX_CONFIG"
  fi
}

case "$MODE" in
  check)
    c4_prepare_run "policy_check"
    FOREGROUND=1 c4_launch policy_check "$PY" -c "import importlib,sys; importlib.import_module('$POLICY_MODULE'); print('import ok: $POLICY_MODULE')" ;;
  smoke)
    c4_prepare_run "policy_smoke"
    write_box_config
    FOREGROUND=1 c4_launch policy_smoke "$PY" -m "$POLICY_MODULE" --config "$BOX_CONFIG" --wm-ckpt "${WM_ARGS[@]}" \
        --output-dir "$OUT_DIR" --smoke
    [[ "$(cat "$RUN_DIR/STATE")" == done ]] || c4_die "policy smoke failed; see $RUN_DIR/launch.log"
    c4_log "policy smoke ok: $OUT_DIR/RUN_SUMMARY.json" ;;
  run)
    c4_require_gpu_idle
    c4_prepare_run "policy"
    write_box_config
    c4_launch policy "$PY" -m "$POLICY_MODULE" --config "$BOX_CONFIG" --wm-ckpt "${WM_ARGS[@]}" --device "$DEVICE" \
        --output-dir "$OUT_DIR" "${EXTRA[@]}" ;;
esac
