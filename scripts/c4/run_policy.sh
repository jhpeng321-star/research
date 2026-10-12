#!/usr/bin/env bash
# C4 imagination policy training inside the world model (stage b).  LF only.
#
#   bash run_policy.sh check                 # import + config parse only (CPU, foreground)
#   bash run_policy.sh run [a.b=c ...]       # one detached training run
#
# Env: CONFIG (default configs/c4_policy.toml), WM_CKPT (world-model checkpoint, REQUIRED for run),
#      POLICY_INIT (optional policy checkpoint to fine-tune), POLICY_MODULE (default wms.control.train_cli),
#      POLICY_WM_KEY (config key that receives WM_CKPT, default wm.checkpoint), POLICY_INIT_KEY
#      (default policy.init_from), RUN_ID, RESTART=1, OMP, FOREGROUND=1.
# Interface assumed (docs/WM_C4_DESIGN.md, wms.control.contract): the CLI takes --config, --out, --set k=v.
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/c4_env.sh"
c4_refuse_crlf "$HERE/c4_env.sh" "$HERE/c4_jobwrap.sh"

MODE=${1:-}; [[ -n "$MODE" ]] || c4_die "usage: run_policy.sh check|run [a.b=c ...]"; shift
case "$MODE" in check|run) ;; *) c4_die "mode must be check or run" ;; esac
c4_require_repo; c4_export_env; cd "$C4_REPO"
CONFIG=${CONFIG:-configs/c4_policy.toml}
POLICY_MODULE=${POLICY_MODULE:-wms.control.train_cli}
[[ -f "$CONFIG" ]] || c4_die "config $CONFIG not found (c4_policy.toml is written by the policy worker)"
SETS=()
if [[ -n "${WM_CKPT:-}" ]]; then
  [[ -f "$WM_CKPT" ]] || c4_die "WM_CKPT $WM_CKPT not found"
  SETS+=(--set "${POLICY_WM_KEY:-wm.checkpoint}=\"$WM_CKPT\"")
fi
if [[ -n "${POLICY_INIT:-}" ]]; then
  [[ -f "$POLICY_INIT" ]] || c4_die "POLICY_INIT $POLICY_INIT not found"
  SETS+=(--set "${POLICY_INIT_KEY:-policy.init_from}=\"$POLICY_INIT\"")
fi
for item in "$@"; do SETS+=(--set "$item"); done
c4_require_disk "${MIN_FREE_GB:-20}"

case "$MODE" in
  check)
    c4_prepare_run "policy_check"
    FOREGROUND=1 c4_launch policy_check "$PY" -c "import importlib,sys; importlib.import_module('$POLICY_MODULE'); print('import ok: $POLICY_MODULE')" ;;
  run)
    [[ -n "${WM_CKPT:-}" ]] || c4_die "WM_CKPT is required (path of the C4 world-model checkpoint)"
    c4_require_gpu_idle
    c4_prepare_run "policy"
    c4_launch policy "$PY" -m "$POLICY_MODULE" --config "$CONFIG" --out "$OUT_DIR" "${SETS[@]}" ;;
esac
