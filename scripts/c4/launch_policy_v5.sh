#!/usr/bin/env bash
# POLICYV5: launch the C4 final policy (Dreamer trainer) for the v5 world model on the GPU host.  LF only.
# Note: this starts a multi-hour GPU training run.
#
#   bash scripts/c4/launch_policy_v5.sh WM_CKPT OUT [extra train_cli args, e.g. --max-updates 50]
#
#   WM_CKPT  the selected v5 WM checkpoint (wm_s*.pt; physics.json and TRAIN_RECEIPT.json next to it)
#   OUT      a NEW output directory (refused if it exists); train.log, pid, LAUNCH.txt go there too
#
# Run from (or with this script inside) the unpacked code root /workspace/c4/v5/repo_<commit>.
# Env overrides: PY (default /root/miniconda3/bin/python), CONFIG (default configs/c4_policy_v5.toml),
#   C4_V5_ROOT (default /workspace/c4/v5), C4_V5_DATA (default $C4_V5_ROOT/data),
#   DEVICE (default cuda), C4_GPU_MEM_FRACTION (optional, passed through), ALLOW_OTHER_REPO=1.
#
# Interface (src/wms/control/train_cli.py, same as scripts/c4/run_policy.sh):
#   python -m wms.control.train_cli --config C --wm-ckpt CKPT --device cuda --output-dir DIR [--max-updates N]
# Start windows: configs/c4_policy_v5.toml -> data.wm_config = configs/c4_wm_v5.toml (data.c4_base) ->
#   WMS_DATA_V5_NATIVE / WMS_DATA_GENRAND_PLAIN (+ GENRAND_BRANCH for the world split) below; no jam scan.
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then echo 'REFUSED: CRLF bytes in script' >&2; exit 2; fi
set -Eeuo pipefail
die() { printf 'REFUSED: %s\n' "$*" >&2; exit 2; }

[[ $# -ge 2 ]] || die "usage: launch_policy_v5.sh WM_CKPT OUT [extra train_cli args]"
WM_CKPT=$1; OUT=$2; shift 2
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
C4_V5_ROOT=${C4_V5_ROOT:-/workspace/c4/v5}
C4_V5_DATA=${C4_V5_DATA:-$C4_V5_ROOT/data}
PY=${PY:-/root/miniconda3/bin/python}
CONFIG=${CONFIG:-configs/c4_policy_v5.toml}
DEVICE=${DEVICE:-cuda}

case "$REPO" in
  "$C4_V5_ROOT"/repo_*) ;;
  *) [[ "${ALLOW_OTHER_REPO:-0}" == 1 ]] || die "code root $REPO is not $C4_V5_ROOT/repo_<commit> (ALLOW_OTHER_REPO=1 to override)" ;;
esac
[[ -f "$REPO/$CONFIG" ]] || die "config $CONFIG not found in $REPO"
[[ -f "$REPO/src/wms/control/c4_starts.py" ]] || die "$REPO predates POLICYV5 (no src/wms/control/c4_starts.py)"
command -v "$PY" >/dev/null 2>&1 || die "python $PY not found"

WM_CKPT=$(readlink -f -- "$WM_CKPT") || die "WM_CKPT not found"
[[ -f "$WM_CKPT" ]] || die "WM_CKPT $WM_CKPT not found"
[[ -f "$(dirname "$WM_CKPT")/physics.json" ]] || die "no physics.json next to $WM_CKPT"

[[ ! -e "$OUT" ]] || die "OUT $OUT already exists (never overwrite a run; choose a new directory)"
OUT_PARENT=$(readlink -f -- "$(dirname -- "$OUT")") || die "parent of OUT does not exist"
[[ -d "$OUT_PARENT" ]] || die "parent of OUT $OUT_PARENT does not exist"
OUT=$OUT_PARENT/$(basename -- "$OUT")

export WMS_DATA_V5_NATIVE=$C4_V5_DATA/v5_native
export WMS_DATA_GENRAND_PLAIN=$C4_V5_DATA/genrand_plain
export WMS_DATA_GENRAND_BRANCH=$C4_V5_DATA/genrand_branch
for d in "$WMS_DATA_V5_NATIVE" "$WMS_DATA_GENRAND_PLAIN" "$WMS_DATA_GENRAND_BRANCH"; do
  [[ -f "$d/C4_EPISODES_MANIFEST.json" ]] || die "no converted C4 source at $d (C4_EPISODES_MANIFEST.json)"
done
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export PYTHONPATH=src:.
# train_cli.checked_path: data, checkpoint and output must lie inside the repo or these roots.
export C4_ALLOWED_ROOTS="$C4_V5_ROOT:$(dirname "$WM_CKPT"):$OUT_PARENT"
[[ -z "${C4_GPU_MEM_FRACTION:-}" ]] || export C4_GPU_MEM_FRACTION

mkdir "$OUT"
cd "$REPO"
CMD=("$PY" -m wms.control.train_cli --config "$CONFIG" --wm-ckpt "$WM_CKPT" --device "$DEVICE"
     --output-dir "$OUT" "$@")
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REPO"
  echo "config=$CONFIG sha256=$(sha256sum "$CONFIG" | cut -d' ' -f1)"
  echo "wm_ckpt=$WM_CKPT sha256=$(sha256sum "$WM_CKPT" | cut -d' ' -f1)"
  echo "WMS_DATA_V5_NATIVE=$WMS_DATA_V5_NATIVE"
  echo "WMS_DATA_GENRAND_PLAIN=$WMS_DATA_GENRAND_PLAIN"
  echo "WMS_DATA_GENRAND_BRANCH=$WMS_DATA_GENRAND_BRANCH"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS C4_ALLOWED_ROOTS=$C4_ALLOWED_ROOTS C4_GPU_MEM_FRACTION=${C4_GPU_MEM_FRACTION:-}"
  printf 'cmd='; printf '%q ' "${CMD[@]}"; echo
} > "$OUT/LAUNCH.txt"

setsid nohup "${CMD[@]}" > "$OUT/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/pid"
echo "launched pid $PID -> $OUT (log $OUT/train.log; stop with: kill $PID)"
