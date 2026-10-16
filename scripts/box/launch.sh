#!/usr/bin/env bash
# Box launcher: export the box's data locations, then call `python -m wms train`.
# Replaces the env-var chain _out/train_r3_chain.sh; all recipe knobs live in configs/*.toml.
#
#   SW_ROOT=/path/to/switching CONFIG=configs/r3.toml bash scripts/box/launch.sh preflight
#   W=/fresh/run/dir CONFIG=configs/r3.toml bash scripts/box/launch.sh run runtime.seed=7
#
# Arguments after the mode are forwarded as `--set` overrides.
set -Eeuo pipefail
die() { printf 'REFUSED: %s\n' "$*" >&2; exit 2; }
if LC_ALL=C grep -q $'\r' "${BASH_SOURCE[0]}"; then die 'launcher has CRLF bytes; require LF'; fi
MODE=${1:-preflight}; [[ $# -eq 0 ]] || shift
case "$MODE" in preflight|run) ;; *) die "mode must be preflight or run, got $MODE" ;; esac
CODE_ROOT=${CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
CONFIG=${CONFIG:-$CODE_ROOT/configs/r3.toml}
PY=${PY:-/workspace/gpu1/env/rd_gpu/bin/python}
command -v "$PY" >/dev/null || die "missing interpreter $PY"
OMP=${OMP:-2}; [[ "$OMP" =~ ^[1-9][0-9]*$ ]] || die 'OMP must be a positive integer'
ORIGINAL_ROOT=${ORIGINAL_ROOT:-/workspace}
SIDECAR_ROOT=${SIDECAR_ROOT:-$ORIGINAL_ROOT/wmn1_a11_20261004T172721Z/work/sidecars}
: "${SW_ROOT:?set SW_ROOT to the TRAIN switching export}"
# Logical dataset name -> box path (WMS_DATA_<NAME>); already-set variables win.
: "${WMS_DATA_PACK_V3:=$ORIGINAL_ROOT/wmn1_m1_45c9032_20261003_A1/work/pack}"
: "${WMS_DATA_PACK_DSPEED:=$ORIGINAL_ROOT/wmn1_a9_data/pack_dspeed}"
: "${WMS_DATA_PACK_DAMPING:=$ORIGINAL_ROOT/wmn1_a10_data/pack_damping}"
: "${WMS_DATA_SIDECARS_PACKV3_V2:=$CODE_ROOT/_out/sidecars_packv3_v2}"
: "${WMS_DATA_IDENTITIES_DSPEED:=$SIDECAR_ROOT/identities_dspeed}"
: "${WMS_DATA_IDENTITIES_DAMPING:=$ORIGINAL_ROOT/wmn1_a10_data/identities_damping}"
: "${WMS_DATA_OVERLAY_DSPEED:=$ORIGINAL_ROOT/wmn1_a9_data/labels_overlay_dspeed}"
: "${WMS_DATA_OVERLAY_DAMPING:=$ORIGINAL_ROOT/wmn1_a10_data/labels_overlay_damping}"
: "${WMS_DATA_LPWO_V3:=$CODE_ROOT/_out/lpwo_v3}"
: "${WMS_DATA_PACK_SWITCHING_TRAIN:=$SW_ROOT/pack_switching_train}"
: "${WMS_DATA_IDENTITIES_SWITCHING_TRAIN:=$SW_ROOT/identities_switching_train}"
: "${WMS_DATA_DRIVER_PARAMS_SWITCHING_TRAIN:=$SW_ROOT/driver_params_switching_train}"
: "${WMS_DATA_WINDOWS_TRAIN_HETDATA:=$SW_ROOT/windows_train_hetdata.json}"
: "${WMS_DATA_ACTION_TIMELINE_INDEX_TRAIN:=$SW_ROOT/ACTION_TIMELINE_INDEX_train.json}"
: "${WMS_DATA_ZERO_SHOT_HELDOUT:=$SW_ROOT/ZERO_SHOT_HELDOUT.json}"
export "${!WMS_DATA_@}"
export OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP" PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(train --config "$CONFIG" --mode "$MODE" --set "runtime.omp=$OMP" --set "runtime.threads=$OMP")
[[ -z "${W:-}" ]] || ARGS+=(--out "$W")
for item in "$@"; do ARGS+=(--set "$item"); done
cd -- "$CODE_ROOT"
exec "$PY" -m wms "${ARGS[@]}"
