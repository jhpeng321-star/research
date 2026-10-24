#!/usr/bin/env bash
# No arguments: CPU config validation only. Future operator commands (separate):
#   --prepare                       git archive HEAD -> repo_r13e_<sha8>; no training
#   --smoke --offline-report PATH   A 2x120, then B 2x120+2x300 from A smoke FINAL
#   --launch-a --offline-report PATH  explicit production A, after BOTH smoke verdicts
#   --launch-b --offline-report PATH  verify A FINAL/checks, re-smoke B from it, then B
# Optional: --variant main|fb (main = E + gs + learned yield = 13E; fb = E + learned
# yield without gs = registered fallback 13E_fb, with disjoint artifacts), and
# --wait (smoke/launch: poll every 60 s until no process holds the GPU, instead of
# refusing with exit 5). The GPU guard never signals any process.
# Invoke smoke/launch commands from the PREPARED copy only. No automatic A->B launch.
# Operator must supply registration_<tag>.json before smoke (see check_registration).
# All monitors can signal only the exact owned production PID/start/--out via pidfd.
set -eu -o pipefail -o noclobber
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PYTHON=${R13E_PYTHON:-$REPO/.venv/bin/python}
if [ -z "${R13E_PYTHON:-}" ] && [ ! -x "$PYTHON" ] && [ -x "$REPO/.venv/Scripts/python.exe" ]; then
    PYTHON=$REPO/.venv/Scripts/python.exe
fi
if [ ! -x "$PYTHON" ] && [ -z "${R13E_PYTHON:-}" ]; then
    PYTHON=/root/miniconda3/bin/python
fi
[ -x "$PYTHON" ] || { echo "set R13E_PYTHON to the deployment Python" >&2; exit 2; }
MODE=config
OFFLINE=
VARIANT=main
WAIT=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --prepare|--smoke|--launch-a|--launch-b)
            [ "$MODE" = config ] || { echo "select exactly one operation" >&2; exit 2; }
            MODE=${1#--}; shift ;;
        --offline-report) [ "$#" -ge 2 ] || exit 2; OFFLINE=$2; shift 2 ;;
        --variant) [ "$#" -ge 2 ] || exit 2; VARIANT=$2; shift 2 ;;
        --wait) WAIT=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$VARIANT" in
    main) TAG=r13e; RUN=c4_wm_v10_round13e
          A=configs/c4_wm_v10round13e.toml; B=configs/c4_wm_v10round13e_stage_b.toml ;;
    fb) TAG=r13e_fb; RUN=c4_wm_v10_round13e_fb
        A=configs/c4_wm_v10round13e_fb.toml; B=configs/c4_wm_v10round13e_fb_stage_b.toml ;;
    *) echo "unknown variant: $VARIANT (main|fb)" >&2; exit 2 ;;
esac
export PYTHONPATH="$REPO/src:$REPO/tests:$REPO" PYTHONDONTWRITEBYTECODE=1
GATE=("$PYTHON" -B -m wms.train.round13e_gate --variant "$VARIANT")

# GPU-busy guard: refuse (exit 5) while ANY compute process holds the GPU, or poll
# with --wait. Read-only: lists PIDs, never signals. nvidia-smi failure fails closed.
gpu_guard() {
    local APPS
    while :; do
        if ! APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); then
            echo "GPU guard: nvidia-smi failed; refusing $MODE (fail closed): $APPS" >&2
            exit 5
        fi
        APPS=$(printf '%s\n' "$APPS" | tr -d ' \r' | sed '/^$/d')
        [ -n "$APPS" ] || return 0
        if [ "$WAIT" != 1 ]; then
            echo "GPU guard: refusing $MODE: GPU held by PID(s) $(printf '%s ' $APPS)(use --wait to poll)" >&2
            exit 5
        fi
        echo "GPU guard: waiting, GPU held by PID(s) $(printf '%s ' $APPS)($(date -Is))" >&2
        sleep 60
    done
}

"${GATE[@]}" config
if [ "$MODE" = config ]; then exit 0; fi
if [ "$MODE" = prepare ]; then exec "${GATE[@]}" prepare; fi
[ -n "$OFFLINE" ] || { echo "$MODE requires --offline-report" >&2; exit 2; }
L=/root/autodl-tmp/c4/v5
CK=$L/runs/c4_wm_v10_round12c/wm_s.pt
EVIDENCE=(--root "$L" --init-checkpoint "$CK" --offline-report "$OFFLINE")
"${GATE[@]}" registered "${EVIDENCE[@]}"
gpu_guard
ulimit -n 65536
export OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
COMMON=(--set loss.loss_share_logging=false --set loss.brake_follow_cav_weight=2.0
    --set optim.batch_size=4 --set runtime.device=cuda)
smoke() {
    local STAGE=$1 CONFIG=$2 INIT=$3 SUFFIX=$4 CODE=0
    local DIR="$L/runs/smoke_${TAG}_$SUFFIX" LOG="$L/smoke_${TAG}_$SUFFIX.log"
    local OVERRIDES=()
    for artifact in "$DIR" "$LOG" "$L/smoke_${TAG}_$SUFFIX.gate.json"; do
        [ ! -e "$artifact" ] && [ ! -L "$artifact" ] || return 4
    done
    if [ "$STAGE" = a ]; then
        OVERRIDES=(--set 'curriculum.stage_steps=[2]')
    else
        OVERRIDES=(--set 'curriculum.stage_steps=[2,2]' --set 'optim.batch_size_per_stage=[4,4]')
    fi
    "$PYTHON" -B -u -m wms train --config "$CONFIG" --mode run --out "$DIR" \
        "${COMMON[@]}" --set "model.init_from=$INIT" "${OVERRIDES[@]}" \
        --set selection.val_max_windows=4 --set curriculum.max_hours=1.5 > "$LOG" 2>&1 || CODE=$?
    "${GATE[@]}" verdict --stage "$STAGE" --smoke-dir "$DIR" --smoke-log "$LOG" \
        --train-exit-code "$CODE" --init-checkpoint "$INIT" \
        --report "$L/smoke_${TAG}_$SUFFIX.gate.json"
}
if [ "$MODE" = smoke ]; then
    CUDA_VISIBLE_DEVICES= "${GATE[@]}" preflight "${EVIDENCE[@]}"
    mkdir "$L/.round${TAG#r}_gate.lock"
    mkdir "$REPO/_out"
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        nice -n 19 "$PYTHON" -B -m pytest -q -p no:cacheprovider --noconftest \
        tests/test_r13e_emergency.py tests/test_r13e_secure_gap.py tests/test_r13e_plan.py \
        tests/test_r13e_offline.py tests/test_r13y_yield.py tests/test_r13y_config.py \
        --basetemp "$REPO/runs/${TAG}_gate_core_tests"
    gpu_guard
    smoke a "$A" "$CK" a
    smoke b "$B" "$L/runs/smoke_${TAG}_a/wm_s.pt" b
    "${GATE[@]}" smoke-complete "${EVIDENCE[@]}"
    echo "Both $VARIANT smoke gates passed. Production requires a separate --launch-a command."
    exit 0
fi
STAGE=${MODE#launch-}
"${GATE[@]}" production-ready "${EVIDENCE[@]}" --stage "$STAGE"
OUT="$L/runs/${RUN}_$STAGE"
for artifact in "$OUT" "$L/production_${TAG}_$STAGE.log" "$L/chk_${TAG}/$STAGE"; do
    [ ! -e "$artifact" ] && [ ! -L "$artifact" ] || { echo "prior production artifact: $artifact" >&2; exit 4; }
done
gpu_guard
mkdir "$L/.round${TAG#r}_production_$STAGE.lock"
if [ "$STAGE" = a ]; then
    CONFIG=$A; INIT=$CK
else
    CONFIG=$B; INIT=$L/runs/${RUN}_a/wm_s.pt
    smoke b "$B" "$INIT" b_final
fi
# Arm both 13C-compatible checks BEFORE exec, against this PID and its start ticks.
START_TICKS=$("$PYTHON" -B -c 'import pathlib,sys; print(pathlib.Path("/proc/"+sys.argv[1]+"/stat").read_text().rsplit(")",1)[1].split()[19])' "$$")
mkdir -p "$L/chk_${TAG}/$STAGE"
for CHECK in midrun postrun; do
    READY="$L/chk_${TAG}/$STAGE/${CHECK}_ready.json"
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WMS_JAM_WORKERS=4 \
        nice -n 19 "$PYTHON" -B -u -m wms.train.round13e_gate --variant "$VARIANT" waiter --root "$L" \
        --stage "$STAGE" --pid "$$" --start-ticks "$START_TICKS" --mode "$CHECK" --ready "$READY" \
        > "$L/chk_${TAG}/$STAGE/${CHECK}_waiter.log" 2>&1 < /dev/null &
    WAITER=$!
    printf '%s\n' "$WAITER" > "$L/chk_${TAG}/$STAGE/${CHECK}_waiter.pid"
    for ((i=0; i<120; i++)); do
        kill -0 "$WAITER"
        if [ -f "$READY" ]; then break; fi
        sleep 1
    done
    test -f "$READY"
    kill -0 "$WAITER"
done
printf '%s\n' "$$" > "$L/production_${TAG}_$STAGE.pid"
date -Is > "$L/production_${TAG}_$STAGE.start"
exec "$PYTHON" -B -u -m wms train --config "$CONFIG" --mode run --out "$OUT" \
    "${COMMON[@]}" --set "model.init_from=$INIT" > "$L/production_${TAG}_$STAGE.log" 2>&1
