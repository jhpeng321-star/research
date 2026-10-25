#!/usr/bin/env bash
# S-R14 gate (patterned on round13e_gate.sh). No arguments: CPU config validation only.
# Future operator commands (separate, explicit):
#   --prepare                          git archive HEAD -> repo_r14_<sha8>; no training
#   --registration-template --offline-report PATH   print the exact registration_r14.json body
#   --smoke --offline-report PATH      A 2x120 from 13E B FINAL, then B 2x120+2x300 from A smoke FINAL
#   --launch-a --offline-report PATH   production A (after both smoke verdicts)
#   --launch-b --offline-report PATH   A production verdict + A MIDRUN CONTINUE, re-smoke B from A FINAL, B
#   --chain --offline-report PATH      --launch-a, wait for A to exit, then --launch-b (same rules)
# Lesson applied (13E): stage B never waits for A's slow POSTRUN check. A's postrun keeps running
# in parallel with B; B's own MIDRUN check requires A postrun == CONTINUE before it measures and
# stops B otherwise (fail closed).  --wait: poll the GPU guard instead of refusing (exit 5).
# The GPU guard and disk guard never signal any process. Invoke from the PREPARED copy only.
# init = 13E stage-B FINAL, accepted only with BOTH 13E B checks CONTINUE (round14_gate.py).
set -eu -o pipefail -o noclobber
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PYTHON=${R14_PYTHON:-$REPO/.venv/bin/python}
if [ -z "${R14_PYTHON:-}" ] && [ ! -x "$PYTHON" ] && [ -x "$REPO/.venv/Scripts/python.exe" ]; then
    PYTHON=$REPO/.venv/Scripts/python.exe
fi
if [ ! -x "$PYTHON" ] && [ -z "${R14_PYTHON:-}" ]; then
    PYTHON=/root/miniconda3/bin/python
fi
[ -x "$PYTHON" ] || { echo "set R14_PYTHON to the deployment Python" >&2; exit 2; }
MODE=config
OFFLINE=
WAIT=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --prepare|--registration-template|--smoke|--launch-a|--launch-b|--chain)
            [ "$MODE" = config ] || { echo "select exactly one operation" >&2; exit 2; }
            MODE=${1#--}; shift ;;
        --offline-report) [ "$#" -ge 2 ] || exit 2; OFFLINE=$2; shift 2 ;;
        --wait) WAIT=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
TAG=r14; RUN=c4_wm_v10_round14
A=configs/c4_wm_v10round14.toml; B=configs/c4_wm_v10round14_stage_b.toml
export PYTHONPATH="$REPO/src:$REPO/tests:$REPO" PYTHONDONTWRITEBYTECODE=1
GATE=("$PYTHON" -B -m wms.train.round14_gate)

# GPU-busy guard (13E): refuse (exit 5) while ANY compute process holds the GPU, or poll with
# --wait. Read-only: lists PIDs, never signals. nvidia-smi failure fails closed.
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
# Disk guard: checkpoints of a run need space on the run filesystem (operational, never deletes).
disk_guard() {
    local FREE_KB
    FREE_KB=$(df -Pk "$L" | awk 'NR==2 {print $4}')
    [ "${FREE_KB:-0}" -ge $(( ${R14_MIN_FREE_GB:-2} * 1024 * 1024 )) ] || {
        echo "disk guard: < ${R14_MIN_FREE_GB:-2} GB free on $L; refusing $MODE" >&2; exit 6; }
}

"${GATE[@]}" config
if [ "$MODE" = config ]; then exit 0; fi
if [ "$MODE" = prepare ]; then exec "${GATE[@]}" prepare; fi
[ -n "$OFFLINE" ] || { echo "$MODE requires --offline-report" >&2; exit 2; }
L=/root/autodl-tmp/c4/v5
CK=$L/runs/c4_wm_v10_round13e_b/wm_s.pt
EVIDENCE=(--root "$L" --init-checkpoint "$CK" --offline-report "$OFFLINE")
if [ "$MODE" = registration-template ]; then exec "${GATE[@]}" registration-template "${EVIDENCE[@]}"; fi
if [ "$MODE" = chain ]; then
    # Explicit operator chain: A production, then B with the same gates as --launch-b.
    bash "$REPO/scripts/round14_gate.sh" --launch-a --offline-report "$OFFLINE" $([ "$WAIT" = 1 ] && echo --wait) || exit $?
    for ((i=0; i<240; i++)); do
        [ -f "$L/chk_${TAG}/a/midrun/RESULT.json" ] && break
        sleep 60
    done
    exec bash "$REPO/scripts/round14_gate.sh" --launch-b --offline-report "$OFFLINE" --wait
fi
"${GATE[@]}" registered "${EVIDENCE[@]}"
gpu_guard
disk_guard
ulimit -n 65536
export OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
export WMS_DATA_LOWPEN_NATIVE="$L/data/lowpen_20261025/lowpen_native"
export WMS_DATA_LOWPEN_GENRAND_PLAIN="$L/data/lowpen_20261025/lowpen_genrand_plain"
export WMS_DATA_LOWPEN_GENRAND_BRANCH="$L/data/lowpen_20261025/lowpen_genrand_branch"
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
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        nice -n 19 "$PYTHON" -B -m pytest -q -p no:cacheprovider --noconftest \
        tests/test_r13e_emergency.py tests/test_r13e_secure_gap.py tests/test_r13e_plan.py \
        tests/test_r13y_yield.py tests/test_r13y_config.py \
        tests/test_r14_lane_secure_gap.py tests/test_r14_lowshare.py tests/test_r14_gate.py \
        --basetemp "$REPO/runs/${TAG}_gate_core_tests"
    gpu_guard
    smoke a "$A" "$CK" a
    smoke b "$B" "$L/runs/smoke_${TAG}_a/wm_s.pt" b
    "${GATE[@]}" smoke-complete "${EVIDENCE[@]}"
    echo "Both R14 smoke gates passed. Production requires a separate --launch-a (or --chain) command."
    exit 0
fi
STAGE=${MODE#launch-}
"${GATE[@]}" production-ready "${EVIDENCE[@]}" --stage "$STAGE"
OUT="$L/runs/${RUN}_$STAGE"
for artifact in "$OUT" "$L/production_${TAG}_$STAGE.log" "$L/chk_${TAG}/$STAGE"; do
    [ ! -e "$artifact" ] && [ ! -L "$artifact" ] || { echo "prior production artifact: $artifact" >&2; exit 4; }
done
gpu_guard
disk_guard
mkdir "$L/.round${TAG#r}_production_$STAGE.lock"
if [ "$STAGE" = a ]; then
    CONFIG=$A; INIT=$CK
else
    CONFIG=$B; INIT=$L/runs/${RUN}_a/wm_s.pt
    smoke b "$B" "$INIT" b_final
fi
# Arm both checks BEFORE exec, against this PID and its start ticks (13E). The postrun check of
# A outlives A and runs in parallel with B; B's midrun check consumes its RESULT.
START_TICKS=$("$PYTHON" -B -c 'import pathlib,sys; print(pathlib.Path("/proc/"+sys.argv[1]+"/stat").read_text().rsplit(")",1)[1].split()[19])' "$$")
mkdir -p "$L/chk_${TAG}/$STAGE"
for CHECK in midrun postrun; do
    READY="$L/chk_${TAG}/$STAGE/${CHECK}_ready.json"
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WMS_JAM_WORKERS=4 \
        nice -n 19 "$PYTHON" -B -u -m wms.train.round14_gate waiter --root "$L" \
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
