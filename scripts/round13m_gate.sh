#!/usr/bin/env bash
# Round-13M launch gate (PREPARED 2026-10-24, NOT launched): round 12 + repair A + the "keep the mean, remove the
# noise" temporal variant (model.temporal_accel_lane_bias: the temporal acceleration emission is a learned per-lane
# constant, initialised at the measured per-lane mean of the init checkpoint's emission, docs/r13m_lane_bias_init.json;
# R13T_DESIGN.md).  Everything else is scripts/round13f_gate.sh (= the round-12 gate up to names; R13M_PRIOR default
# off = round 12's launch choice; init = round-9 final with round 12's init transforms, so round 12 is the paired
# control).  The verdict (python -m wms.train.round13f_gate) selects the lane-bias checks from the config: the extra
# new tensor temporal_lane_bias, its init values in INIT_FROM.json, and the mode in every reloaded checkpoint.
# Usage: bash scripts/round13m_gate.sh [REPO_DIR]
# R13M_ROOT / R13M_PYTHON / R13M_INIT / R13M_PRIOR as R12_* in scripts/round12_gate.sh.
# Exit: 2 setup failure, 3 smoke/verdict failure, 4 prior artifacts/concurrent gate.
set -u -o pipefail -o noclobber
L=${R13M_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:-$L/repo_r13m}
PYTHON=${R13M_PYTHON:-/root/miniconda3/bin/python}
# Default init: round-9 final checkpoint (stage 300), falling back to its wm_s_best.pt; R13M_INIT overrides both.
CK=${R13M_INIT:-}
if [ -z "$CK" ]; then
    CK=$L/runs/c4_wm_v10_round9/wm_s_stage300.pt
    [ -f "$CK" ] || CK=$L/runs/c4_wm_v10_round9/wm_s_best.pt
fi
PRIOR=${R13M_PRIOR:-off}
case "$PRIOR" in
    config) PRIOR_SET=() ;;
    off) PRIOR_SET=(--set loss.driver_prior.weight=0.0) ;;
    weak) PRIOR_SET=(--set loss.driver_prior.weight=0.01) ;;
    *) echo "R13M_PRIOR must be config, off or weak: $PRIOR" >&2; exit 2 ;;
esac
ulimit -n 65536 || exit 2
cd "$REPO" || exit 2
[ -f "$CK" ] || { echo "round-13M init checkpoint missing: $CK" >&2; exit 2; }
for artifact in "$L/runs/smoke_r13m" "$L/runs/c4_wm_v10_round13m" \
    "$L/smoke_r13m.log" "$L/smoke_r13m.verdict" "$L/smoke_r13m.gate.json" "$L/.round13m_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior round-13M artifact exists: $artifact; choose a fresh root" >&2
        exit 4
    fi
done
# Atomic reservation also prevents concurrent launchers. Retained on failure/success:
# even a crash before the first log must not silently reuse this run's artifacts.
mkdir "$L/.round13m_gate.lock" || exit 4
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
# No follower_courtesy_init_zero / head weight / head lr override: the round-13M config carries them.
COMMON=(--set "model.init_from=$CK" --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda
    ${PRIOR_SET[@]+"${PRIOR_SET[@]}"})
if "$PYTHON" -m wms train --config configs/c4_wm_v10round13m.toml --mode run \
    --out "$L/runs/smoke_r13m" "${COMMON[@]}" \
    --set "optim.batch_size_per_stage=[4,4]" \
    --set "curriculum.unroll_s=[120.0,300.0]" --set "curriculum.stage_steps=[2,2]" \
    --set "curriculum.truncation_steps_per_stage=[120,60]" \
    --set selection.val_max_windows=4 \
    --set curriculum.max_hours=1.5 > "$L/smoke_r13m.log" 2>&1; then
    TRAIN_EXIT=0
else
    TRAIN_EXIT=$?
fi
if ! "$PYTHON" -m wms.train.round13f_gate --smoke-dir "$L/runs/smoke_r13m" \
    --smoke-log "$L/smoke_r13m.log" --train-exit-code "$TRAIN_EXIT" \
    --init-checkpoint "$CK" --config configs/c4_wm_v10round13m.toml --expect-prior "$PRIOR" \
    --report "$L/smoke_r13m.gate.json" > "$L/smoke_r13m.verdict" 2>&1; then
    exit 3
fi
# Production: stage-end long-horizon validation capped at 16 windows (operational, as rounds 9-11).
exec "$PYTHON" -m wms train --config configs/c4_wm_v10round13m.toml --mode run \
    --out "$L/runs/c4_wm_v10_round13m" "${COMMON[@]}" --set selection.val_max_windows=16
