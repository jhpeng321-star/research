#!/usr/bin/env bash
# Round-13F launch gate (PREPARED 2026-10-24, NOT launched): round 12 + repair A (R13A_DESIGN.md: courtesy pair
# features keep their state derivative, causal pivot intent, absolute follower term at the parity weight 4.525) + the
# temporal acceleration emission DISABLED (model.temporal_accel_off = true; R13T_DESIGN.md, BRAKE_DIAG.md arm zerotemp).
# Init, schedule, freeze/lr lists, smoke and production arguments are round 12's, so round 12 is the paired control:
# init = round-9 final checkpoint ($L/runs/c4_wm_v10_round9/wm_s_stage300.pt, else wm_s_best.pt) with round 12's
# init transforms (centre driver init, residual accel re-init, fresh zero courtesy head; carried by the config).
# R13F_PRIOR defaults to off = the owner's round-12 launch choice (R12_REVIEW_PACKET.md: R12_PRIOR=off); config|weak
# are accepted as in round 12.  The verdict (python -m wms.train.round13f_gate) runs every round-12 check and then
# requires, in every reloaded checkpoint, temporal_accel_off, follower_courtesy_state_grad, intent = pivot and the
# saved absolute weight of the config.  Only if the verdict passes does it exec the full round (validation capped at
# 16 windows).  Start only after registration + coordinator ack.
# Usage: bash scripts/round13f_gate.sh [REPO_DIR]
# R13F_ROOT / R13F_PYTHON / R13F_INIT / R13F_PRIOR as R12_* in scripts/round12_gate.sh.
# Exit: 2 setup failure, 3 smoke/verdict failure, 4 prior artifacts/concurrent gate.
set -u -o pipefail -o noclobber
L=${R13F_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:-$L/repo_r13f}
PYTHON=${R13F_PYTHON:-/root/miniconda3/bin/python}
# Default init: round-9 final checkpoint (stage 300), falling back to its wm_s_best.pt; R13F_INIT overrides both.
CK=${R13F_INIT:-}
if [ -z "$CK" ]; then
    CK=$L/runs/c4_wm_v10_round9/wm_s_stage300.pt
    [ -f "$CK" ] || CK=$L/runs/c4_wm_v10_round9/wm_s_best.pt
fi
PRIOR=${R13F_PRIOR:-off}
case "$PRIOR" in
    config) PRIOR_SET=() ;;
    off) PRIOR_SET=(--set loss.driver_prior.weight=0.0) ;;
    weak) PRIOR_SET=(--set loss.driver_prior.weight=0.01) ;;
    *) echo "R13F_PRIOR must be config, off or weak: $PRIOR" >&2; exit 2 ;;
esac
ulimit -n 65536 || exit 2
cd "$REPO" || exit 2
[ -f "$CK" ] || { echo "round-13F init checkpoint missing: $CK" >&2; exit 2; }
for artifact in "$L/runs/smoke_r13f" "$L/runs/c4_wm_v10_round13f" \
    "$L/smoke_r13f.log" "$L/smoke_r13f.verdict" "$L/smoke_r13f.gate.json" "$L/.round13f_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior round-13F artifact exists: $artifact; choose a fresh root" >&2
        exit 4
    fi
done
# Atomic reservation also prevents concurrent launchers. Retained on failure/success:
# even a crash before the first log must not silently reuse this run's artifacts.
mkdir "$L/.round13f_gate.lock" || exit 4
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
# No follower_courtesy_init_zero / head weight / head lr override: the round-13F config carries them.
COMMON=(--set "model.init_from=$CK" --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda
    ${PRIOR_SET[@]+"${PRIOR_SET[@]}"})
if "$PYTHON" -m wms train --config configs/c4_wm_v10round13f.toml --mode run \
    --out "$L/runs/smoke_r13f" "${COMMON[@]}" \
    --set "optim.batch_size_per_stage=[4,4]" \
    --set "curriculum.unroll_s=[120.0,300.0]" --set "curriculum.stage_steps=[2,2]" \
    --set "curriculum.truncation_steps_per_stage=[120,60]" \
    --set selection.val_max_windows=4 \
    --set curriculum.max_hours=1.5 > "$L/smoke_r13f.log" 2>&1; then
    TRAIN_EXIT=0
else
    TRAIN_EXIT=$?
fi
if ! "$PYTHON" -m wms.train.round13f_gate --smoke-dir "$L/runs/smoke_r13f" \
    --smoke-log "$L/smoke_r13f.log" --train-exit-code "$TRAIN_EXIT" \
    --init-checkpoint "$CK" --config configs/c4_wm_v10round13f.toml --expect-prior "$PRIOR" \
    --report "$L/smoke_r13f.gate.json" > "$L/smoke_r13f.verdict" 2>&1; then
    exit 3
fi
# Production: stage-end long-horizon validation capped at 16 windows (operational, as rounds 9-11).
exec "$PYTHON" -m wms train --config configs/c4_wm_v10round13f.toml --mode run \
    --out "$L/runs/c4_wm_v10_round13f" "${COMMON[@]}" --set selection.val_max_windows=16
