#!/usr/bin/env bash
# Round-12 launch gate: round 9 (A-only hybrid first-passage merge law, merge_pivot FROZEN) + the BASE CALIBRATION of
# BASE_FOLLOWER_DIAG.md / R12_DESIGN.md + the round-11 contrast courtesy head; fix B (cut-in reaction loss) is DROPPED
# (loss.cutin_reaction.weight = 0, the term is never built).  The config (configs/c4_wm_v10round12.toml) carries every
# switch: driver_param_mode = physical + driver_param_init = centre, residual_accel_reinit = true,
# recorded_replay_present = filled, follower_courtesy_init_zero = true with the zero-init head weight 9.067 and head
# lr x1000; the gate scalars are frozen by the config.  This gate passes NO head-init switch and NO head weight/lr
# overrides (unlike round 11).  Owner decision on the driver prior: R12_PRIOR = config (default, no override) | off
# (--set loss.driver_prior.weight=0.0) | weak (--set loss.driver_prior.weight=0.01); passed to the smoke AND the full
# run, and to the verdict as --expect-prior (off: driver_prior_means must be absent from the log; every mode: the
# learned centre in the checkpoints strictly inside the physical ranges); any other value is a setup failure.
# Init = round-9 final checkpoint ($L/runs/c4_wm_v10_round9/wm_s_stage300.pt, else wm_s_best.pt in that dir).
# Real-data smoke: batch 4 at 120/300 s, 2+2 full training steps (backward + Adam), truncation [120,60], stage-end
# validation capped at 4 windows, max_hours 1.5, stage/final checkpoint save/reload, peak GPU memory; the verdict
# (python -m wms.train.round12_gate) checks the round-9 merge census (merge_pivot frozen), the courtesy loss/contrast
# and head range, the ABSENCE of the cut-in term, fix-D replay telemetry, driver prior means strictly inside the
# physical ranges, the init report (centre init, accel re-init, zero courtesy head, the exact new tensors), strict
# reload of every checkpoint (physical mode, version-2 head, physical bounds) and frozen + merge_pivot bitwise vs init
# with the re-initialised heads and the learned centre moved.  Only if the verdict passes does it exec the full round
# (validation capped at 16 windows).  Start only after the coordinator has decided the launch.
# Usage: bash scripts/round12_gate.sh [REPO_DIR]
# R12_ROOT / R12_PYTHON / R12_INIT (default: $L/runs/c4_wm_v10_round9/wm_s_stage300.pt, else wm_s_best.pt) allow a
# local installation, another init checkpoint, or isolated launcher tests; R12_PRIOR (config|off|weak) as above.
# Exit: 2 setup failure (repo or init checkpoint missing, bad R12_PRIOR), 3 smoke/verdict failure,
# 4 prior artifacts/concurrent gate.
set -u -o pipefail -o noclobber
L=${R12_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:-$L/repo_r12}
PYTHON=${R12_PYTHON:-/root/miniconda3/bin/python}
# Default init: round-9 final checkpoint (stage 300), falling back to its wm_s_best.pt; R12_INIT overrides both.
CK=${R12_INIT:-}
if [ -z "$CK" ]; then
    CK=$L/runs/c4_wm_v10_round9/wm_s_stage300.pt
    [ -f "$CK" ] || CK=$L/runs/c4_wm_v10_round9/wm_s_best.pt
fi
PRIOR=${R12_PRIOR:-config}
case "$PRIOR" in
    config) PRIOR_SET=() ;;
    off) PRIOR_SET=(--set loss.driver_prior.weight=0.0) ;;
    weak) PRIOR_SET=(--set loss.driver_prior.weight=0.01) ;;
    *) echo "R12_PRIOR must be config, off or weak: $PRIOR" >&2; exit 2 ;;
esac
ulimit -n 65536 || exit 2
cd "$REPO" || exit 2
[ -f "$CK" ] || { echo "round-12 init checkpoint missing: $CK" >&2; exit 2; }
for artifact in "$L/runs/smoke_r12" "$L/runs/c4_wm_v10_round12" \
    "$L/smoke_r12.log" "$L/smoke_r12.verdict" "$L/smoke_r12.gate.json" "$L/.round12_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior round-12 artifact exists: $artifact; choose a fresh root" >&2
        exit 4
    fi
done
# Atomic reservation also prevents concurrent launchers. Retained on failure/success:
# even a crash before the first log must not silently reuse this run's artifacts.
mkdir "$L/.round12_gate.lock" || exit 4
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
# No follower_courtesy_init_zero / head weight / head lr override: the round-12 config carries them.
COMMON=(--set "model.init_from=$CK" --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda
    ${PRIOR_SET[@]+"${PRIOR_SET[@]}"})
if "$PYTHON" -m wms train --config configs/c4_wm_v10round12.toml --mode run \
    --out "$L/runs/smoke_r12" "${COMMON[@]}" \
    --set "optim.batch_size_per_stage=[4,4]" \
    --set "curriculum.unroll_s=[120.0,300.0]" --set "curriculum.stage_steps=[2,2]" \
    --set "curriculum.truncation_steps_per_stage=[120,60]" \
    --set selection.val_max_windows=4 \
    --set curriculum.max_hours=1.5 > "$L/smoke_r12.log" 2>&1; then
    TRAIN_EXIT=0
else
    TRAIN_EXIT=$?
fi
if ! "$PYTHON" -m wms.train.round12_gate --smoke-dir "$L/runs/smoke_r12" \
    --smoke-log "$L/smoke_r12.log" --train-exit-code "$TRAIN_EXIT" \
    --init-checkpoint "$CK" --config configs/c4_wm_v10round12.toml --expect-prior "$PRIOR" \
    --report "$L/smoke_r12.gate.json" > "$L/smoke_r12.verdict" 2>&1; then
    exit 3
fi
# Production: stage-end long-horizon validation capped at 16 windows (operational, as rounds 9-11).
exec "$PYTHON" -m wms train --config configs/c4_wm_v10round12.toml --mode run \
    --out "$L/runs/c4_wm_v10_round12" "${COMMON[@]}" --set selection.val_max_windows=16
