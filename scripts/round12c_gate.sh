#!/usr/bin/env bash
# Round-12C launch gate (S-R12C, 2026-10-24): the PHYSICS-ONLY CORE CALIBRATION of round 12.  Round 12
# stopped mid-run (BASE checks rejected); round 12C trains ONLY the driver encoder (physical mode) + the
# shared IDM/MOBIL physics parameters with every learned acceleration emission disabled --
# model.temporal_accel_off = true (R13F), model.residual_accel_off = true and model.follower_courtesy_off =
# true (R12C switches, default off with bit-identity vs 1878f842) -- so the residuals cannot compensate and
# the IDM core must calibrate itself.  The config (configs/c4_wm_v10round12c.toml) carries every switch and
# the restricted freeze list (trainable = driver_encoder.head + driver_encoder.centre_logits only); init,
# losses and the frozen merge pivot are round 12's; diagnostic schedule: single 120 s stage, 30 steps,
# batch 4, val 4 windows, max_hours 2.5.  R12C_PRIOR: the driver prior is OFF (= the round-12 launch choice);
# config|weak are accepted as in round 12 (any other value is a setup failure).
# Init = round-9 final checkpoint ($L/runs/c4_wm_v10_round9/wm_s_stage300.pt, else wm_s_best.pt), exactly as
# round 12 (init transforms carried by the config).  Real-data smoke: batch 4 at 120 s, 2 full training steps
# (backward + Adam), truncation 120, stage-end validation capped at 4 windows, max_hours 1.5, checkpoint
# save/reload, peak GPU memory; the verdict (python -m wms.train.round12c_gate) checks the single-stage merge
# census (merge_pivot frozen), the courtesy loss (term kept, head frozen), the ABSENCE of the cut-in term,
# fix-D replay telemetry, the driver prior ABSENT, the trainable tensor list printed in the smoke log (exactly
# the driver-encoder head + centre), strict reload of every checkpoint (all three emission flags on, physical
# bounds) and frozen bitwise vs init with only driver_encoder.head / centre_logits moved; the residual accel
# row 0 and the courtesy head must still be exactly zero.  Only if the verdict passes does it exec the full
# round (validation capped at 4 windows = the diagnostic schedule).
# Usage: bash scripts/round12c_gate.sh [REPO_DIR]
# R12C_ROOT / R12C_PYTHON / R12C_INIT (default: $L/runs/c4_wm_v10_round9/wm_s_stage300.pt, else wm_s_best.pt)
# allow a local installation, another init checkpoint, or isolated launcher tests; R12C_PRIOR (off|config|weak).
# Exit: 2 setup failure (repo or init checkpoint missing, bad R12C_PRIOR), 3 smoke/verdict failure,
# 4 prior artifacts/concurrent gate.
set -u -o pipefail -o noclobber
L=${R12C_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:-$L/repo_r12c}
PYTHON=${R12C_PYTHON:-/root/miniconda3/bin/python}
# Default init: round-9 final checkpoint (stage 300), falling back to its wm_s_best.pt; R12C_INIT overrides both.
CK=${R12C_INIT:-}
if [ -z "$CK" ]; then
    CK=$L/runs/c4_wm_v10_round9/wm_s_stage300.pt
    [ -f "$CK" ] || CK=$L/runs/c4_wm_v10_round9/wm_s_best.pt
fi
PRIOR=${R12C_PRIOR:-off}
case "$PRIOR" in
    config) PRIOR_SET=() ;;
    off) PRIOR_SET=(--set loss.driver_prior.weight=0.0) ;;
    weak) PRIOR_SET=(--set loss.driver_prior.weight=0.01) ;;
    *) echo "R12C_PRIOR must be off, config or weak: $PRIOR" >&2; exit 2 ;;
esac
ulimit -n 65536 || exit 2
cd "$REPO" || exit 2
[ -f "$CK" ] || { echo "round-12C init checkpoint missing: $CK" >&2; exit 2; }
for artifact in "$L/runs/smoke_r12c" "$L/runs/c4_wm_v10_round12c" \
    "$L/smoke_r12c.log" "$L/smoke_r12c.verdict" "$L/smoke_r12c.gate.json" "$L/.round12c_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior round-12C artifact exists: $artifact; choose a fresh root" >&2
        exit 4
    fi
done
# Atomic reservation also prevents concurrent launchers. Retained on failure/success:
# even a crash before the first log must not silently reuse this run's artifacts.
mkdir "$L/.round12c_gate.lock" || exit 4
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
# Every schedule/switch override comes from the config; the gate only fixes the init, the shared logging
# switches of rounds 9-12 and the device.
COMMON=(--set "model.init_from=$CK" --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda
    ${PRIOR_SET[@]+"${PRIOR_SET[@]}"})
if "$PYTHON" -m wms train --config configs/c4_wm_v10round12c.toml --mode run \
    --out "$L/runs/smoke_r12c" "${COMMON[@]}" \
    --set "optim.batch_size_per_stage=[4]" \
    --set "curriculum.unroll_s=[120.0]" --set "curriculum.stage_steps=[2]" \
    --set "curriculum.truncation_steps_per_stage=[120]" \
    --set selection.val_max_windows=4 \
    --set curriculum.max_hours=1.5 > "$L/smoke_r12c.log" 2>&1; then
    TRAIN_EXIT=0
else
    TRAIN_EXIT=$?
fi
if ! "$PYTHON" -m wms.train.round12c_gate --smoke-dir "$L/runs/smoke_r12c" \
    --smoke-log "$L/smoke_r12c.log" --train-exit-code "$TRAIN_EXIT" \
    --init-checkpoint "$CK" --config configs/c4_wm_v10round12c.toml --expect-prior "$PRIOR" \
    --report "$L/smoke_r12c.gate.json" > "$L/smoke_r12c.verdict" 2>&1; then
    exit 3
fi
# Production: the diagnostic schedule of the config (single 120 s stage, 30 steps, val 4 windows, 2.5 h cap).
exec "$PYTHON" -m wms train --config configs/c4_wm_v10round12c.toml --mode run \
    --out "$L/runs/c4_wm_v10_round12c" "${COMMON[@]}"
