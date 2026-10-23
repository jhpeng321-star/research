#!/usr/bin/env bash
# Round-10 launch gate (round 9 + ONE change: fix B, cut-in reaction supervision; init = round-9 final checkpoint).
# Real-data smoke: batch 4 at 120/300 s, 2+2 full training steps (backward + Adam), stage-end validation capped at
# 4 windows (the 256-window default replays the F2 prefix per eval batch and takes hours; production uses 16),
# stage/final checkpoint save/reload and peak allocated GPU memory in the JSON report. Only if the verdict passes
# does it exec the full round. Start it only after round 9 has finished (the checkpoint below must exist).
# Usage: bash scripts/round10_gate.sh [REPO_DIR]
# R10_ROOT / R10_PYTHON / R10_INIT allow a local installation, another init checkpoint, or isolated launcher tests.
# Exit: 2 setup failure (repo or init checkpoint missing), 3 smoke/verdict failure, 4 prior artifacts/concurrent gate.
set -u -o pipefail -o noclobber
L=${R10_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:-$L/repo_r10}
PYTHON=${R10_PYTHON:-/root/miniconda3/bin/python}
CK=${R10_INIT:-$L/runs/c4_wm_v10_round9/wm_s_stage300.pt}
ulimit -n 65536 || exit 2
cd "$REPO" || exit 2
[ -f "$CK" ] || { echo "round-9 final checkpoint missing: $CK" >&2; exit 2; }
for artifact in "$L/runs/smoke_r10" "$L/runs/c4_wm_v10_round10" \
    "$L/smoke_r10.log" "$L/smoke_r10.verdict" "$L/smoke_r10.gate.json" "$L/.round10_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior round-10 artifact exists: $artifact; choose a fresh root" >&2
        exit 4
    fi
done
# Atomic reservation also prevents concurrent launchers. Retained on failure/success:
# even a crash before the first log must not silently reuse this run's artifacts.
mkdir "$L/.round10_gate.lock" || exit 4
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
COMMON=(--set "model.init_from=$CK" --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda)
if "$PYTHON" -m wms train --config configs/c4_wm_v10round10.toml --mode run \
    --out "$L/runs/smoke_r10" "${COMMON[@]}" \
    --set "optim.batch_size_per_stage=[4,4]" \
    --set "curriculum.unroll_s=[120.0,300.0]" --set "curriculum.stage_steps=[2,2]" \
    --set "curriculum.truncation_steps_per_stage=[120,60]" \
    --set selection.val_max_windows=4 \
    --set curriculum.max_hours=1.5 > "$L/smoke_r10.log" 2>&1; then
    TRAIN_EXIT=0
else
    TRAIN_EXIT=$?
fi
if ! "$PYTHON" -m wms.train.round10_gate --smoke-dir "$L/runs/smoke_r10" \
    --smoke-log "$L/smoke_r10.log" --train-exit-code "$TRAIN_EXIT" \
    --init-checkpoint "$CK" --config configs/c4_wm_v10round10.toml \
    --report "$L/smoke_r10.gate.json" > "$L/smoke_r10.verdict" 2>&1; then
    exit 3
fi
# Production: stage-end long-horizon validation capped at 16 windows (operational, 2026-10-23).
exec "$PYTHON" -m wms train --config configs/c4_wm_v10round10.toml --mode run \
    --out "$L/runs/c4_wm_v10_round10" "${COMMON[@]}" --set selection.val_max_windows=16
