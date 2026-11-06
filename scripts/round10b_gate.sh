#!/usr/bin/env bash
# Round-10b launch gate: round 10 (round 9 + fix B, cut-in reaction) + the LEARNED pre-merge follower courtesy
# (R10B_DESIGN.md); init = round-9 final checkpoint. Real-data smoke: batch 4 at 120/300 s, 2+2 full training steps
# (backward + Adam), stage-end validation capped at 4 windows, max_hours 1.5, stage/final checkpoint save/reload, peak GPU
# memory; the verdict (python -m wms.train.round10b_gate) adds the courtesy checks (finite loss, courtesy rows > 0, the
# only new init tensors are the zero-init head, the head moved, frozen + merge_pivot bitwise). Only if the verdict passes
# does it exec the full round (validation capped at 16 windows). Start only after round 9's final checkpoint exists.
# Usage: bash scripts/round10b_gate.sh [REPO_DIR]
# R10B_ROOT / R10B_PYTHON / R10B_INIT allow a local installation, another init checkpoint, or isolated launcher tests.
# Exit: 2 setup failure (repo or init checkpoint missing), 3 smoke/verdict failure, 4 prior artifacts/concurrent gate.
set -u -o pipefail -o noclobber
L=${R10B_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:-$L/repo_r10b}
PYTHON=${R10B_PYTHON:-/root/miniconda3/bin/python}
CK=${R10B_INIT:-$L/runs/c4_wm_v10_round9/wm_s_stage300.pt}
ulimit -n 65536 || exit 2
cd "$REPO" || exit 2
[ -f "$CK" ] || { echo "round-9 final checkpoint missing: $CK" >&2; exit 2; }
for artifact in "$L/runs/smoke_r10b" "$L/runs/c4_wm_v10_round10b" \
    "$L/smoke_r10b.log" "$L/smoke_r10b.verdict" "$L/smoke_r10b.gate.json" "$L/.round10b_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior round-10b artifact exists: $artifact; choose a fresh root" >&2
        exit 4
    fi
done
# Atomic reservation also prevents concurrent launchers. Retained on failure/success:
# even a crash before the first log must not silently reuse this run's artifacts.
mkdir "$L/.round10b_gate.lock" || exit 4
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
COMMON=(--set "model.init_from=$CK" --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda)
if "$PYTHON" -m wms train --config configs/c4_wm_v10round10b.toml --mode run \
    --out "$L/runs/smoke_r10b" "${COMMON[@]}" \
    --set "optim.batch_size_per_stage=[4,4]" \
    --set "curriculum.unroll_s=[120.0,300.0]" --set "curriculum.stage_steps=[2,2]" \
    --set "curriculum.truncation_steps_per_stage=[120,60]" \
    --set selection.val_max_windows=4 \
    --set curriculum.max_hours=1.5 > "$L/smoke_r10b.log" 2>&1; then
    TRAIN_EXIT=0
else
    TRAIN_EXIT=$?
fi
if ! "$PYTHON" -m wms.train.round10b_gate --smoke-dir "$L/runs/smoke_r10b" \
    --smoke-log "$L/smoke_r10b.log" --train-exit-code "$TRAIN_EXIT" \
    --init-checkpoint "$CK" --config configs/c4_wm_v10round10b.toml \
    --report "$L/smoke_r10b.gate.json" > "$L/smoke_r10b.verdict" 2>&1; then
    exit 3
fi
# Production: stage-end long-horizon validation capped at 16 windows (operational, 2026-10-23).
exec "$PYTHON" -m wms train --config configs/c4_wm_v10round10b.toml --mode run \
    --out "$L/runs/c4_wm_v10_round10b" "${COMMON[@]}" --set selection.val_max_windows=16
