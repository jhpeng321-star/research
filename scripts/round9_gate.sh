#!/usr/bin/env bash
# Round-9 stage-2C launch gate. Run only after the separate support gate passes.
# Real-data smoke: batch 4 at 120/300 s, 2+2 full training steps (backward + Adam),
# stage/final checkpoint save/reload, and peak allocated GPU memory in the JSON report.
# Usage: bash scripts/round9_gate.sh [REPO_DIR]
# R9_ROOT / R9_PYTHON allow a local installation or isolated launcher tests.
# Exit: 2 setup failure, 3 smoke/verdict failure, 4 prior artifacts/concurrent gate.
set -u -o pipefail -o noclobber
L=${R9_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:-$L/repo_r9}
PYTHON=${R9_PYTHON:-/root/miniconda3/bin/python}
ulimit -n 65536 || exit 2
cd "$REPO" || exit 2
for artifact in "$L/runs/smoke_r9" "$L/runs/c4_wm_v10_round9" \
    "$L/smoke_r9.log" "$L/smoke_r9.verdict" "$L/smoke_r9.gate.json" "$L/.round9_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior round-9 artifact exists: $artifact; choose a fresh root" >&2
        exit 4
    fi
done
# Atomic reservation also prevents concurrent launchers. Retained on failure/success:
# even a crash before the first log must not silently reuse this run's artifacts.
mkdir "$L/.round9_gate.lock" || exit 4
CK=$L/runs/c4_wm_v10_round8/wm_s_stage300.pt
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
COMMON=(--set "model.init_from=$CK" --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda)
if "$PYTHON" -m wms train --config configs/c4_wm_v10round9.toml --mode run \
    --out "$L/runs/smoke_r9" "${COMMON[@]}" \
    --set "optim.batch_size_per_stage=[4,4]" \
    --set "curriculum.unroll_s=[120.0,300.0]" --set "curriculum.stage_steps=[2,2]" \
    --set "curriculum.truncation_steps_per_stage=[120,60]" \
    --set curriculum.max_hours=0.5 > "$L/smoke_r9.log" 2>&1; then
    TRAIN_EXIT=0
else
    TRAIN_EXIT=$?
fi
if ! "$PYTHON" -m wms.train.round9_gate --smoke-dir "$L/runs/smoke_r9" \
    --smoke-log "$L/smoke_r9.log" --train-exit-code "$TRAIN_EXIT" \
    --report "$L/smoke_r9.gate.json" > "$L/smoke_r9.verdict" 2>&1; then
    exit 3
fi
exec "$PYTHON" -m wms train --config configs/c4_wm_v10round9.toml --mode run \
    --out "$L/runs/c4_wm_v10_round9" "${COMMON[@]}"
