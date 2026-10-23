#!/usr/bin/env bash
# Round-11 launch gate: round 10b + the STRUCTURAL courtesy fix (R11_DESIGN.md): version-2 courtesy inputs (ramp car
# accel t/t-1, carried merge-zone time), matched-control contrast target, fix-B controls without partnered rows.
# Init = round-10b final checkpoint (head zero-expanded to version 2) unless R11_HEAD_INIT=zero (fresh zero-init head,
# e.g. if the coordinator prefers the r10b_headzero base: then also point R11_INIT at that checkpoint).
# Real-data smoke: batch 4 at 120/300 s, 2+2 full training steps (backward + Adam), stage-end validation capped at 4
# windows, max_hours 1.5, stage/final checkpoint save/reload, peak GPU memory; the verdict (python -m wms.train.round11_gate)
# adds the R11 checks (contrast controls > 0, partnered-control exclusion logged, init report = expanded r10b head or
# zero head as requested, version-2 head after reload, frozen + merge_pivot bitwise vs init, head moved). Only if the
# verdict passes does it exec the full round (validation capped at 16 windows). Start only after round 10b's final
# checkpoint exists AND the coordinator has decided the launch (round-10b reject checks + independent review).
# Usage: bash scripts/round11_gate.sh [REPO_DIR]
# R11_ROOT / R11_PYTHON / R11_INIT / R11_HEAD_INIT (checkpoint|zero) allow a local installation, another init
# checkpoint, the zero-head switch, or isolated launcher tests.
# Exit: 2 setup failure (repo or init checkpoint missing, bad R11_HEAD_INIT), 3 smoke/verdict failure,
# 4 prior artifacts/concurrent gate.
set -u -o pipefail -o noclobber
L=${R11_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:-$L/repo_r11}
PYTHON=${R11_PYTHON:-/root/miniconda3/bin/python}
CK=${R11_INIT:-$L/runs/c4_wm_v10_round10b/wm_s_stage300.pt}
HEAD_INIT=${R11_HEAD_INIT:-checkpoint}
case "$HEAD_INIT" in
    checkpoint) ZERO=false ;;
    zero) ZERO=true ;;
    *) echo "R11_HEAD_INIT must be checkpoint or zero: $HEAD_INIT" >&2; exit 2 ;;
esac
ulimit -n 65536 || exit 2
cd "$REPO" || exit 2
[ -f "$CK" ] || { echo "round-11 init checkpoint missing: $CK" >&2; exit 2; }
for artifact in "$L/runs/smoke_r11" "$L/runs/c4_wm_v10_round11" \
    "$L/smoke_r11.log" "$L/smoke_r11.verdict" "$L/smoke_r11.gate.json" "$L/.round11_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior round-11 artifact exists: $artifact; choose a fresh root" >&2
        exit 4
    fi
done
# Atomic reservation also prevents concurrent launchers. Retained on failure/success:
# even a crash before the first log must not silently reuse this run's artifacts.
mkdir "$L/.round11_gate.lock" || exit 4
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
COMMON=(--set "model.init_from=$CK" --set "model.follower_courtesy_init_zero=$ZERO"
    --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda)
if "$PYTHON" -m wms train --config configs/c4_wm_v10round11.toml --mode run \
    --out "$L/runs/smoke_r11" "${COMMON[@]}" \
    --set "optim.batch_size_per_stage=[4,4]" \
    --set "curriculum.unroll_s=[120.0,300.0]" --set "curriculum.stage_steps=[2,2]" \
    --set "curriculum.truncation_steps_per_stage=[120,60]" \
    --set selection.val_max_windows=4 \
    --set curriculum.max_hours=1.5 > "$L/smoke_r11.log" 2>&1; then
    TRAIN_EXIT=0
else
    TRAIN_EXIT=$?
fi
if ! "$PYTHON" -m wms.train.round11_gate --smoke-dir "$L/runs/smoke_r11" \
    --smoke-log "$L/smoke_r11.log" --train-exit-code "$TRAIN_EXIT" \
    --init-checkpoint "$CK" --config configs/c4_wm_v10round11.toml \
    --expect-courtesy-init "$HEAD_INIT" \
    --report "$L/smoke_r11.gate.json" > "$L/smoke_r11.verdict" 2>&1; then
    exit 3
fi
# Production: stage-end long-horizon validation capped at 16 windows (operational, as round 10b).
exec "$PYTHON" -m wms train --config configs/c4_wm_v10round11.toml --mode run \
    --out "$L/runs/c4_wm_v10_round11" "${COMMON[@]}" --set selection.val_max_windows=16
