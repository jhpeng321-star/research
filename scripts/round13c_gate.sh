#!/usr/bin/env bash
# One-shot 2+2 smoke, fail-closed verdict, then production exec (same PID).
set -eu -o pipefail -o noclobber
L=${R13C_ROOT:-/root/autodl-tmp/c4/v5}
REPO=${1:?repo argument required}
PYTHON=${R13C_PYTHON:-/root/miniconda3/bin/python}
CK=$L/runs/c4_wm_v10_round12c/wm_s.pt
ulimit -n 65536
cd "$REPO"
test -f "$CK"
for artifact in "$L/runs/smoke_r13c" "$L/runs/c4_wm_v10_round13c" \
    "$L/smoke_r13c.log" "$L/smoke_r13c.verdict" "$L/smoke_r13c.gate.json" "$L/.round13c_gate.lock"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "prior R13C artifact exists: $artifact" >&2; exit 4
    fi
done
mkdir "$L/.round13c_gate.lock"
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE="$L/data/v5_native" WMS_DATA_GENRAND_PLAIN="$L/data/genrand_plain"
export WMS_DATA_GENRAND_BRANCH="$L/data/genrand_branch" WMS_DATA_PAIRS300="$L/data/pairs300"
COMMON=(--set "model.init_from=$CK" --set loss.loss_share_logging=false
    --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4 --set runtime.device=cuda)
TRAIN_EXIT=0
"$PYTHON" -u -m wms train --config configs/c4_wm_v10round13c.toml --mode run \
    --out "$L/runs/smoke_r13c" "${COMMON[@]}" \
    --set 'optim.batch_size_per_stage=[4,4]' --set 'curriculum.stage_steps=[2,2]' \
    --set selection.val_max_windows=4 --set curriculum.max_hours=1.5 \
    > "$L/smoke_r13c.log" 2>&1 || TRAIN_EXIT=$?
"$PYTHON" -m wms.train.round13c_gate --smoke-dir "$L/runs/smoke_r13c" \
    --smoke-log "$L/smoke_r13c.log" --train-exit-code "$TRAIN_EXIT" \
    --init-checkpoint "$CK" --config configs/c4_wm_v10round13c.toml \
    --report "$L/smoke_r13c.gate.json" > "$L/smoke_r13c.verdict" 2>&1 || exit 3
cat "$L/smoke_r13c.verdict"
# The registration acknowledgment is written by the operator BEFORE starting this gate.
test -f "$L/registration_r13c.json"
# Arm both monitors against this exact PID/start identity before production exec.
START_TICKS=$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path("/proc/"+sys.argv[1]+"/stat").read_text().rsplit(")",1)[1].split()[19])' "$$")
mkdir -p "$L/chk_r13c"
for MODE in midrun postrun; do
    nohup "$PYTHON" -u -B "$REPO/scripts/round13c_waiter.py" --repo "$REPO" --root "$L" \
        --pid "$$" --start-ticks "$START_TICKS" --mode "$MODE" \
        > "$L/chk_r13c/${MODE}_waiter.log" 2>&1 < /dev/null &
    WAITER=$!
    printf '%s\n' "$WAITER" > "$L/chk_r13c/${MODE}_waiter.pid"
    kill -0 "$WAITER"
done
date -Is > "$L/production_r13c.start"
exec "$PYTHON" -u -m wms train --config configs/c4_wm_v10round13c.toml --mode run \
    --out "$L/runs/c4_wm_v10_round13c" "${COMMON[@]}"
