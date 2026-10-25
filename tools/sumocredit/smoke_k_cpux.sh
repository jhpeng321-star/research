#!/usr/bin/env bash
# S-SUMOCREDIT-SMOKE stages 4-5 (K-candidate plan check; mechanics only; registered remainder of the
# 2,000 sim-s cap: <= 138 sim-s; fresh TRAIN seeds 967603-967604; cpuX ~/sumocredit only).
set -euo pipefail
ROOT=$HOME/sumocredit
CODE=${SC_CODE:-$ROOT/code}
IN=$ROOT/inputs
OUT=${SC_OUT:-$ROOT/out_k}
PY=$HOME/c4/venv/bin/python
export PYTHONPATH=$CODE/src:$CODE
export C4_ALLOWED_ROOTS=$ROOT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$CODE"
mkdir -p "$OUT"
echo "# $(date -u +%FT%TZ) stage 4: K=3 calibration arms (seed 967603; planned 75 sim-s)"
nice -n 10 "$PY" -m traffic_shepherd.c4.collect --policy-pairs --policy "$IN/policy/v8_u600.pt" \
    --settings-wm-ckpt "$IN/wm/wm_s.pt" --settings-wm-config "$IN/wm/config.resolved.json" \
    --registration "$IN/REG_k.json" --demands 2700 --penetrations 0.4 --population textbook \
    --warmup 60 --horizon 63 --branch-first 62 --branch-cadence 60 --branch-points 1 --branch-horizon 3 \
    --workers 1 --no-physics-log --seeds 967603 --gate-horizon 3 --gate-unit fleet \
    --gate-reference native --gate-state belief --gate-candidates 3 --out "$OUT/s4"
"$PY" -m wms.control.real_credit14 gather --collect-root "$OUT/s4" --out-gate "$OUT/gate_s4.jsonl"
"$PY" -m wms.control.confident_gate --records "$OUT/gate_s4.jsonl" --out "$OUT/GATE_TABLE_s4.json" \
    --unit fleet --reference native --horizon 3 --state-source belief --comparison reference
"$PY" -m wms.control.confident_gate --records "$OUT/gate_s4.jsonl" --out "$OUT/GATE_MEAN_TABLE_s4.json" \
    --unit fleet --reference native --horizon 3 --state-source belief --comparison mean
echo "# $(date -u +%FT%TZ) stage 5: deploy policy_gate K=3 (seed 967604; planned 60 sim-s)"
nice -n 10 "$PY" -m wms.control.deploy --controller policy_gate --policy "$IN/policy/v8_u600.pt" \
    --wm-ckpt "$IN/wm/wm_s.pt" --gate-table "$OUT/GATE_TABLE_s4.json" \
    --gate-mean-table "$OUT/GATE_MEAN_TABLE_s4.json" --gate-candidates 3 --state belief \
    --belief-lead 50 --decision-period 1 --seed 967604 --demand 2700 --penetration 0.4 --warmup 55 \
    --horizon 5 --threads 2 --no-physics-log --out "$OUT/s5"
echo "# $(date -u +%FT%TZ) SMOKE K DONE"
