#!/usr/bin/env bash
# S-SUMOCREDIT-SMOKE (registered; <= 2,000 sim-s; fresh TRAIN seeds 967600-967602; cpuX ~/sumocredit only).
# Mechanics only: calibration/credit records, fits, the gated loop arm and deploy policy_gate end to end.
set -euo pipefail
ROOT=$HOME/sumocredit
CODE=${SC_CODE:-$ROOT/code}
IN=$ROOT/inputs
OUT=${SC_OUT:-$ROOT/out}
PY=$HOME/c4/venv/bin/python
export PYTHONPATH=$CODE/src:$CODE
export C4_ALLOWED_ROOTS=$ROOT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$CODE"
mkdir -p "$OUT"
COMMON=(--policy-pairs --policy "$IN/policy/v8_u600.pt" --settings-wm-ckpt "$IN/wm/wm_s.pt"
        --settings-wm-config "$IN/wm/config.resolved.json" --registration "$IN/REG.json"
        --demands 2700 --penetrations 0.4 --population textbook --warmup 300 --branch-first 360
        --branch-cadence 60 --workers 1 --no-physics-log)
echo "# $(date -u +%FT%TZ) stage 1: gate calibration + real credit (seed 967600; planned 1,081 sim-s)"
nice -n 10 "$PY" -m traffic_shepherd.c4.collect "${COMMON[@]}" --seeds 967600 --horizon 360 \
    --branch-points 2 --branch-horizon 60 --gate-horizon 20 --gate-unit group --gate-reference hold \
    --gate-state belief --gate-max-units 2 --credit-horizon 30 --credit-group-max 3 --credit-flips 1 \
    --out "$OUT/s1"
"$PY" -m wms.control.real_credit14 gather --collect-root "$OUT/s1" --out-credit "$OUT/points_s1.jsonl" \
    --out-gate "$OUT/gate_s1.jsonl"
"$PY" -m wms.control.real_credit14 fit --points "$OUT/points_s1.jsonl" --out "$OUT/REAL_CREDIT_s1.json"
"$PY" -m wms.control.confident_gate --records "$OUT/gate_s1.jsonl" --out "$OUT/GATE_TABLE_s1.json" \
    --unit group --reference hold --horizon 20 --state-source belief
echo "# $(date -u +%FT%TZ) stage 2: gated loop arm (seed 967601; planned 451 sim-s)"
nice -n 10 "$PY" -m traffic_shepherd.c4.collect "${COMMON[@]}" --seeds 967601 --horizon 360 \
    --branch-points 1 --branch-horizon 30 --gate-table "$OUT/GATE_TABLE_s1.json" --gate-state belief \
    --out "$OUT/s2"
echo "# $(date -u +%FT%TZ) stage 3: deploy policy_gate (seed 967602; planned 330 sim-s)"
nice -n 10 "$PY" -m wms.control.deploy --controller policy_gate --policy "$IN/policy/v8_u600.pt" \
    --wm-ckpt "$IN/wm/wm_s.pt" --gate-table "$OUT/GATE_TABLE_s1.json" --state belief \
    --seed 967602 --demand 2700 --penetration 0.4 --warmup 300 --horizon 30 --threads 2 \
    --no-physics-log --out "$OUT/s3"
echo "# $(date -u +%FT%TZ) SMOKE DONE"
