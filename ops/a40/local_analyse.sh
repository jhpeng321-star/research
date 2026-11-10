#!/bin/bash
# AMENDMENT 40 laptop analysis from the OFFLOADED files (sha-verified manifest E:/agent-work/a40/offload/OFFLOAD.sha256):
# verify every manifest entry, then the registered analysis (ops/a40_geomvar.py analyse: paired percentile bootstrap
# 10,000 reps seed 20261104, diff = treatment - reference, negative = less delay; contrasts LEARNED_TRIGGER - U0
# [PRIMARY: supported iff CI upper < 0], DUO_HOLD - U0, LEARNED_TRIGGER - DUO_HOLD; total delay, CVaR90 on common
# vehicles, five groups with the 10 s rule, hard brakes, TTC, collisions, teleports, activations) in a wz3to2v80
# process -> E:/research-dense/runs/active/C4-V7FT-20261017/a40_geomvar/
# Usage: bash ops/a40/local_analyse.sh [SETTING]   (default: the host's EVAL_SETTING)
set -uo pipefail
P=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
WT=E:/agent-work/a40-geomvar
O=E:/agent-work/a40/offload
L=$O/root
OUT=E:/research-dense/runs/active/C4-V7FT-20261017/a40_geomvar
SETTING=${1:-$(tr -d '\r\n ' < "$O/host/EVAL_SETTING" 2>/dev/null)}
[ -n "$SETTING" ] || { echo "no EVAL_SETTING offloaded yet; pass the setting"; exit 1; }
mkdir -p "$OUT"
bad=$(cd "$L/evaluation" && sha256sum -c --quiet "$O/OFFLOAD.sha256" 2>&1 | grep -c FAILED)
[ "$bad" = 0 ] || { echo "$bad offloaded files differ from their verified sha"; exit 1; }
cd "$WT" || exit 1
export PYTHONPATH=. COOPMWM_MAP_SPEC=$WT/assets/research/newmap_candidates/wz3to2v80/MAP_SPEC.json
cp "$O/host/IDENTITY_GATE.json" "$O/host/cal/A40_CALSUMMARY.json" "$OUT"/ 2>/dev/null
$P ops/a40_geomvar.py analyse --setting "$SETTING" --final-root "$L" --out "$OUT" || { echo "analyse: not ready / failed"; exit 1; }
(cd "$OUT" && sha256sum *.json > SHA256SUMS)
