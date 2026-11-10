#!/bin/bash
# AMENDMENT 41 laptop re-analysis from the OFFLOADED files (sha-verified manifest): the identity gate, the sealed stage-1
# SELECTION receipt (re-derived and compared) and the stage-2 primary conjunction + secondaries
# -> E:/research-dense/runs/active/C4-V7FT-20261017/a41_d3200bound/
# Usage: bash ops/a41/local_analyse.sh
set -uo pipefail
P=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
WT=E:/agent-work/a41-d3200bound
L=E:/agent-work/a41/offload/root
RUN=E:/research-dense/runs/active/C4-V7FT-20261017
OUT=$RUN/a41_d3200bound
mkdir -p "$OUT"
cd "$WT" || exit 1
export PYTHONPATH=.
cp "$L"/SELECTION_A41.json "$L"/IDENTITY_GATE.json "$OUT"/ 2>/dev/null
$P ops/a41_d3200bound.py reselect --final-root "$L" > "$OUT/RESELECT.json" || echo "reselect: not ready / failed"
$P ops/a41_d3200bound.py analyse --final-root "$L" --out "$OUT" || echo "analyse: not ready / failed"
(cd "$OUT" && sha256sum *.json > SHA256SUMS)
