#!/bin/bash
# AMENDMENT 42 laptop analysis from the OFFLOADED files (sha-verified manifest): admission receipt, identity gate and
# the registered conjunction + secondaries -> E:/research-dense/runs/active/C4-V7FT-20261017/a42_bigp60/
# Usage: bash ops/a42/local_analyse.sh
set -uo pipefail
P=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
WT=E:/agent-work/a42-bigp60
L=E:/agent-work/a42-run/offload/root
RUN=E:/research-dense/runs/active/C4-V7FT-20261017
OUT=$RUN/a42_bigp60
mkdir -p "$OUT"
cd "$WT" || exit 1
export PYTHONPATH=. A42_HOST_BASE=/root/a42v
cp "$L"/ADMISSION_D2800_p60.json "$L"/IDENTITY_GATE.json "$OUT"/ 2>/dev/null
$P ops/a42_bigp60.py analyse --setting a42c_D2800_p60 --final-root "$L" --out "$OUT" || echo "analyse: not ready / failed"
(cd "$OUT" && sha256sum *.json > SHA256SUMS)
