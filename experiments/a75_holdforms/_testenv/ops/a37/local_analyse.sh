#!/bin/bash
# AMENDMENT 37 laptop analysis from the OFFLOADED files (sha-verified manifest): the registered per-cell contrasts,
# the admission receipts, the identity gate and 37b -> E:/research-dense/runs/active/C4-V7FT-20261017/a37_celltailor/
# Usage: bash ops/a37/local_analyse.sh
set -uo pipefail
P=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
WT=E:/agent-work/a37-celltailor
L=E:/agent-work/a37/offload/root
RUN=E:/research-dense/runs/active/C4-V7FT-20261017
OUT=$RUN/a37_celltailor
mkdir -p "$OUT"
cd "$WT" || exit 1
export PYTHONPATH=.
cp "$L"/ADMISSION_*.json "$L"/IDENTITY_GATE.json "$OUT"/ 2>/dev/null
for c in D3200_p20 D2400_p20; do
  $P ops/a37_celltailor.py analyse --setting a35_$c --final-root "$L" --out "$OUT" || echo "analyse $c: not ready / failed"
done
$P ops/a37_celltailor.py analyse-native --setting a33 --final-root "$L" --out "$OUT" --a33-root "$RUN/a33_combined/root" \
  || echo "analyse-native: not ready / failed"
$P ops/a37/render_a37.py "$OUT"
(cd "$OUT" && sha256sum *.json *.md > SHA256SUMS)
