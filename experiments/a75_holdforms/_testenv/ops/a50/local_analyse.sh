#!/bin/bash
# AMENDMENT 50 laptop analysis from the OFFLOADED files (sha-verified manifest): admission receipts, identity gate and
# the registered readings of both parts -> E:/research-dense/runs/active/C4-V7FT-20261017/a50_fullp60/
# Usage: bash ops/a50/local_analyse.sh
set -uo pipefail
P=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
WT=E:/agent-work/a50-fullp60
L=E:/agent-work/a50-run/offload/root
RUN=E:/research-dense/runs/active/C4-V7FT-20261017
OUT=$RUN/a50_fullp60
mkdir -p "$OUT"
cd "$WT" || exit 1
# the records name their bindings by the HOST final-root path; remap that location string only (nm_finaleval
# binding_location, commit 2b7d239f: must equal the offloaded LAUNCHER.json --final-root; binding content stays
# sha-verified and the record identity check stays exact). A50_HOST_BASE rebuilds the host bank paths of the specs.
# MSYS2_ENV_CONV_EXCL keeps Git Bash from rewriting either POSIX path.
export PYTHONPATH=. A50_HOST_BASE=/root/a50v NM_HOST_FINAL_ROOT=/root/a50v/root \
  MSYS2_ENV_CONV_EXCL='NM_HOST_FINAL_ROOT;A50_HOST_BASE'
cp "$L"/ADMISSION_A50_D2800_p60.json "$L"/ADMISSION_A50DIV_D2800_p60.json "$L"/IDENTITY_GATE.json "$OUT"/ 2>/dev/null
$P ops/a50_fullp60.py analyse --setting a50c_D2800_p60 --final-root "$L" --out "$OUT" || echo "analyse (big bank): not ready / failed"
$P ops/a50_divp60.py analyse --setting a50c_D2800_p60 --final-root "$L" --out "$OUT" || echo "analyse (diverse): not ready / failed"
(cd "$OUT" && sha256sum *.json > SHA256SUMS)
