#!/bin/bash
# S-POLSPEED: launch ONE benchmark variant of candidate A's first update (own process, own dir).
# usage: launch_variant.sh NAME RUNTIME_DIR "ENVVARS" [toml override lines...]
set -eu
NAME=$1; RT=$2; EXTRA_ENV=$3; shift 3
B=/root/autodl-tmp/c4/polspeed20261028
R=/root/autodl-tmp/c4/restart20261026/arm/loop/c4prep/resolved_A_MAIN1500_8e75184392b3
L=/root/autodl-tmp/c4/restart20261026/arm/loop
mkdir -p $B/cfg $B/logs
CFG=$B/cfg/$NAME.toml
{ echo "extends = \"$R/POLICY.toml\""; for line in "$@"; do echo -e "$line"; done; } > $CFG
OUT=$B/out/$NAME
mkdir -p $B/out
export PYTHONPATH=$RT/src:$RT C4_ALLOWED_ROOTS=/root/autodl-tmp/c4 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} \
  C4_PROBES_DEVICE=cpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q \
  WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q \
  WMS_SPEED_GRAD_AUDIT_EVERY=25 C4_GPU_MEM_FRACTION=0.30
for kv in $EXTRA_ENV; do export "$kv"; done
cd /root/autodl-tmp/c4/restart20261026/runtime_c4prep_8e75184392b3
nohup setsid /root/miniconda3/bin/python -B -c '
import ctypes, runpy, sys
ctypes.CDLL(None).prctl(0x59616d61, ctypes.c_ulong(-1), 0, 0, 0)  # PR_SET_PTRACER_ANY: allow py-spy on THIS process only
sys.argv = ["policy14_cli"] + sys.argv[1:]
runpy.run_module("wms.control.policy14_cli", run_name="__main__")
' --config $CFG --wm-ckpt $L/wm/c03.a2/wm_s.pt --wm-config $RT/configs/c4_wm_loop14_r16_ft.toml \
  --real-returns $L/returns/returns_c03.a1.npz --longcritic-native-returns $L/returns/native_returns_c03.a1.npz \
  --paircrit-receipt $L/returns/pairing_c03.a1.json --paircrit-policy-sha256 0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e \
  --gate-panel /root/autodl-tmp/c4/restart20261026/arm/gate_panel.json --horizon-gate $R/HORIZON_AUTHORITY.json \
  --looplong-boundary $R/BOUNDARY.json --native-init --phase-relative-signal-gate \
  --real-credit $L/sumocredit/REAL_CREDIT_c03.a1.json --critic-warmup-updates 0 \
  --resume $L/policy/c01.a3/policy_latest.pt --device cuda --seed 965140 --max-updates 150 \
  --output-dir $OUT --compact --prep-dry-run-updates 2 > $B/logs/$NAME.log 2>&1 < /dev/null &
echo "$NAME pid $!"
echo "{\"name\": \"$NAME\", \"pid\": $!, \"env\": \"$EXTRA_ENV\", \"t\": \"$(date -Is)\"}" >> $B/PIDS.jsonl
