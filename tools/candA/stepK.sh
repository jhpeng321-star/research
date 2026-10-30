#!/bin/bash
# S-KEYERR: ordinary same-phase continuation, preserving Adam/head and update-192 gate.
# Stage this tree at RUNTIME and refresh its source/authority manifest before launch.
set -euo pipefail
RUNTIME=${RUNTIME:-/root/autodl-tmp/c4/candA_keyerr}
PRIOR=/root/autodl-tmp/c4/candA_prod_184542ab_lgs
NEW_OUTPUT_DIR=${1:?usage: stepK.sh FRESH_OUTPUT_DIR}
cd "$RUNTIME"
export PYTHONPATH=$PWD/src:$PWD C4_ALLOWED_ROOTS=/root/autodl-tmp/c4
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export C4_PROBES_DEVICE=cpu C4_GPU_MEM_FRACTION=0.45
export WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q
export WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q
export C4_EXECUTION_PARITY=2 C4_EXECUTION_DRIVER=$RUNTIME/configs/execparity_cav_idm_v1.json
export C4_SUMO_LEADER_RULE=1
export WMS_SPEED_FUTURE_WORKERS=4 WMS_SPEED_PARALLEL_FORWARD=1 WMS_SPEED_FUTURE_VJP_SLOTS=4
export WMS_SPEED_FUTURE_MEMORY_FRACTION=.10 WMS_SPEED_FORWARD_MEMORY_FRACTION=.04
export WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION=.022 WMS_SPEED_GRAPH_CVAR=1 WMS_SPEED_ASYNC_SPILL=1
export WMS_WMSPEED_DEDUPE_MOBIL=1 WMS_WMSPEED_MOBIL_LAZY_SAFE=1 WMS_SPEED_PIVOT_DIAG_OFF=1
export WMS_SPEED_FAST_MERGE_LANE=1 WMS_SPEED_FAST_CAV_DRIVE=1 WMS_SPEED_FAST_LONG_HISTORY=1
export WMS_SPEED_GRAD_AUDIT_EVERY=25
set -a
. tools/candA/longmem300_serial_coloc.env
set +a
L=/root/autodl-tmp/c4/restart20261026/arm/loop
exec /root/miniconda3/bin/python -B -m wms.control.policy14_cli \
  --config configs/c4_policy14_candA_longgrad_skip.toml \
  --wm-ckpt "$L/wm/c02/wm_s.pt" \
  --wm-config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml \
  --real-returns "$L/returns/returns_c03.a1.npz" \
  --longcritic-native-returns "$L/returns/native_returns_c03.a1.npz" \
  --paircrit-receipt "$L/returns/pairing_c03.a1.json" \
  --paircrit-policy-sha256 0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e \
  --gate-panel /root/autodl-tmp/c4/restart20261026/arm/gate_panel.json \
  --horizon-gate "$L/longint/gate_wm2/HORIZON_GATE_WM2_REG.json" \
  --looplong-boundary "$RUNTIME/authority_gen/BOUNDARY_WM2.json" \
  --critic-warmup-updates 0 --native-init --phase-relative-signal-gate \
  --resume "$PRIOR/policy_000189.pt" --continue-run "$PRIOR" \
  --device cuda --seed 965140 --max-updates 150 \
  --output-dir "$NEW_OUTPUT_DIR" --compact
