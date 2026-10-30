#!/bin/bash
# Prospective gpuA-alone entry; never invoked by stage.sh.
set -euo pipefail
NEW_OUTPUT_DIR=${1:?usage: launch.sh FRESH_OUTPUT_DIR [pilot|dry]}
MODE=${2:-pilot}
[[ $MODE == pilot || $MODE == dry ]]
RUNTIME=${RUNTIME:?set RUNTIME to the fresh, source-manifest-bound runtime}
cd "$RUNTIME"
[[ ! -e $NEW_OUTPUT_DIR && -d $(dirname "$NEW_OUTPUT_DIR") ]]
export PYTHONPATH=$PWD/src:$PWD C4_ALLOWED_ROOTS=/root/autodl-tmp/c4
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True C4_PROBES_DEVICE=cpu
export WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q
export WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q
export C4_EXECUTION_PARITY=2 C4_EXECUTION_DRIVER=$PWD/configs/execparity_cav_idm_v1.json
export C4_SUMO_LEADER_RULE=1
export WMS_SPEED_FUTURE_WORKERS=1 WMS_SPEED_PARALLEL_FORWARD=1 WMS_SPEED_FORWARD_WORKERS=2
export WMS_SPEED_FUTURE_MEMORY_FRACTION=.24 WMS_SPEED_FORWARD_MEMORY_FRACTION=.24
export WMS_SPEED_FUTURE_BUDGET_CAP=.90 WMS_SPEED_FUTURE_VJP_SLOTS=1
export WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION=.24 WMS_SPEED_GRAPH_CVAR=1 WMS_SPEED_ASYNC_SPILL=1
export WMS_WMSPEED_DEDUPE_MOBIL=1 WMS_WMSPEED_MOBIL_LAZY_SAFE=1 WMS_SPEED_PIVOT_DIAG_OFF=1
export WMS_SPEED_FAST_MERGE_LANE=1 WMS_SPEED_FAST_CAV_DRIVE=1 WMS_SPEED_FAST_LONG_HISTORY=1
export WMS_SPEED_GRAD_AUDIT_EVERY=25
export WMS_LM300_RECOMPUTE_REWARD=2 WMS_LM300_PAIRBIAS_CKPT=1 WMS_LM300_TRACE=1
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
AUTHORITY=${AUTHORITY:-authority_gen/CANDB_AUTHORITY.json}
"$PYTHON" -m tools.candB.authority verify "$AUTHORITY"
# A scoring/training peer (including S-WMREQUAL) must finish first. No signals.
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
mapfile -t INPUTS < <("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1]))["inputs"]; print("\n".join(d[k] for k in ("wm","wm_config","current_policy","horizon_gate","boundary","pairing_receipt","rankcheck","real_returns","native_returns","gate_panel")))' "$AUTHORITY")
[[ ${#INPUTS[@]} == 10 ]]
mapfile -t ALLOW_KEYS < <("$PYTHON" -c 'import torch,sys; from wms.control.train_cli import _read_config; from wms.control.dreamer import DreamerConfig; from wms.control.contract import RewardWeights; from wms.control.phase_resume import resume_diff; r=_read_config("configs/c4_policy14_candB.toml"); p=torch.load(sys.argv[1],map_location="cpu",weights_only=True); print("\n".join(sorted(resume_diff(DreamerConfig(**r["trainer"]),RewardWeights(**r["reward"]),p))))' "${INPUTS[2]}")
ALLOW=()
for key in "${ALLOW_KEYS[@]}"; do [[ -z $key ]] || ALLOW+=(--resume-allow "$key"); done
DRY=()
[[ $MODE != dry ]] || DRY+=(--prep-dry-run-updates 2)
exec "$PYTHON" -B -m wms.control.policy14_cli \
 --config configs/c4_policy14_candB.toml --wm-ckpt "${INPUTS[0]}" --wm-config "${INPUTS[1]}" \
 --real-returns "${INPUTS[7]}" \
 --longcritic-native-returns "${INPUTS[8]}" \
 --paircrit-receipt "${INPUTS[5]}" \
 --paircrit-policy-sha256 0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e \
 --gate-panel "${INPUTS[9]}" \
 --horizon-gate "${INPUTS[3]}" --looplong-boundary "${INPUTS[4]}" --rankcheck "${INPUTS[6]}" \
 --critic-warmup-updates 0 --native-init --phase-relative-signal-gate \
 --resume "${INPUTS[2]}" --new-phase "${ALLOW[@]}" "${DRY[@]}" \
 --device cuda --seed 965140 --max-updates 150 --output-dir "$NEW_OUTPUT_DIR" --compact
