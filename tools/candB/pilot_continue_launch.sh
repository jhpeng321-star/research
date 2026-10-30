#!/bin/bash
set -euo pipefail
OUT=${1:?fresh EXPLORATORY output}
PARENT=${2:?owned stopped phase}
[[ ${3:-} == --exploratory-owner-20261030 && $# == 3 ]]
cd "${RUNTIME:?}"
[[ ! -e $OUT && -d $(dirname "$OUT") && $OUT == *EXPLORATORY* ]]
ulimit -n 65536
export PYTHONPATH=$PWD/src:$PWD C4_ALLOWED_ROOTS=/root/autodl-tmp/c4
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True C4_PROBES_DEVICE=cpu
export WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q
export WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q
export C4_EXECUTION_PARITY=2 C4_EXECUTION_DRIVER=$PWD/configs/execparity_cav_idm_v1.json
export C4_SUMO_LEADER_RULE=1
export WMS_SPEED_FUTURE_WORKERS=1 WMS_SPEED_PARALLEL_FORWARD=0 WMS_SPEED_FORWARD_WORKERS=1
export WMS_SPEED_FUTURE_MEMORY_FRACTION=.24 WMS_SPEED_FORWARD_MEMORY_FRACTION=.24
export WMS_SPEED_FUTURE_BUDGET_CAP=.44 WMS_SPEED_FUTURE_VJP_SLOTS=1
export WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION=.04 WMS_SPEED_GRAPH_CVAR=1 WMS_SPEED_ASYNC_SPILL=1
export WMS_WMSPEED_DEDUPE_MOBIL=1 WMS_WMSPEED_MOBIL_LAZY_SAFE=1 WMS_SPEED_PIVOT_DIAG_OFF=1
export WMS_SPEED_FAST_MERGE_LANE=1 WMS_SPEED_FAST_CAV_DRIVE=1 WMS_SPEED_FAST_LONG_HISTORY=1
export WMS_SPEED_GRAD_AUDIT_EVERY=25
export WMS_LM300_RECOMPUTE_REWARD=2 WMS_LM300_PAIRBIAS_CKPT=1 WMS_LM300_TRACE=1
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
AUTHORITY=${AUTHORITY:-authority_gen/PILOT_EXPLORATORY_AUTHORITY.json}
"$PYTHON" -B -m tools.candB.pilot_authority --exploratory-owner-20261030 verify "$AUTHORITY"
"$PYTHON" -c 'import os,time; assert 0 <= time.time()-float(os.environ["PILOT_ADMISSION_EPOCH"]) < 60; assert int(os.environ["PILOT_ADMISSION_FREE_MIB"]) >= 43000'
exec "$PYTHON" -B -u -m tools.candB.pilot_continue --exploratory-owner-20261030 --authority "$AUTHORITY" --parent "$PARENT" "$OUT"
