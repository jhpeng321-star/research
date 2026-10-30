#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/c4/setupaudit20261030/fix
export PYTHONPATH=src:.:tools/setupaudit C4_ALLOWED_ROOTS=/root/autodl-tmp/c4
export C4_EXECUTION_PARITY=2 C4_SUMO_LEADER_RULE=1
export C4_EXECUTION_DRIVER=$PWD/configs/execparity_cav_idm_v1.json
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export WMS_WMSPEED_DEDUPE_MOBIL=1 WMS_WMSPEED_MOBIL_LAZY_SAFE=1
export WMS_SPEED_PIVOT_DIAG_OFF=1 WMS_SPEED_FAST_MERGE_LANE=1
export WMS_SPEED_FAST_CAV_DRIVE=1 WMS_SPEED_FAST_LONG_HISTORY=1
export WMS_SPEED_GRAPH_CVAR=1
exec /root/miniconda3/bin/python -u "$@"
