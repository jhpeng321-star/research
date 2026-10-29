#!/bin/bash
# S-ZEROSTART: candidate-A phase 1 zero-start. usage: stepZ.sh dry|prod (dense 90 s) | drylong|prodlong (gain-only 300 s long credit + longmem300.env)
# Identical to candA_prod_184542ab_p1 (stepB recipe: nine-row N4, fractions .10/.04/.022 x4 workers,
# segment 2, cap 0.45, leader rule ON, mode-2 law, monitors ON, new-phase from C1 64f70af4)
# except: zero-start (policy14.zero_residual_output), [exploration] rho 0.10->0.50 radius 100, random cooperation group U{2..6}, n_flip 0 + w_team 0 (config af8a8d8a -> see SOURCE.json commit).
set -x
MODE=$1
cd /root/autodl-tmp/c4/candA_184542ab_zs
export PYTHONPATH=$PWD/src:$PWD C4_ALLOWED_ROOTS=/root/autodl-tmp/c4
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export C4_PROBES_DEVICE=cpu C4_GPU_MEM_FRACTION=0.45
export WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q
export WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q
export C4_EXECUTION_PARITY=2 C4_EXECUTION_DRIVER=/root/autodl-tmp/c4/candA_184542ab_zs/configs/execparity_cav_idm_v1.json
export C4_SUMO_LEADER_RULE=1
export WMS_SPEED_FUTURE_WORKERS=4 WMS_SPEED_PARALLEL_FORWARD=1 WMS_SPEED_FUTURE_VJP_SLOTS=4
export WMS_SPEED_FUTURE_MEMORY_FRACTION=.10 WMS_SPEED_FORWARD_MEMORY_FRACTION=.04
export WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION=.022 WMS_SPEED_GRAPH_CVAR=1 WMS_SPEED_ASYNC_SPILL=1
export WMS_WMSPEED_DEDUPE_MOBIL=1 WMS_WMSPEED_MOBIL_LAZY_SAFE=1 WMS_SPEED_PIVOT_DIAG_OFF=1
export WMS_SPEED_FAST_MERGE_LANE=1 WMS_SPEED_FAST_CAV_DRIVE=1 WMS_SPEED_FAST_LONG_HISTORY=1
export WMS_SPEED_GRAD_AUDIT_EVERY=25
L=/root/autodl-tmp/c4/restart20261026/arm/loop
CONFIG=configs/c4_policy14_candA.toml
EXTRA=""
case "$MODE" in
  dry)  EXTRA="--prep-dry-run-updates 2"; OUT=/root/autodl-tmp/c4/candA_dry_184542ab_zs ;;
  prod) OUT=/root/autodl-tmp/c4/candA_prod_184542ab_zs ;;
  drylong|prodlong)
    # gain-only 300 s long credit variant: the S-LONGMEM300 qualified memory recipe (env file) is REQUIRED.
    [ -f tools/candA/longmem300.env ] || { echo "missing tools/candA/longmem300.env (S-LONGMEM300 recipe)"; exit 2; }
    CONFIG=configs/c4_policy14_candA_long300.toml
    set -a; . tools/candA/longmem300.env; set +a
    if [ "$MODE" = drylong ]; then EXTRA="--prep-dry-run-updates 2"; OUT=/root/autodl-tmp/c4/candA_dry_184542ab_zs_long
    else OUT=/root/autodl-tmp/c4/candA_prod_184542ab_zs_long; fi ;;
  *) echo "usage: stepZ.sh dry|prod|drylong|prodlong"; exit 2 ;;
esac
exec /root/miniconda3/bin/python -B -m wms.control.policy14_cli \
  --config $CONFIG \
  --wm-ckpt $L/wm/c02/wm_s.pt \
  --wm-config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml \
  --real-returns $L/returns/returns_c03.a1.npz \
  --longcritic-native-returns $L/returns/native_returns_c03.a1.npz \
  --paircrit-receipt $L/returns/pairing_c03.a1.json \
  --paircrit-policy-sha256 0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e \
  --gate-panel /root/autodl-tmp/c4/restart20261026/arm/gate_panel.json \
  --horizon-gate $L/longint/gate_wm2/HORIZON_GATE_WM2_REG.json \
  --looplong-boundary /root/autodl-tmp/c4/candA_184542ab_zs/authority_gen/BOUNDARY_WM2.json \
  --critic-warmup-updates 0 \
  --native-init --phase-relative-signal-gate \
  --resume $L/policy/c01.a3/policy_latest.pt --new-phase \
  --resume-allow trainer.checkpoint_every --resume-allow trainer.deterministic_crn \
  --resume-allow trainer.entrant_control --resume-allow trainer.entropy_weight \
  --resume-allow trainer.exploit_disable_direct_credit --resume-allow trainer.exploit_monitor \
  --resume-allow trainer.future_samples --resume-allow trainer.imagination_checkpoint_segment \
  --resume-allow trainer.imagination_microbatch --resume-allow trainer.imagination_offload_futures \
  --resume-allow trainer.imagination_offload_saved_tensors \
  --resume-allow trainer.native_anchor_calibration \
  --resume-allow trainer.native_residual_train_group \
  --resume-allow trainer.native_residual_train_trunk --resume-allow trainer.pessimistic_credit \
  --resume-allow trainer.real_credit_individual_floor --resume-allow trainer.real_return_mix \
  --resume-allow trainer.uncertainty_weight \
  --device cuda --seed 965140 --max-updates 150 $EXTRA \
  --output-dir $OUT --compact
