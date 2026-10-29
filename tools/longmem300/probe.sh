#!/bin/bash
# S-LONGMEM300 memory probe / fork launcher. Memory knobs only (env), nothing else differs
# from tools/candA stepZ recipe except [long_credit] enabled (configs/c4_policy14_longmem300.toml).
# usage (single 24 GB GPU, the QUALIFIED serial recipe; knobs are env vars):
#   TAG=fork1 FORK_CKPT=/path/policy_0000NN.pt MAXU=30 bash probe.sh
# usage (gpuA 47 GB, UNMEASURED two-VJP-worker pool recipe; see LONGMEM300.md):
#   TAG=fork1 WORKERS=2 SLOTS=2 FWD_WORKERS=3 ACTIVE=.40 LOW=.02 FWD=.26 CAP=.90 COORD=.10 \
#   FORK_CKPT=/path/policy_0000NN.pt MAXU=30 bash probe.sh
set -x
R=${R:-/root/autodl-tmp/c4/longmem300}          # staging root (outputs under $R/out)
RT=${RT:-$R/rt}                                   # staged runtime tree (cwd of the run)
INP=${INP:-$R/restart20261026}                    # gpuA layout: /root/autodl-tmp/c4/restart20261026
DATA_NATIVE=${DATA_NATIVE:-/root/autodl-tmp/c4/v5/data/v5_native_q}   # gpuA: .../restart20261026_inputs/data/v5_native_q
DATA_GENRAND=${DATA_GENRAND:-/root/autodl-tmp/c4/v5/data/genrand_plain_q}
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:?}
WORKERS=${WORKERS:-1}; SLOTS=${SLOTS:-1}; SEG=${SEG:-2}
ACTIVE=${ACTIVE:-.60}; LOW=${LOW:-.15}; FWD=${FWD:-.15}; CAP=${CAP:-.95}
START=${START:-a190}; UPDATES=${UPDATES:-1}; MAXU=${MAXU:-150}
DRY=""; [ "${DRYRUN:-0}" = 1 ] && DRY="--prep-dry-run-updates 2"
OUT=${OUT:-$R/out/$TAG}
cd $RT
# per-run config = longmem300 config + sweep segment (memory only)
CFG=configs/_lm300_$TAG.toml
cat > $CFG <<EOT
extends = "c4_policy14_longmem300.toml"
$( [ "${LONG:-1}" = 0 ] && printf '[long_credit]\nenabled = false\n' )
[trainer]
imagination_checkpoint_segment = $SEG
[run]
gpu_memory_fraction = $(awk "BEGIN{printf \"%.4f\", ${COORD:-.96}}")
EOT
export PYTHONPATH=$PWD/src:$PWD:$R/pylib C4_ALLOWED_ROOTS=/root/autodl-tmp/c4
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export C4_PROBES_DEVICE=cpu C4_GPU_MEM_FRACTION=${COORD:-.96}
export WMS_DATA_V5_NATIVE_Q=$DATA_NATIVE
export WMS_DATA_GENRAND_PLAIN_Q=$DATA_GENRAND
export C4_EXECUTION_PARITY=2 C4_EXECUTION_DRIVER=$PWD/configs/execparity_cav_idm_v1.json
export C4_SUMO_LEADER_RULE=1
export WMS_SPEED_FUTURE_WORKERS=$WORKERS WMS_SPEED_PARALLEL_FORWARD=1 WMS_SPEED_FUTURE_VJP_SLOTS=$SLOTS
export WMS_SPEED_FUTURE_MEMORY_FRACTION=$ACTIVE WMS_SPEED_FORWARD_MEMORY_FRACTION=$FWD
export WMS_SPEED_REPLAY_FORWARD_MEMORY_FRACTION=$LOW WMS_SPEED_FUTURE_BUDGET_CAP=$CAP
export WMS_SPEED_GRAPH_CVAR=1 WMS_SPEED_ASYNC_SPILL=1
# S-LONGMEM300 exact memory switches (default ON in this launcher; OFF in the library):
#   RECOMPUTE_REWARD=2: per-step reward/TTC recomputed one call at a time inside one autograd node
#   PAIRBIAS_CKPT=1  : WM pair-bias MLP recomputed in backward instead of saved
export WMS_LM300_RECOMPUTE_REWARD=${WMS_LM300_RECOMPUTE_REWARD:-2} WMS_LM300_PAIRBIAS_CKPT=${WMS_LM300_PAIRBIAS_CKPT:-1}
export WMS_LM300_TRACE=${WMS_LM300_TRACE:-1}
[ -n "$FWD_WORKERS" ] && export WMS_SPEED_FORWARD_WORKERS=$FWD_WORKERS
export WMS_WMSPEED_DEDUPE_MOBIL=1 WMS_WMSPEED_MOBIL_LAZY_SAFE=1 WMS_SPEED_PIVOT_DIAG_OFF=1
export WMS_SPEED_FAST_MERGE_LANE=1 WMS_SPEED_FAST_CAV_DRIVE=1 WMS_SPEED_FAST_LONG_HISTORY=1
export WMS_SPEED_GRAD_AUDIT_EVERY=${AUDIT:-25}
[ "$WORKERS" = 1 ] && unset WMS_SPEED_PARALLEL_FORWARD WMS_SPEED_FUTURE_VJP_SLOTS
L=$INP/arm/loop
if [ "$START" = a190 ]; then RESUME=$R/candA_prod_184542ab_p1/policy_000190.pt; else RESUME=$L/policy/c01.a3/policy_latest.pt; fi
[ -n "$FORK_CKPT" ] && RESUME=$FORK_CKPT   # fork from a candidate-A phase checkpoint (START is then irrelevant)
[ -n "$RESUME_OVERRIDE" ] && RESUME=$RESUME_OVERRIDE
# --new-phase from C1 needs the 18 registered keys; a fork from a candidate-A phase checkpoint
# (same trainer config) has NO differences and must pass none (unused keys are refused).
# A different imagination_checkpoint_segment than the checkpoint config needs only that key.
if [ "$START" = c1 ]; then
  ALLOW="--resume-allow trainer.checkpoint_every --resume-allow trainer.deterministic_crn --resume-allow trainer.entrant_control --resume-allow trainer.entropy_weight --resume-allow trainer.exploit_disable_direct_credit --resume-allow trainer.exploit_monitor --resume-allow trainer.future_samples --resume-allow trainer.imagination_checkpoint_segment --resume-allow trainer.imagination_microbatch --resume-allow trainer.imagination_offload_futures --resume-allow trainer.imagination_offload_saved_tensors --resume-allow trainer.native_anchor_calibration --resume-allow trainer.native_residual_train_group --resume-allow trainer.native_residual_train_trunk --resume-allow trainer.pessimistic_credit --resume-allow trainer.real_credit_individual_floor --resume-allow trainer.real_return_mix --resume-allow trainer.uncertainty_weight"
else
  ALLOW="${ALLOW_EXTRA:-}"
  # the p1/zs checkpoints were written with segment 2; any other segment is a declared memory-only diff
  [ "$SEG" != "${CKPT_SEG:-2}" ] && ALLOW="$ALLOW --resume-allow trainer.imagination_checkpoint_segment"
fi
mkdir -p $R/out
bash $HERE/sampler.sh $R/out/$TAG.gpu.csv &
SPID=$!
echo $SPID > $R/out/$TAG.sampler.pid
/root/miniconda3/bin/python -B -m wms.control.policy14_cli \
  --config $CFG \
  --wm-ckpt $L/wm/c02/wm_s.pt \
  --wm-config $INP/runtime_longint/configs/c4_wm_loop14_r16_ft.toml \
  --real-returns $L/returns/returns_c03.a1.npz \
  --longcritic-native-returns $L/returns/native_returns_c03.a1.npz \
  --paircrit-receipt $L/returns/pairing_c03.a1.json \
  --paircrit-policy-sha256 0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e \
  --gate-panel $INP/arm/gate_panel.json \
  --horizon-gate $L/longint/gate_wm2/HORIZON_GATE_WM2_REG.json \
  --looplong-boundary $RT/authority_gen/BOUNDARY_WM2.json \
  --critic-warmup-updates 0 \
  --native-init --phase-relative-signal-gate \
  --resume $RESUME --new-phase \
  $ALLOW \
  --device cuda --seed ${SEED:-965140} --max-updates $MAXU $DRY \
  --output-dir $OUT --compact
RC=$?
touch $R/out/$TAG.gpu.csv.stop
echo "rc=$RC"
exit $RC
