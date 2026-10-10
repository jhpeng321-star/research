#!/usr/bin/env bash
# WM-S-R3-20261010 environment for gpuA. Source me, then run the chain:
#   source /root/autodl-tmp/wms_r3/_out/r3_box/r3_env.sh
#   cd /root/autodl-tmp/wms_r3 && bash _out/train_main_chain.sh preflight
# This file must stay LF: never let it gain CR bytes (the chain refuses CRLF).
# Round-3 recipe (docs/WM_CHANGE_LOG.md): C1 recipe + pack_v3 via lpwo_v3
# lists + sidecars_packv3_v2, R3-1/3/4 guards, 30-60 s history, braking losses
# WITHOUT the rejected residual cap. One GPU, one model; switching data and
# jam oversampling are OFF this round.

export BOX=gpuA ORIGINAL_ROOT=/root/autodl-tmp
export PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python
export PATH=/root/autodl-tmp/gpu1/env/rd_gpu/bin:$PATH

# Code and outputs. CODE_ROOT is created by _out/r3_box/stage_gpuA.sh;
# W is created by the chain itself on first run (never pre-create it).
export CODE_ROOT=/root/autodl-tmp/wms_r3
export W=/root/autodl-tmp/wms_train_r3/r3_main_a1

# Real packs stay at the proven MAIN2 gpuA locations (no re-copy). The
# packv3fix loader resolves the canonical list name pack_v3 to whatever
# basename the base pack is staged under (here: 'pack').
export PACK_V3=$ORIGINAL_ROOT/wmn1_m1_45c9032_20261003_A1/work/pack
export PACK_DSPEED=$ORIGINAL_ROOT/wmn1_a9_data/pack_dspeed
export PACK_DAMPING=$ORIGINAL_ROOT/wmn1_a10_data/pack_damping

# Sidecars: dspeed/damping stay at the real MAIN2 locations; the pack_v3
# identity + TRAIN outcomes sidecars come from the new verified bundle
# uploaded next to this file (sidecars_packv3_v2/).
export SIDECAR_ROOT=/root/autodl-tmp/wms_run2_20261006/work/sidecars
export IDENTITIES_V3=$CODE_ROOT/sidecars_packv3_v2/identities
export TRAIN_OUTCOMES=$CODE_ROOT/sidecars_packv3_v2/outcomes
export IDENTITIES_DSPEED=$SIDECAR_ROOT/identities_dspeed
export IDENTITIES_DAMPING=$ORIGINAL_ROOT/wmn1_a10_data/identities_damping
export OVERLAY_DSPEED=$ORIGINAL_ROOT/wmn1_a9_data/labels_overlay_dspeed
export OVERLAY_DAMPING=$ORIGINAL_ROOT/wmn1_a10_data/labels_overlay_damping
export MERGE_PARAMS=$CODE_ROOT/_out/merge_params.json

# Round-3 window lists (pack_v3 contributing, eval-world disjoint,
# holdout-guarded; receipt-hashed, verified at staging time).
export WINDOWS_TRAIN=$CODE_ROOT/lpwo_v3/windows_train_noPI_noWall.json
export WINDOWS_VAL=$CODE_ROOT/lpwo_v3/windows_val_noPI_noWall.json
export EXCLUDE=$CODE_ROOT/lpwo_v3/excluded_runs_PI_wall_v3.json

# Switching inputs stay staged (chain contract) but are data-off in R3:
# mix 0 + identity oversampling; jam oversampling stays deferred entirely.
export SW_ROOT=/root/autodl-tmp/wmn1_switching_train
export SW_WINDOWS_TRAIN=$SW_ROOT/windows_sw_train_part.json
export SW_WINDOWS_VAL=$SW_ROOT/windows_sw_monitor_part.json
export SWITCH_MIX_RATIO=0
export EVENT_OVERSAMPLE=1

# One structured model, residual rollout, lane head off; C1-style schedule.
export MODEL_KIND=structured ROLLOUT_MODE=residual LANE_HEAD=off
export CAV_DRIVE=teacher
export OMP=2 NUM_WORKERS=0
# Cap above the ~9-10 h budget; the smoke-gate projection must land below it.
export MAX_HOURS=14

# Box patches proven in MAIN2 (env-gated opt-outs; off-box defaults intact).
export SMOKE_CONTENTION_GUARD=0
export RSS_GUARD_BYTES=60000000000

# Braking losses ON, WITHOUT any residual cap (fair test of change #4;
# the hard clamp itself is rejected and must never return).
export ASYM_SPEED_LOSS=2 DECEL_WEIGHT=5 DECEL_THRESHOLD=3
export PROPAGATION_LOSS=1 PROP_WINDOW=3
export PROP_MAX_GAP_M=""
export RESIDUAL_CAP_NEG_MPS2=""
export BRAKING_FLAGS=""

# R3 guard/history flags: the integrated r3 chain must implement them; its
# CLI-help gate refuses unknown tokens before any training starts. Values are
# PROVISIONAL pre-run choices recorded in
# runs/active/WM-S-R3-20261010/REGISTRATION.md - re-confirm against the
# integrated CLI before launch.
#   R3-1 smooth residual bound + residual size penalty (guard vs change #3)
#   R3-3 lane-change-rate loss (guard vs change #6; recorded 1.23 %/HDV-step)
#   R3-4 keep the best long-horizon checkpoint instead of the last
#   R3-5 30-60 s per-driver history (score harness cuts history to 4.5 s)
export FLAGS="--residual-soft-bound-mps2 2.0 --residual-mag-weight 0.01 --lane-rate-weight 0.01 --lane-rate-target 0.0123 --select-by-long-horizon --history-s 60 --history-stride-s 15"

printf 'r3_env: BOX=%s CODE_ROOT=%s W=%s MAX_HOURS=%s model=%s flags=%s\n' \
  "$BOX" "$CODE_ROOT" "$W" "$MAX_HOURS" "$MODEL_KIND" "$FLAGS"
