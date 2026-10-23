#!/usr/bin/env bash
# R12 post-run reject checks, CPU only, gpuR.  Usage: run_all.sh MODE   (MODE = full | mid | dry)
# Structure as tools/r11_checks/run_all.sh; reuses tools/r10b_checks (default-off R12 flags only) and tools/r11_checks.
# Arm mapping of the reused tools (labels in the output files):
#   "r9"       = REFERENCE  = ROUND 9 final checkpoint (R12_REF_CK; legacy driver encoder)
#   "r10b"     = ROUND 12   = round-12 final checkpoint (from R12_RUN; physical driver encoder)
#   "headzero" = round 12 with follower_courtesy.net.4.* zeroed (exact ablation)
# One-step: ALL arms are replayed on the SAME filled-presence replay state (run_onestep.py --replay-present filled, the
# round-12 loss.recorded_replay_present switch) with the round-12 config; the round-9 arm gets its legacy encoder via
# --arm-set (run_onestep refuses an arm whose checkpoint encoder mode differs from its loaded config).
# Steps: rollouts (seed set 1, seeded: --seed-offset 0), [SEEDS2=1: rollouts seed set 2, --seed-offset $SEED2_OFFSET,
# arms r9 + r10b], removal, one-step, gates, M7 head range, analyze.py (M1-M7; M5 proxy vs round 9 on seed set 1),
# M8 (tools/r12_checks/m8_r12.py: absolute bias + behaviour rule, seed-replicated M5 with state-bootstrap CIs) and the
# BASE checks (tools/r12_checks/base_calibration.py, rules BASE-1..4 in its header).
# Combined verdict: $CHK/results/VERDICT_R12_CHECKS_COMBINED_$MODE.md (full mode: copied to $CHK/VERDICT_R12_CHECKS.md).
# Env: R12_RUN, R12_REF_CK, R12_REPO (code), R12_TOOLS (check tools; default $R12_REPO/tools), R12_CHK, STATES (default:
# the round-10b check states), WORKERS (<= 24), SEEDS2 (0|1, default 0), SEED2_OFFSET (default 1000),
# R12_OS_EXTRA (extra run_onestep.py args, e.g. --arm-set for dry-run stand-ins), R12_STEPS (M7 expected steps, 110).
# analyze.py's own M7 line still says "expected 90" (reused unchanged); the M7 head-range json uses R12_STEPS.
# Cost (gpuR, 20-24 workers; scaled from the R11 estimate): full ~75-90 min; SEEDS2=1 adds one rollout pass with two of
# the three arms, ~35 min (rollouts were ~50 min for three arms), i.e. ~110-125 min in total.
set -u -o pipefail
L=/root/autodl-tmp/c4/v5
PY=/root/miniconda3/bin/python
REPO=${R12_REPO:-$L/repo_r12}
TOOLS=${R12_TOOLS:-$REPO/tools}
CHK=${R12_CHK:-$L/chk_r12}
MODE=${1:-full}
WORKERS=${WORKERS:-24}
[ "$WORKERS" -gt 24 ] && WORKERS=24
SEEDS2=${SEEDS2:-0}
SEED2_OFFSET=${SEED2_OFFSET:-1000}
R12_STEPS=${R12_STEPS:-110}   # configs/c4_wm_v10round12.toml stage_steps 40 + 70
CFG=c4_wm_v10round12.toml
REF_CK=${R12_REF_CK:-$L/runs/c4_wm_v10_round9/wm_s_stage300.pt}
[ -f "$REF_CK" ] || REF_CK=$L/runs/c4_wm_v10_round9/wm_s_best.pt
RUN=${R12_RUN:-$L/runs/c4_wm_v10_round12}
STATES=${STATES:-$L/chk_r10b/states192.json.gz}
if [ "$MODE" = dry ]; then
  RUN=${R12_RUN:-$L/runs/smoke_r12}; OUT=/dev/shm/chk_r12_dry; LIM_ROLL="--limit 6 --k 1 --steps 120"; LIM_REM="--limit 6 --k 1"; LIM_OS="--limit 8 --stride-eps 30"; WORKERS=${WORKERS_DRY:-12}
elif [ "$MODE" = mid ]; then
  RUN=${R12_RUN:-$L/runs/smoke_r12}; OUT=/dev/shm/chk_r12_mid; LIM_ROLL="--limit 48 --k 1 --steps 240"; LIM_REM="--limit 48 --k 1"; LIM_OS="--limit 24 --stride-eps 10"
elif [ "$MODE" = full ]; then
  OUT=/dev/shm/chk_r12; LIM_ROLL="--k 2 --steps 240"; LIM_REM="--k 2"; LIM_OS="--stride-eps 4"
else
  echo "MODE must be full | mid | dry"; exit 2
fi
[ -f "$STATES" ] || { echo "states file missing: $STATES"; exit 2; }
mkdir -p "$OUT" "$CHK/results"
if [ "$(cd "$TOOLS/r10b_checks" && pwd -P)" != "$(cd "$CHK" && pwd -P)/r10b_checks" ]; then
  rm -rf "$CHK/r10b_checks" && cp -r "$TOOLS/r10b_checks" "$CHK/r10b_checks" || exit 2
fi
exec > >(tee -a "$CHK/results/run_all_$MODE.log") 2>&1
echo "r12 run_all $MODE start $(date '+%F %T') run=$RUN ref=$REF_CK workers=$WORKERS seeds2=$SEEDS2"
R12_CK=$RUN/wm_s_stage300.pt
[ -f "$R12_CK" ] || R12_CK=$RUN/wm_s_best.pt
[ -f "$R12_CK" ] || { echo "no round-12 checkpoint in $RUN"; exit 2; }
[ -f "$REF_CK" ] || { echo "no round-9 reference checkpoint"; exit 2; }
echo "round-12 checkpoint: $R12_CK ($(sha256sum "$R12_CK" | cut -c1-16)); reference (round 9): $REF_CK ($(sha256sum "$REF_CK" | cut -c1-16))"
HZ_DIR=$OUT/headzero_ck; mkdir -p "$HZ_DIR"
$PY $CHK/r10b_checks/make_headzero.py "$R12_CK" "$HZ_DIR/wm_s_headzero.pt" "$CHK/results/headzero_receipt_$MODE.json" || exit 3
VARS="--var r9=$REF_CK --var r10b=$R12_CK --var headzero=$HZ_DIR/wm_s_headzero.pt"
# round 9 = legacy encoder; its init switches are irrelevant at inference but set to their round-9 values
R9_SET="--arm-set r9=model.driver_param_mode=legacy --arm-set r9=model.driver_param_init=checkpoint --arm-set r9=model.residual_accel_reinit=false"
export PYTHONPATH=$REPO/src:$REPO:$REPO/tools:$CHK/r10b_checks OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO" || exit 2
echo "== rollouts, seed set 1 $(date +%T)"
$PY $CHK/r10b_checks/run_rollouts.py --repo $REPO --states $STATES $VARS --workers $WORKERS $LIM_ROLL --seed-offset 0 --out $OUT/roll 2>&1 | grep -v -i warn | tail -5
SHM2_ARG=""
if [ "$SEEDS2" = 1 ]; then
  echo "== rollouts, seed set 2 (offset $SEED2_OFFSET; r9 + r10b) $(date +%T)"
  $PY $CHK/r10b_checks/run_rollouts.py --repo $REPO --states $STATES --var r9=$REF_CK --var r10b=$R12_CK --workers $WORKERS $LIM_ROLL --seed-offset $SEED2_OFFSET --out $OUT/seeds2/roll 2>&1 | grep -v -i warn | tail -5
  SHM2_ARG="--shm2 $OUT/seeds2"
fi
echo "== removal $(date +%T)"
$PY $CHK/r10b_checks/run_remove.py --repo $REPO --states $STATES $VARS --workers $WORKERS $LIM_REM --out $OUT/remove 2>&1 | grep -v -i warn | tail -5
echo "== one-step (filled replay presence, all arms) $(date +%T)"
# shellcheck disable=SC2086
$PY $CHK/r10b_checks/run_onestep.py --repo $REPO --config $CFG $VARS $R9_SET --replay-present filled --base-cols ${R12_OS_EXTRA:-} --roots $L/data/v5_native $L/data/genrand_plain --workers $WORKERS $LIM_OS --out $OUT/os 2>&1 | grep -v -i warn | grep -v "kin = " | tail -5
echo "== gates $(date +%T)"
$PY $CHK/r10b_checks/gates.py $REPO $OUT/gates.json r9=$REF_CK r10b=$R12_CK 2>&1 | tail -1
echo "== M7 head range (round-12 config; REJECT-ROUND-11 in its output = reject round 12) $(date +%T)"
$PY $TOOLS/r11_checks/m7_head_range.py "$RUN" "$REPO/configs/$CFG" "$CHK/results/M7_head_range_$MODE.json" "$R12_STEPS" 2>&1 | tail -1
echo "== analysis (M1-M7, reference round 9, seed set 1) $(date +%T)"
echo "{\"mode\": \"$MODE\", \"labels\": {\"r9\": \"round 9 reference\", \"r10b\": \"round 12\", \"headzero\": \"round 12 head zeroed\"}, \"r10b_checkpoint\": \"$R12_CK\", \"r9_checkpoint\": \"$REF_CK\", \"onestep_replay_present\": \"filled\", \"seeds2\": $SEEDS2, \"date\": \"$(date '+%F %T')\"}" > $OUT/meta.json
$PY $CHK/r10b_checks/analyze.py --repo $REPO --shm $OUT --meta $OUT/meta.json --m5 $CHK/SUMMARY_R12.md --run $RUN --out $CHK/results/VERDICT_R12_CHECKS_$MODE.md
echo "== M8 absolute bias + behaviour rule, seed-replicated M5 $(date +%T)"
# shellcheck disable=SC2086
$PY $TOOLS/r12_checks/m8_r12.py --shm $OUT $SHM2_ARG --out $CHK/results/M8_R12_$MODE.md --json $CHK/results/M8_R12_$MODE.json 2>&1 | grep -v -i warn | tail -1
echo "== BASE checks (one-step base calibration) $(date +%T)"
$PY $TOOLS/r12_checks/base_calibration.py --os $OUT/os --out $CHK/results/BASE_R12_$MODE.md --json $CHK/results/BASE_R12_$MODE.json 2>&1 | grep -v -i warn | tail -1
echo "r12 run_all $MODE done $(date '+%F %T')"
# One verdict file: label mapping + BASE + M7 head-range rule + M8 / seed-replicated M5 + the reused M1-M7 verdict.
COMBINED=$CHK/results/VERDICT_R12_CHECKS_COMBINED_$MODE.md
{
  echo "# VERDICT_R12_CHECKS ($MODE, $(date '+%F %T'))"
  echo
  echo "Label mapping in the reused sections: r9 = ROUND 9 (reference, $REF_CK), r10b = ROUND 12 ($R12_CK),"
  echo "headzero = round 12 with follower_courtesy.net.4.* zeroed. One-step rows: every arm replayed with the FILLED"
  echo "replay presence. Seed set 2: $([ "$SEEDS2" = 1 ] && echo "yes (offset $SEED2_OFFSET, arms r9 + r10b)" || echo "not run (SEEDS2=0)")."
  echo "Rules: BASE-1..4 (base_calibration.py header), M7 (R11 rule, round-12 config), M8 / M5 (m8_r12.py header),"
  echo "M1-M4 as R11_REVIEW_PACKET.md section 7 with round 9 as the reference. The M7 completion line inside the reused"
  echo "verdict says 'expected 90'; round 12 has $R12_STEPS steps (the M7 head-range json is authoritative)."
  echo
  cat "$CHK/results/BASE_R12_$MODE.md" 2>/dev/null || echo "BASE missing"
  echo
  echo "## M7 head-saturation rule (all steps; REJECT-ROUND-11 = reject round 12)"
  echo
  echo '```'
  head -c 1500 "$CHK/results/M7_head_range_$MODE.json" 2>/dev/null || echo "M7 json missing"
  echo
  echo '```'
  echo
  cat "$CHK/results/M8_R12_$MODE.md" 2>/dev/null || echo "M8 missing"
  echo
  echo "---"
  echo
  cat "$CHK/results/VERDICT_R12_CHECKS_$MODE.md" 2>/dev/null || echo "M1-M7 verdict missing"
} > "$COMBINED"
[ "$MODE" = full ] && cp "$COMBINED" "$CHK/VERDICT_R12_CHECKS.md"
exit 0
