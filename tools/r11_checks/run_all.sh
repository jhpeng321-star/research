#!/usr/bin/env bash
# R11 post-run reject checks (M1-M8; R11_REVIEW_PACKET.md section 7), CPU only, gpuR.  Reuses tools/r10b_checks
# unchanged.  Usage: run_all.sh MODE   (MODE = full | mid | dry)
# Arm mapping of the reused tools (labels in the output files):
#   "r9"       = REFERENCE  = round-10b final checkpoint (R11_REF_CK)
#   "r10b"     = ROUND 11   = round-11 final checkpoint (from R11_RUN)
#   "headzero" = round 11 with follower_courtesy.net.4.* zeroed (exact ablation; make_headzero.py works for the v2 head)
# Rollouts go through model.rollout, so the version-2 merge-zone counter is carried exactly as in training/scoring.
# Env: R11_RUN, R11_REF_CK, R11_REPO (code), R11_TOOLS (check tools; default $R11_REPO/tools), R11_CHK, STATES (default:
# the round-10b check states), WORKERS (<= 24).
set -u -o pipefail
L=/root/autodl-tmp/c4/v5
PY=/root/miniconda3/bin/python
REPO=${R11_REPO:-$L/repo_r11}
TOOLS=${R11_TOOLS:-$REPO/tools}
CHK=${R11_CHK:-$L/chk_r11}
MODE=${1:-full}
WORKERS=${WORKERS:-24}
REF_CK=${R11_REF_CK:-$L/runs/c4_wm_v10_round10b/wm_s_stage300.pt}
[ -f "$REF_CK" ] || REF_CK=$L/runs/c4_wm_v10_round10b/wm_s_best.pt
RUN=${R11_RUN:-$L/runs/c4_wm_v10_round11}
STATES=${STATES:-$L/chk_r10b/states192.json.gz}
if [ "$MODE" = dry ]; then
  RUN=${R11_RUN:-$L/runs/smoke_r11}; OUT=/dev/shm/chk_r11_dry; LIM_ROLL="--limit 6 --k 1 --steps 120"; LIM_REM="--limit 6 --k 1"; LIM_OS="--limit 8 --stride-eps 30"; WORKERS=${WORKERS_DRY:-12}
elif [ "$MODE" = mid ]; then
  RUN=${R11_RUN:-$L/runs/smoke_r11}; OUT=/dev/shm/chk_r11_mid; LIM_ROLL="--limit 48 --k 1 --steps 240"; LIM_REM="--limit 48 --k 1"; LIM_OS="--limit 24 --stride-eps 10"
else
  OUT=/dev/shm/chk_r11; LIM_ROLL="--k 2 --steps 240"; LIM_REM="--k 2"; LIM_OS="--stride-eps 4"
fi
[ -f "$STATES" ] || { echo "states file missing: $STATES"; exit 2; }
mkdir -p "$OUT" "$CHK/results"
if [ "$(cd "$TOOLS/r10b_checks" && pwd -P)" != "$(cd "$CHK" && pwd -P)/r10b_checks" ]; then
  rm -rf "$CHK/r10b_checks" && cp -r "$TOOLS/r10b_checks" "$CHK/r10b_checks" || exit 2
fi
exec > >(tee -a "$CHK/results/run_all_$MODE.log") 2>&1
echo "r11 run_all $MODE start $(date '+%F %T') run=$RUN ref=$REF_CK workers=$WORKERS"
R11_CK=$RUN/wm_s_stage300.pt
[ -f "$R11_CK" ] || R11_CK=$RUN/wm_s_best.pt
[ -f "$R11_CK" ] || { echo "no round-11 checkpoint in $RUN"; exit 2; }
[ -f "$REF_CK" ] || { echo "no round-10b reference checkpoint"; exit 2; }
echo "round-11 checkpoint: $R11_CK ($(sha256sum "$R11_CK" | cut -c1-16)); reference: $REF_CK ($(sha256sum "$REF_CK" | cut -c1-16))"
HZ_DIR=$OUT/headzero_ck; mkdir -p "$HZ_DIR"
$PY $CHK/r10b_checks/make_headzero.py "$R11_CK" "$HZ_DIR/wm_s_headzero.pt" "$CHK/results/headzero_receipt_$MODE.json" || exit 3
VARS="--var r9=$REF_CK --var r10b=$R11_CK --var headzero=$HZ_DIR/wm_s_headzero.pt"
export PYTHONPATH=$REPO/src:$REPO:$REPO/tools:$CHK/r10b_checks OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO" || exit 2
echo "== rollouts $(date +%T)"
$PY $CHK/r10b_checks/run_rollouts.py --repo $REPO --states $STATES $VARS --workers $WORKERS $LIM_ROLL --out $OUT/roll 2>&1 | grep -v -i warn | tail -5
echo "== removal $(date +%T)"
$PY $CHK/r10b_checks/run_remove.py --repo $REPO --states $STATES $VARS --workers $WORKERS $LIM_REM --out $OUT/remove 2>&1 | grep -v -i warn | tail -5
echo "== one-step $(date +%T)"
$PY $CHK/r10b_checks/run_onestep.py --repo $REPO $VARS --roots $L/data/v5_native $L/data/genrand_plain --workers $WORKERS $LIM_OS --out $OUT/os 2>&1 | grep -v -i warn | grep -v "kin = " | tail -5
echo "== gates $(date +%T)"
$PY $CHK/r10b_checks/gates.py $REPO $OUT/gates.json r9=$REF_CK r10b=$R11_CK 2>&1 | tail -1
echo "== M7 head range (R11_REVIEW C1) $(date +%T)"
$PY $TOOLS/r11_checks/m7_head_range.py "$RUN" "$REPO/configs/c4_wm_v10round11.toml" "$CHK/results/M7_head_range_$MODE.json" 2>&1 | tail -1
echo "== analysis $(date +%T)"
echo "{\"mode\": \"$MODE\", \"labels\": {\"r9\": \"round-10b reference\", \"r10b\": \"round 11\", \"headzero\": \"round 11 head zeroed\"}, \"r10b_checkpoint\": \"$R11_CK\", \"r9_checkpoint\": \"$REF_CK\", \"date\": \"$(date '+%F %T')\"}" > $OUT/meta.json
$PY $CHK/r10b_checks/analyze.py --repo $REPO --shm $OUT --meta $OUT/meta.json --m5 $CHK/SUMMARY_R11.md --run $RUN --out $CHK/results/VERDICT_R11_CHECKS_$MODE.md
echo "== M8 absolute bias + keep-10b rule (R11_REVIEW C3) $(date +%T)"
$PY $TOOLS/r11_checks/absolute_bias.py --shm $OUT --out $CHK/results/M8_R11_$MODE.md --json $CHK/results/M8_R11_$MODE.json 2>&1 | grep -v -i warn | tail -1
echo "r11 run_all $MODE done $(date '+%F %T')"
# One verdict file: label mapping + the reused M1-M7 verdict + the M7 head-range rule + M8 (keep-10b rule).
COMBINED=$CHK/results/VERDICT_R11_CHECKS_COMBINED_$MODE.md
{
  echo "# VERDICT_R11_CHECKS ($MODE, $(date '+%F %T'))"
  echo
  echo "Label mapping in the reused M1-M7 section: r9 = ROUND 10b (reference, $REF_CK), r10b = ROUND 11 ($R11_CK),"
  echo "headzero = round 11 with follower_courtesy.net.4.* zeroed. Registered rules: R11_REVIEW_PACKET.md section 7."
  echo
  echo "## M7 head-saturation rule (R11_REVIEW C1, all steps)"
  echo
  echo '```'
  head -c 1500 "$CHK/results/M7_head_range_$MODE.json" 2>/dev/null || echo "M7 json missing"
  echo
  echo '```'
  echo
  cat "$CHK/results/M8_R11_$MODE.md" 2>/dev/null || echo "M8 missing"
  echo
  echo "---"
  echo
  cat "$CHK/results/VERDICT_R11_CHECKS_$MODE.md" 2>/dev/null || echo "M1-M7 verdict missing"
} > "$COMBINED"
[ "$MODE" = full ] && cp "$COMBINED" "$CHK/VERDICT_R11_CHECKS.md"
exit 0
