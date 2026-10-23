#!/usr/bin/env bash
# R10B reviewer checks (M1-M4 + fix-B tails), CPU only.  Usage: run_all.sh MODE  (MODE = full | dry)
# Env: R10B_RUN (round-10b run dir), R9_CK, R10B_CK_OVERRIDE (dry), OUTROOT, WORKERS (<=24).
set -u -o pipefail
L=/root/autodl-tmp/c4/v5
PY=/root/miniconda3/bin/python
REPO=$L/repo_r10b_6a6f2afd
CHK=$L/chk_r10b
MODE=${1:-full}
WORKERS=${WORKERS:-24}
R9_CK=${R9_CK:-$L/runs/c4_wm_v10_round9/wm_s_stage300.pt}
RUN=${R10B_RUN:-$L/runs/c4_wm_v10_round10b}
if [ "$MODE" = dry ]; then
  RUN=${R10B_RUN:-$L/runs/smoke_r10b}; OUT=/dev/shm/chk_r10b_dry; LIM_ROLL="--limit 6 --k 1 --steps 120"; LIM_REM="--limit 6 --k 1"; LIM_OS="--limit 8 --stride-eps 30"; WORKERS=${WORKERS_DRY:-12}
elif [ "$MODE" = mid ]; then
  RUN=${R10B_RUN:-$L/runs/smoke_r10b}; OUT=/dev/shm/chk_r10b_mid; LIM_ROLL="--limit 48 --k 1 --steps 240"; LIM_REM="--limit 48 --k 1"; LIM_OS="--limit 24 --stride-eps 10"
else
  OUT=/dev/shm/chk_r10b; LIM_ROLL="--k 2 --steps 240"; LIM_REM="--k 2"; LIM_OS="--stride-eps 4"
fi
STATES=$CHK/states192.json.gz
mkdir -p "$OUT" "$CHK/results"
exec > >(tee -a "$CHK/results/run_all_$MODE.log") 2>&1
echo "run_all $MODE start $(date '+%F %T') run=$RUN workers=$WORKERS"
R10B_CK=$RUN/wm_s_stage300.pt
[ -f "$R10B_CK" ] || R10B_CK=$RUN/wm_s_best.pt
[ -f "$R10B_CK" ] || { echo "no r10b checkpoint in $RUN"; exit 2; }
echo "r10b checkpoint: $R10B_CK ($(sha256sum "$R10B_CK" | cut -c1-16))"
HZ_DIR=$OUT/headzero_ck; mkdir -p "$HZ_DIR"
$PY $CHK/r10b_checks/make_headzero.py "$R10B_CK" "$HZ_DIR/wm_s_headzero.pt" "$CHK/results/headzero_receipt_$MODE.json" || exit 3
VARS="--var r9=$R9_CK --var r10b=$R10B_CK --var headzero=$HZ_DIR/wm_s_headzero.pt"
export PYTHONPATH=$REPO/src:$REPO:$REPO/tools:$CHK/r10b_checks OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO" || exit 2
echo "== rollouts $(date +%T)"
$PY $CHK/r10b_checks/run_rollouts.py --repo $REPO --states $STATES $VARS --workers $WORKERS $LIM_ROLL --out $OUT/roll 2>&1 | grep -v -i warn | tail -5
echo "== removal $(date +%T)"
$PY $CHK/r10b_checks/run_remove.py --repo $REPO --states $STATES $VARS --workers $WORKERS $LIM_REM --out $OUT/remove 2>&1 | grep -v -i warn | tail -5
echo "== one-step $(date +%T)"
$PY $CHK/r10b_checks/run_onestep.py --repo $REPO $VARS --roots $L/data/v5_native $L/data/genrand_plain --workers $WORKERS $LIM_OS --out $OUT/os 2>&1 | grep -v -i warn | grep -v "kin = " | tail -5
echo "== gates $(date +%T)"
$PY $CHK/r10b_checks/gates.py $REPO $OUT/gates.json r9=$R9_CK r10b=$R10B_CK 2>&1 | tail -1
echo "== analysis $(date +%T)"
M5=$CHK/SUMMARY_R10B.md
echo "{\"mode\": \"$MODE\", \"r10b_checkpoint\": \"$R10B_CK\", \"r9_checkpoint\": \"$R9_CK\", \"date\": \"$(date '+%F %T')\"}" > $OUT/meta.json
$PY $CHK/r10b_checks/analyze.py --repo $REPO --shm $OUT --meta $OUT/meta.json --m5 $M5 --run $RUN --out $CHK/results/VERDICT_R10B_CHECKS_$MODE.md
echo "run_all $MODE done $(date '+%F %T')"
[ "$MODE" = full ] && cp "$CHK/results/VERDICT_R10B_CHECKS_full.md" "$CHK/VERDICT_R10B_CHECKS.md"
exit 0
