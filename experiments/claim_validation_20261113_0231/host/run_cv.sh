#!/bin/bash
# Claim validation 20261113_0231 host chain (amendment 72). ONE STAGE per launch, detached, launched ONCE per stage,
# never inside a retry loop, no automatic replay.   CV_WORKERS=46 run_cv.sh STAGE [search seeds...]
#   ident     identity re-check after the table edits (8 episodes = 9,600 sim-s) -> root/IDENTITY_CV1.json
#   e2tune    5 uniform-hold speeds + U0 on 24 Development worlds -> E2T_RESULTS.json -> E2_SELECTION.json (H*)
#   e2conf    CONF_U0, F, S, H* on the 96 confirmation worlds -> E2C_RESULTS.json -> E2_ANALYSIS.json
#   e1plan    zero SUMO: candidate banks + VERIFY_PLAN.json (needs ORDER_WM_seed<N>.json of all six seeds)
#   e1verify  [seeds]  verification of the listed search seeds (default all six), one block after the other
#   e1bank    zero SUMO: admission + banks + FROZEN_BANKS.json (needs VERIFY_RESULTS of all six seeds)
#   e1conf    the budget-6 banks (+ CONF_U0, shared with e2conf if published in this root) on the 96 worlds -> analysis
# Only /root/cv_20261113_0231 is written. Stop between dispatch rounds: touch /root/cv_20261113_0231/root/STOP_CV1.
STAGE=${1:?stage: ident | e2tune | e2conf | e1plan | e1verify | e1bank | e1conf}
shift
B=/root/cv_20261113_0231
source $B/env_cv.sh
R=$B/root
O=$B/cv/out
REG=$B/cv/REGISTRATION.json
BANK=$B/inputs/A42_BIG_BANK_D2800_p60.json
W=${CV_WORKERS:?set CV_WORKERS}
TAG=$STAGE${1:+_$(echo "$@" | tr ' ' '_')}
C=$B/chain_$TAG.out
LOG=$B/$TAG.out
log() { echo "$(date '+%F %T') $*" >> "$C"; }
drv() { nice -n 19 "$PY" -B $B/cv/box_driver.py "$@" --registration $REG --code-root $B/src --final-root $R; }
py() { nice -n 19 "$PY" -B "$@"; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
ident_ok() { grep -q '"verdict": "PASS"' "$R/IDENTITY_CV1.json" 2>/dev/null; }
[ -f "$C" ] && { echo "$C exists: not relaunching"; exit 1; }
[ -f "$R/BOUND_U0.json" ] && [ -f "$R/prerun_plant/A_checkpoint_000.pt" ] || { echo "run host/setup_cv.sh first"; exit 1; }
log "chain $TAG start pid $$ workers $W host $(hostname)"
echo $$ > "$B/chain_$TAG.pid"
[ -f "$R/STOP_CV1" ] && { log "STOP_CV1 present"; exit 3; }
mkdir -p $O/ident
case "$STAGE" in
ident)
  drv prepare --block ident --waves $B/cv/host/IDENT_WAVES.json --banks-dir $B/inputs >> "$LOG" 2>&1 || { log "PREPARE FAILED"; exit 4; }
  drv run --block ident --waves $B/cv/host/IDENT_WAVES.json --workers 8 --chunk 8 >> "$LOG" 2>&1 || { log "IDENTITY RUN FAILED charged $(charged)"; exit 4; }
  drv collect --block ident --waves $B/cv/host/IDENT_WAVES.json --kind raw --out $O/ident/IDENT_RAW.json >> "$LOG" 2>&1 || { log "IDENTITY COLLECT FAILED"; exit 4; }
  py $B/cv/ident_compare.py --raw $O/ident/IDENT_RAW.json --a42-root /root/a66/refs/a42/root --out $R/IDENTITY_CV1.json >> "$LOG" 2>&1
  log "identity $(tail -1 "$LOG" | cut -c1-400) charged $(charged)"
  ident_ok || { log "IDENTITY FAILED: STOP"; exit 5; }
  log "CV_IDENT_DONE" ;;
e2tune)
  ident_ok || { log "identity is not PASS: nothing runs"; exit 5; }
  M=$O/e2/E2_MANIFEST.json
  drv prepare --block dev --waves $M --banks-dir $O/e2/banks $B/inputs >> "$LOG" 2>&1 || { log "PREPARE FAILED"; exit 6; }
  log "e2 tuning start"
  drv run --block dev --waves $M --workers "$W" --chunk 96 >> "$LOG" 2>&1 || { log "E2TUNE RUN FAILED charged $(charged)"; exit 6; }
  drv collect --block dev --waves $M --kind e2main --out $O/e2/E2T_RESULTS.json >> "$LOG" 2>&1 || { log "E2TUNE COLLECT FAILED"; exit 6; }
  py $B/cv/e2_select.py --out $O --results $O/e2/E2T_RESULTS.json --confirm-worlds 96 >> "$LOG" 2>&1 || { log "E2 SELECT FAILED"; exit 7; }
  log "selection: $(tail -1 "$LOG" | cut -c1-900) charged $(charged)"
  log "CV_E2TUNE_DONE" ;;
e2conf)
  ident_ok || { log "identity is not PASS: nothing runs"; exit 5; }
  S=$O/e2/E2_SELECTION.json
  [ -f "$S" ] || { log "no E2_SELECTION.json"; exit 7; }
  drv prepare --block conf --waves $S --banks-dir $O/e2/banks $B/inputs >> "$LOG" 2>&1 || { log "PREPARE FAILED"; exit 8; }
  log "e2 confirmation start"
  drv run --block conf --waves $S --workers "$W" --chunk 96 >> "$LOG" 2>&1 || { log "E2CONF RUN FAILED charged $(charged)"; exit 8; }
  drv collect --block conf --waves $S --kind e2main --out $O/e2/E2C_RESULTS.json >> "$LOG" 2>&1 || { log "E2CONF COLLECT FAILED"; exit 8; }
  py $B/cv/e2_analyse.py --bank $BANK --out $O --results $O/e2/E2C_RESULTS.json --selection $S >> "$LOG" 2>&1 || { log "E2 ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$LOG" | cut -c1-900) charged $(charged)"
  log "CV_E2CONF_DONE" ;;
e1plan)
  py $B/cv/e1_verify.py plan --out $O --seed-blocks $B/cv/host/SEED_BLOCKS.json --code-root $B/src >> "$LOG" 2>&1 || { log "E1 PLAN FAILED"; exit 10; }
  log "plan: $(tail -1 "$LOG" | cut -c1-600)"
  log "CV_E1PLAN_DONE" ;;
e1verify)
  ident_ok || { log "identity is not PASS: nothing runs"; exit 5; }
  P=$O/e1/verify/VERIFY_PLAN.json
  [ -f "$P" ] || { log "no VERIFY_PLAN.json"; exit 10; }
  for N in ${@:-11 23 42 67 89 103}; do
    [ -f "$R/STOP_CV1" ] && { log "STOP_CV1 present"; exit 3; }
    drv prepare --block e1v_seed$N --waves $P --banks-dir $O/e1/banks >> "$LOG" 2>&1 || { log "PREPARE FAILED seed $N"; exit 11; }
    log "e1 verification seed $N start"
    drv run --block e1v_seed$N --waves $P --workers "$W" --chunk 96 >> "$LOG" 2>&1 || { log "E1VERIFY RUN FAILED seed $N charged $(charged)"; exit 11; }
    drv collect --block e1v_seed$N --waves $P --kind e1verify --out $O/e1/verify/VERIFY_RESULTS_seed$N.json >> "$LOG" 2>&1 || { log "E1VERIFY COLLECT FAILED seed $N"; exit 11; }
    log "CV_E1VERIFY_SEED${N}_DONE charged $(charged)"
  done
  log "CV_E1VERIFY_DONE $*" ;;
e1bank)
  py $B/cv/e1_bank.py admit --out $O --results-dir $O/e1/verify >> "$LOG" 2>&1 || { log "E1 ADMIT FAILED"; exit 12; }
  py $B/cv/e1_bank.py build --out $O --confirm-worlds 96 >> "$LOG" 2>&1 || { log "E1 BUILD FAILED"; exit 12; }
  log "banks: $(tail -1 "$LOG" | cut -c1-900)"
  log "CV_E1BANK_DONE" ;;
e1conf)
  ident_ok || { log "identity is not PASS: nothing runs"; exit 5; }
  F=$O/e1/bank/FROZEN_BANKS.json
  [ -f "$F" ] || { log "no FROZEN_BANKS.json"; exit 12; }
  drv prepare --block conf --waves $F --banks-dir $O/e1/banks >> "$LOG" 2>&1 || { log "PREPARE FAILED"; exit 13; }
  log "e1 confirmation start"
  drv run --block conf --waves $F --workers "$W" --chunk 96 >> "$LOG" 2>&1 || { log "E1CONF RUN FAILED charged $(charged)"; exit 13; }
  drv collect --block conf --waves $F --kind e1confirm --out $O/e1/bank/CONFIRM_RESULTS.json >> "$LOG" 2>&1 || { log "E1CONF COLLECT FAILED"; exit 13; }
  py $B/cv/e1_bank.py analyse --out $O --results-dir $O/e1/bank >> "$LOG" 2>&1 || { log "E1 ANALYSE FAILED"; exit 14; }
  log "analysis: $(tail -1 "$LOG" | cut -c1-900) charged $(charged)"
  log "CV_E1CONF_DONE" ;;
*) log "unknown stage $STAGE"; exit 2 ;;
esac
