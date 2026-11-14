#!/bin/bash
# Claim validation 20261113_0231, E1-FIX arm WM-FIX (POST HOC REPAIR, Amendment 74, outside the A72 Holm family; never replaces
# the registered primary). Additive to run_cv.sh: same root, same driver, same guards, own chain files, own outputs
# (cv/out/e1/fix/ only). ONE STAGE per launch, detached, launched ONCE per stage, never inside a retry loop, no
# automatic replay.   CV_WORKERS=46 run_cv_fix.sh STAGE [search seeds...]
#   e1fixplan    zero SUMO: WM-FIX top six minus the candidates the registered verification or the WM-ACT addendum already ran
#                (needs cv/out/e1/order/ORDER_WMFIX_seed<N>.json of all six seeds) -> fix/VERIFY_PLAN_FIX.json
#   e1fixverify  [seeds]  the NEW candidates only, 24 worlds per seed, the block's registered U0 wave (a root that
#                published E1V<seed>_U0 re-runs nothing for it) -> fix/VERIFY_RESULTS_FIX_seed<N>.json
#   e1fixbank    zero SUMO: admission (registered rule) + budget-6 WM-FIX banks + admitted-count contrast vs Random
#                -> fix/ADMISSION_FIX_seed<N>.json, fix/FROZEN_BANKS_FIX.json
#   e1fixconf    the WM-FIX banks the registered confirmation did not already run, on the 96 confirmation worlds
#                (CONF_U0 reused if published in this root) -> fix/CONFIRM_RESULTS_FIX.json -> fix/E1_FIX_ANALYSIS.json
# Registered inputs (read only): CV_REG_VERIFY = directory of the registered VERIFY_RESULTS_seed<N>.json of all six
# seeds, CV_REG_BANK = directory of the registered FROZEN_BANKS.json, ADMISSION_seed<N>.json, CONFIRM_RESULTS.json.
# Defaults: cv/out/e1/act/registered/{verify,bank} when present, else cv/out/e1/{verify,bank}.
# Only /root/cv_20261113_0231 is written. Stop between dispatch rounds: touch /root/cv_20261113_0231/root/STOP_CV1.
STAGE=${1:?stage: e1fixplan | e1fixverify | e1fixbank | e1fixconf}
shift
B=/root/cv_20261113_0231
source $B/env_cv.sh
R=$B/root
O=$B/cv/out
A=$O/e1/fix
REG=$B/cv/REGISTRATION.json
W=${CV_WORKERS:?set CV_WORKERS}
RV=${CV_REG_VERIFY:-$([ -d $O/e1/act/registered/verify ] && echo $O/e1/act/registered/verify || echo $O/e1/verify)}
RB=${CV_REG_BANK:-$([ -d $O/e1/act/registered/bank ] && echo $O/e1/act/registered/bank || echo $O/e1/bank)}
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
log "chain $TAG start pid $$ workers $W host $(hostname) [WM-FIX: POST HOC REPAIR, Amendment 74]"
echo $$ > "$B/chain_$TAG.pid"
[ -f "$R/STOP_CV1" ] && { log "STOP_CV1 present"; exit 3; }
case "$STAGE" in
e1fixplan)
  py $B/cv/e1_fix.py plan --out $O --code-root $B/src >> "$LOG" 2>&1 || { log "E1FIX PLAN FAILED"; exit 20; }
  log "plan: $(tail -1 "$LOG" | cut -c1-900)"
  log "CV_E1FIXPLAN_DONE" ;;
e1fixverify)
  ident_ok || { log "identity is not PASS: nothing runs"; exit 5; }
  P=$A/VERIFY_PLAN_FIX.json
  [ -f "$P" ] || { log "no VERIFY_PLAN_FIX.json"; exit 20; }
  for N in ${@:-11 23 42 67 89 103}; do
    [ -f "$R/STOP_CV1" ] && { log "STOP_CV1 present"; exit 3; }
    py $B/cv/e1_fix.py has-waves --out $O --seed $N >> "$LOG" 2>&1; H=$?
    [ $H = 10 ] && { log "seed $N: no new candidate, nothing to run"; continue; }
    [ $H = 0 ] || { log "PLAN CHECK FAILED seed $N"; exit 21; }
    drv prepare --block e1v_seed$N --waves $P --banks-dir $A/banks >> "$LOG" 2>&1 || { log "PREPARE FAILED seed $N"; exit 21; }
    log "e1 WM-FIX verification seed $N start charged $(charged)"
    drv run --block e1v_seed$N --waves $P --workers "$W" --chunk 96 >> "$LOG" 2>&1 || { log "E1FIXVERIFY RUN FAILED seed $N charged $(charged)"; exit 21; }
    drv collect --block e1v_seed$N --waves $P --kind e1verify --out $A/VERIFY_RESULTS_FIX_seed$N.json >> "$LOG" 2>&1 || { log "E1FIXVERIFY COLLECT FAILED seed $N"; exit 21; }
    log "CV_E1FIXVERIFY_SEED${N}_DONE charged $(charged)"
  done
  log "CV_E1FIXVERIFY_DONE $*" ;;
e1fixbank)
  py $B/cv/e1_fix.py admit --out $O --results-dir $RV >> "$LOG" 2>&1 || { log "E1FIX ADMIT FAILED (registered results: $RV)"; exit 22; }
  py $B/cv/e1_fix.py build --out $O --registered-bank-dir $RB >> "$LOG" 2>&1 || { log "E1FIX BUILD FAILED (registered banks: $RB)"; exit 22; }
  log "banks: $(tail -1 "$LOG" | cut -c1-1200)"
  log "CV_E1FIXBANK_DONE" ;;
e1fixconf)
  F=$A/FROZEN_BANKS_FIX.json
  [ -f "$F" ] || { log "no FROZEN_BANKS_FIX.json"; exit 22; }
  py $B/cv/e1_fix.py has-waves --out $O --confirm >> "$LOG" 2>&1; H=$?
  if [ $H = 0 ]; then
    ident_ok || { log "identity is not PASS: nothing runs"; exit 5; }
    drv prepare --block conf --waves $F --banks-dir $A/banks >> "$LOG" 2>&1 || { log "PREPARE FAILED"; exit 23; }
    log "e1 WM-FIX confirmation start charged $(charged)"
    drv run --block conf --waves $F --workers "$W" --chunk 96 >> "$LOG" 2>&1 || { log "E1FIXCONF RUN FAILED charged $(charged)"; exit 23; }
    drv collect --block conf --waves $F --kind e1confirm --out $A/CONFIRM_RESULTS_FIX.json >> "$LOG" 2>&1 || { log "E1FIXCONF COLLECT FAILED"; exit 23; }
  elif [ $H = 10 ]; then
    log "every WM-FIX bank is U0 or already confirmed (registered e1conf or WM-ACT): nothing to run"
  else
    log "FROZEN_BANKS_FIX CHECK FAILED"; exit 23
  fi
  py $B/cv/e1_fix.py analyse --out $O --registered-bank-dir $RB >> "$LOG" 2>&1 || { log "E1FIX ANALYSE FAILED (registered confirmation: $RB)"; exit 24; }
  log "analysis: $(tail -1 "$LOG" | cut -c1-1200) charged $(charged)"
  log "CV_E1FIXCONF_DONE" ;;
*) log "unknown stage $STAGE"; exit 2 ;;
esac
