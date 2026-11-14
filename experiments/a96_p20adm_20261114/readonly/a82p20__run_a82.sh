#!/bin/bash
# AMENDMENT 82 (DRAFT) host chain. ONE stage per launch; each physics / sealing stage runs ONCE (its chain file exists ->
# refuse); physics stages are meant to be detached by the caller (setsid -f nohup ... < /dev/null); no retry loop; no
# automatic replay. The base (and so the part) is the directory of this script: /root/a82p20 (part A) or /root/a82four
# (part B). Only the base and /dev/shm (tests) are written; /root/a76p20 is only read.
# Part A (/root/a82p20):  test | synthdry | pin | mech | admit | bank
#   pin   needs A82_TOP_FILE + A82_TOP_SHA256 (the S3b file and its registered sha)
#   mech  first 2 verification episodes of every candidate (<= 24 episodes) -> mech-compare (A82_MECH.json)
#   admit the rest of the <= 12 x 24 verification episodes (cap 345,600 incl. mech); reference NOT re-run
#   bank  zero physics: receipt invariants + a76 admit (re-pointed) -> A82_ADMISSION_D2800_p20.json + bank file
# Part B (/root/a82four): stage | synthdry | test | pin | mech | confirm
#   stage    zero SUMO: the 96 / 192 / 384 panels (write-once, $BASE/panels) + IDENT_U0 bind + A76 bank copy (real root)
#   pin      needs A82_N_WORLDS (96|192|384), A82_BANK_FILE + A82_BANK_SHA256 + A82_BANK_SOURCE,
#            A82_MAPPO20C_POLICY + A82_MAPPO20C_SHA256 + A82_TRAIN_RECEIPT + A82_TRAIN_PLAN [A82F_PANEL_SHA256]
#   mech     8 identity episodes (9,600 sim-s) -> mech-compare (A82F_MECH.json PASS / FAIL)
#   confirm  4 arms x N worlds (raw dropped per round) -> analyse (A82F_RESULTS_D2800_p20.json)
# Physics stages need A82_WORKERS. Stop between dispatch rounds: touch $BASE/root/STOP_A82 (A) / STOP_A82F (B).
STAGE=${1:?stage}
BASE=$(cd "$(dirname "$0")" && pwd)
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK in $BASE"; exit 1; }
MODE=$(cat "$BASE/MODE")
source "$BASE/env_a82.sh"
R=$BASE/root
SR=$BASE/synth/root
cd "$SRC" || exit 2
charged() { python3 -c "import json,os;p='$1/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
if [ "$MODE" = admit ]; then
  DRV=ops/a82_s3b20.py; MECHF=A82_MECH.json; STOPF=STOP_A82
  run() { nice -n 19 "$PY" $DRV "$@" --final-root "$R" --a76-root /root/a76p20/root; }
else
  DRV=ops/a82_fourarm.py; MECHF=A82F_MECH.json; STOPF=STOP_A82F
  REFS="--a76-root /root/a76p20/root --a32-ref $BASE/refs/a32"
  run() { nice -n 19 "$PY" $DRV "$@" --final-root "$R" $REFS; }
fi
case "$MODE:$STAGE" in
admit:test)
  exec nice -n 19 "$PY" -m pytest -q -rs ops/test_a82_s3b20.py ops/test_a82_fourarm.py -p no:cacheprovider --basetemp=/dev/shm/a82_test_$$ ;;
four:test)
  A82F_TEST_ROOT=$SR A82F_HOST_BASE=$BASE/synth exec nice -n 19 "$PY" -m pytest -q -rs ops/test_a82_fourarm.py ops/test_a82_s3b20.py \
    -p no:cacheprovider --basetemp=/dev/shm/a82f_test_$$ ;;
admit:synthdry)
  [ -f "$SR/A82_CANDIDATES.json" ] && { echo "synthetic root already pinned: not rerunning"; exit 1; }
  export A82_HOST_BASE=$BASE/synth A82_TOP_FILE=$SRC/ops/a82/SYNTH_TOP_OFFER_CONDITIONS_D2800_p20.json
  export A82_TOP_SHA256=$(sha256sum "$A82_TOP_FILE" | cut -c1-64) A76_COMP_SHA256=d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca
  srun() { nice -n 19 "$PY" $DRV "$@" --final-root "$SR" --a76-root /root/a76p20/root; }
  { srun pin-top && srun pin && srun prepare && srun dry; } > "$BASE/synth/dry.out" 2>&1 || { echo "SYNTHDRY FAILED"; tail -5 "$BASE/synth/dry.out"; exit 3; }
  srun status >> "$BASE/synth/dry.out" 2>&1
  [ -f "$SR/CHARGES.jsonl" ] && { echo "SYNTHDRY charged SUMO seconds: must not happen"; exit 3; }
  echo "SYNTHDRY_OK charged $(charged "$SR")"; exit 0 ;;
admit:pin)
  [ -n "${A82_TOP_FILE:-}" ] && [ -n "${A82_TOP_SHA256:-}" ] || { echo "A82_TOP_FILE and A82_TOP_SHA256 required"; exit 1; }
  [ -f "$R/A82_CANDIDATES.json" ] && { echo "already pinned: not rerunning"; exit 1; }
  export A76_COMP_SHA256=d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca
  { run pin-top && run pin && run prepare; } > "$BASE/pin.out" 2>&1 || { echo "PIN REFUSED (see $BASE/pin.out, $R/A82_MAPPING_REFUSED.json): STOP"; tail -3 "$BASE/pin.out"; exit 3; }
  run dry > "$BASE/dry.out" 2>&1 || { echo "DRY FAILED"; tail -3 "$BASE/dry.out"; exit 3; }
  cat "$BASE/pin.out"; echo PIN_OK; exit 0 ;;
four:stage)
  [ -f "$BASE/STAGE_OK" ] && { echo "stage already done"; exit 1; }
  { run stage-panels && run bind --which IDENT_U0 && run prepare; } > "$BASE/stage.out" 2>&1 || { echo "STAGE FAILED"; tail -5 "$BASE/stage.out"; exit 3; }
  date "+%F %T" > "$BASE/STAGE_OK"; cat "$BASE/stage.out"; echo STAGE_OK; exit 0 ;;
four:synthdry)
  [ -f "$BASE/STAGE_OK" ] || { echo "run stage first"; exit 1; }
  [ -f "$SR/A82F_MECH.json" ] || [ -f "$SR/CHARGES.jsonl" ] && { echo "synthetic root is not clean: not rerunning"; exit 1; }
  B76=/root/a76p20/root/a76_banks/A76_BANK_D2800_p20.json
  export A82F_HOST_BASE=$BASE/synth A82F_SYNTH_SUBSTITUTE=1 A82_BANK_FILE=$B76 A82_BANK_SHA256=$(sha256sum $B76 | cut -c1-64) A82_BANK_SOURCE=SYNTHDRY_PLACEHOLDER_A76
  srun() { nice -n 19 "$PY" $DRV "$@" --final-root "$SR" $REFS; }
  { srun panel --n 192 --panels-dir "$BASE/panels" && { [ -f "$SR/BOUND_A77_IDENT_U0.json" ] || srun bind --which IDENT_U0; }     && { [ -f "$SR/A82F_BANK.json" ] || srun prepare; } && srun dry --stage i && srun dry --stage c && srun status; } > "$BASE/synth/dry.out" 2>&1 \
    || { echo "SYNTHDRY FAILED"; tail -5 "$BASE/synth/dry.out"; exit 3; }
  [ -f "$SR/CHARGES.jsonl" ] && { echo "SYNTHDRY charged SUMO seconds: must not happen"; exit 3; }
  echo "SYNTHDRY_OK charged $(charged "$SR")"; exit 0 ;;
four:pin)
  [ -f "$BASE/STAGE_OK" ] || { echo "run stage first"; exit 1; }
  for v in A82_N_WORLDS A82_BANK_FILE A82_BANK_SHA256 A82_BANK_SOURCE A82_MAPPO20C_POLICY A82_MAPPO20C_SHA256 A82_TRAIN_RECEIPT A82_TRAIN_PLAN; do
    [ -n "${!v:-}" ] || { echo "$v required"; exit 1; }; done
  [ -f "$R/A82F_PANEL_C.json" ] && { echo "already pinned: not rerunning"; exit 1; }
  { run panel --n "$A82_N_WORLDS" && run bind --which MAPPO20C && run prepare; } > "$BASE/pin.out" 2>&1 || { echo "PIN FAILED"; tail -5 "$BASE/pin.out"; exit 3; }
  { run dry --stage i && run dry --stage c; } > "$BASE/dry.out" 2>&1 || { echo "DRY FAILED"; tail -5 "$BASE/dry.out"; exit 3; }
  cat "$BASE/pin.out"; echo PIN_OK; exit 0 ;;
esac
W=${A82_WORKERS:?set A82_WORKERS}
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$MODE:$STAGE" in admit:mech|four:mech) NEED=2 ;; admit:admit) NEED=4 ;; admit:bank) NEED=1 ;; four:confirm) NEED=6 ;;
  *) echo "unknown stage $MODE:$STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$R/$STOPF" ] && { echo "$STOPF present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G: not launching"; exit 1; }
mechpass() { grep -q '"verdict": "PASS"' "$R/$MECHF" 2>/dev/null; }
export A76_COMP_SHA256=d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca
case "$MODE:$STAGE" in
admit:mech|four:mech) [ -f "$R/$MECHF" ] && { echo "mechanics ran already"; exit 1; } ;;
admit:admit) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; } ;;
admit:bank) grep -q A82_VERIFY_DONE "$BASE/chain_admit.out" 2>/dev/null || { echo "admit stage not done"; exit 5; } ;;
four:confirm) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; } ;;
esac
log "chain $MODE $STAGE start pid $$ workers $W free ${FREE}G"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$MODE:$STAGE" in
admit:mech)
  run mech --workers $(( W < 24 ? W : 24 )) --chunk 24 >> "$BASE/mech.out" 2>&1 || { log "MECH RUN FAILED charged $(charged "$R")"; exit 4; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-2500) charged $(charged "$R")"
  mechpass || { log "MECHANICS FAILED: STOP"; exit 5; }
  log "A82_MECH_DONE" ;;
admit:admit)
  run verify --workers "$W" --chunk 96 --drop-raw --drop-procs 16 >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED charged $(charged "$R")"; exit 6; }
  run drop-raw --drop-procs 16 >> "$BASE/verify.out" 2>&1 || { log "DROP-RAW FAILED"; exit 6; }
  log "verify done; charged $(charged "$R")"
  log "A82_VERIFY_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")" ;;
admit:bank)
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED: $(tail -1 "$BASE/admit.out" | cut -c1-1500)"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-2500)"
  log "A82_BANK_DONE" ;;
four:mech)
  run mech --workers $(( W < 8 ? W : 8 )) --chunk 8 >> "$BASE/mech.out" 2>&1 || { log "MECH RUN FAILED charged $(charged "$R")"; exit 4; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-2500) charged $(charged "$R")"
  mechpass || { log "MECHANICS FAILED: STOP"; exit 5; }
  log "A82F_MECH_DONE" ;;
four:confirm)
  run confirm --workers "$W" --chunk 96 --drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED charged $(charged "$R")"; exit 8; }
  run drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { log "DROP-RAW FAILED"; exit 8; }
  log "confirm done; charged $(charged "$R")"
  run analyse >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 9; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A82F_CONFIRM_DONE" ;;
esac
