#!/bin/bash
# A102TS host chain (Box 3, /root/a102ts). ONE stage per call. prepare / run refuse when their chain file exists;
# dry / prephysics / check are zero SUMO and write only their own .out / chain line.
#   run_a102ts.sh prepare     zero SUMO: write-once bank copies into root/a94_banks (A94 prepare)   -> PREPARE_OK
#   run_a102ts.sh dry         zero SUMO: spec build + identity vs the stored A94 panel records      -> DRY_OK
#   run_a102ts.sh prephysics  zero SUMO: runner pre-physics / executor / digest / reservation / cap -> PREPHYSICS_OK
#   run_a102ts.sh run         SUMO 2 x 1,200 = 2,400 sim-s, 2 workers, raw KEPT; once, no retry     -> A102TS_RUN_DONE
#   run_a102ts.sh check       zero SUMO: a102ts_check.py (executor, J MATCH/MISMATCH, raw files)     -> A102TS_CHECK_DONE
# Writes only /root/a102ts. Stop before the run: touch /root/a102ts/root/STOP_A94 (the A94 runtime's stop file name).
STAGE=${1:?stage: prepare | dry | prephysics | run | check}
BASE=/root/a102ts
source $BASE/env_a102ts.sh
R=$BASE/root
export SRC A94_HAND_SHA256=none A94_BANK_SHA256=577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6
export A81_PANEL_SHA256=d85999ddaaf3eb583775d66dbabca168b4432966bc01e7193dfc79f244f7946c
ARGS="--final-root $R --a66-root /root/a66/root --a75d2-root /root/a75d2c/root"
cd "$SRC" || exit 2
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK"; exit 1; }
drv() { nice -n 19 "$PY" "$BASE/a102_ts.py" "$@" $ARGS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$STAGE" in
dry|prephysics)
  grep -q PREPARE_OK "$BASE/chain_prepare.out" 2>/dev/null || { echo "prepare not done"; exit 5; }
  O=$BASE/$STAGE.out
  drv "$STAGE" > "$O" 2>&1; rc=$?
  [ "$(charged)" = 0 ] || { echo "charged non-zero after $STAGE"; exit 3; }
  [ $rc = 0 ] || { tail -30 "$O"; echo "$STAGE FAILED rc $rc"; exit 3; }
  if [ "$STAGE" = dry ]; then log DRY_OK; else log PREPHYSICS_OK; fi
  tail -c 7000 "$O"; echo; echo "$STAGE OK (charged 0)"; exit 0 ;;
check)
  O=$BASE/check.out
  nice -n 19 python3 "$BASE/a102ts_check.py" > "$O" 2>&1; rc=$?
  cat "$O"; exit $rc ;;
esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$R/STOP_A94" ] && { echo "STOP_A94 present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 2 ] || { echo "free disk ${FREE}G < 2G: not launching"; exit 1; }
case "$STAGE" in
prepare)
  log "chain prepare start pid $$ free ${FREE}G"
  drv prepare > "$BASE/prepare.out" 2>&1 || { log "PREPARE FAILED"; cat "$BASE/prepare.out"; exit 3; }
  [ "$(charged)" = 0 ] || { log "prepare charged non-zero"; exit 3; }
  log "prepare $(tail -1 "$BASE/prepare.out" | cut -c1-600)"
  log PREPARE_OK
  cat "$BASE/prepare.out" ;;
run)
  grep -q DRY_OK "$BASE/chain_dry.out" 2>/dev/null && grep -q PREPHYSICS_OK "$BASE/chain_prephysics.out" 2>/dev/null \
    || { echo "dry / prephysics not OK: not launching"; exit 5; }
  [ "$(charged)" = 0 ] || { echo "root already charged: not launching"; exit 5; }
  log "chain run start pid $$ free ${FREE}G"
  drv run --workers 2 >> "$BASE/run.out" 2>&1; rc=$?
  log "run rc $rc charged $(charged)"
  [ $rc = 0 ] || { log "RUN FAILED rc $rc"; exit 8; }
  log A102TS_RUN_DONE ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
