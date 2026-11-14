#!/bin/bash
# AMENDMENT 98 (A98) host chain (/root/a98). Derived from run_a94.sh. ONE stage per launch (except test / drytest); a
# stage whose chain file exists refuses; no retry; no automatic replay. Writes only /root/a98 (and /dev/shm for tests).
#   test     zero SUMO unit tests
#   drytest  zero SUMO: /dev/shm root (banks prepared there): dry i, dry c, e2e (fake A94 root from A91 records)
#   pin      zero SUMO: prepare (the two pinned bank files) + dry i / c -> PIN_OK
#   mech     rule checks: 2 arms x 2 A70 worlds (4,800 sim-s) -> A98_MECH.json PASS required
#   confirm  2 arms x 192 worlds (460,800 sim-s; raw dropped per round) -> analyse (needs A94Q_DONE + A94 confirm done)
#   analyse  zero SUMO re-run of the analysis ONLY if confirm ran and no results file exists
# Stop between rounds: touch /root/a98/root/STOP_A98.
STAGE=${1:?stage: test | drytest | pin | mech | confirm | analyse}
BASE=/root/a98
source $BASE/env_a98.sh
R=$BASE/root
A91R=/root/a91/root
REFS="--a66-root /root/a66/root --a70-root /root/a75/a70ref/root --ref70-reports /root/a75/ref70 --a75d2-root /root/a75d2c/root"
PANEL=d85999ddaaf3eb583775d66dbabca168b4432966bc01e7193dfc79f244f7946c
cd "$SRC" || exit 2
run() { nice -n 19 "$PY" ops/a98_addon.py "$@" --final-root "$R" $REFS; }
runr() { local root=$1; shift; nice -n 19 "$PY" ops/a98_addon.py "$@" --final-root "$root" $REFS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK"; exit 1; }
export A81_PANEL_SHA256=$PANEL
if [ "$STAGE" = test ]; then
  exec "$PY" -m pytest -q -rs ops/test_a98_addon.py -p no:cacheprovider --basetemp=/dev/shm/a98_test_$$
fi
if [ "$STAGE" = drytest ]; then
  T=/dev/shm/a98_drytest_$$; TR=$T/root; trap 'rm -rf "$T"' EXIT
  mkdir -p "$TR/prerun_plant" "$TR/oracle_banks" "$TR/a32_banks"
  for f in A94_PANEL_C.json A94_PANEL_C.sha256 BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json \
           prerun_plant/A_checkpoint_000.pt oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json \
           oracle_banks/LEARNED_TRIGGER_MODEL.json a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
    cp "$R/$f" "$TR/$f" || { echo "DRYTEST: copy $f failed"; exit 3; }
  done
  echo "== prepare"; runr "$TR" prepare || { echo "DRYTEST: prepare failed"; exit 3; }
  echo; echo "== dry i"; runr "$TR" dry --stage i > "$T/dry_i.out" 2>&1 || { cat "$T/dry_i.out"; echo "DRYTEST: dry i failed"; exit 3; }
  python3 -c "import json;d=json.load(open('$T/dry_i.out'));print({k:d[k] for k in ('arms','episodes','sim_s','stage_cap','charged','cap','families','spec_diff_S4S5_vs_S4')})"
  echo "== dry c"; runr "$TR" dry --stage c > "$T/dry_c.out" 2>&1 || { cat "$T/dry_c.out"; echo "DRYTEST: dry c failed"; exit 3; }
  python3 -c "import json;d=json.load(open('$T/dry_c.out'));print({k:d[k] for k in ('panel_sha256','arms','contrasts','episodes','sim_s','stage_cap','charged','cap')});print([w[:7] for w in d['waves']])"
  echo "== e2e"; nice -n 19 "$PY" ops/a98_e2e.py --a91-root $A91R --final-root "$TR" --shm $T/e2e > "$T/e2e.out" 2>&1 || { cat "$T/e2e.out"; echo "DRYTEST: e2e failed"; exit 3; }
  tail -2 "$T/e2e.out" | cut -c1-1500
  [ -f "$TR/CHARGES.jsonl" ] && { echo "DRYTEST: CHARGES.jsonl appeared"; exit 3; }
  echo "DRYTEST_OK charged 0 (scratch $T removed on exit)"; exit 0
fi
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
stop() { log "CHAIN_STOPPED $*"; printf '%s\n' "{\"stage\": \"$STAGE\", \"time\": \"$(date "+%F %T")\", \"reason\": \"$*\", \"charged\": $(charged), \"chain_log\": \"$C\"}" > "$BASE/CHAIN_STOPPED"; exit "${RC:-9}"; }
case "$STAGE" in pin|analyse) NEED=1 ;; mech) NEED=2 ;; confirm) NEED=5 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$BASE/CHAIN_STOPPED" ] && { echo "CHAIN_STOPPED present: not launching"; exit 1; }
[ -f "$R/STOP_A98" ] && { echo "STOP_A98 present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G: not launching"; exit 1; }
mechpass() { grep -q '"verdict": "PASS"' "$R/A98_MECH.json" 2>/dev/null; }
case "$STAGE" in
mech) grep -q PIN_OK "$BASE/chain_pin.out" 2>/dev/null || { echo "not pinned"; exit 5; } ;;
confirm) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; } ;;
analyse) grep -q A98_CONFIRM_RUN_DONE "$BASE/chain_confirm.out" 2>/dev/null || { echo "confirm not done"; exit 5; }
         [ -f "$R/A98_RESULTS_D2800_p60.json" ] && { echo "results exist: not rerunning"; exit 5; } ;;
esac
log "chain $STAGE start pid $$ workers ${A98_WORKERS:-} free ${FREE}G"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
pin)
  run prepare > "$BASE/pin_prepare.out" 2>&1 || { RC=3 stop "PREPARE FAILED"; }
  run dry --stage i > "$BASE/dry_i.out" 2>&1 || { RC=3 stop "DRY I FAILED"; }
  run dry --stage c > "$BASE/dry_c.out" 2>&1 || { RC=3 stop "DRY C FAILED"; }
  [ "$(charged)" = 0 ] || { RC=3 stop "pin charged non-zero"; }
  log "banks $(python3 -c "import json;d=json.load(open('$BASE/pin_prepare.out'));print({k:v['file_sha256'][:12] for k,v in d.items()})")"
  log "PIN_OK charged 0" ;;
mech)
  W=${A98_WORKERS:?set A98_WORKERS}
  run mech --workers 4 --chunk 4 >> "$BASE/mech.out" 2>&1 || { RC=4 stop "MECH RUN FAILED"; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-1500) charged $(charged)"
  mechpass || { RC=5 stop "MECHANICS FAILED"; }
  log "A98_MECH_DONE" ;;
confirm)
  W=${A98_WORKERS:?set A98_WORKERS}
  log "confirm start"
  run confirm --workers "$W" --chunk 96 --drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { RC=8 stop "CONFIRM FAILED"; }
  run drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { RC=8 stop "DROP-RAW FAILED"; }
  log "confirm done; charged $(charged)"
  log "A98_CONFIRM_RUN_DONE"
  grep -q A94Q_DONE /root/a94queue.log 2>/dev/null && grep -q A94_CONFIRM_RUN_DONE /root/a94/chain_confirm.out 2>/dev/null \
    || { RC=9 stop "A94 not finished: analysis not run (run_a98.sh analyse later)"; }
  run analyse >> "$BASE/analyse.out" 2>&1 || { RC=9 stop "ANALYSE FAILED"; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A98_CONFIRM_DONE" ;;
analyse)
  run analyse >> "$BASE/analyse.out" 2>&1 || { RC=9 stop "ANALYSE FAILED"; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A98_ANALYSE_DONE" ;;
esac
