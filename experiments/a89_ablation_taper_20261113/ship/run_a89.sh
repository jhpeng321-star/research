#!/bin/bash
# AMENDMENT 89 (A89) host chain. Derived from run_a81.sh (e217897c); A89 changes: names/paths, the zero-SUMO "pin"
# stage (panel-c + prepare + dry i/c), the CHAIN_STOPPED receipt, required A89_ARMS_SHA256 / A89_PANEL_SHA256.
# ONE STAGE per launch; a stage whose chain file exists refuses; no retry loop; no automatic replay.
#   run_a89.sh test    : zero SUMO unit tests (temporary files in /dev/shm only)
#   run_a89.sh pin     : zero SUMO: panel-c (write-once) + prepare (bank copies) + dry i / c  -> PIN_OK
#   run_a89.sh mech    : 3 identity arms + 4 plan arms x 2 A70 worlds (16,800 sim-s) -> A89_MECH.json PASS / FAIL
#   run_a89.sh confirm : 6 arms x 96 fresh worlds (691,200 sim-s; raw archives dropped per round) -> analyse
#   run_a89.sh analyse : zero SUMO re-run of the registered analysis ONLY if confirm ran and no results file exists
# Writes only /root/a89 (and /dev/shm for the tests). Stop between rounds: touch /root/a89/root/STOP_A89.
STAGE=${1:?stage: test | pin | mech | confirm | analyse}
BASE=/root/a89
source $BASE/env_a89.sh
R=$BASE/root
A70R=/root/a75/a70ref/root
REF70=/root/a75/ref70
A75D2R=/root/a75d2c/root
REFS="--a66-root /root/a66/root --a70-root $A70R --ref70-reports $REF70 --a75d2-root $A75D2R"
cd "$SRC" || exit 2
run() { nice -n 19 "$PY" ops/a89_sixarm.py "$@" --final-root "$R" $REFS; }
charged() { python3 -c "import json,os;p='$R/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK: run setup first"; exit 1; }
if [ "$STAGE" = test ]; then
  exec "$PY" -m pytest -q -rs ops/test_a89_sixarm.py -p no:cacheprovider --basetemp=/dev/shm/a89_test_$$
fi
[ -n "${A89_ARMS_SHA256:-}" ] || { echo "A89_ARMS_SHA256 unset (the registered arms-bank sha): not launching"; exit 5; }
export A89_ARMS_SHA256
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
stop() { log "CHAIN_STOPPED $*"; printf '%s\n' "{\"stage\": \"$STAGE\", \"time\": \"$(date "+%F %T")\", \"reason\": \"$*\", \"charged\": $(charged), \"chain_log\": \"$C\"}" > "$BASE/CHAIN_STOPPED"; exit "${RC:-9}"; }
case "$STAGE" in pin) NEED=1 ;; mech) NEED=2 ;; confirm) NEED=6 ;; analyse) NEED=1 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$BASE/CHAIN_STOPPED" ] && { echo "CHAIN_STOPPED present: not launching"; exit 1; }
[ -f "$R/STOP_A89" ] && { echo "STOP_A89 present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G needed by stage $STAGE: not launching"; exit 1; }
DROP="--drop-raw --drop-procs 16"
mechpass() { grep -q '"verdict": "PASS"' "$R/A89_MECH.json" 2>/dev/null; }
case "$STAGE" in
mech) grep -q PIN_OK "$BASE/chain_pin.out" 2>/dev/null || { echo "pin not done"; exit 5; } ;;
confirm) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; }
         [ -n "${A89_PANEL_SHA256:-}" ] || { echo "A89_PANEL_SHA256 unset (the registered panel sha): not launching"; exit 5; } ;;
analyse) grep -q A89_CONFIRM_RUN_DONE "$BASE/chain_confirm.out" 2>/dev/null || { echo "confirm not done"; exit 5; }
         [ -f "$R/A89_RESULTS_D2800_p60.json" ] && { echo "results exist: not rerunning"; exit 5; } ;;
esac
log "chain $STAGE start pid $$ workers ${A89_WORKERS:-} free ${FREE}G arms ${A89_ARMS_SHA256} panel ${A89_PANEL_SHA256:-unset}"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
pin)
  mkdir -p "$R"
  [ -f "$R/A89_PANEL_C.json" ] || run panel-c > "$BASE/pin_panel_c.out" 2>&1 || { RC=3 stop "PANEL-C FAILED"; }
  run prepare > "$BASE/pin_prepare.out" 2>&1 || { RC=3 stop "PREPARE FAILED"; }
  run dry --stage i > "$BASE/dry_i.out" 2>&1 || { RC=3 stop "DRY I FAILED"; }
  run dry --stage c > "$BASE/dry_c.out" 2>&1 || { RC=3 stop "DRY C FAILED"; }
  [ "$(charged)" = 0 ] || { RC=3 stop "pin charged non-zero"; }
  log "panel $(cat "$R/A89_PANEL_C.sha256")"
  log "PIN_OK charged $(charged)" ;;
mech)
  W=${A89_WORKERS:?set A89_WORKERS}
  run mech --workers $(( W < 14 ? W : 14 )) --chunk 14 >> "$BASE/mech.out" 2>&1 || { RC=4 stop "MECH RUN FAILED"; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-1500) charged $(charged)"
  mechpass || { RC=5 stop "MECHANICS FAILED"; }
  log "A89_MECH_DONE" ;;
confirm)
  W=${A89_WORKERS:?set A89_WORKERS}
  export A81_PANEL_SHA256=$A89_PANEL_SHA256          # read by the (unchanged) A81 setup() as the registered panel sha
  log "confirm start"
  run confirm --workers "$W" --chunk 96 $DROP >> "$BASE/confirm.out" 2>&1 || { RC=8 stop "CONFIRM FAILED free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"; }
  run drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { RC=8 stop "DROP-RAW FAILED"; }
  log "confirm done; charged $(charged)"
  log "A89_CONFIRM_RUN_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"
  run analyse >> "$BASE/analyse.out" 2>&1 || { RC=9 stop "ANALYSE FAILED"; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A89_CONFIRM_DONE" ;;
analyse)
  export A81_PANEL_SHA256=${A89_PANEL_SHA256:?}
  run analyse >> "$BASE/analyse.out" 2>&1 || { RC=9 stop "ANALYSE FAILED"; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A89_ANALYSE_DONE" ;;
esac
