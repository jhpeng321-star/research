#!/bin/bash
# A99 PANEL host chain (/root/a99panel). ONE stage per launch; a physics / sealing stage whose chain file exists refuses;
# no retry loop. Only the base (and /dev/shm for pytest temp) is written.
#   test      zero SUMO unit tests
#   stage     zero SUMO, once: panel (A96(c) file 0bb84f3c, panel 6c161a56) into the root -> STAGE_OK
#   pin       zero SUMO, once: byte copies of the two A99 banks (S4 682c01ae, hand 59acb747) + A99P_BANKS.json + dry
#             -> PIN_OK, PINNED.env
#   dry       zero SUMO: arms, runner-loaded executor sha per spec, spec diff vs the U0 construction, sim-s, caps
#   mech      first 2 panel worlds of each of the 4 arms (8 episodes, 9,600 sim-s) -> mech-compare (A99P_MECH.json PASS)
#   confirm   the remaining 760 episodes (cap 921,600 total; raw dropped per round) -> analyse
#   analyse   zero SUMO re-run of the analysis ONLY if confirm ran and no results file exists
# Stop between rounds: touch /root/a99panel/root/STOP_A99P.
STAGE=${1:?stage}
BASE=$(cd "$(dirname "$0")" && pwd)
[ "$BASE" = /root/a99panel ] || { echo "base must be /root/a99panel"; exit 2; }
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK in $BASE"; exit 1; }
source "$BASE/env_a99p.sh"
[ "$A96_PART" = c ] && [ "$A96_HOST_BASE" = "$BASE" ] || { echo "env is not the A99P env"; exit 2; }
R=$BASE/root
cd "$SRC" || exit 2
run() { nice -n 19 "$PY" ops/a99_panel.py "$@" --final-root "$R"; }
charged() { python3 -c "import json,os;p='$1/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
case "$STAGE" in
test) exec nice -n 19 "$PY" -m pytest -q -rs ops/test_a99_panel.py -p no:cacheprovider --basetemp=/dev/shm/a99p_test_$$ ;;
dry) run dry; exit $? ;;
esac
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
stop() { log "CHAIN_STOPPED $*"; printf '%s\n' "{\"stage\": \"$STAGE\", \"time\": \"$(date "+%F %T")\", \"reason\": \"$*\", \"charged\": $(charged "$R"), \"chain_log\": \"$C\"}" > "$BASE/CHAIN_STOPPED"; exit "${RC:-9}"; }
case "$STAGE" in stage|pin|analyse) NEED=1 ;; mech) NEED=2 ;; confirm) NEED=8 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$BASE/CHAIN_STOPPED" ] && { echo "CHAIN_STOPPED present: not launching"; exit 1; }
[ -f "$R/STOP_A99P" ] && { echo "STOP_A99P present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G: not launching"; exit 1; }
mechpass() { grep -q '"verdict": "PASS"' "$R/A99P_MECH.json" 2>/dev/null; }
pinned_ok() { grep -q PIN_OK "$BASE/chain_pin.out" 2>/dev/null && [ -f "$BASE/PINNED.env" ]; }
case "$STAGE" in
pin) grep -q STAGE_OK "$BASE/chain_stage.out" 2>/dev/null || { echo "stage not done"; exit 5; } ;;
mech) pinned_ok || { echo "not pinned"; exit 5; } ;;
confirm) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; }; pinned_ok || { echo "not pinned"; exit 5; } ;;
analyse) grep -q A99P_CONFIRM_RUN_DONE "$BASE/chain_confirm.out" 2>/dev/null || { echo "confirm not done"; exit 5; }
         [ -f "$R/A99P_RESULTS_D2800_p20.json" ] && { echo "results exist: not rerunning"; exit 5; } ;;
esac
log "chain $STAGE start pid $$ workers ${A99P_WORKERS:-} free ${FREE}G"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$STAGE" in
stage)
  run stage-panel > "$BASE/stage.out" 2>&1 || { RC=3 stop "STAGE FAILED"; }
  [ "$(charged "$R")" = 0 ] || { RC=3 stop "stage charged non-zero"; }
  log "panel $(cat "$R/A99P_PANEL_C.sha256")"
  log "STAGE_OK charged 0" ;;
pin)
  run prepare > "$BASE/pin.out" 2>&1 || { RC=3 stop "PREPARE FAILED"; }
  run dry > "$BASE/dry.out" 2>&1 || { RC=3 stop "DRY FAILED"; }
  [ "$(charged "$R")" = 0 ] || { RC=3 stop "pin charged non-zero"; }
  P=$(cut -d' ' -f5 "$R/A99P_PANEL_C.sha256")
  S4=$(sha256sum "$R/a99p_banks/A99P_BANK_S4_ON20C_D2800_p20.json" | cut -c1-64)
  H=$(sha256sum "$R/a99p_banks/A99P_BANK_HAND_ON20C_D2800_p20.json" | cut -c1-64)
  echo "export A99P_PANEL_SHA256=$P A99P_S4_BANK_SHA256=$S4 A99P_HAND_BANK_SHA256=$H" > "$BASE/PINNED.env"
  log "pinned $(cat "$BASE/PINNED.env")"
  log "PIN_OK charged 0" ;;
mech)
  run mech --workers 8 --chunk 8 >> "$BASE/mech.out" 2>&1 || { RC=4 stop "MECH RUN FAILED"; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-1500) charged $(charged "$R")"
  mechpass || { RC=5 stop "MECHANICS FAILED"; }
  log "A99P_MECH_DONE" ;;
confirm)
  W=${A99P_WORKERS:?set A99P_WORKERS}
  run confirm --workers "$W" --chunk 96 --drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { RC=8 stop "CONFIRM FAILED"; }
  run drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { RC=8 stop "DROP-RAW FAILED"; }
  log "confirm done; charged $(charged "$R")"
  log "A99P_CONFIRM_RUN_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"
  run analyse >> "$BASE/analyse.out" 2>&1 || { RC=9 stop "ANALYSE FAILED"; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A99P_CONFIRM_DONE" ;;
analyse)
  run analyse >> "$BASE/analyse.out" 2>&1 || { RC=9 stop "ANALYSE FAILED"; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A99P_ANALYSE_DONE" ;;
esac
