#!/bin/bash
# AMENDMENT 96 part (b) host chain. ONE stage per launch; each physics / sealing stage runs ONCE (chain file exists ->
# refuse); physics stages are detached by the caller (queue scripts); no retry loop; no automatic replay.
# Reference base (/root/a96ref; synth: /root/a96synth_ref):
#   panel   zero SUMO: A96_PANEL_V.json 988900-988923 + A96_REF_PINS.json (write-once)
#   bind    zero SUMO: BOUND_A77_MAPPO20C.json; needs A96_MAPPO20C_POLICY A96_MAPPO20C_SHA256 A96_TRAIN_RECEIPT A96_TRAIN_PLAN
#   dry     zero SUMO
#   synthdry (synth base only) panel + IDENT_U0 bind + dry
#   ref     PHYSICS 24 episodes MAPPO20C (28,800 sim-s cap in code), raw dropped -> seal (A96_REF.json)
# Admission base (/root/a96adm_hand|_s4|_s5; synth: /root/a96synth_adm, ref = /root/a96synth_ref/root):
#   pin     zero SUMO: needs A80_TOP_FILE + A80_TOP_SHA256: pin-top (A80 MAPPING_RULE at D2800_p20) + pin + prepare + dry
#   pinref  zero SUMO: after the reference seal: pin-ref + dry (reference loaded, reports ok)
#   mech    first 2 verification episodes of every candidate -> mech-compare (A96_MECH.json PASS / FAIL)
#   admit   the rest of the verification episodes (cap 12 x 24 x 1,200 = 345,600 incl. mech), raw dropped
#   bank    zero physics: receipt invariants + a76 admit (re-pointed) -> A96_ADMISSION_D2800_p20.json + a96_banks/A96_BANK_D2800_p20.json
# Physics stages need A96_WORKERS. Stop between dispatch rounds: touch $BASE/root/STOP_A96.
STAGE=${1:?stage}
BASE=$(cd "$(dirname "$0")" && pwd)
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK in $BASE"; exit 1; }
source "$BASE/env_a96.sh"
R=$BASE/root
cd "$SRC" || exit 2
case "$BASE" in /root/a96ref|/root/a96synth_ref) MODE=ref ;; /root/a96adm_*) MODE=adm; REFR=/root/a96ref/root ;; /root/a96synth_adm) MODE=adm; REFR=/root/a96synth_ref/root ;; *) echo "unknown base"; exit 2 ;; esac
case "$BASE" in /root/a96synth_*) export A96_SYNTH=1 ;; esac
charged() { python3 -c "import json,os;p='$1/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
if [ $MODE = ref ]; then run() { nice -n 19 "$PY" ops/a96_p20adm.py "$@" --final-root "$R"; }
else run() { nice -n 19 "$PY" ops/a96_p20adm.py "$@" --final-root "$R" --ref-root "$REFR"; }; fi
case "$MODE:$STAGE" in
ref:panel)
  [ -f "$R/A96_PANEL_V.json" ] && { echo "panel exists"; exit 1; }
  run ref-panel | tee "$BASE/panel.out"; exit ${PIPESTATUS[0]} ;;
ref:synthdry)
  [ -n "${A96_SYNTH:-}" ] || { echo "synthdry only in a synth base"; exit 1; }
  { [ -f "$R/A96_PANEL_V.json" ] || run ref-panel; } && { [ -f "$R/BOUND_A77_IDENT_U0.json" ] || run ref-bind --which IDENT_U0; } && run ref-dry > "$BASE/dry.out" 2>&1 \
    || { echo "SYNTHDRY FAILED"; tail -5 "$BASE/dry.out"; exit 3; }
  [ -f "$R/CHARGES.jsonl" ] && { echo "SYNTHDRY charged SUMO seconds"; exit 3; }
  cat "$BASE/dry.out"; echo SYNTHDRY_OK; exit 0 ;;
ref:bind)
  [ -n "${A96_SYNTH:-}" ] && { echo "real bind not in a synth base"; exit 1; }
  for v in A96_MAPPO20C_POLICY A96_MAPPO20C_SHA256 A96_TRAIN_RECEIPT A96_TRAIN_PLAN; do [ -n "${!v:-}" ] || { echo "$v required"; exit 1; }; done
  run ref-bind --which MAPPO20C > "$BASE/bind.out" 2>&1 || { echo "BIND FAILED"; tail -3 "$BASE/bind.out"; exit 3; }
  run ref-dry > "$BASE/dry.out" 2>&1 || { echo "DRY FAILED"; tail -3 "$BASE/dry.out"; exit 3; }
  cat "$BASE/bind.out" "$BASE/dry.out"; echo BIND_OK; exit 0 ;;
ref:dry) run ref-dry; exit $? ;;
adm:pin)
  [ -n "${A80_TOP_FILE:-}" ] && [ -n "${A80_TOP_SHA256:-}" ] || { echo "A80_TOP_FILE and A80_TOP_SHA256 required"; exit 1; }
  [ -f "$R/A96_CANDIDATES.json" ] && { echo "already pinned: not rerunning"; exit 1; }
  { run pin-top && run pin && run prepare; } > "$BASE/pin.out" 2>&1 || { echo "PIN REFUSED (see $BASE/pin.out, $R/A96_MAPPING_REFUSED.json): STOP"; tail -3 "$BASE/pin.out"; exit 3; }
  run dry > "$BASE/dry.out" 2>&1 || { echo "DRY FAILED"; tail -3 "$BASE/dry.out"; exit 3; }
  [ -f "$R/CHARGES.jsonl" ] && { echo "pin charged SUMO seconds: must not happen"; exit 3; }
  cat "$BASE/pin.out"; echo PIN_OK; exit 0 ;;
adm:pinref)
  [ -f "$R/A96_REFPIN.json" ] && { echo "reference already pinned"; exit 1; }
  { run pin-ref && run dry; } > "$BASE/pinref.out" 2>&1 || { echo "PINREF FAILED"; tail -3 "$BASE/pinref.out"; exit 3; }
  cat "$BASE/pinref.out"; echo PINREF_OK; exit 0 ;;
esac
W=${A96_WORKERS:?set A96_WORKERS}
[ -n "${A96_SYNTH:-}" ] && { echo "no physics in a synth base"; exit 1; }
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
case "$MODE:$STAGE" in ref:ref|adm:mech) NEED=2 ;; adm:admit) NEED=4 ;; adm:bank) NEED=1 ;; *) echo "unknown stage $MODE:$STAGE"; exit 2 ;; esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$R/STOP_A96" ] && { echo "STOP_A96 present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G: not launching"; exit 1; }
mechpass() { grep -q '"verdict": "PASS"' "$R/A96_MECH.json" 2>/dev/null; }
case "$MODE:$STAGE" in
ref:ref) [ -f "$R/BOUND_A77_MAPPO20C.json" ] || { echo "MAPPO20C not bound"; exit 1; } ;;
adm:mech) [ -f "$R/A96_MECH.json" ] && { echo "mechanics ran already"; exit 1; }; [ -f "$R/A96_REFPIN.json" ] || { echo "reference not pinned"; exit 1; } ;;
adm:admit) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; } ;;
adm:bank) grep -q A96_VERIFY_DONE "$BASE/chain_admit.out" 2>/dev/null || { echo "admit stage not done"; exit 5; } ;;
esac
log "chain $MODE $STAGE start pid $$ workers $W free ${FREE}G"
echo $$ > "$BASE/chain_$STAGE.pid"
case "$MODE:$STAGE" in
ref:ref)
  run ref-run --workers $(( W < 24 ? W : 24 )) --chunk 24 >> "$BASE/ref.out" 2>&1 || { log "REF RUN FAILED charged $(charged "$R")"; exit 4; }
  run ref-seal >> "$BASE/ref.out" 2>&1 || { log "REF SEAL FAILED charged $(charged "$R")"; exit 4; }
  log "ref sealed: $(tail -1 "$BASE/ref.out" | cut -c1-600) charged $(charged "$R")"
  log "A96_REF_DONE" ;;
adm:mech)
  run mech --workers $(( W < 24 ? W : 24 )) --chunk 24 >> "$BASE/mech.out" 2>&1 || { log "MECH RUN FAILED charged $(charged "$R")"; exit 4; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-2500) charged $(charged "$R")"
  mechpass || { log "MECHANICS FAILED: STOP"; exit 5; }
  log "A96_MECH_DONE" ;;
adm:admit)
  run verify --workers "$W" --chunk 96 --drop-procs 16 >> "$BASE/verify.out" 2>&1 || { log "VERIFY FAILED charged $(charged "$R")"; exit 6; }
  run drop-raw --drop-procs 16 >> "$BASE/verify.out" 2>&1 || { log "DROP-RAW FAILED"; exit 6; }
  log "verify done; charged $(charged "$R")"
  log "A96_VERIFY_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")" ;;
adm:bank)
  run admit >> "$BASE/admit.out" 2>&1 || { log "ADMIT FAILED: $(tail -1 "$BASE/admit.out" | cut -c1-1500)"; exit 7; }
  log "admit: $(tail -1 "$BASE/admit.out" | cut -c1-2500)"
  log "A96_BANK_DONE" ;;
esac
