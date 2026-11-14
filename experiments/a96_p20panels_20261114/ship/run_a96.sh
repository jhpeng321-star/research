#!/bin/bash
# AMENDMENT 96 parts (c) / (d) host chain (base = the directory of this script: /root/a96pc or /root/a96pd; part from
# $BASE/PART). Derived from run_a82.sh (part B) + run_a94.sh. ONE stage per launch; a stage whose chain file exists
# refuses; no retry loop; no automatic replay. Only the base and /dev/shm (test / drytest) are written.
#   test          zero SUMO unit tests (incl. builder stand-ins)
#   stage         zero SUMO, once: [c: build the 192-world panel 984400-984591 in $BASE/panels] + panel into the root
#                 [d: byte copy of /root/a82four/panels/A82F_PANEL_C_192.json] + IDENT_U0 bind -> STAGE_OK
#   bank-hand     zero SUMO: BANK_HAND_P20 from A96_HAND_ROOTS="ROOT=SHA|none [ROOT=SHA|none]" -> src/ops/a96 -> BANK_OK
#   bank-searched zero SUMO, part c only: from A96_SEARCHED_ROOTS="S4ROOT=SHA|none S5ROOT=SHA|none" -> BANK_OK
#                 (exit 8 NO_SEARCHED: no searched plan anywhere -> the panel is not run)
#   drytest       zero SUMO: /dev/shm roots with stand-in banks (searched: the A82 S3b bank; hand: the A76 bank) and the
#                 IDENT_U0 binding in place of MAPPO20C: prepare, dry i, dry c, mech rule-check e2e, analysis e2e, for
#                 the 4-arm AND the 3-arm layout -> DRYTEST_OK
#   pin           zero SUMO: bind MAPPO20C + prepare (bank copies) + dry i / c -> PIN_OK, writes $BASE/PINNED.env
#                 needs A96_BANK_FILE A96_BANK_SHA256 A96_HAND_FILE (unless hand none) A96_HAND_SHA256
#                 A96_MAPPO20C_POLICY A96_MAPPO20C_SHA256 A96_TRAIN_RECEIPT A96_TRAIN_PLAN
#   mech          identity (8) + bank rule checks (2 per bank arm) = 12 / 10 episodes (cap 14,400) -> PASS required
#   confirm       4 (3) arms x 192 worlds (921,600 / 691,200; raw dropped per round) -> analyse
#   analyse       zero SUMO re-run of the analysis ONLY if confirm ran and no results file exists
# Stop between rounds: touch $BASE/root/STOP_A96P<C|D>.
STAGE=${1:?stage}
BASE=$(cd "$(dirname "$0")" && pwd)
[ -f "$BASE/SETUP_OK" ] || { echo "no SETUP_OK in $BASE"; exit 1; }
source "$BASE/env_a96.sh"
PART=$(cat "$BASE/PART"); PU=${PART^^}
[ "$A96_PART" = "$PART" ] || { echo "env part $A96_PART != $PART"; exit 2; }
R=$BASE/root
A91R=/root/a91/root
A82BANK=/root/a82p20/root/a82_banks/A82_BANK_D2800_p20.json
A82BANK_SHA=3609634dace1d68778ec89392bc125f41ebd05b409a4e5777e6e80932283feb1
A76BANK=/root/a76p20/root/a76_banks/A76_BANK_D2800_p20.json
A76BANK_SHA=f8b5237eacfe2fecc04ab0c3d7f08432b5eda5e1825805316a98b05c94f22d80
REFS="--a76-root /root/a76p20/root --a32-ref $BASE/refs/a32"
cd "$SRC" || exit 2
run() { nice -n 19 "$PY" ops/a96_p20.py "$@" --final-root "$R" $REFS; }
runr() { local root=$1; shift; nice -n 19 "$PY" ops/a96_p20.py "$@" --final-root "$root" $REFS; }
charged() { python3 -c "import json,os;p='$1/CHARGES.jsonl';print(sum(json.loads(l)['sim_s'] for l in open(p)) if os.path.exists(p) else 0)"; }
builder() { nice -n 19 "$PY" -I -c "import sys; sys.path[:0]=['$SRC/pytools','$SRC/ops','$SRC']; sys.argv=sys.argv[1:]; exec(open('$SRC/ops/a96/build_bank_a96.py').read())" b "$@" --cell D2800_p20 --prefix A96$PU; }
sha() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
case "$STAGE" in
test)
  exec nice -n 19 "$PY" -m pytest -q -rs ops/test_a96_p20.py \
    -p no:cacheprovider --basetemp=/dev/shm/a96${PART}_test_$$ ;;
drytest)
  grep -q STAGE_OK "$BASE/chain_stage.out" 2>/dev/null || { echo "run stage first"; exit 5; }
  T=/dev/shm/a96${PART}_drytest_$$; trap 'rm -rf "$T"' EXIT
  [ "$(sha $A82BANK)" = $A82BANK_SHA ] && [ "$(sha $A76BANK)" = $A76BANK_SHA ] || { echo "stand-in bank pins"; exit 3; }
  export A96_SYNTH_SUBSTITUTE=1 A96_BANK_FILE=$A82BANK A96_BANK_SHA256=$A82BANK_SHA
  for H in $A76BANK_SHA none; do
    export A96_HAND_SHA256=$H A96_HAND_FILE=$A76BANK; [ $H = none ] && unset A96_HAND_FILE
    TR=$T/root_${H:0:8}; echo "==================== part $PART layout A96_HAND_SHA256=$H (stand-ins: searched = A82 S3b bank, hand = A76 bank, MAPPO20C = IDENT_U0)"
    mkdir -p "$TR/prerun_plant" "$TR/oracle_banks" "$TR/a32_banks"
    for f in A96${PU}_PANEL_C.json A96${PU}_PANEL_C.sha256 BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json \
             prerun_plant/A_checkpoint_000.pt oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json \
             oracle_banks/LEARNED_TRIGGER_MODEL.json a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json BOUND_A77_IDENT_U0.json; do
      cp "$R/$f" "$TR/$f" || { echo "DRYTEST: copy $f failed"; exit 3; }
    done
    echo "== prepare"; runr "$TR" prepare > "$T/prep.out" 2>&1 || { cat "$T/prep.out"; echo "DRYTEST: prepare failed"; exit 3; }
    python3 -c "import json;d=json.load(open('$T/prep.out'));print({k:(v.get('families'),v.get('executor_priority')) for k,v in d.items()})"
    echo "== dry i"; runr "$TR" dry --stage i > "$T/dry_i.out" 2>&1 || { cat "$T/dry_i.out"; echo "DRYTEST: dry i failed"; exit 3; }
    python3 -c "import json;d=json.load(open('$T/dry_i.out'));print({k:d[k] for k in ('arms','episodes','sim_s','caps','charged','spec_diff_vs_stored_driver','rule_check_waves')});print('stored',d['stored'])"
    echo "== dry c"; runr "$TR" dry --stage c > "$T/dry_c.out" 2>&1 || { cat "$T/dry_c.out"; echo "DRYTEST: dry c failed"; exit 3; }
    python3 -c "import json;d=json.load(open('$T/dry_c.out'));print({k:d[k] for k in ('panel_sha256','arms','contrasts','episodes','sim_s','caps','charged','synthetic_mappo20c_substitute')});print([w[:6] for w in d['waves']])"
    echo "== mech rule-check e2e"; nice -n 19 "$PY" ops/a96_e2e.py mech --a91-root $A91R --shm $T/me > "$T/me.out" 2>&1 || { cat "$T/me.out"; echo "DRYTEST: mech e2e failed"; exit 3; }
    tail -2 "$T/me.out"
    echo "== analysis e2e"; nice -n 19 "$PY" ops/a96_e2e.py analyse --a91-root $A91R --final-root "$TR" --shm $T/e2e > "$T/e2e.out" 2>&1 || { cat "$T/e2e.out"; echo "DRYTEST: e2e failed"; exit 3; }
    tail -2 "$T/e2e.out"
    [ -f "$TR/CHARGES.jsonl" ] && { echo "DRYTEST: CHARGES.jsonl appeared"; exit 3; }
  done
  echo "DRYTEST_OK charged 0 (scratch $T removed on exit)"; exit 0 ;;
esac
C=$BASE/chain_$STAGE.out
log() { echo "$(date "+%F %T") $*" >> "$C"; }
stop() { log "CHAIN_STOPPED $*"; printf '%s\n' "{\"stage\": \"$STAGE\", \"time\": \"$(date "+%F %T")\", \"reason\": \"$*\", \"charged\": $(charged "$R"), \"chain_log\": \"$C\"}" > "$BASE/CHAIN_STOPPED"; exit "${RC:-9}"; }
case "$STAGE" in stage|bank-hand|bank-searched|pin|analyse) NEED=1 ;; mech) NEED=2 ;; confirm) NEED=8 ;; *) echo "unknown stage $STAGE"; exit 2 ;; esac
case "$STAGE" in pin|mech|confirm|analyse)
  [ ${#A96_BANK_SHA256} = 64 ] && { [ ${#A96_HAND_SHA256} = 64 ] || [ "${A96_HAND_SHA256:-}" = none ]; } || { echo "A96_BANK_SHA256 / A96_HAND_SHA256 (sha or none) required"; exit 5; }
  export A96_BANK_SHA256 A96_HAND_SHA256; unset A96_SYNTH_SUBSTITUTE ;;
esac
[ -f "$C" ] && { echo "$C exists: stage $STAGE already launched; not relaunching"; exit 1; }
[ -f "$BASE/CHAIN_STOPPED" ] && { echo "CHAIN_STOPPED present: not launching"; exit 1; }
[ -f "$R/STOP_A96P$PU" ] && { echo "STOP_A96P$PU present: not launching"; exit 1; }
FREE=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$NEED" ] || { echo "free disk ${FREE}G < ${NEED}G: not launching"; exit 1; }
mechpass() { grep -q '"verdict": "PASS"' "$R/A96${PU}_MECH.json" 2>/dev/null; }
pinned_ok() { grep -q PIN_OK "$BASE/chain_pin.out" 2>/dev/null && grep -q "^export A96_BANK_SHA256=$A96_BANK_SHA256 A96_HAND_SHA256=$A96_HAND_SHA256 " "$BASE/PINNED.env"; }
case "$STAGE" in
bank-hand) [ -n "${A96_HAND_ROOTS:-}" ] || { echo "A96_HAND_ROOTS unset"; exit 5; } ;;
bank-searched) [ "$PART" = c ] || { echo "part (d) uses the A82 S3b bank file directly"; exit 5; }
               [ -n "${A96_SEARCHED_ROOTS:-}" ] || { echo "A96_SEARCHED_ROOTS unset"; exit 5; } ;;
pin) grep -q STAGE_OK "$BASE/chain_stage.out" 2>/dev/null && grep -q BANK_OK "$BASE/chain_bank-hand.out" 2>/dev/null || { echo "stage / bank-hand not done"; exit 5; }
     [ "$PART" = d ] || grep -q BANK_OK "$BASE/chain_bank-searched.out" 2>/dev/null || { echo "bank-searched not done"; exit 5; }
     grep -q "sha256 A96${PU}_BANK_HAND_D2800_p20.json $A96_HAND_SHA256" "$BASE/chain_bank-hand.out" || { echo "A96_HAND_SHA256 is not the built hand bank (or none)"; exit 5; }
     for v in A96_BANK_FILE A96_MAPPO20C_POLICY A96_MAPPO20C_SHA256 A96_TRAIN_RECEIPT A96_TRAIN_PLAN; do [ -n "${!v:-}" ] || { echo "$v required"; exit 5; }; done
     [ "$A96_HAND_SHA256" = none ] || [ -n "${A96_HAND_FILE:-}" ] || { echo "A96_HAND_FILE required"; exit 5; } ;;
mech) pinned_ok || { echo "not pinned with these bank shas"; exit 5; } ;;
confirm) mechpass || { echo "mechanics is not PASS: nothing runs"; exit 5; }
         pinned_ok || { echo "not pinned with these bank shas"; exit 5; }
         [ -n "${A96_PANEL_SHA256:-}" ] || { echo "A96_PANEL_SHA256 unset"; exit 5; } ;;
analyse) grep -q A96_CONFIRM_RUN_DONE "$BASE/chain_confirm.out" 2>/dev/null || { echo "confirm not done"; exit 5; }
         [ -f "$R/A96${PU}_RESULTS_D2800_p20.json" ] && { echo "results exist: not rerunning"; exit 5; } ;;
esac
log "chain $STAGE part $PART start pid $$ workers ${A96_WORKERS:-} free ${FREE}G bank ${A96_BANK_SHA256:-unset} hand ${A96_HAND_SHA256:-unset} panel ${A96_PANEL_SHA256:-unset}"
echo $$ > "$BASE/chain_$STAGE.pid"
bankfail() { local T; T=$(date +%Y%m%d_%H%M%S); log "BANK BUILD STOPPED: $(grep -E 'STOP|NO_SEARCHED' "$1" | cut -c1-600)"
  mv "$C" "$C.stopped_$T"; mv "$1" "$1.stopped_$T"; tail -3 "$1.stopped_$T"; exit "${2:-3}"; }
case "$STAGE" in
stage)
  if [ "$PART" = c ]; then
    A96_HAND_SHA256=none run stage-panels > "$BASE/stage.out" 2>&1 || { RC=3 stop "STAGE-PANELS FAILED"; }
  fi
  { A96_HAND_SHA256=none run panel --n 192 --panels-dir "$BASE/panels" && A96_HAND_SHA256=none run bind --which IDENT_U0; } >> "$BASE/stage.out" 2>&1 || { RC=3 stop "STAGE FAILED"; }
  [ "$(charged "$R")" = 0 ] || { RC=3 stop "stage charged non-zero"; }
  log "panel $(cat "$R/A96${PU}_PANEL_C.sha256")"
  log "STAGE_OK charged 0" ;;
bank-hand)
  O=$BASE/bank_hand.out; ARGS=; for x in $A96_HAND_ROOTS; do ARGS="$ARGS --hand $x"; done
  builder hand $ARGS --out-dir "$SRC/ops/a96" > "$O" 2>&1 && grep -q BUILD_BANK_A96_OK "$O" || bankfail "$O"
  log "roots $A96_HAND_ROOTS; $(grep -E '^(BANK_|sha256 )' "$O" | tr '\n' ';')"
  log "BANK_OK" ;;
bank-searched)
  O=$BASE/bank_searched.out; ARGS=; for x in $A96_SEARCHED_ROOTS; do ARGS="$ARGS --src $x"; done
  builder searched $ARGS --out-dir "$SRC/ops/a96" > "$O" 2>&1; r=$?
  [ $r -eq 8 ] && bankfail "$O" 8
  [ $r -eq 0 ] && grep -q BUILD_BANK_A96_OK "$O" || bankfail "$O"
  log "roots $A96_SEARCHED_ROOTS; $(grep -E '^(BANK_|sha256 )' "$O" | tr '\n' ';')"
  log "BANK_OK" ;;
pin)
  { run bind --which MAPPO20C && run prepare; } > "$BASE/pin.out" 2>&1 || { RC=3 stop "PIN (bind / prepare) FAILED"; }
  { run dry --stage i && run dry --stage c; } > "$BASE/dry.out" 2>&1 || { RC=3 stop "DRY FAILED"; }
  [ "$(charged "$R")" = 0 ] || { RC=3 stop "pin charged non-zero"; }
  P=$(cut -d' ' -f5 "$R/A96${PU}_PANEL_C.sha256")
  echo "export A96_BANK_SHA256=$A96_BANK_SHA256 A96_HAND_SHA256=$A96_HAND_SHA256 A96_PANEL_SHA256=$P A96_BANK_FILE=$A96_BANK_FILE A96_HAND_FILE=${A96_HAND_FILE:-}" > "$BASE/PINNED.env"
  log "pinned $(cat "$BASE/PINNED.env")"
  log "PIN_OK charged 0" ;;
mech)
  W=${A96_WORKERS:?set A96_WORKERS}
  run mech --workers $(( W < 12 ? W : 12 )) --chunk 12 >> "$BASE/mech.out" 2>&1 || { RC=4 stop "MECH RUN FAILED"; }
  run mech-compare >> "$BASE/mech.out" 2>&1
  log "mech $(tail -1 "$BASE/mech.out" | cut -c1-1500) charged $(charged "$R")"
  mechpass || { RC=5 stop "MECHANICS FAILED"; }
  log "A96_MECH_DONE" ;;
confirm)
  W=${A96_WORKERS:?set A96_WORKERS}
  log "confirm start"
  run confirm --workers "$W" --chunk 96 --drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { RC=8 stop "CONFIRM FAILED"; }
  run drop-raw --drop-procs 16 >> "$BASE/confirm.out" 2>&1 || { RC=8 stop "DROP-RAW FAILED"; }
  log "confirm done; charged $(charged "$R")"
  log "A96_CONFIRM_RUN_DONE free $(df -BG --output=avail "$R" | tail -1 | tr -d " ")"
  run analyse >> "$BASE/analyse.out" 2>&1 || { RC=9 stop "ANALYSE FAILED"; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A96_CONFIRM_DONE" ;;
analyse)
  run analyse >> "$BASE/analyse.out" 2>&1 || { RC=9 stop "ANALYSE FAILED"; }
  log "analysis: $(tail -1 "$BASE/analyse.out" | cut -c1-1500)"
  log "A96_ANALYSE_DONE" ;;
esac
