#!/bin/bash
# AMENDMENT 52 host chain (detached, launched ONCE; never inside a retry loop; no automatic replay).
#   0. wait until the A50 chain has ended (its chain.pid dead, no a50 driver python left) and require its final
#      line A50_ALL_DONE (else stop: nothing run)
#   1. identity: 4 episodes (A50 identity worlds) with the u0 export bound through the A52 path; compare with the
#      A50 WAVE_A50_IDENT_U0 records (copied read-only); FAILED -> stop, MAPPO60 not run
#   2. MAPPO60 on the 96 A50 confirmation worlds (one dispatch)
#   3. read-only copy of the finished A50 confirmation receipts / reports -> analysis A52_RESULTS_D2800_p60.json
# Only /root/a52 is written. Stop between stages: touch /root/a52/root/STOP_A52.
BASE=/root/a52
source "$BASE/env_a52.sh"
R=$BASE/root
REF=$BASE/a50ref/root
A50=/root/a50v
W=46
C=$BASE/chain.out
D=ops/a52_mappo60.py
log() { echo "$(date '+%F %T') $*" >> "$C"; }
run() { nice -n 19 "$PY" $D "$@" --final-root "$R"; }
[ -f "$C" ] && { echo "chain.out exists: not relaunching"; exit 1; }
log "chain start pid $$"
echo $$ > "$BASE/chain.pid"

# 0. wait for the A50 chain (read-only look at its pid / log)
while :; do
  p=$(cat $A50/chain.pid 2>/dev/null)
  if { [ -n "$p" ] && kill -0 "$p" 2>/dev/null; } || pgrep -f '[o]ps/a50_(divp60|fullp60).py' >/dev/null; then sleep 60; continue; fi
  break
done
last=$(tail -1 $A50/chain.out)
log "A50 chain ended: $last"
case "$last" in *A50_ALL_DONE*) ;; *) log "A50 did not end with A50_ALL_DONE: STOP (nothing run)"; exit 2 ;; esac

# copy (read-only source) the A50 identity records
mkdir -p "$REF"
cp -p $A50/root/WAVE_A50_IDENT_U0__*.json "$REF/" || { log "copy ident FAILED"; exit 2; }

# 1. identity
[ -f "$R/STOP_A52" ] && { log "STOP_A52 before identity"; exit 3; }
log "identity start"
run identity --workers 4 --chunk 4 >> "$BASE/identity.out" 2>&1 || { log "IDENTITY RUN FAILED"; exit 4; }
run identity-compare --a50-root "$REF" >> "$BASE/identity.out" 2>&1 || { log "IDENTITY COMPARE ERROR"; exit 4; }
log "identity $(tail -1 "$BASE/identity.out")"
grep -q '"verdict": "PASS"' "$R/IDENTITY_A52.json" || { log "IDENTITY FAILED: STOP (MAPPO60 not run)"; exit 5; }

# 2. MAPPO60
[ -f "$R/STOP_A52" ] && { log "STOP_A52 before confirm"; exit 3; }
log "confirm MAPPO60 start"
run confirm --workers $W --chunk 97 >> "$BASE/confirm.out" 2>&1 || { log "CONFIRM FAILED"; exit 6; }
log "confirm done; charged $(python3 -c "import json;print(sum(json.loads(l)['sim_s'] for l in open('$R/CHARGES.jsonl')))")"

# 3. A50 confirmation receipts + bindings + cached vehicle reports (read-only copy), analysis
for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json LAUNCHER.json ADMISSION_A50_D2800_p60.json \
         a50_banks/A50_BIG_BANK_D2800_p60.json prerun_plant/CONFIG.json oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json          oracle_banks/LEARNED_TRIGGER_MODEL.json a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
  mkdir -p "$REF/$(dirname "$f")"; cp -p "$A50/root/$f" "$REF/$f" || { log "copy $f FAILED"; exit 7; }
done
cp -p $A50/root/WAVE_A50C_U0__*.json $A50/root/WAVE_A50C_DUO_HOLD__*.json $A50/root/WAVE_A50C_LEARNED_TRIGGER__*.json \
      $A50/root/WAVE_A50C_BIG_BANK__*.json "$REF/" || { log "copy waves FAILED"; exit 7; }
for d in $A50/root/evaluation/a50c_*; do
  n=$(basename "$d"); mkdir -p "$REF/evaluation/$n"
  if [ -f "$d/vehicles_report.json" ]; then cp -p "$d/vehicles_report.json" "$REF/evaluation/$n/"
  else cp -p "$d/vehicles.jsonl.gz" "$REF/evaluation/$n/" 2>/dev/null || log "no report/log in $n"; fi
done
log "a50 copy: $(ls "$REF"/WAVE_A50C_* | wc -l) receipts, $(ls -d "$REF"/evaluation/a50c_* | wc -l) episode dirs"
run analyse --a50-root "$REF" --out "$R" >> "$BASE/analyse.out" 2>&1 || { log "ANALYSE FAILED"; exit 8; }
log "done: $(tail -1 "$BASE/analyse.out")"
log "A52_ALL_DONE"
