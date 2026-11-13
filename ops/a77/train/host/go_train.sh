#!/bin/bash
# AMENDMENT 77 (S-MAPPO20C) training chain - ONE launch, detached, own PIDs only, no retry loop, no relaunch.
#   bash /home/ubuntu/a77ship/go_train.sh <registration file containing the A77 PLAN sha>
# The call returns at once; the chain runs under setsid/nohup and logs to /home/ubuntu/a77/chain.out:
#   0 gates      SETUP_OK, registration text holds PLAN 57f51364 + both sanity panel shas, disk >= 15 GB,
#                MemAvailable >= 40 GB, 1-min load < 6 (else A77_ALLOW_SHARED=1), fresh run dir
#   1 plan       launch.py plan (option ft84_u0, cpuV, 46 workers) -> PLAN.json must hash to 57f51364 -> ack
#   2 check      launch.py check: fake-env smoke of the exact config + strict u0 warm start (zero SUMO)
#   3 identity   host reproducibility: A65 sanity panel cb64b674 (989588-989599, D2800_p60, native + u0) re-run and
#                compared with the stored A65 rows 3fe97fb7 field by field; any difference -> STOP   (21,600 sim-s)
#   4 sanity20   descriptive pre panel 439d0c2d (987588-987599, D2800_p20, native + u0)            (21,600 sim-s)
#   5 smoke      launch.py smoke (1 it x 2 ep x 330 s) + smoke gate                                      (660 sim-s)
#   6 production launch.py run: 84 it x 46 ep x 900 s, detached supervisor                         (3,477,600 sim-s)
#   7 post       launch.py bind --iteration 84 (BOUND_00084.json) + sanity_post (policy_00084 on 439d0c2d;
#                10,800 sim-s) + SHA256SUMS -> TRAIN_DONE
# Every stage is precharged in ledger/CHARGES.jsonl BEFORE it starts (cap 3,560,000; planned 3,532,260).
# Stop between stages: touch /home/ubuntu/a77/STOP (production: launch.py stop --run-dir /home/ubuntu/a77/run84).
set -uo pipefail
B=/home/ubuntu/a77
SHIP=$(cd "$(dirname "$0")" && pwd)
REG=${1:?registration file}
EXPECTED_PLAN=57f51364e2d56f89539d483fb445704c2bbb6e7fce8a824626082e25eb3d9f5d
P20_PANEL=439d0c2dc55f2b03079de220cc276a075bcf1c8309ea1980003d504aeb5ff92d
P60_PANEL=cb64b674d80c91d5ca2698ae8ef5dd4a382cda74d992fd40e4c811fcace6b71b
if [ "${A77_CHAIN:-}" != 1 ]; then
  [ -f "$B/chain.out" ] && { echo "$B/chain.out exists: one launch only, not relaunching"; exit 1; }
  REG=$(readlink -f "$REG")
  A77_CHAIN=1 setsid -f nohup bash "$0" "$REG" < /dev/null > "$B/go_train.nohup" 2>&1
  echo "A77 training chain started detached; follow $B/chain.out"
  exit 0
fi
C=$B/chain.out
V=$B/venv
PY=$V/bin/python
CODE=$B/code
RUN=$B/run84
L=$CODE/tools/nm_mappo/launch.py
CFG20=$CODE/configs/c4_mappo_speed_wz3to2_d2800p20_a77.toml
CFG60=$CODE/configs/c4_mappo_speed_wz3to2_d2800p60.toml
INIT=$B/inputs/mappo_u0_policy_00169.pt
export SUMO_HOME=$V/lib/python3.12/site-packages/sumo
export PATH="$SUMO_HOME/bin:$V/bin:$PATH" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date '+%F %T') $*" >> "$C"; }
stop() { log "STOP: $*"; echo "$*" > "$B/CHAIN_STOPPED"; exit "${2:-1}"; }
gate_stop() { [ -f "$B/STOP" ] && stop "STOP file before $1" 9; return 0; }
charge() { "$PY" "$B/ledger/charge.py" "$1" "$2" "$3" >> "$C" 2>&1 || stop "ledger refused stage $1"; }
reg_receipt() {   # sanity_eval registration receipt: panel sha + budget (+ the registration file sha)
  echo "{\"panel_sha256\": \"$2\", \"budget_sim_s\": $3, \"stage\": \"$4\", \"registration_sha256\": \"$(sha256sum "$REG" | cut -c1-64)\"}" > "$1"; }
sanity() {        # sanity <out> <config> <receipt> <args...>   (24 jobs / workers, detached child of this chain)
  local out=$1 cfg=$2 rec=$3; shift 3
  (cd "$CODE" && nice -n 19 "$PY" -u tools/nm_mappo/sanity_eval.py --config "$cfg" --out "$out" --registered "$rec" \
      --workers 24 "$@" > "$out.log" 2>&1)
}
wait_exit() {     # wait for the supervisor's EXIT_<stage>_*.json (own run dir only; polls every 120 s)
  while ! ls "$RUN/$1"/EXIT_"$1"_*.json > /dev/null 2>&1; do sleep 120; done
  cat "$RUN/$1"/EXIT_"$1"_*.json >> "$C"; echo >> "$C"
}
echo $$ > "$B/chain.pid"
log "chain start pid $$ host $(hostname) reg $REG ($(sha256sum "$REG" | cut -c1-16))"

# ---------------------------------------------------------------- 0 gates
[ -f "$B/SETUP_OK" ] || stop "SETUP_OK missing (run setup_a77_train.sh)"
for s in $EXPECTED_PLAN $P20_PANEL $P60_PANEL; do grep -q "$s" "$REG" || stop "registration file lacks $s"; done
[ -e "$RUN" ] && stop "$RUN exists: fresh run only"
FREE=$(df -BG --output=avail "$B" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge 15 ] || stop "free disk ${FREE}G < 15G"
MEM=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "${MEM:-0}" -ge 40 ] || stop "MemAvailable ${MEM}G < 40G"
LOAD=$(cut -d' ' -f1 /proc/loadavg)
if [ "$(awk -v l="$LOAD" 'BEGIN{print (l>=6)}')" = 1 ] && [ "${A77_ALLOW_SHARED:-}" != 1 ]; then
  stop "load $LOAD: another job is running (A77 needs the box; A77_ALLOW_SHARED=1 to override, disclosed)"; fi
log "gates ok: disk ${FREE}G mem ${MEM}G load $LOAD nproc $(nproc)"

# ---------------------------------------------------------------- 1 plan + ack
"$PY" "$L" plan --option ft84_u0 --host cpuV --run-dir "$RUN" --init "$INIT" --python "$PY" --sumo-home "$SUMO_HOME" \
  --config "$CFG20" --wall-cap-h 30 --phase-cap-sim-s 3560000 --other-charges-sim-s 54000 \
  --ledger "$B/ledger/CHARGES.jsonl" >> "$C" 2>&1 || stop "plan failed"
GOT=$(sha256sum "$RUN/PLAN.json" | cut -c1-64)
[ "$GOT" = "$EXPECTED_PLAN" ] || stop "PLAN sha $GOT != registered $EXPECTED_PLAN (nothing simulated; diff with $SHIP/EXPECTED_PLAN_a77_run84.json)"
cp "$REG" "$RUN/REGISTRATION_BLOCK.md"
"$PY" "$L" ack --run-dir "$RUN" --registration "$RUN/REGISTRATION_BLOCK.md" >> "$C" 2>&1 || stop "ack failed"
log "PLAN $GOT registered"

# ---------------------------------------------------------------- 2 check (zero SUMO)
(cd "$CODE" && "$PY" "$L" check --run-dir "$RUN" > "$B/check.out" 2>&1) || stop "fake-env check failed (see check.out)"
log "check rc 0: $(tail -1 "$B/check.out" | cut -c1-200)"

# ---------------------------------------------------------------- 3 host identity (A65 panel, exact rows)
gate_stop identity
charge identity 21600 "host identity: A65 sanity panel cb64b674, 12 worlds x native,u0 x (300+600) s at D2800_p60"
reg_receipt "$B/IDENTITY_REGISTERED.json" $P60_PANEL 21600 identity
sanity "$B/identity" "$CFG60" "$B/IDENTITY_REGISTERED.json" --policy u0="$INIT" || stop "identity sanity_eval rc $?"
"$PY" "$SHIP/compare_rows.py" "$B/inputs/a65_sanity_pre4_rows.jsonl" "$B/identity/rows.jsonl" "$B/HOST_IDENTITY.json" >> "$C" 2>&1 \
  || stop "HOST IDENTITY FAILED: this host does not reproduce the A65 rows (nothing trained)"
"$PY" "$B/ledger/charge.py" --reconcile identity 21600 "PASS: 24/24 rows equal to A65 sanity_pre4" >> "$C" 2>&1
log "host identity PASS"

# ---------------------------------------------------------------- 4 sanity_pre20 (descriptive)
gate_stop sanity_pre20
charge sanity_pre20 21600 "12 TRAIN worlds 987588-987599 x native,u0 x (300+600) s at D2800_p20 (panel 439d0c2d)"
reg_receipt "$RUN/SANITY_REGISTERED.json" $P20_PANEL 21600 sanity_pre20
sanity "$B/sanity_pre20" "$CFG20" "$RUN/SANITY_REGISTERED.json" --policy u0="$INIT" || stop "sanity_pre20 rc $?"
grep -q '"ok": false' "$B/sanity_pre20/rows.jsonl" && stop "sanity_pre20 has failed rows (infrastructure)"
"$PY" "$B/ledger/charge.py" --reconcile sanity_pre20 21600 "completed (descriptive)" >> "$C" 2>&1
log "sanity_pre20 done: $(head -c 300 "$B/sanity_pre20/SUMMARY.json" | tr '\n' ' ')"

# ---------------------------------------------------------------- 5 smoke
gate_stop smoke
charge smoke 660 "run84: 1 it x 2 ep x (300+30) s"
(cd "$CODE" && "$PY" "$L" smoke --run-dir "$RUN" >> "$C" 2>&1) || stop "smoke launch refused"
wait_exit smoke
"$PY" - "$L" "$RUN" >> "$C" 2>&1 <<'EOF' || stop "SMOKE GATE FAILED"
import importlib.util, sys
from pathlib import Path
s = importlib.util.spec_from_file_location('L', sys.argv[1]); L = importlib.util.module_from_spec(s); s.loader.exec_module(L)
ok, why = L.smoke_passed(Path(sys.argv[2])); print('smoke gate', ok, why); sys.exit(0 if ok else 1)
EOF
log "smoke PASS"

# ---------------------------------------------------------------- 6 production (84 fixed updates)
gate_stop production
charge production84 3477600 "84 it x 46 ep x (300+600) s; trainer --max-sumo-seconds 2318400 controlled"
(cd "$CODE" && "$PY" "$L" run --run-dir "$RUN" >> "$C" 2>&1) || stop "production launch refused"
log "production started; waiting for EXIT_production"
wait_exit production
grep -q '"rc": 0' "$RUN"/production/EXIT_production_*.json || stop "production exit rc != 0 (no relaunch; report)"
"$PY" "$B/ledger/charge.py" --reconcile production84 3477600 "completed rc 0" >> "$C" 2>&1
log "production done: $(grep -o '"iteration": [0-9]*' "$RUN/production/status.json")"

# ---------------------------------------------------------------- 7 bind 84 + sanity_post + sums
"$PY" "$L" bind --run-dir "$RUN" --iteration 84 >> "$C" 2>&1 || stop "bind 84 failed"
charge sanity_post 10800 "12 worlds x 1 arm (wz policy_00084) x (300+600) s at D2800_p20 (panel 439d0c2d)"
reg_receipt "$RUN/SANITY_POST_REGISTERED.json" $P20_PANEL 10800 sanity_post
sanity "$B/sanity_post" "$CFG20" "$RUN/SANITY_POST_REGISTERED.json" --policy wz="$RUN/production/policy_00084.pt" \
  --skip-native --prior-rows "$B/sanity_pre20/rows.jsonl" || log "sanity_post rc $? (descriptive; reported)"
"$PY" "$B/ledger/charge.py" --reconcile sanity_post 10800 "completed (descriptive)" >> "$C" 2>&1
(cd "$B" && find run84 sanity_pre20 sanity_post identity ledger HOST_IDENTITY.json -type f ! -path '*/scenarios/*' \
   ! -path '*/checkpoints/*' -size -64M -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS_A77_TRAIN)
log "TRAIN_DONE policy_00084 $(sha256sum "$RUN/production/policy_00084.pt" | cut -c1-64)"
date '+%F %T' > "$B/TRAIN_DONE"
