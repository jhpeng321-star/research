#!/usr/bin/env bash
# S-R14B auto-launch chain (gpuR).  Arm from the PREPARED repo_r14b_<sha8> copy with
#   nohup setsid bash scripts/round14b_autochain.sh B2_PID B2_START_TICKS > LOG 2>&1 < /dev/null &
# 1. wait until the 13E b2 training (exact PID + start ticks) has exited and b2 RUN_SUMMARY.json
#    exists (never signals it; never reads or waits for the slow b2 postrun);
# 2. register once (registration_r14.json: code/config/init sha; the gate verifies the b2 FINAL,
#    the 13E B midrun recheck CONTINUE and that no b2 postrun STOP exists);
# 3. fail-closed smoke (A 2x120 from b2 FINAL, B 2x120 + 2x300 from A smoke FINAL);
# 4. arm `watch --name b2` (a REAL b2 postrun STOP SIGTERMs the owned R14 production by pidfd);
# 5. production A, then B immediately (round14_gate.sh --chain; A's checks run in parallel).
# Any failed step stops the chain (nothing is retried).  Own PIDs only; no name-based signals.
set -eu -o pipefail -o noclobber
B2_PID=${1:?b2 training PID}
B2_TICKS=${2:?b2 training start ticks}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
L=/root/autodl-tmp/c4/v5
PY=${R14_PYTHON:-/root/miniconda3/bin/python}
OFFLINE=$L/diag_r14/R14_OFFLINE_REPORT.json
CK=$L/runs/c4_wm_v10_round13e_b2/wm_s.pt
export R14_PYTHON=$PY PYTHONPATH="$REPO/src:$REPO/tests:$REPO" PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
log() { echo "[r14b-autochain $(date -Is)] $*"; }
mkdir "$L/.round14b_autochain.lock"
printf '%s\n' "$$" > "$L/.round14b_autochain.lock/pid"
"$PY" -B -m wms.train.round14_gate config

ticks() { "$PY" -B -c 'import pathlib,sys
try: print(pathlib.Path("/proc/"+sys.argv[1]+"/stat").read_text().rsplit(")",1)[1].split()[19])
except FileNotFoundError: print("gone")' "$1"; }
log "waiting for b2 training PID $B2_PID (ticks $B2_TICKS) to exit"
for ((i=0; i<16*60; i++)); do
    [ "$(ticks "$B2_PID")" = "$B2_TICKS" ] || break
    sleep 60
done
[ "$(ticks "$B2_PID")" != "$B2_TICKS" ] || { log "b2 still running after 16 h; stop"; exit 10; }
for ((i=0; i<10; i++)); do [ -f "$L/runs/c4_wm_v10_round13e_b2/RUN_SUMMARY.json" ] && break; sleep 30; done
[ -f "$L/runs/c4_wm_v10_round13e_b2/RUN_SUMMARY.json" ] || { log "b2 exited without RUN_SUMMARY.json; stop"; exit 11; }
log "b2 ended with RUN_SUMMARY.json; registering"
"$PY" -B -m wms.train.round14_gate register --root "$L" --init-checkpoint "$CK" --offline-report "$OFFLINE"
log "smoke"
bash "$REPO/scripts/round14_gate.sh" --smoke --offline-report "$OFFLINE" --wait
log "arming the b2 postrun STOP watcher"
mkdir -p "$L/chk_r14"
CUDA_VISIBLE_DEVICES= nohup nice -n 19 "$PY" -B -u -m wms.train.round14_gate watch --root "$L" --name b2 \
    > "$L/chk_r14/b2_watch.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$L/chk_r14/b2_watch.pid"
log "production A then B"
bash "$REPO/scripts/round14_gate.sh" --chain --offline-report "$OFFLINE" --wait
log "chain finished (B production exited)"
