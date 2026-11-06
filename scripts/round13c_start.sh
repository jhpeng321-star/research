#!/usr/bin/env bash
# Operator one-shot wrapper. Registration exists locally before this script is shipped.
set -eu -o pipefail -o noclobber
L=/root/autodl-tmp/c4/v5
REPO=$L/repo_r13c_bd3c16a8
test -d "$REPO"
test ! -e "$L/gate_r13c.pid"
test ! -e "$L/registration_r13c.json"
printf '%s\n' '{"registration":"S-R13C, appended 2026-10-24 before launch", "commit":"bd3c16a8b8fce3f88e0e39ba44e492ee793d5b9f", "config_sha256":"8d3b156c2016040c0e975a25c7702a90d37d2198b8e78499d430c7d81b499f15", "init_sha256":"c76c338dea6355f85cfaee63cdb9077fb129ed25db0ab582d77eda2961822091"}' > "$L/registration_r13c.json"
sha256sum -c <(printf '%s  %s\n' c76c338dea6355f85cfaee63cdb9077fb129ed25db0ab582d77eda2961822091 "$L/runs/c4_wm_v10_round12c/wm_s.pt")
nohup bash "$REPO/scripts/round13c_gate.sh" "$REPO" > "$L/gate_r13c.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$L/gate_r13c.pid"
date -Is > "$L/gate_r13c.start"
ps -p "$PID" -o pid,lstart,args
