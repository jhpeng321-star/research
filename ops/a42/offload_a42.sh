#!/bin/bash
# AMENDMENT 42 continuous offloader (laptop, Git Bash):  bash ops/a42/offload_a42.sh HOST PORT [USER]
# Every pass: (1) the small root files (WAVE_* / RESERVED_* / CHARGES / ADMISSION / IDENTITY_GATE / a42_banks / logs;
# never *.pt) and the host chain logs; (2) every file of CLOSED episode dirs (RESULT*, observable, vehicles.jsonl.gz,
# vehicles_report.json, CLAIM/CLOSE) not yet in the manifest, sha256-verified host vs laptop before it is recorded in
# E:/agent-work/a42-run/offload/OFFLOAD.sha256. Remote files are pruned ONLY with PRUNE=1 (RESULT*.gz/observable only,
# after a verified copy). Stops after the final pass once the chain has ended, or on E:/agent-work/a42-run/STOP_OFFLOAD.
set -uo pipefail
HOST=${1:?HOST}; PORT=${2:?PORT}; USR=${3:-root}
PP=""; [ "$PORT" = - ] || PP="-p $PORT"; PS=""; [ "$PORT" = - ] || PS="-P $PORT"
OPT="${SSH_KEY:+-i $SSH_KEY} -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new"
SSH="ssh $PP $OPT $USR@$HOST"
B=/root/a42v; R=$B/root
O=E:/agent-work/a42-run/offload; L=$O/root; MAN=$O/OFFLOAD.sha256
mkdir -p "$L/evaluation" "$O/host"; touch "$MAN"
SUDO=""; [ "$($SSH 'id -u' 2>/dev/null)" = 0 ] || SUDO="sudo -n"
say() { echo "$(date '+%F %T') $*"; }
final=0
while [ ! -f E:/agent-work/a42-run/STOP_OFFLOAD ]; do
  if ! $SSH true 2>/dev/null; then say "host unreachable"; sleep 120; continue; fi
  st=$($SSH "$SUDO bash -c 'p=\$(cat $B/chain.pid 2>/dev/null); if [ -z \"\$p\" ]; then echo nochain; elif kill -0 \$p 2>/dev/null; then echo alive; else echo ended; fi'" 2>/dev/null | tr -d '\r')
  $SSH "$SUDO tar -C $R -cf - --exclude=./evaluation --exclude='*.pt' --exclude='*.lock' ." 2>/dev/null \
    | tar --force-local -xf - -C "$L" || say "root small-file pull failed"
  $SSH "$SUDO bash -c 'cd $B && tar -cf - --ignore-failed-read *.out setup.log analysis WORKERS chain.pid 2>/dev/null'" 2>/dev/null \
    | tar --force-local -xf - -C "$O/host" 2>/dev/null
  $SSH "$SUDO bash -c 'cd $R/evaluation 2>/dev/null && for d in *; do [ -f \$d/CLOSE.json ] || continue; for f in \$d/*; do case \$f in *.json|*.json.gz|*.jsonl.gz) echo \$f;; esac; done; done'" \
    2>/dev/null | tr -d '\r' | sort > "$O/remote.list"
  cut -c67- "$MAN" | sort -u > "$O/have.list"
  comm -23 "$O/remote.list" "$O/have.list" > "$O/todo.list"
  n=$(wc -l < "$O/todo.list")
  if [ "$n" -gt 0 ]; then
    split -l 150 "$O/todo.list" "$O/batch."
    for bf in "$O"/batch.*; do
      files=$(tr '\n' ' ' < "$bf")
      remote=$($SSH "$SUDO bash -c 'cd $R/evaluation && sha256sum $files'" 2>/dev/null | tr -d '\r')
      $SSH "$SUDO tar -C $R/evaluation -cf - $files" 2>/dev/null | tar --force-local -xf - -C "$L/evaluation"
      ok=""
      while read -r h f; do
        [ -z "${f:-}" ] && continue
        f=${f#\*}
        lh=$(sha256sum "$L/evaluation/$f" 2>/dev/null | cut -c1-64)
        if [ "$lh" = "$h" ]; then echo "$h  $f" >> "$MAN"; ok="$ok $f"; else say "MISMATCH $f (retried next pass)"; fi
      done <<< "$remote"
      if [ "${PRUNE:-0}" = 1 ] && [ -n "$ok" ]; then
        pr=$(for f in $ok; do case $f in */RESULT.json.gz|*/observable.jsonl.gz) echo -n "$f ";; esac; done)
        [ -n "$pr" ] && $SSH "$SUDO bash -c 'cd $R/evaluation && rm -f $pr'"
      fi
      rm -f "$bf"
    done
    say "offloaded $(($(wc -l < "$MAN"))) files total (+$n listed this pass)"
  fi
  if [ "$final" = 1 ]; then say "final pass done; chain ended: $(tail -1 "$O/host/chain.out" 2>/dev/null)"; break; fi
  [ "$st" = ended ] && { final=1; say "chain ended; one more pass"; }
  sleep 60
done
