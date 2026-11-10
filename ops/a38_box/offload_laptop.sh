#!/bin/bash
# A38 offloader (laptop, git bash). Usage: HOST=<ip> PORT=<port> [USER_=root] bash offload_laptop.sh
# Every 3 min: list result-bearing files under the box root (RESULT*.json.gz, vehicles*.jsonl.gz, vehicles_report.json,
# observable.jsonl.gz, WAVE_*/RESERVED_* records, CHARGES.jsonl, *_LOG.jsonl, chain/analysis outputs) with remote sha256;
# pull only new/changed files; verify local sha == remote sha; append to OFFLOAD.sha256. NEVER deletes anything on the
# box. Stop: touch E:/agent-work/a38/STOP_OFFLOAD.
: "${HOST:?HOST}" "${PORT:?PORT}"; U=${USER_:-root}
SSH="ssh -p $PORT -o ConnectTimeout=20 -o ServerAliveInterval=30 $U@$HOST"
R=/root/a38
L=E:/agent-work/a38/offload
LOG=E:/agent-work/a38/offload.log
mkdir -p "$L"; touch "$L/OFFLOAD.sha256"
while [ ! -f E:/agent-work/a38/STOP_OFFLOAD ]; do
  remote=$(timeout 300 $SSH "cd $R && find root chain.out tests.out IDENTITY_GATE.json CALSUMMARY.json cal analysis -type f \
    \( -name 'RESULT*.json.gz' -o -name 'vehicles*.jsonl.gz' -o -name 'vehicles_report.json' -o -name 'observable.jsonl.gz' \
       -o -name 'WAVE_*.json' -o -name 'RESERVED_*.json' -o -name 'CHARGES.jsonl' -o -name '*_LOG.jsonl' -o -name 'CLOSE.json' \
       -o -name 'A38_*.json' -o -name '*.out' -o -name '*.json' -path '*cal*' -o -name 'IDENTITY_GATE.json' -o -name 'CALSUMMARY.json' \) \
       -print0 2>/dev/null | xargs -0 -r sha256sum" 2>>"$LOG")
  if [ -z "$remote" ]; then echo "$(date '+%F %T') no listing" >> "$LOG"; sleep 180; continue; fi
  need=$(while read -r h f; do [ -z "$f" ] && continue
           lh=$( [ -f "$L/$f" ] && sha256sum "$L/$f" | cut -c1-64 ); [ "$lh" = "$h" ] || echo "$f"; done <<< "$remote" | head -400)
  if [ -n "$need" ]; then
    echo "$need" | timeout 1800 $SSH "cd $R && tar cf - -T -" | ( cd "$L" && tar xf - ) 2>>"$LOG"
    n=0; bad=0
    while read -r f; do h=$(awk -v f="$f" '$2==f{print $1}' <<< "$remote")
      if [ "$(sha256sum "$L/$f" 2>/dev/null | cut -c1-64)" = "$h" ]; then echo "$h  $f" >> "$L/OFFLOAD.sha256"; n=$((n+1))
      else bad=$((bad+1)); echo "$(date '+%F %T') MISMATCH $f (live file may have changed; retried next pass)" >> "$LOG"; fi
    done <<< "$need"
    echo "$(date '+%F %T') pulled $n verified, $bad mismatched" >> "$LOG"
  fi
  sleep 180
done
echo "$(date '+%F %T') stopped" >> "$LOG"
