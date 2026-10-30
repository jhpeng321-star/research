#!/bin/bash
# Owned waiter: no GPU allocation until the owned primary scorer has completed.
set -euo pipefail
root=/root/autodl-tmp/c4/setupaudit20261030/fix
while true; do
  case "$(ps -p 433299 -o stat=)" in
    ''|Z*) break ;;
    *) sleep 60 ;;
  esac
done
bash "$root/remote2.sh" summarize2.py panel2_complete.json --out panel2_pooled.json
while test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; do
  sleep 60
done
nvidia-smi
exec bash "$root/remote2.sh" strategic2.py --out strategic2_complete.json --device cuda
