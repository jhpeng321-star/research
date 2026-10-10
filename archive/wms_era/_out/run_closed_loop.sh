#!/usr/bin/env bash
# WM-N1-REPL-CL-20261005 Part C launcher (coordinator launches; do not run unapproved).
# Usage: bash run_closed_loop.sh <workers> [policies] [extra args...]
#   e.g. bash run_closed_loop.sh 4 pl01_a11,fleet_wm,always_native,always_pi,random --threads 4
#   re-invoke the same command to pick up worlds whose base state appeared later.
set -euo pipefail
WORKERS="${1:?workers}"; shift
POLICIES="${1:-pl01_a11,fleet_wm,always_native,always_pi,random}"; [ $# -gt 0 ] && shift
exec "${PYTHON:-python}" "$(dirname "$0")/run_closed_loop.py" \
  --workers "$WORKERS" --policies "$POLICIES" "$@"
