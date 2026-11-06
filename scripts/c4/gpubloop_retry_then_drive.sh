#!/usr/bin/env bash
# gpubloop_retry_then_drive.sh CONFIG PHASE -- `loop14.py retry PHASE` (new .aN attempt), then on success
# hand over to gpubloop_driver.sh.  Run from the staged code root.  Never signals any process.
set -u
CFG="$1"; PHASE="$2"; P="${P:-/root/miniconda3/bin/python}"
cd "$(cd "$(dirname "$0")/../.." && pwd)"
echo "$(date -u +%FT%TZ) retry $PHASE (pid $$)"
"$P" scripts/c4/loop14.py retry --config "$CFG" "$PHASE"; rc=$?
echo "$(date -u +%FT%TZ) retry exit $rc"
if [ "$rc" = 0 ] || [ "$rc" = 4 ]; then exec bash scripts/c4/gpubloop_driver.sh "$CFG"; fi
"$P" scripts/c4/loop14.py status --config "$CFG"; exit "$rc"
