#!/usr/bin/env bash
# DEFERRED by owner decision (coordinator message 2026-10-12): MAPPO training is out of scope for now.
# Kept as a stub so the interface is recorded.  LF only.
#
# When re-enabled it would run (CPU box, SUMO workers):
#   python -m wms.control.mappo.train --config configs/c4_mappo.toml --workers N --out <fresh dir>
# through the same c4_prepare_run / c4_launch machinery as run_policy.sh.
echo "run_mappo.sh: DEFERRED (MAPPO is out of scope for the C4 run plan); nothing started." >&2
exit 3
