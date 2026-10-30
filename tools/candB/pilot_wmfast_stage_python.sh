#!/bin/bash
set -euo pipefail
[[ $# -ge 3 && $1 == -m && $2 == tools.candB.bundle ]]
shift 2
exec "${PILOT_REAL_PYTHON:?}" -m tools.candB.pilot_bundle "$@"
