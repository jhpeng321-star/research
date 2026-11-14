#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo 'Usage: python_bridge.sh PYTHON_ARGS (GENRAND-only population adapter; BOXOPS_REAL_PYTHON required)'
    exit 0
fi
: "${BOXOPS_REAL_PYTHON:?set BOXOPS_REAL_PYTHON}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ${1:-} == -m && ${2:-} == traffic_shepherd.c4.genrand_driver ]]; then
    shift 2
    exec "$BOXOPS_REAL_PYTHON" "$HERE/ops.py" genrand "$@"
fi
exec "$BOXOPS_REAL_PYTHON" "$@"
