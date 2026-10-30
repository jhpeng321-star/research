#!/bin/bash
# Existing stage.sh transport; only committed Git blobs enter the runtime.
set -euo pipefail
[[ ${1:-} == --exploratory-owner-20261030 ]]
shift
export PILOT_REAL_PYTHON=${PYTHON:-python}
export PILOT_COMMITTED_STAGE=--exploratory-owner-20261030
export PYTHON=$PWD/tools/candB/pilot_python.sh
exec bash tools/candB/stage.sh "$@"
