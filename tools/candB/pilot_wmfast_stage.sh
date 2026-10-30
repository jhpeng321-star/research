#!/bin/bash
set -euo pipefail
[[ ${1:-} == --exploratory-owner-20261030 ]]
shift
export CANDB_STAGE_COMMIT=${CANDB_STAGE_COMMIT:?immutable committed source required}
export PILOT_REAL_PYTHON=${PYTHON:-python}
export PYTHON=$PWD/tools/candB/pilot_wmfast_stage_python.sh
exec bash tools/candB/stage.sh "$@"
