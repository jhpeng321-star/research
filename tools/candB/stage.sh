#!/bin/bash
# Source packaging only by default; explicit gpuA target stages into a fresh dir.
set -euo pipefail
ARCHIVE=${1:?usage: stage.sh FRESH_ARCHIVE [gpuA FRESH_RUNTIME]}
PYTHON=${PYTHON:-python}
if [[ -n ${CANDB_STAGE_COMMIT:-} ]]; then
  "$PYTHON" -m tools.candB.bundle "$ARCHIVE" --commit "$CANDB_STAGE_COMMIT"
else
  "$PYTHON" -m tools.candB.bundle "$ARCHIVE"
fi
if [[ $# -gt 1 ]]; then
  [[ $# == 3 && $2 == gpuA && $3 == /root/autodl-tmp/c4/candB_* ]]
  RUNTIME=$3
  ssh gpuA "test -d /root/autodl-tmp/c4 && test ! -e '$RUNTIME' && mkdir '$RUNTIME'"
  scp "$ARCHIVE" "gpuA:$RUNTIME/SOURCE.tar.gz"
  ssh gpuA "tar -xzf '$RUNTIME/SOURCE.tar.gz' -C '$RUNTIME'"
fi
