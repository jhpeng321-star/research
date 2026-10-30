#!/bin/bash
set -euo pipefail
[[ ${PILOT_COMMITTED_STAGE:-} == --exploratory-owner-20261030 ]]
[[ $# == 3 && $1 == -m && $2 == tools.candB.bundle ]]
exec "${PILOT_REAL_PYTHON:?}" -m tools.candB.pilot_bundle "$3"
