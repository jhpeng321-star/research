#!/usr/bin/env bash
# Paired SUMO comparison of C4 controllers on the Linux GPU box.  LF only.
#   bash scripts/c4/run_compare.sh run --controllers native damping pi mappo=/path/policy.pt \
#        --seeds 4107-4114 --out runs/active/C4-COMPARE-x [--stochastic]
#   bash scripts/c4/run_compare.sh summarize --out runs/active/C4-COMPARE-x
set -Eeuo pipefail
REPO=${C4_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -z "${SUMO_HOME:-}" ]]; then
  SUMO_HOME=$("$PY" -c 'import sumo; print(sumo.SUMO_HOME)' 2>/dev/null || true)
  export SUMO_HOME
fi
[[ -z "${SUMO_HOME:-}" ]] || export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
exec "$PY" tools/c4_compare.py "$@"
