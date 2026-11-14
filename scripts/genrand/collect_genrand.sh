#!/usr/bin/env bash
# GENRAND undirected random-action collection on a many-core Linux box (GENRAND_NOTES.md).
#
#   scripts/genrand/collect_genrand.sh OUT PLAIN_SEEDS BRANCH_SEEDS WORKERS [driver args...]
#   e.g. scripts/genrand/collect_genrand.sh runs/active/C4-GENRAND-20261014 \
#            961000-961479 961480-961515 46 --speed-factor 1.3
#
# Resumable: re-run the same command after an interruption (finished episodes are skipped).
# Prints the wall-time estimate first; writes OUT/GENRAND_MANIFEST.json, OUT/COLLECT_SUMMARY.json,
# per-phase OUT/{plain,branch}/COLLECT_SUMMARY.json and converts into OUT/c4_episodes at the end.
set -euo pipefail
if [ "$#" -lt 4 ]; then
  sed -n '2,10p' "$0"; exit 2
fi
OUT="$1"; PLAIN="$2"; BRANCH="$3"; WORKERS="$4"; shift 4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$HERE:$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
command -v sumo >/dev/null || { echo "sumo not on PATH" >&2; exit 3; }
"$PY" -m traffic_shepherd.c4.genrand_driver --out "$OUT" --plain-seeds "$PLAIN" \
  --branch-seeds "$BRANCH" --workers "$WORKERS" --estimate-only "$@"
exec "$PY" -m traffic_shepherd.c4.genrand_driver --out "$OUT" --plain-seeds "$PLAIN" \
  --branch-seeds "$BRANCH" --workers "$WORKERS" "$@"
