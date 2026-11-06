#!/bin/sh
# Wait for the inference pool to close, then verify sequentially on one CPU.
set -eu
ROOT=/root/autodl-tmp/c4/v5/diag_yield
test -d "$ROOT/verification"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/code_final/src"
attempt=0
while test ! -f "$ROOT/r13y_offline_final/summary.json"; do
  attempt=$((attempt + 1))
  test "$attempt" -le 180
  sleep 10
done
P=/root/miniconda3/bin/python
nice -n 19 "$P" -B "$ROOT/code_final/scripts/r13y_bitidentity.py" \
  --repo "$ROOT/base_eb75010c" --out "$ROOT/verification/base_final.pt"
nice -n 19 "$P" -B "$ROOT/code_final/scripts/r13y_bitidentity.py" \
  --repo "$ROOT/code_final" --out "$ROOT/verification/off_final.pt"
nice -n 19 "$P" -B "$ROOT/code_final/scripts/r13y_bitidentity.py" \
  --compare "$ROOT/verification/base_final.pt" "$ROOT/verification/off_final.pt"
nice -n 19 "$P" -B -m pytest -q -p no:cacheprovider \
  --basetemp "$ROOT/verification/pytest_final" \
  "$ROOT/code_final/tests/test_r13y_yield.py" "$ROOT/code_final/tests/test_r13y_config.py" \
  "$ROOT/code_final/tests/test_checkpoint_forward_exact.py" "$ROOT/code_final/tests/unit/test_v6merge.py" \
  -k 'not off_path_bit_identical and not config_plumbing'
printf 'R13Y verification complete\n'
