#!/bin/bash
# WM-C zero-SUMO checks for ONE WM checkpoint on gpuA (shared GPU: small capped workers, nice 10):
#   (a) PLANGEN rank check: 40 moments, solos for S-labelled variants, K 8, H 300
#   (b) claim-4 regime benefit: 135 tape states x feasible S-REGIME hold strategies, T only, K 8, H 300
# usage: bash run_ckpt.sh LABEL CHECKPOINT [NWORK_A] [NWORK_B]
set -u
LABEL=$1; CK=$2; NA=${3:-4}; NB=${4:-4}
B=/root/autodl-fs/wmc_rankcheck_20261106
OUT=$B/$LABEL
mkdir "$OUT" || { echo "REFUSED: $OUT exists"; exit 2; }
cd $B/code_v2
sha256sum "$CK" > "$OUT/CHECKPOINT_SHA256.txt"
cp CODE_SHAS.txt "$OUT/"
/root/miniconda3/bin/python - "$NA" "$NB" "$OUT" <<'PY'
import json, sys
na, nb, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
a = [m['sid'] for m in json.load(open('data/plangen_MOMENTS_OBS.json')) if m.get('valid')]
b = [m['sid'] for m in json.load(open('data/regime_STATES.json')) if m.get('valid')]
open(f'{out}/groups_a.txt', 'w').write('\n'.join(','.join(a[i::na]) for i in range(na)) + '\n')
open(f'{out}/groups_b.txt', 'w').write('\n'.join(','.join(b[i::nb]) for i in range(nb)) + '\n')
PY
COMMON="--wm wma:WMA_READY.json --wm-arg checkpoint=$CK --wm-arg map_spec=maps/wz3to2/MAP_SPEC.json --wm-arg device=cuda --wm-arg max_rows=24 --K 8 --H 300 --mem-fraction 0.08"
i=0
while read -r G; do
  i=$((i+1))
  PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup nice -n 10 /root/miniconda3/bin/python -B -m tools.wzprop score \
    --moments data/plangen_MOMENTS_OBS.json --out $OUT/RANK_part$i.jsonl $COMMON --sids $G \
    --solos-labelled-by data/PG_SUMMARY.json > $OUT/rank_part$i.log 2>&1 &
  echo "$! rank_part$i $(date +%s)" >> $OUT/PIDS.txt
done < $OUT/groups_a.txt
i=0
while read -r G; do
  i=$((i+1))
  PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup nice -n 10 /root/miniconda3/bin/python -B -m tools.wzprop score \
    --moments data/regime_STATES.json --out $OUT/REGIME_part$i.jsonl $COMMON --sids $G --no-solos \
    > $OUT/regime_part$i.log 2>&1 &
  echo "$! regime_part$i $(date +%s)" >> $OUT/PIDS.txt
done < $OUT/groups_b.txt
cat $OUT/PIDS.txt
