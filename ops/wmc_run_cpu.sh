#!/bin/bash
# WM-C zero-SUMO checks on the gpuA CPU (GPU reserved for WM-E): T only (no solos), K draws, H 300, 1 thread per worker.
# usage: bash run_cpu.sh LABEL CHECKPOINT MOMENTS_JSON NWORK K
set -u
LABEL=$1; CK=$2; MOM=$3; N=${4:-10}; K=${5:-2}
B=/root/autodl-fs/wmc_rankcheck_20261106
OUT=$B/$LABEL
mkdir "$OUT" || { echo "REFUSED: $OUT exists"; exit 2; }
cd $B/code_v3
sha256sum "$CK" > "$OUT/CHECKPOINT_SHA256.txt"
cp CODE_SHAS.txt "$OUT/"
/root/miniconda3/bin/python - "$N" "$OUT" "$MOM" <<'PY'
import json, sys
n, out, mom = int(sys.argv[1]), sys.argv[2], sys.argv[3]
a = [m['sid'] for m in json.load(open(mom)) if m.get('valid')]
open(f'{out}/groups.txt', 'w').write('\n'.join(','.join(a[i::n]) for i in range(n)) + '\n')
PY
i=0
while read -r G; do
  i=$((i+1))
  PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup nice -n 15 /root/miniconda3/bin/python -B -m tools.wzprop score \
    --moments "$MOM" --out $OUT/SCORES_part$i.jsonl --wm wma:WMA_READY.json --wm-arg checkpoint=$CK \
    --wm-arg map_spec=maps/wz3to2/MAP_SPEC.json --wm-arg device=cpu --wm-arg max_rows=24 --K $K --H 300 --no-solos \
    --sids $G > $OUT/part$i.log 2>&1 &
  echo "$! part$i $(date +%s)" >> $OUT/PIDS.txt
done < $OUT/groups.txt
cat $OUT/PIDS.txt
