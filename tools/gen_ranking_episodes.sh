#!/usr/bin/env bash
# Generate the SUMO episodes used by tools/wm_controller_ranking.py (C4 v7 WM evaluation, owner 2026-10-17).
#
#   bash tools/gen_ranking_episodes.sh [--out ROOT] [--workers N] [--seeds A-B] [--mappo CKPT] [--only a,b]
#
# Controllers: native, pi, damping, maxaccel_lc513 (built-ins of traffic_shepherd.c4.collect) and the two
# Dreamer checkpoints (dreamer_preloop, dreamer_loop); mappo only when --mappo CKPT is given.
# Demand 2400, penetrations 0.1,0.4, seeds 969912-969935 (C4_LOOP_TRAIN block), warmup 300, horizon 600,
# textbook population, --cav-vtype normal_car.  Workers <= 12 (18 caused TraCI start failures on the laptop).
# Each controller writes ROOT/<controller>/ (collect resumes complete episodes; failures are recorded in
# ROOT/<controller>/collect_log.jsonl and ROOT/GEN_LOG.txt and are NOT retried automatically; re-running
# the same command is an explicit, logged second attempt of the missing jobs only - collect keeps the
# failed directory as <job>.incomplete-<n> and records the attempt number; logs are appended).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe}"
OUT="E:/research-dense/runs/active/C4-DIAG-20261016/ranking"
WORKERS=12
SEEDS="969912-969935"
MAPPO=""
ONLY=""
CKPT_DIR="${CKPT_DIR:-E:/research-dense/runs/active/C4-V5-20261014/final_20261016}"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --mappo) MAPPO="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
if [ "$WORKERS" -gt 12 ]; then echo "workers capped at 12"; WORKERS=12; fi
# never the 964000-964199 evaluation block
if ! "$PY" -c "
import sys
for tok in sys.argv[1].split(','):
    a, _, b = tok.partition('-'); a = int(a); b = int(b or a)
    if a <= 964199 and b >= 964000: sys.exit(1)
" "$SEEDS"; then
  echo "refusing seeds in the 964000-964199 block"; exit 2
fi
export PYTHONPATH="src;."
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT"
declare -a NAMES SPECS
NAMES+=(native pi damping maxaccel_lc513 dreamer_preloop dreamer_loop)
SPECS+=(native pi damping maxaccel_lc513 "$CKPT_DIR/dreamer_v5m_preloop.pt" "$CKPT_DIR/dreamer_v5m_loop.pt")
if [ -n "$MAPPO" ]; then NAMES+=(mappo); SPECS+=("$MAPPO"); fi
cd "$REPO" || exit 2
rc_all=0
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; spec="${SPECS[$i]}"
  if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$name,"; then continue; fi
  echo "[gen] $(date -Is) start $name ($spec)" | tee -a "$OUT/GEN_LOG.txt"
  "$PY" -m traffic_shepherd.c4.collect --policy "$spec" --penetrations 0.1,0.4 --demands 2400 \
      --seeds "$SEEDS" --warmup 300 --horizon 600 --population textbook --cav-vtype normal_car \
      --workers "$WORKERS" --out "$OUT/$name" >> "$OUT/$name.log" 2>&1
  rc=$?
  summ=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"run={d['episodes_run']} complete={d['episodes_complete']} failed={d['episodes_failed']}\")" "$OUT/$name/COLLECT_SUMMARY.json" 2>/dev/null)
  echo "[gen] $(date -Is) end $name rc=$rc $summ" | tee -a "$OUT/GEN_LOG.txt"
  [ "$rc" -ne 0 ] && rc_all=1
done
exit $rc_all
