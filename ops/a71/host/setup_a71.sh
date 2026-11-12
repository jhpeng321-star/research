#!/bin/bash
# AMENDMENT 71 host setup (Box 3; run as root; zero SUMO; never deletes; writes only /root/a71 and NEW files in
# /root/a70/src/ops: the A71 driver + ops/a71/). The code tree is the A70 tree (identity IDENTITY_A70.json covers it).
#   usage: setup_a71.sh tree | select
set -uo pipefail
SHIP=$(cd "$(dirname "$0")" && pwd)
BASE=/root/a71
S=/root/a70/src
R=$BASE/root
die() { echo "SETUP FAILED: $*"; exit 1; }
[ "$(id -u)" = 0 ] || die "must run as root"
put() { if [ -f "$2" ]; then cmp -s "$1" "$2" || die "$2 exists with different content"; else cp "$1" "$2" || die "copy $2"; fi; }
source /root/a70/env_a70.sh || die env
run() { "$PY" ops/a71_wideadmit.py "$@" --final-root "$R" --a68-root /root/a68/root --a70-root /root/a70/root --a66-root /root/a66/root; }
case "${1:?tree | select}" in
tree)
  mkdir -p "$BASE" "$S/ops/a71" "$R/prerun_plant" "$R/oracle_banks" "$R/a32_banks" "$R/a71_banks"
  put "$SHIP/a71_wideadmit.py" "$S/ops/a71_wideadmit.py"
  for f in BOUND_U0.json ORACLE_BINDING.json PRERUN_PLAN.json prerun_plant/CONFIG.json prerun_plant/A_checkpoint_000.pt \
           oracle_banks/GUIDANCE_R6_WMONLY_PRIORITY.json oracle_banks/LEARNED_TRIGGER_MODEL.json \
           a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json; do
    [ -f "$R/$f" ] || cp "/root/a70/root/$f" "$R/$f" || die "mirror $f"
    cmp -s "/root/a68/root/$f" "$R/$f" || die "$R/$f differs from the A68 root"
  done
  put "$SHIP/run_a71.sh" "$BASE/run_a71.sh"
  diff -rq /root/a68/src "$S" -x __pycache__ | cut -c1-200
  date '+%F %T' > "$BASE/SETUP_TREE_OK"; echo SETUP_TREE_OK ;;
select)
  F=$S/ops/a71/CANDIDATES_D2800_p60.json
  SEARCH=/home/ubuntu/a68search/STRATEGY_SEARCH_D2800_p60.json
  [ "$(sha256sum $SEARCH | cut -c1-64)" = 840be64d2014c1d0a304ab7dff5e08cb9658d1db9253655c198dcf574ff286d1 ] || die "search sha"
  if [ ! -f "$F" ]; then "$PY" ops/a71_wideadmit.py select --search "$SEARCH" --out "$F" || die select; fi
  DOC=$("$PY" -c "import json;print(json.load(open('$F'))['doc_sha256'])")
  P=$S/ops/a71/CANDIDATES_D2800_p60.sha256
  if [ -f "$P" ]; then [ "$(cut -c1-64 "$P")" = "$DOC" ] || die "pin differs"; else echo "$DOC  CANDIDATES_D2800_p60.json" > "$P"; fi
  echo "doc_sha256 $DOC"; sha256sum "$F" "$S/ops/a71_wideadmit.py"
  "$PY" -c "
import json;d=json.load(open('$F'))
print({k:d[k] for k in ('n_eligible','eligible_by_size','n_reused','n_skipped','n_candidates','candidates_by_size','n_distinct_descriptors','flags')})
print([(s['name'],s['skip']) for s in d['skipped']])"
  run prepare || die prepare
  run dry --stage v > "$BASE/dry_v.json" || die "dry v"
  "$PY" -c "
import json;d=json.load(open('$BASE/dry_v.json'))
print({k:d[k] for k in ('stage','panel_sha256','n_waves','sim_s','stage_cap','charged','cap')}); print(json.dumps(d['first'])[:1500])"
  date '+%F %T' > "$BASE/SETUP_OK"; echo SETUP_OK ;;
*) die "unknown mode" ;;
esac
