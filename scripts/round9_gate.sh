#!/usr/bin/env bash
# v10 WM ROUND 9 launch gate (A-only first_passage + hybrid width + F2 conditioning, init round 8 stage300).
# Runs a short REAL-DATA smoke (60/120 s x 2+3 steps) on gpuR, checks the train log, and ONLY if every check
# passes launches the full round. Do not start this until the support gate has passed (S-SUP / tools/r9_*).
#
#   bash scripts/round9_gate.sh [REPO_DIR]        # default REPO_DIR=$L/repo_r9 (use the shipped repo_r9_<sha8>)
#
# Verdict file: $L/smoke_r9.verdict ("smoke: ok" / "smoke: bad ..." / "smoke: traceback").
# Exit codes: 2 repo missing, 3 smoke failed, 4 verdict/log files already exist (never overwrite a prior run).
L=/root/autodl-tmp/c4/v5
REPO=${1:-$L/repo_r9}
ulimit -n 65536
cd "$REPO" || exit 2
[ ! -e $L/runs/smoke_r9 ] && [ ! -e $L/runs/c4_wm_v10_round9 ] && [ ! -e $L/smoke_r9.log ] || { echo "prior round-9 smoke/run artifacts exist under $L; choose fresh names" >&2; exit 4; }
CK=$L/runs/c4_wm_v10_round8/wm_s_stage300.pt
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE=$L/data/v5_native WMS_DATA_GENRAND_PLAIN=$L/data/genrand_plain WMS_DATA_GENRAND_BRANCH=$L/data/genrand_branch WMS_DATA_PAIRS300=$L/data/pairs300
COMMON="--set model.init_from=$CK --set loss.loss_share_logging=false --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4"
/root/miniconda3/bin/python -m wms train --config configs/c4_wm_v10round9.toml --mode run --out $L/runs/smoke_r9 $COMMON \
  --set "curriculum.unroll_s=[60.0,120.0]" --set "curriculum.stage_steps=[2,3]" --set "curriculum.truncation_steps_per_stage=[120,120]" \
  --set curriculum.max_hours=0.5 > $L/smoke_r9.log 2>&1
OK=$(/root/miniconda3/bin/python - <<'PY'
import json, math, sys
try:
    rows = [json.loads(x) for x in open("/root/autodl-tmp/c4/v5/runs/smoke_r9/train_log.jsonl")]
except Exception:
    print("no_log"); sys.exit()

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

fin = bool(rows) and all(finite(r.get("loss")) for r in rows)
onev = sum(r.get("merge_pivot_on_events", 0) or 0 for r in rows)
tfpos = sum(r.get("merge_pivot_tf_positive_mass_events", 0) or 0 for r in rows)
zero = sum((r.get("merge_pivot_tf_zero_mass_events", 0) or 0) + (r.get("merge_pivot_on_zero_mass_events", 0) or 0) for r in rows)
scored = sum(r.get("merge_pivot_on_scored_seconds", 0) or 0 for r in rows if (r.get("stage_s", 0) or 0) >= 120)
tau_ok = bool(rows) and all(finite(r.get("merge_pivot_tau")) and r["merge_pivot_tau"] > 0 for r in rows)
w_ok = bool(rows) and all(finite(r.get("merge_pivot_w")) and r["merge_pivot_w"] > 0 for r in rows)
ok = len(rows) >= 4 and fin and onev > 0 and tfpos > 0 and zero == 0 and scored > 0 and tau_ok and w_ok
print("ok" if ok else f"bad steps={len(rows)} finite={fin} on_events={onev} tf_pos={tfpos} zero_mass={zero} "
      f"scored_s_120={scored} tau_ok={tau_ok} w_ok={w_ok}")
PY
)
echo "smoke: $OK" > $L/smoke_r9.verdict
grep -q "Traceback" $L/smoke_r9.log && echo "smoke: traceback" >> $L/smoke_r9.verdict
if [ "$OK" != "ok" ] || grep -q "Traceback" $L/smoke_r9.log; then exit 3; fi
exec /root/miniconda3/bin/python -m wms train --config configs/c4_wm_v10round9.toml --mode run --out $L/runs/c4_wm_v10_round9 $COMMON
