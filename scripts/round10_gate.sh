#!/usr/bin/env bash
# v10 WM ROUND 10 launch gate: round 9 + ONE change (fix B, cut-in reaction supervision), init round 9 final checkpoint.
# Runs a short REAL-DATA smoke (120/300 s x 2+2 steps, batch 4, as the round-9 gate) on gpuR, checks the train log, the init report and the frozen
# tensors, and ONLY if every check passes launches the full round. Start it only after round 9 has finished and been
# scored (the checkpoint below must exist). Never run it on a box with a prior round-10 artifact (exit 4).
#
#   bash scripts/round10_gate.sh [REPO_DIR]       # default REPO_DIR=$L/repo_r10
#
# Verdict file: $L/smoke_r10.verdict ("smoke: ok" / "smoke: bad ..." / "smoke: traceback").
# Exit codes: 2 repo or init checkpoint missing, 3 smoke failed, 4 prior round-10 artifacts exist.
L=/root/autodl-tmp/c4/v5
REPO=${1:-$L/repo_r10}
CK=${R10_INIT:-$L/runs/c4_wm_v10_round9/wm_s_stage300.pt}
ulimit -n 65536
cd "$REPO" || exit 2
[ -f "$CK" ] || { echo "round-9 final checkpoint missing: $CK" >&2; exit 2; }
[ ! -e $L/runs/smoke_r10 ] && [ ! -e $L/runs/c4_wm_v10_round10 ] && [ ! -e $L/smoke_r10.log ] || { echo "prior round-10 smoke/run artifacts exist under $L; choose fresh names" >&2; exit 4; }
export PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=12 WMS_LATEST_EVERY=20
export WMS_DATA_V5_NATIVE=$L/data/v5_native WMS_DATA_GENRAND_PLAIN=$L/data/genrand_plain WMS_DATA_GENRAND_BRANCH=$L/data/genrand_branch WMS_DATA_PAIRS300=$L/data/pairs300
COMMON="--set model.init_from=$CK --set loss.loss_share_logging=false --set loss.brake_follow_cav_weight=2.0 --set optim.batch_size=4"
/root/miniconda3/bin/python -m wms train --config configs/c4_wm_v10round10.toml --mode run --out $L/runs/smoke_r10 $COMMON \
  --set "optim.batch_size_per_stage=[4,4]" \n  --set "curriculum.unroll_s=[120.0,300.0]" --set "curriculum.stage_steps=[2,2]" --set "curriculum.truncation_steps_per_stage=[120,60]" \
  --set curriculum.max_hours=0.5 > $L/smoke_r10.log 2>&1
OK=$(CK="$CK" SMOKE="$L/runs/smoke_r10" /root/miniconda3/bin/python - <<'PY'
import glob, json, math, os, sys
SMOKE, CK = os.environ["SMOKE"], os.environ["CK"]
try:
    rows = [json.loads(x) for x in open(SMOKE + "/train_log.jsonl")]
except Exception:
    print("no_log"); sys.exit()

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def total(key, only=None):
    return sum((r.get(key, 0) or 0) for r in rows if only is None or only(r))

fin = bool(rows) and all(finite(r.get("loss")) for r in rows)
# round-9 FINAL merge checks (teacher-forced + F2 only; the on-policy objective is retired, flag must be 1 on
# every row; zero mass is checked inside the loss, after the counted exclusion).  merge_pivot is frozen in round 10,
# so no pivot gradient is required.
onflag = bool(rows) and all(r.get("merge_pivot_on_objective_disabled") == 1 for r in rows)
tfpos = total("merge_pivot_tf_positive_mass_events")
zero = total("merge_pivot_tf_zero_mass_events")
scored = total("merge_pivot_tf_scored_seconds", lambda r: (r.get("stage_s", 0) or 0) >= 120)
tfcars = total("merge_pivot_tf_scored_cars")
tfexcl = total("merge_pivot_tf_excluded_motion")
tau_ok = bool(rows) and all(finite(r.get("merge_pivot_tau")) and r["merge_pivot_tau"] > 0 for r in rows)
w_ok = bool(rows) and all(finite(r.get("merge_pivot_w")) and r["merge_pivot_w"] > 0 for r in rows)
# fix-B checks: recorded cut-in events exist, rows are scored, the loss is finite and positive.
cev = total("cutin_reaction_n_events")
cevrows = total("cutin_reaction_n_event_rows")
cctl = total("cutin_reaction_n_control_rows")
cfin = bool(rows) and all(finite(r.get("cutin_reaction")) and finite(r.get("cutin_reaction_event_loss"))
                          and finite(r.get("cutin_reaction_control_loss")) for r in rows)
cpos = sum(r.get("cutin_reaction", 0) or 0 for r in rows) > 0
# init report: round-9 checkpoint must load completely (no new tensor, no shape mismatch).
try:
    init = json.load(open(SMOKE + "/INIT_FROM.json"))
    init_ok = not init["new_in_model"] and not init["shape_mismatch"] and not init["unused_from_checkpoint"]
except Exception:
    init, init_ok = {}, False
# frozen tensors stay bitwise equal; merge_pivot (round-9 fit) never moves; something trainable does.
frozen_ok = moved = pivot_ok = False
try:
    import torch
    from wms.config.load import load_train_config
    cfg = load_train_config("configs/c4_wm_v10round10.toml")
    prefixes = tuple(cfg.optim.freeze_prefixes)
    before = torch.load(CK, map_location="cpu", weights_only=True)["model"]
    ckpts = sorted(glob.glob(SMOKE + "/wm_s_stage*.pt"), key=os.path.getmtime)
    after = torch.load(ckpts[-1], map_location="cpu", weights_only=True)["model"]
    changed = [k for k in after if k in before and not torch.equal(after[k], before[k])]
    frozen_ok = not [k for k in changed if k.startswith(prefixes)]
    pivot_ok = not [k for k in changed if k.startswith("merge_pivot")] and any(k.startswith("merge_pivot") for k in after)
    moved = any(not k.startswith(prefixes) for k in changed)
except Exception as exc:
    print(f"bad ckpt_check_failed={type(exc).__name__}: {exc}"); sys.exit()
ok = (len(rows) >= 4 and fin and onflag and tfpos > 0 and tfcars > 0 and zero == 0 and scored > 0 and tau_ok and w_ok
      and cev > 0 and cevrows > 0 and cctl > 0 and cfin and cpos and init_ok and frozen_ok and pivot_ok and moved)
print("ok" if ok else f"bad steps={len(rows)} finite={fin} on_objective_disabled={onflag} tf_pos={tfpos} tf_cars={tfcars} tf_excluded_motion={tfexcl} zero_mass={zero} "
      f"tf_scored_s_120={scored} tau_ok={tau_ok} w_ok={w_ok} cutin_events={cev} cutin_event_rows={cevrows} "
      f"cutin_control_rows={cctl} cutin_finite={cfin} cutin_positive={cpos} init_ok={init_ok} "
      f"frozen_ok={frozen_ok} pivot_unchanged={pivot_ok} trainable_moved={moved}")
PY
)
echo "smoke: $OK" > $L/smoke_r10.verdict
grep -q "Traceback" $L/smoke_r10.log && echo "smoke: traceback" >> $L/smoke_r10.verdict
if [ "$OK" != "ok" ] || grep -q "Traceback" $L/smoke_r10.log; then exit 3; fi
exec /root/miniconda3/bin/python -m wms train --config configs/c4_wm_v10round10.toml --mode run --out $L/runs/c4_wm_v10_round10 $COMMON
