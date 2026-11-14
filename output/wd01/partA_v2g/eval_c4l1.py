"""WD-01-C4L1 (registered frozen look, zero simulation): the FIRST Part A
v2 general look at the chunk-4 CALIBRATION worlds, exactly per
output/wd01/REGISTRATION_C4L1.md (LF-sha256 5e4a7721...b124).

Adapted from the A2 harness (eval_a2.py; same engine, truth type and
metrics) with the registered C4L1 protocol:

- WORLDS: exactly the 9 chunk-4 `CALIBRATION` worlds of
  results_plan_chunk4/WORLD_PLAN.json; refused worlds are reported and
  never replaced.
- ACCESS CONTROL: the loader refuses the 9 CAL worlds unless it is
  given the frozen registration file with the matching LF-sha256
  (verify_registration).  HOLDOUT is refused via
  E:/research-dense/tools/holdout_guard.py +
  protocols/holdout_exclusions.json (excluded roots per owner ruling,
  plus any HOLDOUT-split world).  The opened-file list is asserted at
  close-out to contain only the 9 CAL worlds' __native_base run files
  plus the plan file.
- FREEZE: refuses to start unless the registration matches its recorded
  LF-sha256 and C4L1_FREEZE.json matches reality (engine commit, params
  sha, the sha256 of every eval script).
- ONE pass: raw rows are saved BEFORE aggregation; aggregation re-rolls
  nothing.  A crash before any row may be restarted once (disclosed); a
  crash after rows is reported as is; no re-run after seeing any number.
- The output directory results/c4l1 must not exist before the pass and
  is never overwritten.
- Registered comparators in the same pass: constant-speed persistence
  and the v1 p3o replica (frozen knobs); labelled v2g_guards_on variant
  every --guards-on-every-th world (always reported, never gate).
- No T/a sweep (not registered for C4L1).  No parameter change, no
  variants beyond the labelled guards_on, no re-scoring here.
- CPU: exactly one worker process (--workers 1 enforced), below-normal
  priority, WD_FORBID_D=1 installs the forbid-D audit hook.

Verdict (computed from the SAVED rows only), frozen rule at H = 30 s:
PASS iff (a) v2g collisions <= 1 per 1000 vehicle-steps AND
(c) v2g delay MAE < const-speed delay MAE on the same windows AND
|mean signed v2g delay error| <= 15 % of the mean truth tracked delay
AND (d) v2g v_RMSE <= 1.05 x min(const-speed, p3o replica) v_RMSE.
(b) lane-change ratios are REPORTED ONLY (owner decision, LC1 option 1).

Usage:
    python eval_c4l1.py [--registration PATH] [--freeze PATH]
                        [--guards-on-every K] [--data-root ROOT]
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_P3D_ROOT = os.environ.get("WD01_DATA_ROOT", "").strip() or \
    str(Path("E:/rd-wt/wd01/output/wd01"))
os.environ.setdefault("WD01_DATA_ROOT", _P3D_ROOT)

if os.environ.get("WD_FORBID_D", "").strip() == "1":
    import forbid_d_hook  # noqa: E402

    forbid_d_hook.install()

import wdpaths  # noqa: E402

DATA_ROOT = Path(os.environ.get("WD01_DATA_ROOT_TRUTH", "").strip() or
                 wdpaths.RESEARCH_DENSE_ROOT)

from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as A2  # noqa: E402
from partA_v2g.engine import load_params, param_set_id  # noqa: E402

ANALYSIS_ID = "WD-01-C4L1"
CHUNK = 4
RUNS_DIR = "campaign_stage1_chunk4"
TRUTH_HORIZON = 540.0                    # truth window (t0, t0 + 540]
PLAN = DATA_ROOT / "output" / "campaign_stage1_v3" / \
    "results_plan_chunk4" / "WORLD_PLAN.json"
REGISTRATION = WD01 / "REGISTRATION_C4L1.md"
FREEZE = HERE / "C4L1_FREEZE.json"
RESULTS_DIR = HERE / "results" / "c4l1"
GUARDS_ON = dict(cooldown_s=2.0, lc_min_speed_mps=1.0)

REGISTRATION_LF_SHA = "5e4a772137b8b427823b4860a20440eaa65abcbea735551eff681aa67b80b124"
PARAMS_NOMINAL_LF_SHA = ("079a55768f803b23c92af188680c1df91dc63f04a"
                         "cbbbac948870b46e75935f7")
CAL_WORLD_NAMES = frozenset({
    "w8621m2700p02t240", "w8622m2700p02t360", "w8639m2700p05t360",
    "w8640m2000p05t540", "w8657m2700p10t300", "w8658m2600p10t360",
    "w8674m2600p20t300", "w8675m2700p20t300", "w8692m2600p40t300",
})
EVAL_SCRIPTS = ("eval_c4l1.py", "eval_a2.py", "eval_a.py", "engine.py",
                "metrics.py", "cavctrl.py")


class AccessRefused(RuntimeError):
    """The loader refused a world set (registration/holdout/split rule)."""


# ----------------------------------------------------------- hash guards

def lf_sha256(path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def raw_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_registration(path) -> str:
    """Refuse unless `path` IS the frozen registration (LF-sha256)."""
    p = Path(path)
    if not p.exists():
        raise AccessRefused(f"registration file missing: {p}")
    got = lf_sha256(p)
    if got != REGISTRATION_LF_SHA:
        raise AccessRefused(
            f"registration LF-sha256 mismatch: got {got}, expected "
            f"{REGISTRATION_LF_SHA} ({p}); the 9 CAL worlds stay closed")
    return got


def engine_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(HERE),
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise AccessRefused(f"git rev-parse HEAD failed: {out.stderr}")
    return out.stdout.strip()


def verify_freeze(freeze_path=FREEZE, registration_path=REGISTRATION,
                  root: Path = HERE, engine_commit_sha: Optional[str] = None):
    """Refuse to start unless the freeze matches reality: the
    registration's LF-sha, the params LF-sha, every eval script's
    sha256, and the engine commit."""
    fp = Path(freeze_path)
    if not fp.exists():
        raise AccessRefused(f"freeze file missing: {fp}")
    fz = json.loads(fp.read_text(encoding="utf-8"))
    reg_sha = verify_registration(registration_path)
    if fz.get("registration_lf_sha256") != reg_sha:
        raise AccessRefused("freeze/registration LF-sha mismatch")
    params_sha = lf_sha256(Path(root) / "params_nominal.json")
    if fz.get("params_nominal_lf_sha256") != params_sha:
        raise AccessRefused(
            f"freeze params sha mismatch: freeze "
            f"{fz.get('params_nominal_lf_sha256')} vs actual {params_sha}")
    head = engine_commit_sha if engine_commit_sha is not None \
        else engine_commit()
    if fz.get("engine_commit") != head:
        # the engine is the source state at the recorded commit; HEAD may
        # have moved on (records-only commits) as long as it descends
        # from it -- the script hashes below pin the actual code.
        anc = subprocess.run(
            ["git", "merge-base", "--is-ancestor",
             str(fz.get("engine_commit")), head], cwd=str(HERE),
            capture_output=True, text=True)
        if anc.returncode != 0:
            raise AccessRefused(
                f"freeze engine commit mismatch: freeze "
                f"{fz.get('engine_commit')} vs HEAD {head}")
    for name, want in (fz.get("eval_scripts") or {}).items():
        got = raw_sha256(Path(root) / name)
        if got != want:
            raise AccessRefused(
                f"freeze script sha mismatch for {name}: freeze {want} "
                f"vs actual {got}")
    return fz


# --------------------------------------------------------- world access

def load_holdout_data():
    """The authoritative exclusions (registered HOLDOUT mechanism)."""
    sys.path.insert(0, str(Path(wdpaths.RESEARCH_DENSE_ROOT) / "tools"))
    import holdout_guard
    return holdout_guard, holdout_guard.load_holdout_exclusions()


def select_cal_worlds(plan: dict, holdout_data: Optional[dict] = None,
                      holdout_mod=None) -> List[dict]:
    """The 9 registered CAL worlds from the plan, split-checked,
    holdout-refused; refuses any world-set deviation."""
    sys.path.insert(0, str(Path(wdpaths.RESEARCH_DENSE_ROOT) / "tools"))
    hm = holdout_mod
    if hm is None:
        hm, _ = load_holdout_data()
    holdout_split = [w["world"] for w in plan["worlds"]
                     if w.get("split") == "HOLDOUT"
                     and w["world"] in CAL_WORLD_NAMES]
    if holdout_split:
        raise AccessRefused(f"HOLDOUT-split world selected: {holdout_split}")
    cal = [w for w in plan["worlds"] if w.get("split") == "CALIBRATION"]
    names = {w["world"] for w in cal}
    if names != set(CAL_WORLD_NAMES):
        raise AccessRefused(
            f"chunk-4 CALIBRATION set mismatch: plan has {sorted(names)}, "
            f"registration requires {sorted(CAL_WORLD_NAMES)}")
    assert hm is not None
    hm.assert_final_holdout_eligible(sorted(names),
                                     data=holdout_data)  # excluded roots
    out = []
    for w in cal:
        entry = dict(w)
        entry["chunk"] = CHUNK
        m = re.search(r"w\d+m(\d+)p(\d+)t(\d+)", w["world"])
        entry["demand"] = int(m.group(1)) if m else -1
        entry["penetration"] = int(m.group(2)) / 100.0 if m else \
            float("nan")
        out.append(entry)
    out.sort(key=lambda w: w["world"])
    return out


def load_cal_worlds(root: Path = DATA_ROOT, registration_path=REGISTRATION,
                    plan_path: Optional[Path] = None,
                    opened: Optional[set] = None) -> List[dict]:
    """verify_registration FIRST, then plan + screening.  The 9 CAL runs
    are touched only after the registration hash matched."""
    verify_registration(registration_path)
    opened = EA.OPENED_RUN_FILES if opened is None else opened
    plan = EA._open_json(plan_path or
                         (Path(root) / "output" / "campaign_stage1_v3" /
                          "results_plan_chunk4" / "WORLD_PLAN.json"))
    opened.add(str(plan_path or
                   (Path(root) / "output" / "campaign_stage1_v3" /
                    "results_plan_chunk4" / "WORLD_PLAN.json")))
    _hm, holdout_data = load_holdout_data()
    entries = select_cal_worlds(plan, holdout_data=holdout_data)
    usable = []
    for entry in entries:
        pp = A2._run_paths(entry["world"], Path(root))
        if not pp["physics"].exists() or not pp["summary"].exists():
            entry["skip_reason"] = "native_base run files missing"
            continue
        opened.add(str(pp["summary"]))
        summary = json.loads(pp["summary"].read_text(encoding="utf-8"))
        status = None
        if pp["receipt"].exists():
            opened.add(str(pp["receipt"]))
            status = json.loads(
                pp["receipt"].read_text(encoding="utf-8")).get("exit_status")
        dur = float(summary["duration_s"])
        t_end = A2._last_time(pp["physics"])
        if status == "complete" and abs(t_end - dur) <= 1.0:
            entry["native_run"] = pp["physics"]
            entry["t_end_s"] = t_end
            usable.append(entry)
        else:
            entry["skip_reason"] = f"receipt={status} t_end={t_end} dur={dur}"
    return usable


_RUN_PATH = re.compile(
    r"runs[/\\]campaign_stage1_chunk4[/\\](w\d+?\S*?)__native_base[/\\]")


def assert_cal_only(opened_files, refused: List[str]):
    """Close-out assertion: opened = plan file(s) + the 9 CAL worlds'
    __native_base run files ONLY; nothing HOLDOUT, nothing else."""
    opened_files = [str(p) for p in opened_files]
    bad = []
    for p in opened_files:
        if "results_plan_chunk4" in p and p.endswith("WORLD_PLAN.json"):
            continue
        m = _RUN_PATH.search(p)
        if m and m.group(1) in CAL_WORLD_NAMES:
            continue
        bad.append(p)
    assert not bad, f"opened files outside the registered CAL set: {bad}"
    opened_worlds = {m.group(1) for p in opened_files
                     if (m := _RUN_PATH.search(p))}
    assert opened_worlds <= CAL_WORLD_NAMES
    assert not (opened_worlds & set(refused)), \
        f"refused worlds' run files were opened: {opened_worlds & set(refused)}"
    return dict(opened_worlds=sorted(opened_worlds),
                cal_worlds_without_opened_run=sorted(
                    CAL_WORLD_NAMES - opened_worlds))


# --------------------------------------------------------------- worker

def _worker(args):
    """One worker (C4L1 runs exactly one): own shard file.  Returns
    (shard_path, opened_files, per-world stats)."""
    rank, entries, params, opts = args
    out_dir = RESULTS_DIR
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"shard_{rank}.jsonl"
    opened = set()
    EA.OPENED_RUN_FILES.clear()
    EA.OPENED_RUN_FILES.update(opened)
    EA.MAPPER = EA.ModelMapper()
    replica = EA.V1P3OReplica()
    guard_params = None
    if opts["guards_on_every"]:
        guard_params = json.loads(json.dumps(params))
        guard_params["mobil"].update(GUARDS_ON)
    stats = []
    horizon = None
    with open(shard, "w", encoding="utf-8") as raw:
        for wi, entry in enumerate(entries):
            tw = time.perf_counter()
            try:
                log = EA.NativeLog(entry)
            except FileNotFoundError as e:
                print(f"  [w{rank}] {entry['world']}: SKIP ({e})",
                      flush=True)
                continue
            # registered window rule: score only s + H <= t0 + 540
            # (attempt-2 receipt: eval_a2's hs_ok uses log.t_end = the
            # t0+600 log end, which overshoots the registered horizon)
            log.t_end = min(log.t_end, log.t0 + TRUTH_HORIZON)
            opened.update(EA.OPENED_RUN_FILES)
            horizon = log.t0 + TRUTH_HORIZON
            rows_world = 0
            for k in range(1, 19):
                w = log.t0 + 30.0 * k
                if w > horizon + 1e-9:
                    break
                rows, _t = A2.window_rows(
                    params, log, entry, w, replica,
                    hook_of=lambda ww, vehs: A2.native_control_hook(
                        ww, params, set(vehs) & log.cav_ids))
                if not rows:
                    break
                for r in rows:
                    raw.write(json.dumps(r) + "\n")
                rows_world += len(rows)
            if guard_params is not None and opts["guards_on_every"] and \
                    wi % opts["guards_on_every"] == 0:
                for k in range(1, 19):
                    w = log.t0 + 30.0 * k
                    if w > horizon + 1e-9:
                        break
                    rows, _t = A2.window_rows(
                        guard_params, log, entry, w, None,
                        hook_of=lambda ww, vehs: A2.native_control_hook(
                            ww, guard_params, set(vehs) & log.cav_ids),
                        emit_const=False)
                    if not rows:
                        break
                    for r in rows:
                        r["model"] = "v2g_guards_on"
                        raw.write(json.dumps(r) + "\n")
                    rows_world += len(rows)
            raw.flush()
            opened.update(EA.OPENED_RUN_FILES)
            stats.append(dict(world=entry["world"], rows=rows_world,
                              wall_s=round(time.perf_counter() - tw, 1)))
            print(f"  [w{rank}] [{wi + 1}/{len(entries)}] "
                  f"{entry['world']}: {rows_world} rows, "
                  f"{time.perf_counter() - tw:.1f}s", flush=True)
    side = shard.with_suffix(".opened.json")
    side.write_text(json.dumps(sorted(opened)), encoding="utf-8")
    return str(shard), sorted(opened), stats


# ------------------------------------------------------------ aggregation

def extra_tables(df):
    """n_free_road stats + sign diagnostics (aggregate_eval_a2
    definitions) from the saved rows."""
    import statistics as st
    tr = df[df.model == "v2g"]
    free = {}
    for h in (10, 30, 60, 120):
        g = tr[tr.H == h]
        if not len(g):
            continue
        fp = list(g.n_free_road_pred)
        ft = list(g.n_free_road_true)
        free[f"H{h}"] = dict(
            n=len(g),
            n_free_road_assumed_mean=st.fmean(fp),
            n_free_road_assumed_median=st.median(fp),
            n_free_road_assumed_max=max(fp),
            share_windows_with_any=sum(1 for x in fp if x > 0) / len(g),
            n_free_road_true_mean=st.fmean(ft),
            n_free_road_true_max=max(ft))
    sign = {}
    for h in (60, 120):
        g = tr[tr.H == h]
        if not len(g):
            continue
        errs = list(g.delay_err)
        sign[f"H{h}"] = dict(
            n=len(g), n_nonnegative=sum(1 for e in errs if e >= 0),
            n_negative=sum(1 for e in errs if e < 0),
            delay_err_mean=st.fmean(errs),
            delay_err_mae=st.fmean([abs(e) for e in errs]))
    return free, sign


def criteria_verdict(per_mh: dict, missing_arrivals: dict) -> dict:
    """The frozen C4L1 rule at H30, from the SAVED-row aggregation."""
    v = per_mh.get("v2g|H30")
    cs = per_mh.get("const_speed|H30")
    p3 = per_mh.get("v1_p3o_replica|H30")
    curve = missing_arrivals.get("H30", {})
    if not (v and cs and p3):
        return dict(passed=False,
                    reason="missing v2g/const_speed/p3o H30 cells")
    a_ok = v["collisions_per_1000"] <= 1.0
    c_mae_ok = v["delay_err_mae"] < cs["delay_err_mae"]
    mean_truth_delay = curve.get("delay_true_tracked_mean")
    bias = abs(v["delay_err_mean"])
    c_bias_ok = (mean_truth_delay is not None and
                 bias <= 0.15 * mean_truth_delay)
    d_lim = 1.05 * min(cs["v_rmse_mean"], p3["v_rmse_mean"])
    d_ok = v["v_rmse_mean"] <= d_lim
    return dict(
        H=30,
        a=dict(value=v["collisions_per_1000"], limit="<= 1.0",
               ok=a_ok),
        c=dict(delay_mae_v2g=v["delay_err_mae"],
               delay_mae_const_speed=cs["delay_err_mae"],
               mae_ok=c_mae_ok,
               mean_signed_delay_err=v["delay_err_mean"],
               mean_truth_tracked_delay=mean_truth_delay,
               bias_limit=0.15 * (mean_truth_delay or float("nan")),
               bias_ok=c_bias_ok),
        d=dict(v_rmse_v2g=v["v_rmse_mean"],
               v_rmse_const_speed=cs["v_rmse_mean"],
               v_rmse_p3o_replica=p3["v_rmse_mean"],
               limit=d_lim, ok=d_ok),
        passed=bool(a_ok and c_mae_ok and c_bias_ok and d_ok))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", type=str, default=str(REGISTRATION))
    ap.add_argument("--freeze", type=str, default=str(FREEZE))
    ap.add_argument("--guards-on-every", type=int, default=3,
                    help="labelled v2g_guards_on variant every k-th "
                         "world (0=off)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--data-root", type=str, default="")
    args = ap.parse_args()
    if args.workers != 1:
        sys.exit("C4L1 is registered at --workers 1 (one process only)")
    if args.data_root:
        globals()["DATA_ROOT"] = Path(args.data_root)
    t_start = time.perf_counter()

    # ---- freeze gate (refuses to start on any mismatch)
    fz = verify_freeze(args.freeze, args.registration)
    print(f"{ANALYSIS_ID}: freeze verified (engine {fz['engine_commit'][:12]},"
          f" registration {fz['registration_lf_sha256'][:12]}, params "
          f"{fz['params_nominal_lf_sha256'][:12]}, "
          f"{len(fz['eval_scripts'])} eval scripts pinned)")

    # ---- below-normal priority (single CPU process)
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        print("priority: BELOW_NORMAL")
    except Exception as e:  # noqa: BLE001
        print(f"priority: not set ({e})")

    if (RESULTS_DIR / "raw_rows.jsonl").exists() or \
            (RESULTS_DIR / "shards").exists():
        sys.exit(f"refusing: {RESULTS_DIR} already holds pass output "
                 "(one pass, never overwrite)")
    # exist_ok: the dir may already hold only the aborted-attempt record;
    # the guard above refuses when any pass output exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    params = load_params()
    assert lf_sha256(HERE / "params_nominal.json").upper() == \
        PARAMS_NOMINAL_LF_SHA.upper(), \
        "params_nominal.json is not the frozen C4L1 set (LF-sha)"
    EA.MAPPER = EA.ModelMapper()
    worlds = load_cal_worlds(globals()["DATA_ROOT"], args.registration)
    refused = sorted(CAL_WORLD_NAMES - {e["world"] for e in worlds})
    print(f"{ANALYSIS_ID}: {len(worlds)}/9 CAL worlds usable "
          f"(refused {refused}); workers 1; p3o ON; guards_on every "
          f"{args.guards_on_every}", flush=True)

    results = [_worker((0, worlds, params, dict(
        guards_on_every=args.guards_on_every)))]

    # ---- merge the shard in world order; raw rows saved BEFORE
    # aggregation (aggregation re-rolls nothing)
    shard, opened_worker, stats = results[0]
    world_order = [e["world"] for e in worlds]
    raw_path = RESULTS_DIR / "raw_rows.jsonl"
    n_rows = 0
    with open(raw_path, "w", encoding="utf-8") as out, \
            open(shard, encoding="utf-8") as fh:
        keep = []
        for line in fh:
            r = json.loads(line)
            if r["world"] in world_order:
                keep.append((world_order.index(r["world"]), line))
        for _i, line in sorted(keep, key=lambda t: t[0]):
            out.write(line if line.endswith("\n") else line + "\n")
            n_rows += 1
    opened_all = sorted(set(opened_worker) | set(EA.OPENED_RUN_FILES))
    EA.OPENED_RUN_FILES.clear()
    EA.OPENED_RUN_FILES.update(opened_all)
    cal_only = assert_cal_only(opened_all, refused)

    import pandas as pd
    df = pd.read_json(raw_path, lines=True)
    summary = A2.aggregate(df)
    free_tab, sign_tab = extra_tables(df)
    verdict = criteria_verdict(summary["per_model_h"],
                               summary["missing_arrivals"])
    meta = dict(
        analysis_id=ANALYSIS_ID,
        status="completed",
        registration="output/wd01/REGISTRATION_C4L1.md",
        registration_lf_sha256=verify_registration(args.registration),
        freeze_file=str(args.freeze),
        freeze_sha256=raw_sha256(args.freeze),
        freeze=fz,
        truth="chunk-4 <world>__native_base runs (true fleet-native); "
              "CAV ids from the physics rows' cav_ids",
        data_root=str(DATA_ROOT),
        plan=str(PLAN),
        n_worlds_run=len(worlds),
        refused_worlds=refused,
        per_world_stats=stats,
        workers=1,
        truth_horizon_s=TRUTH_HORIZON,
        windows_per_world="starts t0+30k, kept when w+H <= t0+540 "
                          "(exact counts in per_model_h n)",
        guards_on_variant=f"labelled design-deviation comparison "
                          f"(cooldown 2.0, lc_min_speed 1.0), every "
                          f"{args.guards_on_every}-th world",
        param_set_id=param_set_id(),
        wall_clock_s=time.perf_counter() - t_start,
        opened_run_files=opened_all,
        assert_cal_only=cal_only,
        criteria_verdict=verdict,
        summary=summary,
        n_free_road=free_tab,
        h60_h120_sign_diagnostics=sign_tab,
    )
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(meta, indent=1, default=str), encoding="utf-8")
    (RESULTS_DIR / "criteria_verdict.json").write_text(
        json.dumps(verdict, indent=1), encoding="utf-8")
    (RESULTS_DIR / "opened_run_files_full.json").write_text(
        json.dumps(opened_all, indent=1), encoding="utf-8")
    print("per-model H rows:")
    for k, v in summary["per_model_h"].items():
        print(" ", k, "n", v["n"], "s_rmse", round(v["s_rmse_mean"], 3),
              "v_rmse", round(v["v_rmse_mean"], 3), "delay_err",
              round(v["delay_err_mean"], 3), "MAE",
              round(v["delay_err_mae"], 3), "disc_lc",
              f"{v['lc_disc_pred']}/{v['lc_disc_true']}", "coll/1000",
              round(v["collisions_per_1000"], 3))
    print("CRITERIA VERDICT (frozen rule, H30):",
          "PASS" if verdict["passed"] else "FAIL")
    print(json.dumps(verdict, indent=1))
    print("DONE", ANALYSIS_ID, f"wall {meta['wall_clock_s']:.0f}s, "
          f"raw rows {n_rows} -> {raw_path}")


if __name__ == "__main__":
    main()
