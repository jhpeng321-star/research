"""WD-01-PB-V04A: build the v0.4a dataset root (ZERO SIMULATION).

Root: E:/research-dense/output/wd01_partB_v0data_v04a (NEW; all older
roots are read-only).  Every source of v04a_sources.enumerate_specs is
built FROM THE RAW RUNS with the v0.3.2 label/feature code
(v0lib.build_run_trace / window_label, featurize_v0.featurize_at with
the v0.4 action tokens at CAV_RAW 20, deployable_features.SourceObs for
the two sensing variants); nothing is hard-linked (the root is written
file by file, sha256 recorded per file).

Per source (world, source):
  * arm verification (actions_v04.ActionTimeline.verify_arm): the policy
    is checked against switch_schedule.json + kind=='command' owners; a
    mismatch REFUSES the source (recorded in the MANIFEST, not built);
  * decision times: chunk-4 base / native_base: t0, t0+10, ..., t0+570
    ("after t0"); other bases: the v0.3.2 grid (120 .. t0+570); branch
    arms: t0 (featurized from the shared base trace = the saved state)
    then t0+10 .. t0+570 from the branch trace;
  * labels: window_label per (t, tau); rows at t == t0 of a base-type
    trace use skip_first_tick=True so the damping arm's t0 label is on
    the same interval convention as a restored branch (which cannot
    contain the pair ending at t0+0.5);
  * per-row policy conditioning: canonical policy class + the
    planner-known schedule summary pi_native/pi_damping/pi_pi
    (ActionTimeline.action_shares: fleet runs exact one-hot);
  * files: <world>__<source>_feat_dep_strict.npz (HEADLINE),
    _feat_dep_det.npz (SENSITIVITY), _labels.parquet, _td.parquet, and a
    _done.json marker (resumable); MANIFEST.json is assembled from the
    markers; SHA256SUMS covers MANIFEST + worlds/.

Never read: dispatcher_rows.jsonl, insertions.jsonl.  HOLDOUT is never
enumerated (plan split first) and an open-audit hook asserts it.

Usage:
  python build_v04a.py --data-root E:/research-dense --workers 4
  python build_v04a.py --data-root <tmp> --out <tmp/root> --worlds w8623m2600p02t540 --chunks chunk4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import actions_v04 as A  # noqa: E402
import build_v0data as B  # noqa: E402
import featurize_v0 as F  # noqa: E402
import v04a_sources as S  # noqa: E402
import v0lib as V  # noqa: E402
import wdpaths  # noqa: E402
from deployable_features import SourceObs, VARIANTS  # noqa: E402

SCHEMA = "wd01_partB_v0data/1"          # loader-compatible
V04A_SCHEMA = "wd01_partB_v0data_v04a/1"
T0W = time.time()

LABEL_COLS = B.LABEL_COLS + ["pi_native", "pi_damping", "pi_pi", "role",
                             "aux", "arm_kind"]
TAG = {"detector": "_feat_dep_det.npz", "strict": "_feat_dep_strict.npz"}
SHARE_COLS = ("pi_native", "pi_damping", "pi_pi")


def log(msg):
    print(f"[{time.time() - T0W:7.1f}s] {msg}", flush=True)


def sha256_file(path, block=1 << 20):
    return V.sha256_file(path)


# ---------------------------------------------------------- grids -------

def decision_times(spec, rt, t0: float) -> np.ndarray:
    """Decision times of one source (see module docstring).  Branch arms
    return the t0 row FIRST (featurized from the base trace)."""
    hi = t0 + 600.0 - V.COMMIT_S
    if spec.type == "base_like":
        lo = t0 if spec.chunk == "chunk4" else V.BASE_DECISION_MIN_T
        ts = np.arange(lo, hi + 1e-9, V.DECIDE_EVERY_S)
        return ts[ts <= rt.data_end + 1e-9]
    ts = V.decision_grid(t0, rt.data_end, "native")
    return np.concatenate([[t0], ts])


def _has_snapshot(rt, t: float) -> bool:
    k = int(round((t - rt.t_origin) / V.DT))
    return 0 <= k <= rt.n_ticks and rt.tick_last[k] > rt.tick_first[k]


# ------------------------------------------------------- one source -----

def build_one(spec, rt_base, base_obs_sched, timeline_cache, sched_cache,
              out_dir: Path, geom, resolved: dict) -> dict:
    """Build one source; returns its MANIFEST entry.  rt_base = the shared
    damping-base trace (origin 0) of the world."""
    world, source = spec.world, spec.source
    t0 = float(spec.t0_s)
    arm = Path(wdpaths.remap(spec.arm_dir))
    base_dir = Path(wdpaths.remap(spec.base_dir))
    ok, receipt, reason = V.receipt_status(arm)
    if not ok:
        return dict(key=spec.key(), refused=True,
                    reason=f"incomplete run: {reason}", spec=spec.__dict__)
    tl = A.ActionTimeline.from_run_dir(arm, schedule_authoritative=True)
    try:
        arm_ev = tl.verify_arm(spec.verify_kind)
    except A.ArmPolicyMismatch as e:
        return dict(key=spec.key(), refused=True,
                    reason=f"ArmPolicyMismatch: {e}", spec=spec.__dict__)

    loading = int(spec.demand_veh_h)
    pen = float(spec.penetration)
    p40 = bool(abs(pen - 0.4) < 1e-9)

    if spec.type == "base_like":
        if arm == base_dir and rt_base is not None:
            rt = rt_base
        else:
            rt = V.build_run_trace(world, "base", arm, t0, t_ins_pre={},
                                   sched_cache=sched_cache)
        t_pre = {}
    else:
        t_pre = V.build_t_ins_pre(rt_base, t0)
        rt = V.build_run_trace(world, "native", arm, t0, t_ins_pre=t_pre,
                               sched_cache=sched_cache,
                               base_arm_dir=base_dir)
    rt.demand_loading = loading

    ts_all = decision_times(spec, rt, t0)
    items, keep_t = [], []
    for i, t in enumerate(ts_all):
        first_branch = (spec.type == "branch" and i == 0)
        src_rt = rt_base if first_branch else rt
        if _has_snapshot(src_rt, float(t)):
            items.append((src_rt, float(t)))
            keep_t.append(float(t))
    n_dropped = int(len(ts_all) - len(keep_t))
    ts = np.asarray(keep_t, np.float64)
    tau_avail = rt.data_end - ts
    speed_mode, mode_source = B.resolve_speed_mode(
        receipt, rt.stats, spec.default_speed_mode)
    pre_fix = bool(speed_mode == 31)

    feats = F.build_features_multi(items, rt, tau_avail, geom, actions=tl)
    feats["chunk_id"] = np.str_(spec.chunk)
    # ---- per-row labels + pi conditioning -------------------------------
    raw_off = feats["raw_off"]
    vid_tab = feats["vid_table"]
    fleet_share = None
    if tl.fleet_policy is not None:
        fleet_share = np.zeros(3)
        fleet_share[A.SHARE_ORDER.index(tl.fleet_policy)] = 1.0
    lab_rows, td_rows = [], []
    aux = bool(spec.role == "aux")
    pol_kw = dict(role=spec.role, aux=aux, arm_kind=spec.verify_kind)
    for i, t in enumerate(ts):
        t = float(t)
        ta = float(tau_avail[i])
        at_t0 = abs(t - t0) < 1e-9
        skip = bool(at_t0 and spec.type == "base_like")
        sl = slice(int(raw_off[i]), int(raw_off[i + 1]))
        cav_v = [str(vid_tab[k]) for k in
                 feats["raw_vid"][sl][feats["raw_is_cav"][sl] == 1]]
        per_t = {}

        def label(tau):
            if tau in per_t:
                return per_t[tau]
            row = V.window_label(rt, t, tau, skip_first_tick=skip)
            if at_t0 and spec.type == "branch":
                row["n_vehicles_present"] = int(feats["raw_off"][i + 1]
                                                - feats["raw_off"][i])
            per_t[tau] = row
            return row

        for tau in V.TAUS:
            if tau > ta + 1e-9:
                continue
            sh = fleet_share if fleet_share is not None else \
                tl.action_shares(cav_v, t, tau)
            lab_rows.append(dict(
                world=world, source=source, chunk_id=spec.chunk,
                split=spec.split, demand_veh_h=loading, penetration=pen,
                p40=p40, pre_fix=pre_fix, t0_s=t0, policy=spec.policy,
                t=t, tau=float(tau), tau_available=ta,
                cav_speed_mode=speed_mode, **label(tau),
                pi_native=float(sh[0]), pi_damping=float(sh[1]),
                pi_pi=float(sh[2]), **pol_kw))
        if at_t0:
            continue                    # no TD tuples at the shared t0 row
        kn = int(round((t + V.COMMIT_S - rt.t_origin) / V.DT))
        if not (0 <= kn <= rt.n_ticks and
                rt.tick_last[kn] > rt.tick_first[kn]):
            continue
        d30 = label(V.COMMIT_S)["total_delay_ref_s"]
        for tau in V.TAUS:
            if tau < V.COMMIT_S - 1e-9:
                continue
            complete = tau <= ta + 1e-9
            mc = label(tau) if complete else None
            td_rows.append(dict(
                world=world, source=source, chunk_id=spec.chunk,
                split=spec.split, demand_veh_h=loading, penetration=pen,
                p40=p40, pre_fix=pre_fix, t0_s=t0, t=t,
                t_next=float(t + V.COMMIT_S), tau=float(tau),
                tau_available=ta, d30_s=d30,
                label_mc_s=(mc["total_delay_ref_s"] if complete
                            else np.nan),
                label_mc_complete=int(complete),
                cav_speed_mode=speed_mode))
    lab_df = pd.DataFrame(lab_rows, columns=LABEL_COLS)
    td_df = pd.DataFrame(td_rows, columns=B.TD_COLS)

    # ---- deployable sidecars (HEADLINE strict, SENSITIVITY detector) ----
    ins = dict(rt.t_ins_pre)
    ins.update(rt.t_ins)
    pairs = sorted(ins.items())
    obs = SourceObs(rt_base.sched_vid, rt_base.sched_t, rt_base.sched_ent,
                    rt_base.sched_is_cav,
                    [v for v, _ in pairs], [t for _, t in pairs])
    blocks = {v: obs.blocks(ts, t0, float(loading), variant=v)
              for v in VARIANTS}
    dem_det, ctx_det = blocks["detector"]
    dem_str, ctx_str = blocks["strict"]
    orc = feats["context"]
    for c in (0, 3, 6, 7):
        if not np.allclose(ctx_det[:, c], orc[:, c], rtol=0.0, atol=1e-6):
            raise SystemExit(f"VERIFICATION FAIL {spec.key()}: detector "
                             f"ctx col {c} != oracle context")
    for c in (0, 3):
        if not np.allclose(ctx_str[:, c], orc[:, c], rtol=0.0, atol=1e-6):
            raise SystemExit(f"VERIFICATION FAIL {spec.key()}: strict "
                             f"ctx col {c} != oracle context")
    assert np.array_equal(dem_str, dem_det[:, 6:])
    assert (ctx_det[:, 1] == 0).all() and (ctx_str[:, 1] == 0).all()
    assert (ctx_str[:, 6] == 0).all() and (ctx_str[:, 7] == 0).all()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{world}__{source}"
    files = {}
    for v in VARIANTS:
        dem, ctx = blocks[v]
        payload = {k: feats[k] for k in feats
                   if k not in ("demand", "context")}
        payload["demand"], payload["context"] = dem, ctx
        payload["feature_set"] = np.str_(f"deployable_{v}")
        payload["sensing_variant"] = np.str_(v)
        payload["world"] = np.str_(world)
        payload["source"] = np.str_(source)
        payload["world_t0"] = np.float64(t0)
        payload["data_end"] = np.float64(rt.data_end)
        payload["policy"] = np.str_(spec.policy)
        payload["action_inputs"] = np.str_(
            "v04a: switch_schedule.json (planned action = token) + "
            "kind=='command' owners (verification) + fleet manifest "
            "policy; never dispatcher_rows / insertions")
        dst = Path(str(stem) + TAG[v])
        tmp = Path(str(dst) + ".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(dst)
        files[f"worlds/{spec.chunk}/{dst.name}"] = sha256_file(dst)
    f_lab = Path(str(stem) + "_labels.parquet")
    f_td = Path(str(stem) + "_td.parquet")
    lab_df.to_parquet(f_lab, index=False)
    td_df.to_parquet(f_td, index=False)
    files[f"worlds/{spec.chunk}/{f_lab.name}"] = sha256_file(f_lab)
    files[f"worlds/{spec.chunk}/{f_td.name}"] = sha256_file(f_td)

    tok = feats["cav_tokens"]
    msk = feats["cav_mask"] > 0
    act_slot = tok[..., 5:10].argmax(-1)
    n_tok = int(msk.sum())
    act_counts = {A.ACTIONS_V04[k]: int(((act_slot == k) & msk).sum())
                  for k in range(len(A.ACTIONS_V04))}
    entry = dict(
        world=world, source=source, chunk_id=spec.chunk, split=spec.split,
        role=spec.role, aux=aux, demand_veh_h=loading, penetration=pen,
        p40=p40, pre_fix=pre_fix, t0_s=t0,
        run=str(arm), base_run=str(base_dir),
        policy=spec.policy, policy_v04a=spec.policy,
        arm_kind=spec.verify_kind, source_type=spec.type,
        cav_speed_mode=speed_mode, speed_mode_source=mode_source,
        physics_sha256=sha256_file(arm / "run" / "physics_steps.jsonl"),
        files=files,
        rows=dict(n_decision_times=int(len(ts)),
                  n_label_rows=int(len(lab_df)), n_td_rows=int(len(td_df)),
                  n_dropped_decision_ticks=n_dropped),
        token_actions=dict(n_tokens=n_tok, by_action=act_counts),
        arm_verification=arm_ev,
        label_sums=dict(
            total_delay_s=float(lab_df["total_delay_ref_s"].sum())
            if len(lab_df) else 0.0,
            collision_any=int(lab_df["collision_any"].sum())
            if len(lab_df) else 0),
        trace_stats=rt.stats, note=spec.note,
        built_utc=datetime.now(timezone.utc).isoformat())
    B.atomic_write_json(out_dir / f"{world}__{source}_done.json", entry)
    return entry


def verify_done(root: Path, chunk: str, world: str, source: str):
    p = root / "worlds" / chunk / f"{world}__{source}_done.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return None
    for rel, sha in d.get("files", {}).items():
        f = root / rel
        if not f.is_file() or sha256_file(f) != sha:
            return None
    return d


def build_world(task) -> list:
    """All sources of one (chunk, world): shared base trace + every
    spec.  Runs in a pool worker; returns [entry | refusal, ...]."""
    root, specs, rebuild = task
    root = Path(root)
    S.OpenAudit.start()
    S.OpenAudit.paths = []
    B.lower_priority()
    res = _build_world(root, specs, rebuild)
    opened = sorted({p.replace("\\", "/") for p in S.OpenAudit.paths
                     if "/runs/" in p.replace("\\", "/")
                     or "/output/campaign" in p.replace("\\", "/")})
    S.OpenAudit.paths = []
    res.append(dict(_opened=opened))
    return res


def _build_world(root, specs, rebuild) -> list:
    geom = V.cached_geometry()
    sched_cache: dict = {}
    out_dir = root / "worlds" / specs[0]["chunk"]
    out: list = []
    todo = []
    for sd in specs:
        d = None if rebuild else verify_done(root, sd["chunk"], sd["world"],
                                             sd["source"])
        if d is not None:
            d["resumed"] = True
            out.append(d)
        else:
            todo.append(sd)
    if not todo:
        return out
    first = S.SrcSpec(**todo[0])
    base_dir = Path(wdpaths.remap(first.base_dir))
    ok, _r, reason = V.receipt_status(base_dir)
    if not ok:
        return out + [dict(key=S.SrcSpec(**sd).key(), refused=True,
                           reason=f"base run incomplete: {reason}",
                           spec=sd) for sd in todo]
    t0 = float(first.t0_s)
    rt_base = V.build_run_trace(first.world, "base", base_dir, t0,
                                t_ins_pre={}, sched_cache=sched_cache)
    rt_base.demand_loading = int(first.demand_veh_h)
    for sd in todo:
        spec = S.SrcSpec(**sd)
        t1 = time.time()
        try:
            e = build_one(spec, rt_base, None, None, sched_cache, out_dir,
                          geom, {})
        except Exception as ex:               # noqa: BLE001
            e = dict(key=spec.key(), refused=True,
                     reason=f"build error {type(ex).__name__}: {ex}",
                     spec=sd)
        out.append(e)
        if not e.get("refused"):
            log(f"built {spec.key()}: dt={e['rows']['n_decision_times']} "
                f"labels={e['rows']['n_label_rows']} "
                f"({time.time() - t1:.1f}s)")
        else:
            log(f"REFUSED {spec.key()}: {e['reason']}")
    return out


# ------------------------------------------------------------ finalize --

def write_sha256sums(root: Path) -> int:
    rels = ["MANIFEST.json"] + [
        p.relative_to(root).as_posix()
        for p in sorted((root / "worlds").rglob("*")) if p.is_file()]
    lines = [f"{sha256_file(root / rel)}  {rel}" for rel in rels]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
    return len(lines)


def assemble_manifest(root: Path, results: list, specs: list, holdout,
                      arm_verification_path, argv) -> dict:
    entries, refused = {}, []
    for e in results:
        if "_opened" in e:
            continue
        if e.get("refused"):
            refused.append(dict(key=e["key"], reason=e["reason"]))
            continue
        key = f"{e['chunk_id']}/{e['world']}__{e['source']}"
        entries[key] = e
    chunks = {}
    for c in sorted({s.chunk for s in specs}):
        keys = sorted(k for k in entries if k.startswith(c + "/"))
        chunks[c] = dict(
            chunk_id=c, plan=str(S.PLAN[c]),
            plan_sha256=sha256_file(Path(wdpaths.remap(S.PLAN[c]))),
            runs_root=str(S.RUNS[c]),
            default_speed_mode=S.DEFAULT_SPEED_MODE[c],
            skipped_holdout=[dict(world=w, split="HOLDOUT")
                             for w in sorted(holdout.get(c, ()))],
            skipped_incomplete=[r for r in refused
                                if r["key"].startswith(c + "/")],
            world_keys=keys, n_sources=len(keys))
    pol_counts: dict = {}
    for e in entries.values():
        k = (e["policy"], e["role"], e["split"])
        pol_counts["|".join(k)] = pol_counts.get("|".join(k), 0) + 1
    man = dict(
        schema=SCHEMA, v04a_schema=V04A_SCHEMA, zero_simulation=True,
        tau_s=list(V.TAUS), commit_window_s=V.COMMIT_S,
        decide_every_s=V.DECIDE_EVERY_S,
        note=("WD-01-PB-V04A root: true-policy sources + per-CAV action "
              "tokens (CAV_RAW 20) + policy conditioning; built from the "
              "raw runs; auxiliary sources flagged aux=true (training "
              "only)"),
        v04a=dict(
            registration="output/wd01/REGISTRATION_PARTB_V04A.md",
            policy_classes=list(S.POLICY_CLASSES),
            feature_sets=["deployable_strict", "deployable_detector"],
            no_oracle_feature_files=True,
            per_cav_action_sources=["switch_schedule.json",
                                    "suffix_commands.jsonl (kind=='command')",
                                    "telemetry.jsonl (kind=='command', "
                                    "fleet base runs)",
                                    "manifest.json controller (fleet)"],
            dispatcher_rows_read=False, insertions_jsonl_read=False,
            holdout_accessed=False,
            decision_grid=dict(
                chunk4_base_like="t0 .. t0+570 step 10 (after t0)",
                other_base="120 .. t0+570 step 10 (v0.3.2 grid)",
                branch="t0 (shared base state) then t0+10 .. t0+570"),
            t0_label_convention=("base-type rows at t == t0 use "
                                 "skip_first_tick=True (branch traces "
                                 "cannot contain the pair ending at "
                                 "t0+0.5)"),
            arm_verification_file=str(arm_verification_path),
            n_refused=len(refused), refused=refused,
            counts_by_policy_role_split=pol_counts),
        chunks=chunks, worlds=entries,
        build=dict(built_utc=datetime.now(timezone.utc).isoformat(),
                   builder="partB_v0data/build_v04a.py",
                   n_sources=len(entries), command=" ".join(argv),
                   holdout_accessed=False))
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(wdpaths.RESEARCH_DENSE_ROOT))
    ap.add_argument("--out", default=None,
                    help="root dir (default <data-root>/output/"
                         "wd01_partB_v0data_v04a)")
    ap.add_argument("--chunks", default=",".join(S.ALL_CHUNKS))
    ap.add_argument("--worlds", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--arm-verification", default=None,
                    help="arm_verify_v04a.py output json to record in the "
                         "MANIFEST")
    args = ap.parse_args(argv)
    rd = Path(args.data_root)
    root = Path(args.out or rd / "output" / "wd01_partB_v0data_v04a")
    chunks = tuple(c for c in args.chunks.split(",") if c)
    want = set(w for w in args.worlds.split(",") if w) or None
    root.mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    S.OpenAudit.start()
    holdout = {c: S.chunk_worlds(c)[1] for c in chunks}
    specs = S.enumerate_specs(chunks, want)
    assert all(s.split in S.KEEP_SPLITS for s in specs)
    assert not ({s.world for s in specs}
                & {w for h in holdout.values() for w in h})
    by_world: dict = {}
    for s in specs:
        by_world.setdefault((s.chunk, s.world), []).append(s.__dict__)
    tasks = [(str(root), v, args.rebuild) for v in by_world.values()]
    log(f"v04a build: {len(specs)} sources in {len(tasks)} worlds, "
        f"{args.workers} workers -> {root}")
    results: list = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(build_world, tasks,
                                                        chunksize=1), 1):
                results.extend(res)
                if i % 10 == 0 or i == len(tasks):
                    log(f"  worlds {i}/{len(tasks)}")
    else:
        for i, t in enumerate(tasks, 1):
            results.extend(build_world(t))
            log(f"  worlds {i}/{len(tasks)}")
    hold_all = {w for h in holdout.values() for w in h}
    opened = sorted({p for e in results for p in e.get("_opened", ())})
    bad = [p for p in opened if any(w in p for w in hold_all)]
    assert not bad, f"HOLDOUT path opened by a worker: {bad[:5]}"
    assert not [p for p in opened if "dispatcher_rows" in p
                or "insertions.jsonl" in p], "forbidden log opened"
    (root / "results" / "opened_run_files.json").write_text(
        json.dumps(dict(n=len(opened), holdout_worlds_checked=len(hold_all),
                        paths=opened), indent=0), encoding="utf-8")
    n_open = S.OpenAudit.assert_no_holdout(hold_all)
    S.OpenAudit.stop()
    arm_path = args.arm_verification or ""
    man = assemble_manifest(root, results, specs, holdout, arm_path,
                            sys.argv[1:])
    man["v04a"]["open_audit"] = dict(
        n_paths_recorded_in_parent=n_open,
        note=("parent-process audit only; pool workers apply the same "
              "spec list (no HOLDOUT world can reach a spec)"))
    B.atomic_write_json(root / "MANIFEST.json", man)
    n_sums = write_sha256sums(root)
    log(f"MANIFEST: {len(man['worlds'])} sources, "
        f"{man['v04a']['n_refused']} refused; SHA256SUMS {n_sums} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
