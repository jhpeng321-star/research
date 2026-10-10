"""WD-01 Part B round 0 builder: V0 dataset from chunk-1 native runs.

ZERO SIMULATION.  CHUNK-AGNOSTIC: every run takes --chunk (worlds land in
worlds/<chunk>/, MANIFEST carries a per-chunk section chunks.<chunk> with
the chunk's plan/runs-root/speed-mode config), so chunk-2 worlds can be
appended per world later without touching chunk 1.  cav_speed_mode is
resolved per run: run receipt (config.speed_mode, as chunk 2 records, or
the chunk-3-style top-level cav_speed_mode key) -> speed modes observed in
the CAV physics rows -> all physics rows -> per-chunk config default
(--default-speed-mode); pre_fix = (cav_speed_mode == 31) and p40 =
(penetration == 0.4) are stamped on every label/TD row and npz scalar.

Per (world, source) writes three files under
  <out>/worlds/<chunk>/<world>__<source>_feat.npz      features per decision time
                                  <world>__<source>_labels.parquet  (t, tau) labels
                                  <world>__<source>_td.parquet      TD tuples
plus a <world>__<source>_done.json marker holding the files' sha256s
(keys are out-relative paths); on restart completed sources are skipped
(marker + hash re-verified).  A global MANIFEST.json lists every source
(flat keys "<chunk>/<world>__<source>") with per-chunk sections;
HOLDOUT worlds are listed as skipped and never built.

Single process, below-normal priority (the box is shared with SUMO).

Usage:
  python build_v0data.py                     # chunk1 (default): everything buildable
  python build_v0data.py --worlds w8407m2400p02t360
  python build_v0data.py --sources native
  python build_v0data.py --chunk chunk2 --plan <chunk2 WORLD_PLAN.json> \
                         --runs <chunk2 runs root> --default-speed-mode 27
  python build_v0data.py --rebuild --worlds <w>   # ignore done markers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v0lib as V  # noqa: E402
import featurize_v0 as F  # noqa: E402

sys.path.insert(0, str(HERE.parent))         # output/wd01 (wdpaths)
import wdpaths  # noqa: E402

DEFAULT_PLAN = str(Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" /
                   "campaign_stage1_chunk1" / "results_plan_wm44" /
                   "WORLD_PLAN.json")
DEFAULT_RUNS = str(Path(wdpaths.RESEARCH_DENSE_ROOT) / "runs" /
                   "campaign_stage1_chunk1")
SCHEMA = "wd01_partB_v0data/1"
DEFAULT_CHUNK = "chunk1"
DEFAULT_SPEED_MODE = 31            # chunk-1 fleet; chunk 2 records 27
T0W = time.time()

LABEL_COLS = ["world", "source", "chunk_id", "split", "demand_veh_h",
              "penetration", "p40", "pre_fix", "t0_s", "policy", "t", "tau",
              "tau_available", "cav_speed_mode", "time_on_section_s",
              "free_flow_ref_s", "traversed_m", "wait_mainline_s",
              "wait_ramp_s", "total_delay_ref_s", "ttc_lt3_veh_s",
              "hard_brake_s", "collision_any", "collision_n_veh",
              "teleports", "n_queued_at_horizon", "n_vehicles_present",
              "n_unmatched_insertions"]
TD_COLS = ["world", "source", "chunk_id", "split", "demand_veh_h",
           "penetration", "p40", "pre_fix", "t0_s", "t", "t_next", "tau",
           "tau_available", "d30_s", "label_mc_s", "label_mc_complete",
           "cav_speed_mode"]


def resolve_speed_mode(receipt: dict | None, rt_stats: dict,
                       default: int | None):
    """(cav_speed_mode, source) per run, in priority order:
    1. the run receipt's recorded config speed mode (chunk 2 records 27);
    2. the chunk-3-style top-level receipt key cav_speed_mode;
    3. the speed modes observed in the CAV physics rows (single distinct
       value only -- chunk 2 receipts record null while CAV rows show 27,
       HDVs 31);
    4. the speed modes observed over ALL physics rows (chunk-1 behaviour);
    5. the per-chunk config default (--default-speed-mode).
    'mixed' observed sets fall back to the next tier; the observed sets
    stay in the trace stats for transparency."""
    if receipt:
        try:
            m = receipt["evaluate_summary"]["config"]["speed_mode"]
            if m is not None:
                return int(m), "receipt"
        except (KeyError, TypeError, ValueError):
            pass
        try:
            m = receipt.get("cav_speed_mode")
            if m is not None:
                return int(m), "receipt"
        except (AttributeError, TypeError, ValueError):
            pass
    cav = [float(v) for v in (rt_stats.get("cav_smode_values") or [])]
    if len(cav) == 1:
        return int(cav[0]), "observed-cav"
    vals = [float(v) for v in (rt_stats.get("smode_values") or [])]
    if len(vals) == 1:
        return int(vals[0]), "observed"
    if default is not None:
        return int(default), "config"
    return int(DEFAULT_SPEED_MODE), "config-default"


def log(msg):
    print(f"[{time.time() - T0W:7.1f}s] {msg}", flush=True)


def lower_priority():
    BELOW_NORMAL = 0x4000
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
        log("priority set to below-normal (psutil)")
        return
    except Exception:
        pass
    try:
        import ctypes
        if os.name == "nt":
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.kernel32.SetPriorityClass(h, BELOW_NORMAL)
            log(f"priority set to below-normal (ctypes, ok={bool(ok)})")
        else:
            os.nice(10)
            log("priority set to nice 10")
    except Exception as e:                        # best effort only
        log(f"priority unchanged ({e})")


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def sha_of(path: Path):
    return V.sha256_file(path)


def t_ins_pre_from_base_physics(base_arm: Path, t0: float) -> dict:
    """Shared pre-t0 insertions from the base run's physics_steps
    departed_ids (v0.3.2 fix #2: the ONLY derivation path -- the old
    insertions.jsonl fallback is REMOVED; that log is unreliable and its
    empty pre-t0 history inflated the w8473 fix27 labels ~8.8x).  A light
    streaming pass collecting first insertions at or before t0, with the
    same grid/off-grid handling as v0lib.build_run_trace."""
    out: dict = {}
    p = base_arm / "run" / "physics_steps.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"base run physics missing: {p} -- pre-t0 insertion history "
            f"must come from the base run's departed_ids (the "
            f"insertions.jsonl fallback was removed, v0.3.2 fix #2)")
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t = float(r["time_s"])
            if t > t0 + 1e-9:
                continue
            k = round(t / V.DT)
            if abs(k * V.DT - t) > 1e-6:
                continue                       # off-grid row: trace skips it
            for vid_ in r["departed_ids"]:
                if vid_ not in out:
                    out[vid_] = t
    return out


def insertion_crosscheck(rt: V.RunTrace, arm_dir: Path) -> dict:
    """Physics-derived insertion times vs the run's insertions.jsonl."""
    p = arm_dir / "run" / "insertions.jsonl"
    logged: dict = {}
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                logged[r["vehicle_id"]] = float(r["inserted_s"])
    exact = mismatch = log_only = 0
    for vid, ti in rt.t_ins.items():
        if vid not in logged:
            continue
        if abs(logged[vid] - ti) <= 1e-6:
            exact += 1
        else:
            mismatch += 1
    log_only = sum(1 for vid in logged if vid not in rt.t_ins)
    return dict(n_physics_derived=len(rt.t_ins), n_logged=len(logged),
                n_exact=exact, n_mismatch=mismatch, n_logged_only=log_only)


def build_source(world: dict, source: str, arm_dir: Path, runs_root: Path,
                 out_dir: Path, geom, sched_cache: dict,
                 base_rt_cache: dict, chunk_id: str = DEFAULT_CHUNK,
                 default_speed_mode: int | None = DEFAULT_SPEED_MODE,
                 receipt: dict | None = None,
                 bases_root: Path | None = None) -> dict:
    """Build one (world, source) into worlds/<chunk_id>/; returns its
    manifest entry (files keyed by out-relative paths)."""
    wname = world["world"]
    t0 = float(world["t0_s"])
    split = world["split"]
    if str(split).upper() == "HOLDOUT":   # in-code guard: main() already
        raise AssertionError(             # skips HOLDOUT before any run
            f"refusing to build HOLDOUT world {wname!r}: its run files "
            f"must never be opened")      # file is touched
    loading = int(world["loading_veh_h"])
    pen = float(world["penetration"])
    p40 = bool(abs(pen - 0.4) < 1e-9)

    base_ref = (world.get("base_arm") or {}).get("run") \
        or f"{wname}__base"
    base_arm = (runs_root if bases_root is None else bases_root) / base_ref
    # base trace is shared (labels of the native branch need its pre-t0
    # insertion history; the base source needs it as its own trace)
    if source == "base":
        rt = V.build_run_trace(wname, "base", arm_dir, t0,
                               t_ins_pre={}, sched_cache=sched_cache,
                               bases_root=bases_root)
        base_rt_cache[wname] = rt
        t_pre = {}
    else:
        cached = base_rt_cache.get(wname)
        if cached is not None:
            t_pre = V.build_t_ins_pre(cached, t0)
        else:
            # v0.3.2 fix #2: ALWAYS derive the shared pre-t0 history from
            # the base run's physics departed_ids (bases_root aware); the
            # insertions.jsonl fallback that inflated w8473 is removed.
            t_pre = t_ins_pre_from_base_physics(base_arm, t0)
        rt = V.build_run_trace(wname, "native", arm_dir, t0,
                               t_ins_pre=t_pre, sched_cache=sched_cache,
                               bases_root=bases_root)
    rt.demand_loading = loading

    ts = V.decision_grid(t0, rt.data_end, source)
    # decision times need the snapshot tick to exist
    keep = []
    for t in ts:
        k = int(round((t - rt.t_origin) / V.DT))
        if 0 <= k <= rt.n_ticks and rt.tick_last[k] > rt.tick_first[k]:
            keep.append(t)
    n_dropped_ticks = int(len(ts) - len(keep))
    ts = np.asarray(keep, np.float64)
    tau_avail = rt.data_end - ts
    cav_speed_mode, mode_source = resolve_speed_mode(
        receipt, rt.stats, default_speed_mode)
    pre_fix = bool(cav_speed_mode == 31)

    # ---- label rows (complete horizons only) --------------------------
    lab_rows, td_rows = [], []
    for t, ta in zip(ts, tau_avail):
        lab = V.window_label(rt, t, V.COMMIT_S)          # D30 reused below
        per_t = {V.COMMIT_S: lab}
        for tau in V.TAUS:
            if tau > ta + 1e-9:
                continue                                 # horizon exceeds data
            row = per_t.get(tau) or V.window_label(rt, t, tau)
            per_t[tau] = row
            lab_rows.append(dict(
                world=wname, source=source, chunk_id=chunk_id, split=split,
                demand_veh_h=loading, penetration=pen, p40=p40,
                pre_fix=pre_fix, t0_s=t0,
                policy=V.SOURCE_POLICY[source], t=float(t), tau=float(tau),
                tau_available=float(ta),
                cav_speed_mode=cav_speed_mode, **row))
        # TD tuples: next snapshot tick must exist
        kn = int(round((t + V.COMMIT_S - rt.t_origin) / V.DT))
        if not (0 <= kn <= rt.n_ticks and
                rt.tick_last[kn] > rt.tick_first[kn]):
            continue
        d30 = lab["total_delay_ref_s"]
        for tau in V.TAUS:
            if tau < V.COMMIT_S - 1e-9:
                continue
            complete = tau <= ta + 1e-9
            mc = per_t.get(tau)
            td_rows.append(dict(
                world=wname, source=source, chunk_id=chunk_id, split=split,
                demand_veh_h=loading, penetration=pen, p40=p40,
                pre_fix=pre_fix, t0_s=t0,
                t=float(t), t_next=float(t + V.COMMIT_S), tau=float(tau),
                tau_available=float(ta),
                d30_s=d30,
                label_mc_s=(mc["total_delay_ref_s"] if complete else np.nan),
                label_mc_complete=int(complete),
                cav_speed_mode=cav_speed_mode))

    lab_df = pd.DataFrame(lab_rows, columns=LABEL_COLS)
    td_df = pd.DataFrame(td_rows, columns=TD_COLS)

    # ---- features ------------------------------------------------------
    feats = F.attach_meta(rt, F.build_features(rt, ts, tau_avail, geom))
    feats["chunk_id"] = np.str_(chunk_id)
    feats["pre_fix"] = np.bool_(pre_fix)
    feats["p40"] = np.bool_(p40)

    out_dir.mkdir(parents=True, exist_ok=True)
    f_npz = out_dir / f"{wname}__{source}_feat.npz"
    f_lab = out_dir / f"{wname}__{source}_labels.parquet"
    f_td = out_dir / f"{wname}__{source}_td.parquet"
    tmp = f_npz.with_name(f_npz.name + ".tmp.npz")   # numpy appends .npz
    np.savez_compressed(tmp, **feats)
    os.replace(tmp, f_npz)
    lab_df.to_parquet(f_lab, index=False)
    td_df.to_parquet(f_td, index=False)

    rel = lambda p: f"worlds/{chunk_id}/{p.name}"  # noqa: E731
    entry = dict(
        world=wname, source=source, chunk_id=chunk_id, split=split,
        demand_veh_h=loading, penetration=pen, p40=p40, pre_fix=pre_fix,
        t0_s=t0,
        run=str(arm_dir), policy=V.SOURCE_POLICY[source],
        cav_speed_mode=cav_speed_mode, speed_mode_source=mode_source,
        physics_sha256=sha_of(arm_dir / "run" / "physics_steps.jsonl"),
        files={rel(p): sha_of(p) for p in (f_npz, f_lab, f_td)},
        rows=dict(n_decision_times=int(len(ts)), n_label_rows=int(len(lab_df)),
                  n_td_rows=int(len(td_df)),
                  n_dropped_decision_ticks=n_dropped_ticks),
        label_sums=dict(
            total_delay_s=float(lab_df["total_delay_ref_s"].sum())
            if len(lab_df) else 0.0,
            wait_mainline_s=float(lab_df["wait_mainline_s"].sum())
            if len(lab_df) else 0.0,
            wait_ramp_s=float(lab_df["wait_ramp_s"].sum())
            if len(lab_df) else 0.0,
            ttc_lt3_veh_s=float(lab_df["ttc_lt3_veh_s"].sum())
            if len(lab_df) else 0.0,
            hard_brake_s=float(lab_df["hard_brake_s"].sum())
            if len(lab_df) else 0.0,
            collision_any=int(lab_df["collision_any"].sum())
            if len(lab_df) else 0),
        trace_stats=rt.stats,
        insertion_crosscheck=insertion_crosscheck(rt, arm_dir),
        built_utc=datetime.now(timezone.utc).isoformat(),
    )
    if source == "native":
        entry["n_pre_t0_shared_insertions"] = len(t_pre)
    atomic_write_json(out_dir / f"{wname}__{source}_done.json",
                      dict(entry, flags_backfilled=False))
    return entry


def verify_done(out_root: Path, chunk_dir: Path, wname: str,
                source: str) -> dict | None:
    """Existing done marker whose recorded hashes still match -> entry.
    done.json 'files' keys are out-relative paths."""
    done = chunk_dir / f"{wname}__{source}_done.json"
    if not done.exists():
        return None
    try:
        d = json.loads(done.read_text(encoding="utf-8"))
    except Exception:
        return None
    for fname, sha in d.get("files", {}).items():
        p = out_root / fname
        if not p.exists() or sha_of(p) != sha:
            return None
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=DEFAULT_PLAN,
                    help="WORLD_PLAN.json for THIS chunk")
    ap.add_argument("--runs", default=DEFAULT_RUNS,
                    help="runs root for THIS chunk")
    ap.add_argument("--bases-root", default=None,
                    help="root holding the BASE arm dirs when they live "
                         "outside --runs (chunk1_fix27: the verify27 "
                         "mode-27 bases); default: --runs")
    ap.add_argument("--chunk", default=DEFAULT_CHUNK,
                    help="chunk id (worlds land in worlds/<chunk>/)")
    ap.add_argument("--default-speed-mode", type=int,
                    default=DEFAULT_SPEED_MODE,
                    help="cav_speed_mode used when neither the run receipt "
                         "nor the physics rows record one (chunk 2: 27)")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--worlds", default="")
    ap.add_argument("--sources", default="base,native")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    lower_priority()
    plan_path, runs_root = Path(args.plan), Path(args.runs)
    bases_root = Path(args.bases_root) if args.bases_root else None
    out_dir = Path(args.out)
    chunk_id = str(args.chunk)
    chunk_dir = out_dir / "worlds" / chunk_id
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    want = set(args.worlds.split(",")) if args.worlds else None

    plan, worlds = V.load_plan(plan_path)
    geom = V.cached_geometry()
    worlds_dir = chunk_dir
    worlds_dir.mkdir(parents=True, exist_ok=True)
    sched_cache: dict = {}
    base_rt_cache: dict = {}

    manifest_path = out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) \
        if manifest_path.exists() else dict(
            schema=SCHEMA, zero_simulation=True,
            tau_s=list(V.TAUS), commit_window_s=V.COMMIT_S,
            decide_every_s=V.DECIDE_EVERY_S,
            base_decision_min_t_s=V.BASE_DECISION_MIN_T,
            note="V0 native-policy value dataset (round 0); base source is "
                 "the campaign incumbent arm (damping) per DESIGN PartB v2 "
                 "§4 Step 0 ('chunk-1 native branches and bases'); see "
                 "DATA_CARD_V0.md for provenance and caveats",
            chunks={}, worlds={})
    manifest.setdefault("chunks", {})
    manifest.setdefault("worlds", {})

    chunk_sec = manifest["chunks"].setdefault(chunk_id, dict(
        chunk_id=chunk_id,
        default_speed_mode=int(args.default_speed_mode),
        skipped_holdout=[], skipped_incomplete=[], world_keys=[]))
    chunk_sec.update(plan=str(plan_path), plan_sha256=sha_of(plan_path),
                     runs_root=str(runs_root))
    if bases_root is not None:
        chunk_sec["bases_root"] = str(bases_root)

    entries = manifest["worlds"]
    prefix = f"{chunk_id}/"
    # world keys of OTHER chunks must survive; this chunk's stale keys are
    # rebuilt from what we verify/build below
    for k in [k for k in entries if k.startswith(prefix)]:
        entries.pop(k, None)
    chunk_sec["world_keys"] = []

    skipped_holdout, skipped_incomplete = [], []
    n_built = 0
    for w in worlds:
        wname = w["world"]
        if w["split"] == "HOLDOUT":          # always listed, never built
            if wname not in [s["world"] for s in skipped_holdout]:
                skipped_holdout.append(dict(world=wname, split=w["split"]))
            continue
        if want and wname not in want:
            continue
        t0, runs = V.world_sources(w, runs_root, bases_root)
        for source in sources:
            arm = runs[source]
            ok, receipt, reason = V.receipt_status(arm)
            if not ok:
                skipped_incomplete.append(dict(world=wname, source=source,
                                               run=str(arm), reason=reason))
                continue
            key = f"{prefix}{wname}__{source}"
            if not args.rebuild:
                done = verify_done(out_dir, chunk_dir, wname, source)
                if done is not None:
                    # done.json carries the FULL entry (stats included), so
                    # resume never degrades the manifest; IDENTITY fields
                    # are stamped unconditionally from the plan + arm key
                    # (a stale/incorrect marker must not win -- the chunk-1
                    # backfill carried source='base' onto native entries)
                    e = dict(done)
                    e["world"] = wname
                    e["source"] = source
                    e["split"] = w["split"]
                    e["demand_veh_h"] = int(w["loading_veh_h"])
                    e["penetration"] = float(w["penetration"])
                    e["p40"] = bool(abs(float(w["penetration"]) - 0.4)
                                    < 1e-9)
                    e.setdefault("pre_fix",
                                 bool(int(done.get(
                                     "cav_speed_mode",
                                     DEFAULT_SPEED_MODE)) == 31))
                    e["t0_s"] = t0
                    e["policy"] = V.SOURCE_POLICY[source]
                    e["chunk_id"] = chunk_id
                    e["resumed"] = True
                    entries[key] = e
                    continue
            t1 = time.time()
            entry = build_source(w, source, arm, runs_root, chunk_dir,
                                 geom, sched_cache, base_rt_cache,
                                 chunk_id=chunk_id,
                                 default_speed_mode=args.default_speed_mode,
                                 receipt=receipt, bases_root=bases_root)
            entries[key] = entry
            n_built += 1
            log(f"built {key}: dt={entry['rows']['n_decision_times']} "
                f"labels={entry['rows']['n_label_rows']} "
                f"td={entry['rows']['n_td_rows']} "
                f"mode={entry['cav_speed_mode']}({entry['speed_mode_source']})"
                f" ({time.time() - t1:.1f}s + trace "
                f"{entry['trace_stats']['wall_s']}s)")
            _finalize(manifest, chunk_sec, chunk_id, entries,
                      skipped_holdout, skipped_incomplete, n_built,
                      manifest_path, sys.argv)

    _finalize(manifest, chunk_sec, chunk_id, entries, skipped_holdout,
              skipped_incomplete, n_built, manifest_path, sys.argv)
    log(f"chunk {chunk_id}: {len(chunk_sec['world_keys'])} sources complete, "
        f"{len(skipped_incomplete)} incomplete skipped, "
        f"{len(skipped_holdout)} HOLDOUT worlds listed, "
        f"{n_built} built this run; "
        f"MANIFEST total {len(entries)} sources")
    return 0


def _finalize(manifest, chunk_sec, chunk_id, entries, skipped_holdout,
              skipped_incomplete, n_built, manifest_path, argv):
    chunk_sec["world_keys"] = sorted(
        k for k in entries
        if k.startswith(f"{chunk_id}/"))
    chunk_sec["skipped_holdout"] = skipped_holdout
    chunk_sec["skipped_incomplete"] = skipped_incomplete
    chunk_sec["n_sources"] = len(chunk_sec["world_keys"])
    chunk_sec["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["build"] = dict(
        built_utc=datetime.now(timezone.utc).isoformat(),
        n_sources_built_this_run=n_built,
        n_sources_complete=len(entries),
        chunks=sorted(manifest["chunks"]),
        command=" ".join(argv))
    atomic_write_json(manifest_path, manifest)


if __name__ == "__main__":
    sys.exit(main())
