"""WM-N1 label-coverage v0.4a LABELS-ONLY builder (owner-approved
2026-10-04, runs/active/WM-N1-LABELCOV-20261004; ZERO SIMULATION).

Builds ONLY the ``_labels.parquet`` files that the frozen v0.4a root lacks,
into a FRESH root, for the (world, source) pairs that pack_v3 TRAIN
zero-row items resolve to (classes M and N of T3's LABEL_COVERAGE_PLAN;
the R class reuses the existing frozen parquets).  Labels are recomputed
from the frozen raw ``physics_steps.jsonl`` with the exact
``build_v04a.build_one`` label conventions:

* arm verification (``actions_v04.ActionTimeline.verify_arm``) before any
  trace is built; a mismatch REFUSES the source;
* branch decision grid ``[t0] + (t0+10 .. t0+570 step 10)`` capped by the
  snapshot filter (``_has_snapshot``);
* ``v0lib.window_label`` with ``skip_first_tick=False`` (every coverage
  source is branch-type, so the v0.3.2 interval convention applies
  verbatim);
* the branch ``t == t0`` row's ``n_vehicles_present`` is taken from the
  SHARED BASE trace snapshot at t0, exactly as ``build_one`` takes it from
  the t0-row features (count of snapshot records);
* per-row policy conditioning ``pi_native/pi_damping/pi_pi``: the fleet
  one-hot when the run manifest pins a fleet policy, else
  ``ActionTimeline.action_shares`` over the snapshot CAVs (the t0 row's
  CAV list comes from the BASE snapshot, matching the t0-row features);
  wall arms have no fleet manifest and the wall token has no share slot,
  so their three share columns are 0.0 (documented: the planner-known
  continuation summary does not cover the wall token).

Differences from ``build_v04a.build_one`` (by design, labels unchanged):
the feature ``.npz`` sidecars and the ``_td.parquet`` are NOT built; no
feature tensor is computed.  TRAIN-only: every spec whose plan split is
not TRAIN is refused outright; the targets come from the pack_v3 manifest
+ INDEX metadata (zero-row TRAIN items, owner-(b) arms excluded), so no
CAL/HOLDOUT/chunk-5/J4B/INC1 world can reach a raw run.

The frozen v0.4a root is never written.  The output root must be FRESH
(resume within the same root via ``<world>__<source>_done.json`` markers,
whose file SHAs are re-verified).  A root ``MANIFEST.json`` +
``SHA256SUMS`` (labels + markers only) is assembled at the end.

Usage:
  python build_v04a_cov.py --pack D:/rd-staging/wmn1_feat/pack_v3 \
      --out E:/research-dense/output/wd01_partB_v0data_v04a_cov_20261004 \
      --workers 10
"""
from __future__ import annotations

import argparse
import json
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
import v04a_sources as S  # noqa: E402
import v0lib as V  # noqa: E402
import wdpaths  # noqa: E402

SCHEMA = "wd01_partB_v0data_v04a_cov/1"
LABEL_COLS = B.LABEL_COLS + ["pi_native", "pi_damping", "pi_pi", "role",
                             "aux", "arm_kind"]
OWNER_EXCLUDED_ARMS = ("base", "native_base")
T0W = time.time()


def log(msg):
    print(f"[{time.time() - T0W:7.1f}s] {msg}", flush=True)


def sha256_file(path, block=1 << 20):
    return V.sha256_file(path)


# ------------------------------------------------------ target selection --

def cov_targets(pack_dir: Path, v04a_root: Path) -> list[dict]:
    """The (batch, chunk, world, arm, policy, source) build targets.

    Derived from pack manifest + INDEX METADATA ONLY: every TRAIN item
    whose packed ``outcome_rows`` is empty and whose arm is not owner-(b)
    excluded, resolved through ``features.outcomes.label_source``; the
    target survives only when the resolved label parquet is absent from
    the frozen v0.4a root (R class is not rebuilt).
    """
    sys.path.insert(0, str(HERE.parent / "wm_n1"))
    from features.outcomes import labels_path, label_source
    man = json.loads((pack_dir / "manifest.json").read_text(
        encoding="utf-8"))
    items = [it for it in man.get("items", [])
             if str(it.get("split", "")).upper() == "TRAIN"]
    out: dict = {}
    for it in items:
        arm = str(it["arm"])
        if arm in OWNER_EXCLUDED_ARMS:
            continue
        if any(int(p) >= 0 for p in it.get("outcome_rows", [])):
            continue                      # the pack already holds its rows
        batch = str(it.get("batch") or it["chunk"])
        src = label_source(batch, str(it["chunk"]), arm, it.get("policy"))
        if src is None:
            raise SystemExit(
                f"TRAIN zero-row item {batch}/{it['world']}__{arm} does "
                "not resolve to a v0.4a source; refusing (do not guess)")
        if labels_path(str(it["chunk"]), str(it["world"]), src,
                       v04a_root).is_file():
            continue                      # R class: frozen parquet exists
        key = (str(it["chunk"]), str(it["world"]), src)
        rec = out.setdefault(key, dict(
            batch=batch, chunk=key[0], world=key[1], source=key[2], arm=arm,
            policy=it.get("policy"), shard_dirs={}))
        rec["shard_dirs"][key[2]] = None                # filled below
    # attach the INDEX shard_dir per target (metadata only; the corpus
    # featurizer recorded the authoritative raw run there)
    idx = json.loads((pack_dir / "INDEX.json").read_text(encoding="utf-8"))
    by_key = {(r.get("batch") or r["chunk"], r["world"], r["arm"]): r
              for r in idx.get("runs", [])}
    for rec in out.values():
        # the pack shard is keyed by the pack ARM (e.g. ``native``), while
        # the resolved label source may differ (``swnative``)
        r = by_key.get((rec["batch"], rec["world"], rec["arm"]))
        rec["shard_dirs"][rec["source"]] = None if r is None \
            else str(r.get("shard_dir") or "")
    return sorted(out.values(), key=lambda d: (d["chunk"], d["world"],
                                               d["source"]))


# ------------------------------------------------------------ one source --

def _has_snapshot(rt, t: float) -> bool:
    k = int(round((t - rt.t_origin) / V.DT))
    return 0 <= k <= rt.n_ticks and rt.tick_last[k] > rt.tick_first[k]


def decision_times(rt, t0: float) -> np.ndarray:
    """Branch grid: t0 first (shared base state), then t0+10 .. t0+570."""
    ts = V.decision_grid(t0, rt.data_end, "native")
    return np.concatenate([[t0], ts])


def resolve_run_dir(spec: S.SrcSpec, shard_dir: str | None):
    """(arm_dir, ok, receipt, reason, raw_source) for one target.

    Primary: the registry arm dir with a complete RUN_RECEIPT.json.  When
    the run receipt is absent (the chunk1 p40 worlds superseded by fix27:
    their branch runs live only in the D:/rd-staging ``raw_chunk1_p40``
    staging tree, without receipts), the run directory recorded by the
    pack shard ``meta.json:input_physics`` is used instead and VOUCHED by
    that shard: the physics file must exist and the meta identity must
    equal the spec identity.  ``raw_source`` records which path was taken.
    """
    arm = Path(wdpaths.remap(spec.arm_dir))
    ok, receipt, reason = V.receipt_status(arm)
    if ok:
        return arm, ok, receipt, reason, "run_receipt"
    if shard_dir:
        mp = Path(shard_dir) / "meta.json"
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            ip = meta.get("input_physics")
        except Exception:                                 # noqa: BLE001
            meta, ip = None, None
        if ip:
            p = Path(ip)
            alt = p.parent.parent          # <arm_root>/run/physics_steps.jsonl
            run_arm = alt.name.rsplit("__", 1)[-1]
            if p.is_file() and alt.is_dir() \
                    and str(meta.get("world")) == str(spec.world) \
                    and str(meta.get("arm")) == run_arm \
                    and str(meta.get("chunk")) == str(spec.chunk) \
                    and abs(float(meta.get("t0_s", np.nan))
                            - float(spec.t0_s)) < 1e-6:
                return alt, True, None, "", "shard_input_physics"
            reason = (f"run receipt missing and shard-vouched fallback "
                      f"unusable (input_physics={ip!r}, meta identity "
                      f"world={meta.get('world') if meta else None!r} "
                      f"arm={meta.get('arm') if meta else None!r} != "
                      f"run dir arm {run_arm!r})")
        else:
            reason = f"run receipt missing and shard meta unreadable: {mp}"
    else:
        reason = f"run receipt missing and no pack shard recorded: {reason}"
    return arm, False, None, reason, "run_receipt"


def label_rows_for_source(spec: S.SrcSpec, rt_base, sched_cache, *,
                          arm_dir: Path | None = None, receipt=None,
                          raw_source: str = "run_receipt") -> dict:
    """The label loop of ``build_v04a.build_one`` for one branch source,
    minus features/TD.  Returns its manifest entry (with the label frame)."""
    world, source = spec.world, spec.source
    t0 = float(spec.t0_s)
    arm = Path(wdpaths.remap(spec.arm_dir)) if arm_dir is None \
        else Path(arm_dir)
    base_dir = Path(wdpaths.remap(spec.base_dir))
    if receipt is None and raw_source != "shard_input_physics":
        return dict(key=spec.key(), refused=True,
                    reason="incomplete run: no receipt and no shard voucher")
    tl = A.ActionTimeline.from_run_dir(arm, schedule_authoritative=True)
    try:
        arm_ev = tl.verify_arm(spec.verify_kind)
    except A.ArmPolicyMismatch as e:
        return dict(key=spec.key(), refused=True,
                    reason=f"ArmPolicyMismatch: {e}")

    t_pre = V.build_t_ins_pre(rt_base, t0)
    rt = V.build_run_trace(world, "native", arm, t0, t_ins_pre=t_pre,
                           sched_cache=sched_cache, base_arm_dir=base_dir)
    rt.demand_loading = int(spec.demand_veh_h)

    ts_all = decision_times(rt, t0)
    keep_t = [float(t) for t in ts_all
              if _has_snapshot(rt_base if abs(float(t) - t0) < 1e-9 else rt,
                               float(t))]
    n_dropped = int(len(ts_all) - len(keep_t))
    ts = np.asarray(keep_t, np.float64)
    tau_avail = rt.data_end - ts
    speed_mode, mode_source = B.resolve_speed_mode(
        receipt, rt.stats, spec.default_speed_mode)
    pre_fix = bool(speed_mode == 31)

    fleet_share = None
    if tl.fleet_policy is not None:
        fleet_share = np.zeros(3)         # wall fleet token: no share slot
        if tl.fleet_policy in A.SHARE_ORDER:
            fleet_share[A.SHARE_ORDER.index(tl.fleet_policy)] = 1.0
    wall_arm = bool(spec.verify_kind == "wall")
    lab_rows = []
    aux = bool(spec.role == "aux")
    pol_kw = dict(role=spec.role, aux=aux, arm_kind=spec.verify_kind)
    for i, t in enumerate(ts):
        t = float(t)
        ta = float(tau_avail[i])
        at_t0 = abs(t - t0) < 1e-9
        src_rt = rt_base if at_t0 else rt
        sl = src_rt.snapshot_at(t)
        cav_v = [str(src_rt.vid_table[k]) for k in
                 sl["vid"][sl["is_cav"] == 1]] if sl else []
        per_t = {}

        def label(tau):
            if tau in per_t:
                return per_t[tau]
            row = V.window_label(rt, t, tau, skip_first_tick=False)
            if at_t0:
                snap = rt_base.snapshot_at(t0)
                row["n_vehicles_present"] = int(0 if snap is None
                                                else len(snap["vid"]))
            per_t[tau] = row
            return row

        for tau in V.TAUS:
            if tau > ta + 1e-9:
                continue
            if wall_arm:
                sh = np.zeros(3)          # documented: no share slot for wall
            else:
                sh = fleet_share if fleet_share is not None else \
                    tl.action_shares(cav_v, t, tau)
            lab_rows.append(dict(
                world=world, source=source, chunk_id=spec.chunk,
                split=spec.split, demand_veh_h=int(spec.demand_veh_h),
                penetration=float(spec.penetration),
                p40=bool(abs(float(spec.penetration) - 0.4) < 1e-9),
                pre_fix=pre_fix, t0_s=t0, policy=spec.policy,
                t=t, tau=float(tau), tau_available=ta,
                cav_speed_mode=speed_mode, **label(tau),
                pi_native=float(sh[0]), pi_damping=float(sh[1]),
                pi_pi=float(sh[2]), **pol_kw))
    lab_df = pd.DataFrame(lab_rows, columns=LABEL_COLS)
    return dict(
        key=spec.key(), refused=False,
        world=world, source=source, chunk_id=spec.chunk, split=spec.split,
        role=spec.role, aux=aux, demand_veh_h=int(spec.demand_veh_h),
        penetration=float(spec.penetration), t0_s=t0,
        run=str(arm), base_run=str(base_dir),
        policy=spec.policy, arm_kind=spec.verify_kind, source_type="branch",
        raw_source=raw_source,
        cav_speed_mode=speed_mode, speed_mode_source=mode_source,
        physics_sha256=sha256_file(arm / "run" / "physics_steps.jsonl"),
        rows=dict(n_decision_times=int(len(ts)),
                  n_label_rows=int(len(lab_df)),
                  n_dropped_decision_ticks=n_dropped),
        arm_verification=arm_ev,
        label_sums=dict(
            total_delay_s=float(lab_df["total_delay_ref_s"].sum())
            if len(lab_df) else 0.0,
            collision_any=int(lab_df["collision_any"].sum())
            if len(lab_df) else 0),
        trace_stats=rt.stats, note=spec.note,
        labels_frame=lab_df,
        built_utc=datetime.now(timezone.utc).isoformat())


def verify_done(root: Path, chunk: str, world: str, source: str):
    """Resume: a done marker whose recorded label-file SHA still verifies."""
    p = root / "worlds" / chunk / f"{world}__{source}_done.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return None
    rel = f"worlds/{chunk}/{world}__{source}_labels.parquet"
    sha = (d.get("files") or {}).get(rel)
    f = root / rel
    if not isinstance(sha, str) or not f.is_file() \
            or sha256_file(f) != sha:
        return None
    return d


# ------------------------------------------------------------- per world --

def build_world(task) -> list:
    root, chunk, world, want, shard_dirs, rebuild = task
    root = Path(root)
    S.OpenAudit.start()
    S.OpenAudit.paths = []
    B.lower_priority()
    res = _build_world(root, chunk, world, want, shard_dirs, rebuild)
    opened = sorted({p.replace("\\", "/") for p in S.OpenAudit.paths
                     if "/runs/" in p.replace("\\", "/")
                     or "/output/campaign" in p.replace("\\", "/")})
    S.OpenAudit.paths = []
    res.append(dict(_opened=opened))
    return res


def _build_world(root: Path, chunk: str, world: str, want: set,
                 shard_dirs: dict, rebuild: bool) -> list:
    sched_cache: dict = {}
    out: list = []
    todo = []
    for src in sorted(want):
        d = None if rebuild else verify_done(root, chunk, world, src)
        if d is not None:
            d["resumed"] = True
            out.append(d)
        else:
            todo.append(src)
    if not todo:
        return out
    specs = [sp for sp in S.enumerate_specs(chunks=(chunk,),
                                            worlds={world},
                                            include_superseded=True)
             if sp.split == "TRAIN" and sp.world == world
             and sp.source in want]
    by_src = {sp.source: sp for sp in specs}
    missing = [s for s in todo if s not in by_src]
    if missing:
        raise SystemExit(
            f"no TRAIN spec for {chunk}/{world} sources {missing}; "
            "refusing (do not guess)")
    first = by_src[todo[0]]
    base_dir = Path(wdpaths.remap(first.base_dir))
    ok, _r, reason = V.receipt_status(base_dir)
    if not ok:
        return out + [dict(key=f"{chunk}/{world}__{s}", refused=True,
                            reason=f"base run incomplete: {reason}")
                      for s in todo]
    rt_base = V.build_run_trace(world, "base", base_dir, float(first.t0_s),
                                t_ins_pre={}, sched_cache=sched_cache)
    rt_base.demand_loading = int(first.demand_veh_h)
    for src in todo:
        sp = by_src[src]
        arm_dir, _ok, receipt, reason, raw_source = resolve_run_dir(
            sp, shard_dirs.get(src))
        t1 = time.time()
        try:
            if not _ok:
                raise RuntimeError(reason)
            e = label_rows_for_source(sp, rt_base, sched_cache,
                                      arm_dir=arm_dir, receipt=receipt,
                                      raw_source=raw_source)
        except Exception as ex:               # noqa: BLE001
            e = dict(key=sp.key(), refused=True,
                     reason=f"build error {type(ex).__name__}: {ex}")
        if not e.get("refused"):
            frame = e.pop("labels_frame")
            out_dir = root / "worlds" / sp.chunk
            out_dir.mkdir(parents=True, exist_ok=True)
            f_lab = out_dir / f"{sp.world}__{sp.source}_labels.parquet"
            tmp = Path(str(f_lab) + ".tmp.parquet")
            frame.to_parquet(tmp, index=False)
            tmp.replace(f_lab)
            e["files"] = {f"worlds/{sp.chunk}/{f_lab.name}":
                          sha256_file(f_lab)}
            B.atomic_write_json(
                out_dir / f"{sp.world}__{sp.source}_done.json", e)
            log(f"built {sp.key()}: dt={e['rows']['n_decision_times']} "
                f"labels={e['rows']['n_label_rows']} "
                f"({time.time() - t1:.1f}s)")
        else:
            log(f"REFUSED {e['key']}: {e['reason']}")
        out.append(e)
    return out


# ------------------------------------------------------------- finalize --

def write_sha256sums(root: Path) -> int:
    rels = ["MANIFEST.json"] + [
        p.relative_to(root).as_posix()
        for p in sorted((root / "worlds").rglob("*")) if p.is_file()]
    lines = [f"{sha256_file(root / rel)}  {rel}" for rel in rels]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
    return len(lines)


def assemble_manifest(root: Path, results: list, targets: list,
                      holdout: set, argv) -> dict:
    entries, refused = {}, []
    for e in results:
        if "_opened" in e:
            continue
        if e.get("refused"):
            refused.append(dict(key=e["key"], reason=e["reason"]))
            continue
        key = f"{e['chunk_id']}/{e['world']}__{e['source']}"
        entries[key] = e
    by_chunk: dict = {}
    for t in targets:
        by_chunk.setdefault(t["chunk"], set()).add((t["world"], t["source"]))
    chunks = {}
    for c in sorted(by_chunk):
        keys = sorted(k for k in entries if k.startswith(c + "/"))
        chunks[c] = dict(
            chunk_id=c,
            runs_root=str(S.RUNS.get(c, S.WALLS_ROOT)),
            target_world_sources=len(by_chunk[c]),
            skipped_incomplete=[r for r in refused
                                if r["key"].startswith(c + "/")],
            world_keys=keys, n_sources=len(keys))
    return dict(
        schema=SCHEMA, zero_simulation=True, labels_only=True,
        note=("WM-N1 label-coverage labels-only v0.4a root: only the "
              "label parquets the frozen v0.4a root lacks; exact "
              "build_v04a.build_one label conventions; TRAIN targets "
              "from pack_v3 metadata; frozen root never written"),
        tau_s=list(V.TAUS), commit_window_s=V.COMMIT_S,
        decide_every_s=V.DECIDE_EVERY_S,
        owner_excluded_arms=list(OWNER_EXCLUDED_ARMS),
        holdout_worlds_checked=sorted(holdout),
        holdout_accessed=False, train_only=True,
        chunks=chunks, worlds=entries,
        build=dict(built_utc=datetime.now(timezone.utc).isoformat(),
                   builder="partB_v0data/build_v04a_cov.py",
                   n_sources=len(entries), n_refused=len(refused),
                   command=" ".join(argv)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", default="D:/rd-staging/wmn1_feat/pack_v3")
    ap.add_argument("--v04a-root", default=None)
    ap.add_argument("--out",
                    default="E:/research-dense/output/"
                            "wd01_partB_v0data_v04a_cov_20261004")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit-worlds", default="",
                    help="comma-separated world filter (smoke tests)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(HERE.parent / "wm_n1"))
    from features.outcomes import V04A_ROOT
    pack_dir = Path(args.pack)
    v04a_root = Path(args.v04a_root or V04A_ROOT)
    root = Path(args.out)
    if root.exists():
        raise SystemExit(f"refusing to touch existing output root {root}; "
                         "pick a fresh one (never overwrite)")
    root.mkdir(parents=True)
    (root / "worlds").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    targets = cov_targets(pack_dir, v04a_root)
    limit = {w for w in args.limit_worlds.split(",") if w}
    if limit:
        targets = [t for t in targets if t["world"] in limit]
    log(f"cov targets: {len(targets)} (world, source) pairs; root={root}")
    by_world: dict = {}
    for t in targets:
        v = by_world.setdefault((t["chunk"], t["world"]),
                                dict(sources=set(), shard_dirs={}))
        v["sources"].add(t["source"])
        v["shard_dirs"].update(t["shard_dirs"])
    tasks = [(str(root), c, w, v["sources"], v["shard_dirs"], args.rebuild)
             for (c, w), v in sorted(by_world.items())
             if not limit or w in limit]

    S.OpenAudit.start()
    holdout = {w for c in ("chunk1", "chunk1_fix27", "chunk2", "chunk3")
               for w in S.chunk_worlds(c)[1]}
    holdout |= set(S.chunk_worlds("chunk4")[1])
    holdout |= set(S.walls_plan_worlds()[1])
    hold_all = frozenset(holdout)
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
    opened = sorted({p for e in results for p in e.get("_opened", ())})
    bad = [p for p in opened if any(w in p for w in hold_all)]
    assert not bad, f"HOLDOUT path opened by a worker: {bad[:5]}"
    assert not [p for p in opened if "dispatcher_rows" in p
                or "insertions.jsonl" in p], "forbidden log opened"
    (root / "results" / "opened_run_files.json").write_text(
        json.dumps(dict(n=len(opened),
                        holdout_worlds_checked=len(hold_all),
                        paths=opened), indent=0), encoding="utf-8")
    S.OpenAudit.stop()
    man = assemble_manifest(root, results, targets, holdout, sys.argv[1:])
    B.atomic_write_json(root / "MANIFEST.json", man)
    n_sums = write_sha256sums(root)
    log(f"MANIFEST: {len(man['worlds'])} sources, "
        f"{man['build']['n_refused']} refused; SHA256SUMS {n_sums} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
