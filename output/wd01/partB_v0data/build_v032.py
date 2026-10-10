"""WD-01 Part B v0.3.2: corrected derived dataset root from v0.3.

    v0.3.2 = v0.3 (labels/TD/features copied or hard-linked, hash-verified,
             MANIFEST sha256 recorded as the parent)
           + v0.3.2 fixes (never modifies v0.2/v0.3/v0.3.1):
    #1  deployable context col 1 zeroed (ctx[0]=t/1000 + ctx[1]=(t-t0)/600
        made the campaign branch time t0 exactly recoverable; t0 is a
        campaign design event, not a deployable input);
    #2  chunk1_fix27/w8473m2700p40t540__native REBUILT from the raw runs
        (the removed insertions.jsonl fallback had returned an EMPTY
        shared pre-t0 insertion history and inflated its labels ~8.8x in
        tau30 mean wait); pre-t0 history now always comes from the base
        run's physics_steps departed_ids (bases_root aware);
    #3  per-source policy relabels: the campaign '__native'/'__pi' arms of
        chunks 1-3 + chunk1_fix27 are random per-CAV switching schedules
        (actions from switch_schedule.json / suffix_commands.jsonl -- NEVER
        dispatcher_rows.jsonl), so every entry/label row is relabelled
        switching_mix provenance (canonical policy map POLICY_OF_SOURCE);
    #5  BOTH deployable sensing variants ship as sidecars:
        <world>__<source>_feat_dep_det.npz    (detector: all-vehicle
             entrance counts + CAV counts; labelled SENSITIVITY) and
        <world>__<source>_feat_dep_strict.npz (strict: CAV self-reports
             only, entrance queues zeroed; HEADLINE).

The v0.3 root is NEVER written.  HOLDOUT is never opened (only MANIFEST
sources are touched; a world-name guard asserts against the parent
MANIFEST's skipped_holdout lists).  ZERO SIMULATION: run logs are read as
data (one light streaming pass per arm over physics_steps.jsonl; the
w8473 rebuild uses the same RunTrace path as the original v0.3 build).

Per-source verification (hard failure): the recomputed deployable context
must reproduce the stored oracle context on every RSU-observable column
(detector: t, loading, q_main, q_ramp = cols 0,3,6,7; strict: cols 0,3 --
col 1 is dropped since fix #1 zeroes it) at 1e-6.  For w8473__native the
stored context is the REBUILT (physics-correct) one.

Stages (idempotent):
  link     hard-link (fallback: verified copy) every parent file listed in
           the v0.3 MANIFEST + the per-world done markers, sha256-verified
           before AND after, EXCEPT the rebuild source's three files;
  relabel  rewrite every label parquet's policy column to the v0.3.2
           provenance prose (atomic tmp+replace), re-hashed into the
           MANIFEST;
  rebuild  rebuild w8473m2700p40t540__native (fix #2) and hard-check its
           tau30 mean-wait ratio vs its base into [0.3, 1.0] (the other
           fix27 natives sit in that range; the inflated build was 8.8x);
  dep      build + verify BOTH sidecar variants per source (multiprocess
           over worlds);
  finalize v0.3.2 MANIFEST (parent sha, fixes applied, per-source policy
           labels, feature_sets), SHA256SUMS over MANIFEST + worlds/,
           README_DATA_NEVER_DELETE.md, results/build_v032_last.json.

Usage:
  python build_v032.py                      # default v0.3 -> v0.3.2
  python build_v032.py --workers 4
  python build_v032.py --v03-root <tiny> --out <tiny32> --parent-sha ""
  python build_v032.py --no-rebuild         # synthetic roots w/o w8473
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import v0lib as V  # noqa: E402  (schedule loader + DT; read-only reuse)
from deployable_features import (STRICT_DEMAND_NAMES,  # noqa: E402
                                 VARIANTS, SourceObs)
import actions_v04 as A  # noqa: E402  (v0.4 per-CAV action reconstruction)

sys.path.insert(0, str(HERE.parent))         # output/wd01 (wdpaths)
import wdpaths  # noqa: E402
import build_v0data as B  # noqa: E402  (lower_priority, atomic_write_json,
#                         build_source for the #2 rebuild)

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
DEFAULT_V03 = str(RD / "output" / "wd01_partB_v0data_v03")
DEFAULT_OUT = str(RD / "output" / "wd01_partB_v0data_v032")
PARENT_MANIFEST_SHA = ("5cf9fa479f1ba19a209c1ed2c486663ed033866a5545eea3f"
                       "468de6c457f37a1")
REBUILD_KEY = "chunk1_fix27/w8473m2700p40t540__native"
T0W = time.time()

# v0.3.2 fix #3: canonical per-source policy labels.  The campaign
# '__native'/'__pi' arms of chunks 1-3 + chunk1_fix27 are random per-CAV
# switching schedules, NOT fleet native/PI.  Per-CAV actions are
# attributed only from switch_schedule.json / suffix_commands.jsonl --
# dispatcher_rows.jsonl is never read ( asserted by tests ).
POLICY_OF_SOURCE = {"base": "damping_fleet", "native": "switching_mix",
                    "pi": "switching_mix"}

FIXES = [
    "#1 deployable context col 1 zeroed (t0 not recoverable from the "
    "deployable blocks); oracle files unchanged",
    "#2 insertions.jsonl fallback removed everywhere; pre-t0 insertion "
    "history always from the base run's physics departed_ids "
    "(bases_root aware); chunk1_fix27/w8473m2700p40t540__native "
    "labels/td/feat rebuilt from the raw runs",
    "#3 '__native'/'__pi' arms relabelled switching_mix (random per-CAV "
    "switching schedules; actions only from switch_schedule.json / "
    "suffix_commands.jsonl, never dispatcher_rows.jsonl); label parquet "
    "policy column + MANIFEST policy prose updated",
    "#5 both deployable sensing variants built: _feat_dep_det.npz "
    "(all-vehicle entrance counts; labelled SENSITIVITY) and "
    "_feat_dep_strict.npz (CAV self-reports only; HEADLINE; no "
    "all-vehicle entrance counts, no penetration-revealing ratio)",
]


def log(msg):
    print(f"[{time.time() - T0W:7.1f}s] {msg}", flush=True)


def sha256_file(path, block=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


_sha = sha256_file


# ------------------------------------------------------- light pass ----

def light_pass(physics_path: Path, t_origin: float) -> tuple[dict, set]:
    """(vid -> first realised insertion time, designated-CAV vid set) from
    one streaming pass over physics_steps.jsonl -- EXACTLY the fields
    v0lib.build_run_trace extracts for t_ins/cav_ids (same pre-origin and
    off-grid row skips, first insertion per vid), and nothing else."""
    t_ins: dict = {}
    cav_ids: set = set()
    with open(physics_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t = float(r["time_s"])
            if t <= t_origin + 1e-9:
                continue                     # native branch grid starts > t0
            k = (t - t_origin) / V.DT
            if abs(round(k) * V.DT - (t - t_origin)) > 1e-6:
                continue                     # off-grid row: trace skips it
            for vid_ in r["departed_ids"]:
                if vid_ not in t_ins:
                    t_ins[vid_] = t
            if r["cav_ids"]:
                cav_ids.update(r["cav_ids"])
    return t_ins, cav_ids


# ------------------------------------------------- per-source sidecar --

def build_sidecars(out_root: Path, entry: dict, sched, ins_vids, ins_time,
                   variants=VARIANTS) -> list[tuple[str, str, int]]:
    """Write BOTH sensing-variant sidecars for one MANIFEST entry and
    verify them against the stored (local) oracle context.

    detector: RSU-observable oracle ctx cols (0, 3, 6, 7) at 1e-6; col 1
    is dropped because fix #1 zeroes it in deployable blocks.
    strict: cols (0, 3) plus the CAV count columns must equal the
    detector demand block's CAV columns; entrance-queue cols are 0 by
    construction.  Returns [(rel, sha, n_t), ...] in VARIANTS order."""
    world, source = entry["world"], entry["source"]
    rel_in = f"worlds/{entry['chunk_id']}/{world}__{source}_feat.npz"
    z = np.load(Path(wdpaths.remap(out_root / rel_in)),
                allow_pickle=False)
    ts = z["t"]
    obs = SourceObs(sched["vids"], sched["t"], sched["ent"], sched["cav"],
                    ins_vids, ins_time)
    t0_s = float(entry["t0_s"])
    loading = float(entry["demand_veh_h"])
    blocks = {v: obs.blocks(ts, t0_s, loading, variant=v)
              for v in variants}
    dem_det, ctx_det = blocks["detector"]
    dem_str, ctx_str = blocks["strict"]

    # hard verification against the stored oracle context: the light pass
    # must reproduce every RSU-observable column exactly
    orc = z["context"]

    def _mismatch(ctx, cols):
        return [c for c in cols
                if not np.allclose(ctx[:, c], orc[:, c], rtol=0.0,
                                   atol=1e-6)]

    bad = _mismatch(ctx_det, (0, 3, 6, 7))
    if bad:
        c = bad[0]
        i = int(np.flatnonzero(~np.isclose(ctx_det[:, c], orc[:, c],
                                           rtol=0.0, atol=1e-6))[0])
        raise SystemExit(
            f"VERIFICATION FAIL {world}__{source}: detector context "
            f"col {c} differs from the stored oracle context at row {i} "
            f"({ctx_det[i, c]} vs {orc[i, c]}) -- the light insertion "
            f"pass disagrees with the stored build")
    bad = _mismatch(ctx_str, (0, 3))
    if bad:
        raise SystemExit(
            f"VERIFICATION FAIL {world}__{source}: strict context cols "
            f"{bad} differ from the stored oracle context")
    # strict/detector internal consistency + fix invariants
    if not np.array_equal(dem_str, dem_det[:, 6:]):
        raise SystemExit(f"VERIFICATION FAIL {world}__{source}: strict "
                         f"demand block != detector CAV columns")
    if (ctx_det[:, 1] != 0.0).any() or (ctx_str[:, 1] != 0.0).any():
        raise SystemExit(f"VERIFICATION FAIL {world}__{source}: deployable "
                         f"context col 1 must be identically 0 (fix #1: "
                         f"t0 must not be recoverable)")
    if (ctx_str[:, 6] != 0.0).any() or (ctx_str[:, 7] != 0.0).any():
        raise SystemExit(f"VERIFICATION FAIL {world}__{source}: strict "
                         f"context entrance-queue cols must be 0")
    for dem in (dem_det, dem_str):
        if not np.isfinite(dem).all() or (dem < 0).any():
            raise SystemExit(f"VERIFICATION FAIL {world}__{source}: "
                             f"deployable demand block has non-finite/"
                             f"negative entries")
    # no penetration-revealing ratio anywhere: no strict column may be a
    # function of an all-vehicle count (they are CAV counts / 0-1 flags)
    assert dem_str.shape[1] == len(STRICT_DEMAND_NAMES)

    out = []
    tag = {"detector": "_feat_dep_det.npz", "strict": "_feat_dep_strict.npz"}
    for v in variants:
        dem, ctx = blocks[v]
        payload = {k: z[k] for k in z.files}
        payload["demand"] = dem
        payload["context"] = ctx
        payload["feature_set"] = np.str_(f"deployable_{v}")
        payload["sensing_variant"] = np.str_(v)
        payload["built_utc"] = np.str_(
            datetime.now(timezone.utc).isoformat())
        rel_out = (f"worlds/{entry['chunk_id']}/{world}__{source}"
                   f"{tag[v]}")
        dst = out_root / rel_out
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(dst) + ".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(dst)                      # .npz suffix added by numpy
        out.append((rel_out, _sha(dst), int(len(ts))))
    z.close()
    return out


def build_action_sidecar(out_root: Path, entry: dict, arm: Path) -> tuple:
    """--with-actions: per-source v0.4 action sidecar
    ``<world>__<source>_feat_act.npz`` -- the parent oracle payload PLUS
    per-decision-time per-CAV action ids / unknown mask aligned with the
    cav_mask token order (present CAVs sorted by vid, exactly the
    featurize_v0 ordering).  Actions come only from
    switch_schedule.json / suffix_commands.jsonl / telemetry.jsonl
    owners and the fleet manifest policy -- dispatcher_rows.jsonl is
    never read.  Fleet arms are policy-verified (loud)."""
    world, source = entry["world"], entry["source"]
    tl = A.ActionTimeline.from_run_dir(arm)
    fleet_check = None
    if tl.fleet_policy is not None:
        fleet_check = tl.verify_fleet_policy()
    elif tl.n_command_rows == 0 and tl.n_schedule_events == 0:
        raise SystemExit(
            f"ACTION SIDECAR FAIL {world}__{source}: no action evidence "
            f"in {arm} (no switch_schedule.json, no command rows, no "
            "fleet manifest) -- everything would be 'unknown'")
    rel_in = f"worlds/{entry['chunk_id']}/{world}__{source}_feat.npz"
    z = np.load(Path(wdpaths.remap(out_root / rel_in)),
                allow_pickle=False)
    ts = z["t"]
    raw_off = z["raw_off"]
    raw_vid, raw_is_cav = np.asarray(z["raw_vid"]), np.asarray(z["raw_is_cav"])
    vid_table = np.asarray(z["vid_table"])
    n_t, max_cavs = int(z["cav_mask"].shape[0]), int(z["cav_mask"].shape[1])
    ids = np.zeros((n_t, max_cavs), np.int8)
    unk = np.ones((n_t, max_cavs), np.float32)
    for i in range(n_t):
        sl = slice(int(raw_off[i]), int(raw_off[i + 1]))
        idx = raw_vid[sl][raw_is_cav[sl] == 1]
        idx = idx[np.argsort(idx, kind="stable")]   # featurizer token order
        vids = [str(vid_table[k]) for k in idx]
        if vids:
            ids[i, :len(vids)], unk[i, :len(vids)] = \
                tl.action_ids(vids, float(ts[i]))
    payload = {k: z[k] for k in z.files}
    payload["action_ids"] = ids
    payload["action_unknown"] = unk
    payload["action_inputs"] = np.str_("v04: switch_schedule.json + "
                                       "suffix_commands.jsonl/telemetry"
                                       ".jsonl owners + fleet manifest "
                                       "policy; never dispatcher_rows")
    payload["action_evidence"] = np.str_(json.dumps(dict(
        sources=tl.sources, fleet_policy=tl.fleet_policy,
        owner_counts=tl.command_owner_counts(),
        fleet_check=fleet_check)))
    z.close()
    rel_out = (f"worlds/{entry['chunk_id']}/{world}__{source}"
               f"_feat_act.npz")
    dst = out_root / rel_out
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dst) + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(dst)
    return rel_out, _sha(dst)


def build_world(task) -> list:
    """Both sidecar variants for every source of one world: light passes
    + sidecars (+ v0.4 action sidecars when the task carries
    with_actions).  Runs in a worker process; returns
    [(rel, sha, n_t), ...] and the per-source canonical policy labels."""
    out_root, chunk_runs, chunk_bases, entries, holdout, with_actions = task
    for key, entry in entries.items():
        assert entry["world"] not in holdout, \
            f"HOLDOUT world reached the builder: {key}"
    out_root = Path(out_root)
    world = next(iter(entries.values()))["world"]

    def arm_of(entry: dict) -> Path:
        run = entry.get("run")
        if run:
            return Path(wdpaths.remap(run))
        return Path(wdpaths.remap(chunk_runs)) / \
            f"{entry['world']}__{entry['source']}"

    # the base arm (schedule + CAV designation + shared insertion history)
    # exists in the runs tree even when the MANIFEST lacks the base entry
    base_entry = next((e for e in entries.values()
                       if e["source"] == "base"), None)
    if base_entry is not None:
        base_arm = arm_of(base_entry)
    elif chunk_bases:
        base_arm = Path(wdpaths.remap(chunk_bases)) / f"{world}__base"
    else:
        base_arm = Path(wdpaths.remap(chunk_runs)) / f"{world}__base"
    t0 = float(next(iter(entries.values()))["t0_s"])

    # base pass (t from 0): CAV designation set + shared insertion history
    base_t_ins, base_cav = light_pass(
        base_arm / "run" / "physics_steps.jsonl", 0.0)
    sched_raw = V.load_schedule(base_arm / "run")
    vids = sorted(sched_raw)
    sched = dict(
        vids=vids,
        t=np.array([sched_raw[x][0] for x in vids], np.float64),
        ent=np.array([sched_raw[x][1] for x in vids], np.int64),
        cav=np.array([1 if x in base_cav else 0 for x in vids], np.int8))

    out, policies = [], []
    for entry in sorted(entries.values(), key=lambda e: e["source"]):
        source = entry["source"]
        if source == "base":
            ins = dict(base_t_ins)
        else:
            src_t_ins, _ = light_pass(
                arm_of(entry) / "run" / "physics_steps.jsonl", t0)
            ins = dict(src_t_ins)
            for vid, ti in base_t_ins.items():   # shared pre-t0 history
                if ti <= t0 + 1e-9:
                    ins.setdefault(vid, ti)
        pairs = sorted(ins.items())
        recs = build_sidecars(
            out_root, entry, sched,
            [v for v, _ in pairs], [t for _, t in pairs])
        out.extend(recs)
        if with_actions:
            rel, sha = build_action_sidecar(out_root, entry,
                                            arm_of(entry))
            out.append((rel, sha, 0))
        policies.append((f"{entry['chunk_id']}/{world}__{source}",
                         POLICY_OF_SOURCE.get(source, source)))
    return out, policies


# ------------------------------------------------------------ stages --

def link_parent_files(v03_root: Path, out_root: Path, expect_sha: str,
                      skip_files: set) -> dict:
    """Hard-link (fallback copy) every parent-MANIFEST file + the per-world
    done markers into the v0.3.2 root, hash-verified before AND after.
    `skip_files` (out-relative) are NOT linked -- they are rebuilt."""
    man_path = v03_root / "MANIFEST.json"
    v03_sha = _sha(man_path)
    if expect_sha and v03_sha != expect_sha:
        raise SystemExit(
            f"REFUSAL: {man_path} sha256 {v03_sha} != expected parent "
            f"MANIFEST sha {expect_sha} (pass --parent-sha '' only for "
            f"synthetic test roots)")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    holdouts = {s["world"] for sec in man.get("chunks", {}).values()
                for s in sec.get("skipped_holdout", [])}
    jobs: list[tuple[Path, Path, str]] = []
    n_skip = 0
    for key, e in sorted(man["worlds"].items()):
        for rel, sha in sorted(e["files"].items()):
            if any(f"/{w}__" in rel for w in holdouts):
                raise SystemExit(f"REFUSAL: HOLDOUT file listed: {rel}")
            if rel in skip_files:
                n_skip += 1
                continue
            jobs.append((v03_root / rel, out_root / rel, sha))
        marker = f"{key.split('/')[1]}_done.json"
        if any(w in marker for w in holdouts):
            raise SystemExit(f"REFUSAL: HOLDOUT marker listed: {marker}")
        src_m = v03_root / "worlds" / e["chunk_id"] / marker
        if src_m.is_file():
            d = json.loads(src_m.read_text(encoding="utf-8"))
            assert d.get("files") == e["files"], \
                f"done marker disagrees with MANIFEST: {key}"
            jobs.append((src_m, out_root / "worlds" / e["chunk_id"] / marker,
                         ""))
    log(f"link stage: parent MANIFEST sha OK ({v03_sha[:12]}...); "
        f"{len(man['worlds'])} sources, {len(jobs)} files "
        f"({n_skip} skipped for rebuild), "
        f"{len(holdouts)} HOLDOUT worlds listed (never opened)")
    n_link = n_copy = 0
    for src, dst, sha in jobs:
        if dst.is_file() and (not sha or _sha(dst) == sha):
            continue                       # idempotent re-run
        assert src.is_file(), f"missing in parent root: {src}"
        if sha:
            assert _sha(src) == sha, f"source hash mismatch: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        linked = False
        try:
            os.link(src, dst)
            linked = True
            n_link += 1
        except OSError:
            import shutil
            shutil.copy2(src, dst)
            n_copy += 1
        got = _sha(dst)
        assert got == (sha or _sha(src)), f"link hash mismatch: {dst}"
        _ = linked
    rec = dict(v03_root=str(v03_root), v03_manifest_sha256=v03_sha,
               n_sources=len(man["worlds"]), n_files=len(jobs),
               n_skipped_for_rebuild=n_skip,
               n_hardlinks=n_link, n_copies=n_copy, mismatches=0,
               n_holdout_listed=len(holdouts), holdout_accessed=False,
               linked_utc=datetime.now(timezone.utc).isoformat())
    B.atomic_write_json(out_root / ".parent_link_verified.json", rec)
    print(f"link: {len(jobs)} files sha256-verified before AND after "
          f"({n_link} hard links, {n_copy} copies); 0 mismatches; "
          f"{n_skip} rebuild files skipped; holdout_accessed=false",
          flush=True)
    return rec


def relabel_policy(out_root: Path, man: dict) -> dict:
    """v0.3.2 fix #3: rewrite every label parquet's `policy` column to the
    v0.3.2 provenance prose (atomic tmp+replace) and return the relabel
    record {rel: {old_sha, new_sha}}.  TD tables carry no policy column."""
    rec: dict = {}
    n_rows = 0
    for key, e in sorted(man["worlds"].items()):
        rel = next((r for r in e["files"] if r.endswith("_labels.parquet")),
                   None)
        if rel is None:
            continue
        p = Path(wdpaths.remap(out_root / rel))
        df = pd.read_parquet(p)
        if "policy" not in df.columns or not len(df):
            continue
        new_prose = V.SOURCE_POLICY.get(e["source"])
        if new_prose is None or (df["policy"] == new_prose).all():
            continue                        # idempotent re-run
        old_sha = _sha(p)
        df = df.assign(policy=new_prose)
        tmp = Path(str(p) + ".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
        rec[rel] = dict(old_sha256=old_sha, new_sha256=_sha(p),
                        canonical_policy=POLICY_OF_SOURCE.get(
                            e["source"], e["source"]),
                        n_rows=int(len(df)))
        n_rows += int(len(df))
    log(f"relabel stage: {len(rec)} label parquets rewritten "
        f"({n_rows} rows) to the v0.3.2 provenance prose")
    return rec


def rebuild_w8473(v03_root: Path, out_root: Path, man: dict,
                  key: str = REBUILD_KEY) -> tuple:
    """v0.3.2 fix #2: rebuild `key`'s labels/td/feat from the raw runs
    (bases_root aware; pre-t0 history from the base physics pass) and
    hard-check the tau30 mean-wait ratio vs its base into [0.3, 1.0].
    Returns (new entry, check record)."""
    chunk_id, src_full = key.split("/", 1)
    world, source = src_full.rsplit("__", 1)
    e_old = man["worlds"][key]
    assert e_old["world"] == world and e_old["source"] == source
    holdouts = {s["world"] for sec in man.get("chunks", {}).values()
                for s in sec.get("skipped_holdout", [])}
    assert world not in holdouts
    sec = man["chunks"][chunk_id]
    plan_path = Path(wdpaths.remap(sec["plan"]))
    runs_root = Path(wdpaths.remap(sec["runs_root"]))
    bases_root = (Path(wdpaths.remap(sec["bases_root"])
                       if sec.get("bases_root") else None)
                  if sec.get("bases_root") else None)
    plan, worlds = V.load_plan(plan_path)
    w = next((w for w in worlds if w["world"] == world), None)
    assert w is not None, f"{world} missing from {plan_path}"
    t0, runs = V.world_sources(w, runs_root, bases_root)
    arm = runs[source]
    ok, receipt, reason = V.receipt_status(arm)
    assert ok, f"rebuild source run incomplete: {arm} ({reason})"

    log(f"rebuild stage: {key} from {arm} "
        f"(bases_root={bases_root})")
    t1 = time.time()
    entry = B.build_source(
        w, source, arm, runs_root, out_root / "worlds" / chunk_id,
        V.cached_geometry(), {}, {}, chunk_id=chunk_id,
        default_speed_mode=int(sec.get("default_speed_mode", 31)),
        receipt=receipt, bases_root=bases_root)
    log(f"rebuild stage: built in {time.time() - t1:.1f}s "
        f"(trace {entry['trace_stats']['wall_s']}s, "
        f"n_pre_t0_shared_insertions="
        f"{entry.get('n_pre_t0_shared_insertions')})")
    assert entry.get("n_pre_t0_shared_insertions", 0) > 0, \
        "rebuilt w8473 still has an EMPTY pre-t0 insertion history"

    # wait-ratio acceptance vs the base arm: tau30 mean wait on the
    # ALIGNED decision-time window (base restricted to t >= t0 + 10 s,
    # where the switching arm's grid lives -- the metric the review's
    # 8.8x finding used: inflated native 14711 s vs base 1670 s)
    def _tau30_wait(e, t_min=None):
        rel = next(r for r in e["files"] if r.endswith("_labels.parquet"))
        df = pd.read_parquet(Path(wdpaths.remap(out_root / rel)))
        g = df[np.isclose(df["tau"], 30.0)]
        if t_min is not None:
            g = g[g["t"] >= t_min - 1e-9]
        assert len(g), f"no tau30 rows for {rel} (t_min={t_min})"
        return float((g["wait_mainline_s"] + g["wait_ramp_s"]).mean())

    base_key = key.replace("__native", "__base")
    e_base = man["worlds"][base_key]
    t0w = float(entry["t0_s"])
    r_new = _tau30_wait(entry)
    r_base = _tau30_wait(e_base, t_min=t0w + 10.0)
    ratio = r_new / r_base if r_base else float("inf")
    check = dict(key=key, tau30_mean_wait_new_s=r_new,
                 tau30_mean_wait_base_s=r_base, ratio=ratio,
                 metric=("tau30 mean (wait_mainline+wait_ramp); base "
                         "restricted to the aligned window t >= t0+10 s"),
                 inflated_native_tau30_s=14711.0, inflated_ratio=8.8,
                 n_pre_t0_shared_insertions=int(
                     entry.get("n_pre_t0_shared_insertions", 0)),
                 ok=bool(0.3 <= ratio <= 1.0))
    log(f"rebuild stage: tau30 mean wait {r_new:.0f} s vs base "
        f"{r_base:.0f} s -> ratio {ratio:.2f} "
        f"({'OK' if check['ok'] else 'OUT OF [0.3, 1.0]'})")
    if not check["ok"]:
        raise SystemExit(f"REBUILD CHECK FAIL: {key} tau30 mean-wait "
                         f"ratio {ratio:.2f} outside [0.3, 1.0]")
    entry["rebuilt_v032"] = dict(
        reason=("v0.3.2 fix #2: the removed insertions.jsonl fallback "
                "produced an EMPTY shared pre-t0 insertion history "
                "(n_pre_t0_shared_insertions=0 in the parent root) and "
                "inflated the labels ~8.8x in tau30 mean wait"),
        parent_entry_policy=str(e_old.get("policy", "")),
        bases_root=str(bases_root or runs_root))
    return key, entry, check


def write_sha256sums(out_root: Path) -> int:
    rels = ["MANIFEST.json"] + \
        [p.relative_to(out_root).as_posix()
         for p in sorted((out_root / "worlds").rglob("*")) if p.is_file()]
    lines = [f"{_sha(out_root / rel)}  {rel}" for rel in rels]
    (out_root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    return len(lines)


def write_readme(out_root: Path, summary: dict, n_sums: int) -> None:
    sha = _sha(out_root / "MANIFEST.json")
    lines = [
        "# DATA 鈥?NEVER DELETE: WD-01 Part B V0 value dataset v0.3.2 "
        "(corrected derived root)", "",
        "v0.3.2 = the v0.3 root (labels/TD/features, hash-verified links)",
        "plus the v0.3.2 review fixes:",
        "- #1 deployable context col 1 zeroed (campaign branch time t0 is",
        "  NOT recoverable from any deployable block).",
        "- #2 insertions.jsonl is never read by label/feature code; the",
        "  chunk1_fix27/w8473m2700p40t540__native source is rebuilt from",
        "  the raw runs (its v0.3 labels were ~8.8x inflated by the",
        "  removed fallback's empty pre-t0 insertion history).",
        "- #3 '__native'/'__pi' arms relabelled switching_mix (random",
        "  per-CAV switching schedules; per-CAV actions only from",
        "  switch_schedule.json / suffix_commands.jsonl -- never",
        "  dispatcher_rows.jsonl).",
        "- #5 BOTH deployable sensing variants ship per source:",
        "  `<world>__<source>_feat_dep_strict.npz` (HEADLINE: CAV",
        "  self-reports only -- no all-vehicle entrance counts, entrance",
        "  queues zeroed, no penetration-revealing ratio) and",
        "  `<world>__<source>_feat_dep_det.npz` (labelled SENSITIVITY:",
        "  all-vehicle entrance counts + CAV counts, entrance queues).",
        "", "- **Do not delete, move or rewrite** anything here.",
        "- **Train the deployable models with:**",
        "  `python -m partB_v0 train --value-model gbm --gbm-variant auto",
        "  --features deployable_strict --data <this root> --out <dir>`",
        "  (HEADLINE) and `--features deployable_detector` (labelled",
        "  SENSITIVITY).",
        "- **Integrity:**",
        f"  - `MANIFEST.json`: sha256 {sha} (raw bytes).",
        f"  - `SHA256SUMS`: {n_sums} entries (MANIFEST + every worlds/ ",
        "    file, including both sidecar variants and relabelled label",
        "    parquets).",
        "- **Parent:** v0.3 root",
        f"  {summary['v03_root']}",
        f"  (MANIFEST sha256 {summary['v03_manifest_sha256']}; recorded in",
        "  MANIFEST.json `parent` and verified at link time).",
        "- **Supersession** is inherited from v0.3 (chunk-1 p40 TRAIN/CAL",
        "  sources marked superseded_by='chunk1_fix27'; files kept; the",
        "  trainer's default --exclude-superseded drops them).",
        "- **HOLDOUT** worlds are never built and never opened (listed in",
        "  the parent MANIFEST skipped_holdout; guarded in code).",
        "- **Backup:** E:/research-dense-backups/wd01/partB_v0data_v032/",
        ""]
    (out_root / "README_DATA_NEVER_DELETE.md").write_text(
        "\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------- main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data-root", default=str(RD),
                    help="research root holding output/ (default "
                         "E:/research-dense)")
    ap.add_argument("--v03-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--parent-sha", default=PARENT_MANIFEST_SHA,
                    help="expected sha256 of the v0.3 MANIFEST (refusal "
                         "guard; pass '' for synthetic test roots)")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="skip the w8473 rebuild stage (synthetic roots "
                         "without that source)")
    ap.add_argument("--rebuild-key", default=REBUILD_KEY)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--with-actions", action="store_true",
                    help="ALSO write per-source v0.4 action sidecars "
                         "<world>__<source>_feat_act.npz (per-CAV action "
                         "ids + unknown mask from switch_schedule.json / "
                         "suffix_commands.jsonl / the fleet manifest "
                         "policy -- never dispatcher_rows.jsonl).  "
                         "Default OFF: outputs are byte-identical to the "
                         "v0.3.2 builder.  The v0.4 dataset rebuild that "
                         "consumes them is a separate registered task.")
    args = ap.parse_args(argv)

    B.lower_priority()
    rd = Path(args.data_root)
    v03_root = Path(args.v03_root or
                    rd / "output" / "wd01_partB_v0data_v03")
    out_root = Path(args.out or rd / "output" / "wd01_partB_v0data_v032")
    if out_root.resolve() == v03_root.resolve():
        raise SystemExit("REFUSAL: --out must differ from the v0.3 root")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "results").mkdir(exist_ok=True)

    man = json.loads((v03_root / "MANIFEST.json").read_text("utf-8"))
    holdout = {s["world"] for sec in man.get("chunks", {}).values()
               for s in sec.get("skipped_holdout", [])}
    skip: set = set()
    rebuild_rec = None
    if not args.no_rebuild:
        if args.rebuild_key in man["worlds"]:
            skip |= set(man["worlds"][args.rebuild_key]["files"])
        else:
            log(f"rebuild stage: key {args.rebuild_key} absent from the "
                f"parent MANIFEST; skipping (synthetic root)")

    # ---- stage: link (parent files except the rebuild target) ---------
    link_rec = link_parent_files(v03_root, out_root, args.parent_sha, skip)

    # ---- stage: rebuild (fix #2) -- BEFORE relabel: the rebuilt labels
    #      carry the v0.3.2 prose already and relabel reads every parquet
    new_key = new_entry = None
    if not args.no_rebuild and args.rebuild_key in man["worlds"]:
        new_key, new_entry, rebuild_rec = rebuild_w8473(
            v03_root, out_root, man, key=args.rebuild_key)

    # ---- stage: policy relabels (fix #3, label parquets) ---------------
    relabel_rec = relabel_policy(out_root, man)

    # ---- stage: both deployable sidecars (fix #5) ----------------------
    worlds: dict = {}
    for key, e in sorted(man["worlds"].items()):
        chunk_id, src_name = key.split("/", 1)
        sec = man["chunks"][chunk_id]
        wkey = f"{chunk_id}/{e['world']}"
        w = worlds.setdefault(wkey, dict(
            out_root=str(out_root),
            chunk_runs=wdpaths.remap(sec["runs_root"]),
            chunk_bases=(wdpaths.remap(sec["bases_root"])
                         if sec.get("bases_root") else None),
            entries={}, holdout=holdout))
        w["entries"][src_name] = e
    tasks = [(w["out_root"], w["chunk_runs"], w["chunk_bases"],
              w["entries"], w["holdout"], args.with_actions)
             for w in worlds.values()]
    log(f"dep stage: {len(tasks)} worlds x base+native x 2 variants"
        + (" + v0.4 action sidecars" if args.with_actions else "")
        + f", {args.workers} workers")
    done: list = []
    policies: dict = {}
    t0 = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, (res, pol) in enumerate(
                    pool.imap_unordered(build_world, tasks, chunksize=1),
                    1):
                done.extend(res)
                policies.update(pol)
                if i % 20 == 0 or i == len(tasks):
                    log(f"  sidecars: {i}/{len(tasks)} worlds "
                        f"({time.time() - t0:.0f}s)")
    else:
        for i, task in enumerate(tasks, 1):
            res, pol = build_world(task)
            done.extend(res)
            policies.update(pol)
            if i % 20 == 0 or i == len(tasks):
                log(f"  sidecars: {i}/{len(tasks)} worlds "
                    f"({time.time() - t0:.0f}s)")
    n_det = sum(1 for rel, _s, _n in done if rel.endswith("_det.npz"))
    n_str = sum(1 for rel, _s, _n in done if rel.endswith("_strict.npz"))
    n_act = sum(1 for rel, _s, _n in done if rel.endswith("_act.npz"))
    log(f"dep stage: {n_det} detector + {n_str} strict sidecars "
        f"written+verified"
        + (f" + {n_act} v0.4 action sidecars" if args.with_actions
           else ""))

    # ---- finalize: v0.3.2 MANIFEST -------------------------------------
    man_out = json.loads(json.dumps(man))          # deep copy of the parent
    man_out["schema"] = man.get("schema", "wd01_partB_v0data/1")
    man_out["parent"] = dict(
        root=str(v03_root),
        manifest_sha256=link_rec["v03_manifest_sha256"],
        link_verified=True,
        n_files_linked=link_rec["n_files"],
        n_files_skipped_for_rebuild=link_rec["n_skipped_for_rebuild"],
        hardlinks=link_rec["n_hardlinks"], copies=link_rec["n_copies"])
    man_out["policies"] = dict(
        canonical_by_source=dict(POLICY_OF_SOURCE),
        prose_by_source=dict(V.SOURCE_POLICY),
        note=("v0.3.2 fix #3: the campaign '__native'/'__pi' arms of "
              "chunks 1-3 + chunk1_fix27 are random per-CAV switching "
              "schedules, NOT a fleet-wide native/PI policy; per-CAV "
              "actions are attributed only from switch_schedule.json / "
              "suffix_commands.jsonl -- dispatcher_rows.jsonl is never "
              "read.  True fleet-native pairs (chunk-4 __native_base) "
              "are out of scope for this dataset."),
        per_cav_action_sources=["switch_schedule.json",
                                "suffix_commands.jsonl"],
        dispatcher_rows_read=False)
    man_out["feature_sets"] = dict(
        oracle=("per-source <src>_feat.npz (historical; demand/context "
                "carry the exact future-arrival schedule -- offline "
                "upper bound, PARTB_LEAK_AUDIT.md)"),
        deployable_strict=("per-source <src>_feat_dep_strict.npz "
                           "(v0.3.2 HEADLINE, CAV-STRICT): CAV "
                           "self-report entrance counts over (t-60,t], "
                           "(t-180,t], (t-300,t] only -- no all-vehicle "
                           "entrance counts, entrance queues zeroed, no "
                           "penetration-revealing ratio; configured "
                           "demand level; no future schedule, no future "
                           "CAV share, no time-to-branch, no t0 (fix "
                           "#1); every input has timestamp <= t"),
        deployable_detector=("per-source <src>_feat_dep_det.npz "
                             "(v0.3.2 labelled SENSITIVITY): all-vehicle "
                             "entrance counts + CAV counts and the "
                             "entrance queues -- otherwise identical to "
                             "the strict blocks"),
        action_tokens_v04=("per-source <src>_feat_act.npz (--with-actions "
                           "PREVIEW for the separate registered v0.4 "
                           "rebuild): parent payload + per-CAV action "
                           "ids/unknown mask from switch_schedule.json / "
                           "command owners / fleet manifest policy; "
                           "never dispatcher_rows.jsonl"),
        default="oracle (backward compat; select with --features "
                "deployable_strict | deployable_detector)")
    if not args.with_actions:
        # flag OFF: the manifest stays byte-identical to the previous
        # builder (no action keys, no extra feature_set prose)
        man_out["feature_sets"].pop("action_tokens_v04", None)
    man_out["build"] = dict(
        built_utc=datetime.now(timezone.utc).isoformat(),
        fixes=list(FIXES),
        n_sidecars_detector=n_det,
        n_sidecars_strict=n_str,
        n_sidecars_actions=n_act,
        n_decision_times_with_sidecar=int(sum(r[2] for r in done)),
        n_label_parquets_relabelled=len(relabel_rec),
        relabel=relabel_rec,
        rebuild=rebuild_rec,
        builder="partB_v0data/build_v032.py",
        command=" ".join(sys.argv[1:]),
        holdout_accessed=False)
    if args.with_actions:
        man_out["build"]["with_actions"] = True
    else:
        man_out["build"].pop("n_sidecars_actions", None)   # old schema
    # per-entry corrections: policy prose + canonical label + new hashes
    for key, e in man_out["worlds"].items():
        e["policy"] = V.SOURCE_POLICY.get(e["source"], e.get("policy", ""))
        e["policy_v032"] = POLICY_OF_SOURCE.get(e["source"],
                                                e.get("source", ""))
        for rel in relabel_rec:
            if rel in e["files"]:
                e["files"][rel] = relabel_rec[rel]["new_sha256"]
                e["policy_relabelled_v032"] = True
        if key == new_key and new_entry is not None:
            cleared = dict(e)
            cleared.clear()
            cleared.update(new_entry)
            man_out["worlds"][key] = cleared
    for rel, sha, _n in done:
        # rel = worlds/<chunk>/<world>__<source>_feat_dep_{det,str}.npz
        #                                  or ..._feat_act.npz
        parts = rel.split("/")
        chunk_id, fname = parts[1], parts[2]
        world, rest = fname.rsplit("__", 1)
        for suffix in ("_feat_dep_det.npz", "_feat_dep_strict.npz",
                       "_feat_act.npz"):
            rest = rest.replace(suffix, "")
        source = rest
        wkey = f"{chunk_id}/{world}__{source}"
        assert wkey in man_out["worlds"], f"sidecar without entry: {rel}"
        man_out["worlds"][wkey]["files"][rel] = sha
    man_path = out_root / "MANIFEST.json"
    B.atomic_write_json(man_path, man_out)

    n_sums = write_sha256sums(out_root)
    summary = dict(
        schema="wd01_partB_v0data_v032_build/1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        v03_root=str(v03_root),
        v03_manifest_sha256=link_rec["v03_manifest_sha256"],
        link=dict(n_files=link_rec["n_files"],
                  n_skipped_for_rebuild=link_rec["n_skipped_for_rebuild"],
                  n_hardlinks=link_rec["n_hardlinks"],
                  n_copies=link_rec["n_copies"],
                  mismatches=link_rec["mismatches"]),
        n_sources=len(man_out["worlds"]),
        n_sidecars_detector=n_det,
        n_sidecars_strict=n_str,
        n_label_parquets_relabelled=len(relabel_rec),
        rebuild=rebuild_rec,
        policies=man_out["policies"]["canonical_by_source"],
        n_sha256sums_entries=n_sums,
        n_holdout_listed=len(holdout),
        holdout_accessed=False,
        command=" ".join(sys.argv[1:]))
    if args.with_actions:
        summary["n_sidecars_actions"] = n_act
    B.atomic_write_json(out_root / "results" / "build_v032_last.json",
                        summary)
    write_readme(out_root, summary, n_sums)

    print("V032 BUILD SUMMARY", flush=True)
    print(f"  root: {out_root}", flush=True)
    print(f"  parent: {v03_root} (MANIFEST sha256 "
          f"{link_rec['v03_manifest_sha256'][:12]}...)", flush=True)
    print(f"  link: {link_rec['n_files']} files verified "
          f"({link_rec['n_hardlinks']} hard links, "
          f"{link_rec['n_copies']} copies; "
          f"{link_rec['n_skipped_for_rebuild']} skipped for the rebuild), "
          f"0 mismatches", flush=True)
    print(f"  relabel: {len(relabel_rec)} label parquets -> v0.3.2 "
          f"provenance prose (switching_mix)", flush=True)
    if rebuild_rec:
        print(f"  rebuild: {rebuild_rec['key']} tau30 mean wait "
              f"{rebuild_rec['tau30_mean_wait_new_s']:.0f} s vs base "
              f"{rebuild_rec['tau30_mean_wait_base_s']:.0f} s -> ratio "
              f"{rebuild_rec['ratio']:.2f}", flush=True)
    print(f"  sidecars: {n_det} detector (SENSITIVITY) + {n_str} strict "
          f"(HEADLINE) written and context-verified", flush=True)
    if args.with_actions:
        print(f"  action sidecars: {n_act} v0.4 <src>_feat_act.npz "
              "(preview; the v0.4 rebuild is a separate registered task)",
              flush=True)
    print(f"  SHA256SUMS: {n_sums} entries", flush=True)
    print("  holdout_accessed=false", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
