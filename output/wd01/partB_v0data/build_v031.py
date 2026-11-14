"""WD-01 Part B v0.3.1: derive the DEPLOYABLE dataset root from v0.3.

    v0.3.1 = v0.3 (labels/TD/features copied or hard-linked, hash-verified,
             MANIFEST sha256 recorded as the parent) \
           + per-source deployable sidecars <world>__<source>_feat_dep.npz \
             (same state blocks; demand/context rebuilt from RSU-observable
             inputs only -- deployable_features.py; NO future schedule, NO
             future CAV share, NO time-to-branch, NO true penetration).

The v0.3 root is NEVER written.  HOLDOUT is never opened (only MANIFEST
sources are touched; a world-name guard asserts against the parent
MANIFEST's skipped_holdout lists).  ZERO SIMULATION: run logs are read as
data (one light streaming pass per arm over physics_steps.jsonl collecting
realised insertion times + the CAV designation set -- the same fields the
v0.3 builder's RunTrace used, nothing else).

Per-source verification (hard failure): the recomputed deployable context
must reproduce the stored oracle context on every RSU-observable column
(t, since-t0, loading, q_main, q_ramp) at 1e-6 -- proving the light pass
sees exactly the schedule/insertion state the v0.3 build saw.

Stages (idempotent):
  link    hard-link (fallback: verified copy) every parent file listed in
          the v0.3 MANIFEST + the per-world done markers, sha256-verified
          before AND after.
  dep     build + verify the sidecars (multiprocess over worlds).
  finalize  v0.3.1 MANIFEST (parent sha recorded, sidecars hashed into the
          per-source files dicts), SHA256SUMS over MANIFEST + worlds/,
          README_DATA_NEVER_DELETE.md, results/build_v031_last.json.

Usage:
  python build_v031.py                      # default v0.3 -> v0.3.1
  python build_v031.py --workers 8
  python build_v031.py --v03-root <tiny> --out <tiny31> --parent-sha ""   # test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import v0lib as V  # noqa: E402  (schedule loader + DT; read-only reuse)
from deployable_features import SourceObs  # noqa: E402

sys.path.insert(0, str(HERE.parent))         # output/wd01 (wdpaths)
import wdpaths  # noqa: E402
import build_v0data as B  # noqa: E402  (lower_priority, atomic_write_json)

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
DEFAULT_V03 = str(RD / "output" / "wd01_partB_v0data_v03")
DEFAULT_OUT = str(RD / "output" / "wd01_partB_v0data_v031")
PARENT_MANIFEST_SHA = ("5cf9fa479f1ba19a209c1ed2c486663ed033866a5545eea3f"
                       "468de6c457f37a1")
T0W = time.time()
# RSU-observable oracle context cols.  v0.3.2 fix #1: col 1 ((t-t0)/600)
# is no longer reproduced by the deployable blocks (they zero it so the
# campaign branch time t0 is not recoverable), so it is dropped here.
_CTX_COLS_CHECKED = (0, 3, 6, 7)


def log(msg):
    print(f"[{time.time() - T0W:7.1f}s] {msg}", flush=True)


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


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


def ins_cav_from_log(base_arm: Path) -> set:
    """Vids whose logged insertion row says is_cav (designation
    corroboration).  DEPRECATED v0.3.2 fix #2: kept for signature
    compatibility only and returns the EMPTY set -- the physics cav_ids
    fleet is the sole designation source (insertions.jsonl is unreliable
    and no label/feature code may read it)."""
    return set()


# ------------------------------------------------- per-source sidecar --

def build_sidecar(v03_root: Path, out_root: Path, entry: dict,
                  sched, ins_vids, ins_time,
                  ins_nopre=None) -> tuple[str, str, int, str | None]:
    """Write <world>__<source>_feat_dep.npz for one MANIFEST entry and
    verify the deployable context against the stored oracle context.
    Returns (rel path, sha256, n decision times, note|None).

    Known v0.3 builder artifact (accepted with a note, never silently):
    some chunk1_fix27 native sources were built with the
    insertions-file fallback returning an EMPTY pre-t0 history, so their
    STORED context q_main/q_ramp double-count the whole pre-t0 demand as
    queued (and their labels are likewise inflated -- reported, not
    touched here).  When the stored q matches the empty-history recompute
    instead, the sidecar ships the physics-correct q and the source is
    recorded in the build summary."""
    world, source = entry["world"], entry["source"]
    rel_in = f"worlds/{entry['chunk_id']}/{world}__{source}_feat.npz"
    z = np.load(Path(wdpaths.remap(v03_root / rel_in)), allow_pickle=False)
    ts = z["t"]
    obs = SourceObs(sched["vids"], sched["t"], sched["ent"], sched["cav"],
                    ins_vids, ins_time)
    dem, ctx = obs.blocks(ts, float(entry["t0_s"]),
                          float(entry["demand_veh_h"]))

    # hard verification against the stored oracle context: the light pass
    # must reproduce every RSU-observable column exactly
    orc = z["context"]

    def _mismatch(cols):
        return [c for c in cols
                if not np.allclose(ctx[:, c], orc[:, c], rtol=0.0,
                                   atol=1e-6)]

    note = None
    bad = _mismatch(_CTX_COLS_CHECKED)
    if bad and set(bad) <= {6, 7} and ins_nopre is not None:
        vids_np, t_np = ins_nopre
        obs_np = SourceObs(sched["vids"], sched["t"], sched["ent"],
                           sched["cav"], vids_np, t_np)
        _dem_np, ctx_np = obs_np.blocks(ts, float(entry["t0_s"]),
                                        float(entry["demand_veh_h"]))
        if all(np.allclose(ctx_np[:, c], orc[:, c], rtol=0.0, atol=1e-6)
               for c in (6, 7)):
            note = (f"{world}__{source}: stored context q cols were built "
                    f"without the shared pre-t0 insertion history "
                    f"(v0.3 insertions-file fallback); the sidecar ships "
                    f"the physics-correct q from the base-run trace")
            bad = _mismatch((0, 1, 3))
    if bad:
        c = bad[0]
        i = int(np.flatnonzero(~np.isclose(ctx[:, c], orc[:, c],
                                           rtol=0.0, atol=1e-6))[0])
        raise SystemExit(
            f"VERIFICATION FAIL {world}__{source}: deployable "
            f"context col {c} differs from the stored oracle context "
            f"at row {i} ({ctx[i, c]} vs {orc[i, c]}) -- the "
            f"light insertion pass disagrees with the v0.3 build")
    if not np.isfinite(dem).all() or (dem < 0).any():
        raise SystemExit(f"VERIFICATION FAIL {world}__{source}: deployable "
                         f"demand block has non-finite/negative entries")

    payload = {k: z[k] for k in z.files}
    payload["demand"] = dem
    payload["context"] = ctx
    payload["feature_set"] = np.str_("deployable")
    payload["built_utc"] = np.str_(
        datetime.now(timezone.utc).isoformat())
    rel_out = f"worlds/{entry['chunk_id']}/{world}__{source}_feat_dep.npz"
    dst = out_root / rel_out
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dst) + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(dst)                          # .npz suffix added by numpy
    z.close()
    return rel_out, sha256_file(dst), int(len(ts)), note


def build_world(task) -> list:
    """Both sources of one world: light passes + sidecars.  Runs in a
    worker process; returns [(rel, sha, n_t), ...]."""
    v03_root, out_root, chunk_runs, chunk_bases, entries, holdout = task
    for key, entry in entries.items():
        assert entry["world"] not in holdout, \
            f"HOLDOUT world reached the builder: {key}"
    v03_root, out_root = Path(v03_root), Path(out_root)
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
    cav_desig = base_cav | ins_cav_from_log(base_arm)
    sched_raw = V.load_schedule(base_arm / "run")
    vids = sorted(sched_raw)
    sched = dict(
        vids=vids,
        t=np.array([sched_raw[x][0] for x in vids], np.float64),
        ent=np.array([sched_raw[x][1] for x in vids], np.int64),
        cav=np.array([1 if x in cav_desig else 0 for x in vids], np.int8))

    out, notes = [], []
    for entry in sorted(entries.values(), key=lambda e: e["source"]):
        source = entry["source"]
        ins_nopre = None
        if source == "base":
            ins = dict(base_t_ins)
        else:
            src_t_ins, _ = light_pass(
                arm_of(entry) / "run" / "physics_steps.jsonl", t0)
            ins = dict(src_t_ins)
            ins_nopre = (list(src_t_ins.keys()),
                         list(src_t_ins.values()))
            for vid, ti in base_t_ins.items():   # shared pre-t0 history
                if ti <= t0 + 1e-9:
                    ins.setdefault(vid, ti)
        pairs = sorted(ins.items())
        rel, sha, n_t, note = build_sidecar(
            v03_root, out_root, entry, sched,
            [v for v, _ in pairs], [t for _, t in pairs], ins_nopre)
        out.append((rel, sha, n_t))
        if note:
            notes.append(note)
    return out, notes


# ------------------------------------------------------------ stages --

def link_parent_files(v03_root: Path, out_root: Path, expect_sha: str) -> dict:
    """Hard-link (fallback copy) every parent-MANIFEST file + the per-world
    done markers into the v0.3.1 root, hash-verified before AND after."""
    man_path = v03_root / "MANIFEST.json"
    v03_sha = sha256_file(man_path)
    if expect_sha and v03_sha != expect_sha:
        raise SystemExit(
            f"REFUSAL: {man_path} sha256 {v03_sha} != expected parent "
            f"MANIFEST sha {expect_sha} (pass --parent-sha '' only for "
            f"synthetic test roots)")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    holdouts = {s["world"] for sec in man.get("chunks", {}).values()
                for s in sec.get("skipped_holdout", [])}
    jobs: list[tuple[Path, Path, str]] = []
    for key, e in sorted(man["worlds"].items()):
        for rel, sha in sorted(e["files"].items()):
            if any(f"/{w}__" in rel for w in holdouts):
                raise SystemExit(f"REFUSAL: HOLDOUT file listed: {rel}")
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
        f"{len(man['worlds'])} sources, {len(jobs)} files, "
        f"{len(holdouts)} HOLDOUT worlds listed (never opened)")
    n_link = n_copy = 0
    for src, dst, sha in jobs:
        if dst.is_file() and (not sha or sha256_file(dst) == sha):
            continue                       # idempotent re-run
        assert src.is_file(), f"missing in parent root: {src}"
        if sha:
            assert sha256_file(src) == sha, f"source hash mismatch: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        linked = False
        try:
            os.link(src, dst)
            linked = True
            n_link += 1
        except OSError:
            shutil.copy2(src, dst)
            n_copy += 1
        got = sha256_file(dst)
        assert got == (sha or sha256_file(src)), f"link hash mismatch: {dst}"
        _ = linked
    rec = dict(v03_root=str(v03_root), v03_manifest_sha256=v03_sha,
               n_sources=len(man["worlds"]), n_files=len(jobs),
               n_hardlinks=n_link, n_copies=n_copy, mismatches=0,
               n_holdout_listed=len(holdouts), holdout_accessed=False,
               linked_utc=datetime.now(timezone.utc).isoformat())
    B.atomic_write_json(out_root / ".parent_link_verified.json", rec)
    print(f"link: {len(jobs)} files sha256-verified before AND after "
          f"({n_link} hard links, {n_copy} copies); 0 mismatches; "
          f"holdout_accessed=false", flush=True)
    return rec


def write_sha256sums(out_root: Path) -> int:
    rels = ["MANIFEST.json"] + \
        [p.relative_to(out_root).as_posix()
         for p in sorted((out_root / "worlds").rglob("*")) if p.is_file()]
    lines = [f"{sha256_file(out_root / rel)}  {rel}" for rel in rels]
    (out_root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    return len(lines)


def write_readme(out_root: Path, summary: dict, n_sums: int) -> None:
    sha = sha256_file(out_root / "MANIFEST.json")
    lines = [
        "# DATA — NEVER DELETE: WD-01 Part B V0 value dataset v0.3.1 "
        "(deployable derived root)", "",
        "v0.3.1 = the v0.3 root (labels/TD/features, hash-verified links)",
        "plus a per-source deployable sidecar `<world>__<source>_feat_dep"
        ".npz`: the SAME state blocks with demand/context rebuilt from",
        "RSU-observable inputs only (recent observed entrance/CAV-entry",
        "counts, configured demand level; NO future-arrival schedule, NO",
        "future CAV share, NO time-to-branch, NO true penetration; every",
        "input has timestamp <= t).  See PARTB_LEAK_AUDIT.md and",
        "deployable_features.py.", "",
        "- **Do not delete, move or rewrite** anything here.",
        "- **Train the deployable model with:**",
        "  `python -m partB_v0 train --value-model gbm --features",
        "  deployable --data <this root> --out <dir>`",
        "- **Integrity:**",
        f"  - `MANIFEST.json`: sha256 {sha} (raw bytes).",
        f"  - `SHA256SUMS`: {n_sums} entries (MANIFEST + every worlds/ "
        "file,",
        "    including sidecars and kept superseded files).  Verify with:",
        "    `tr -d '\\r' < SHA256SUMS | sha256sum -c --quiet`",
        "- **Parent:** v0.3 root",
        f"  {summary['v03_root']}",
        f"  (MANIFEST sha256 {summary['v03_manifest_sha256']}; recorded in",
        "  MANIFEST.json `parent` and verified at link time).",
        "- **Supersession** is inherited from v0.3 (chunk-1 p40 TRAIN/CAL",
        "  sources marked superseded_by='chunk1_fix27'; files kept; the",
        "  trainer's default --exclude-superseded drops them).",
        "- **HOLDOUT** worlds are never built and never opened (listed in",
        "  the parent MANIFEST skipped_holdout; guarded in code).",
        "- **Backup:** E:/research-dense-backups/wd01/partB_v0data_v031/",
        ""]
    (out_root / "README_DATA_NEVER_DELETE.md").write_text(
        "\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------- main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--v03-root", default=DEFAULT_V03)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--parent-sha", default=PARENT_MANIFEST_SHA,
                    help="expected sha256 of the v0.3 MANIFEST (refusal "
                         "guard; pass '' for synthetic test roots)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    B.lower_priority()
    v03_root, out_root = Path(args.v03_root), Path(args.out)
    if out_root.resolve() == v03_root.resolve():
        raise SystemExit("REFUSAL: --out must differ from the v0.3 root")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "results").mkdir(exist_ok=True)

    link_rec = link_parent_files(v03_root, out_root, args.parent_sha)
    man = json.loads((v03_root / "MANIFEST.json").read_text("utf-8"))
    holdout = {s["world"] for sec in man.get("chunks", {}).values()
               for s in sec.get("skipped_holdout", [])}

    worlds: dict = {}
    for key, e in sorted(man["worlds"].items()):
        chunk_id, src_name = key.split("/", 1)
        sec = man["chunks"][chunk_id]
        wkey = f"{chunk_id}/{e['world']}"
        w = worlds.setdefault(wkey, dict(
            v03_root=str(v03_root), out_root=str(out_root),
            chunk_runs=wdpaths.remap(sec["runs_root"]),
            chunk_bases=(wdpaths.remap(sec["bases_root"])
                         if sec.get("bases_root") else None),
            entries={}, holdout=holdout))
        w["entries"][src_name] = e
    tasks = [(w["v03_root"], w["out_root"], w["chunk_runs"],
              w["chunk_bases"], w["entries"], w["holdout"])
             for w in worlds.values()]
    log(f"dep stage: {len(tasks)} worlds x base+native, "
        f"{args.workers} workers")
    done: list = []
    notes: list = []
    t0 = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, (res, wnotes) in enumerate(
                    pool.imap_unordered(build_world, tasks, chunksize=1),
                    1):
                done.extend(res)
                notes.extend(wnotes)
                if i % 20 == 0 or i == len(tasks):
                    log(f"  sidecars: {i}/{len(tasks)} worlds "
                        f"({time.time() - t0:.0f}s)")
    else:
        for i, task in enumerate(tasks, 1):
            res, wnotes = build_world(task)
            done.extend(res)
            notes.extend(wnotes)
            if i % 20 == 0 or i == len(tasks):
                log(f"  sidecars: {i}/{len(tasks)} worlds "
                    f"({time.time() - t0:.0f}s)")
    for n in notes:
        log(f"  NOTE {n}")
    log(f"dep stage: {len(done)} sidecars written+verified "
        f"({len(notes)} accepted-with-note sources)")

    # ---- finalize: v0.3.1 MANIFEST -------------------------------------
    man_out = json.loads(json.dumps(man))          # deep copy of the parent
    man_out["schema"] = man.get("schema", "wd01_partB_v0data/1")
    man_out["parent"] = dict(
        root=str(v03_root),
        manifest_sha256=link_rec["v03_manifest_sha256"],
        link_verified=True,
        n_files_linked=link_rec["n_files"],
        hardlinks=link_rec["n_hardlinks"], copies=link_rec["n_copies"])
    man_out["feature_sets"] = dict(
        oracle=("per-source <src>_feat.npz (historical; demand/context "
                "carry the exact future-arrival schedule -- offline "
                "upper bound, PARTB_LEAK_AUDIT.md)"),
        deployable=("per-source <src>_feat_dep.npz sidecar (v0.3.1): "
                    "RSU-observable inputs only -- recent observed "
                    "entrance/CAV-entry counts over (t-60,t], (t-180,t], "
                    "(t-300,t], configured demand level, q_main/q_ramp; "
                    "no future schedule, no future CAV share, no "
                    "time-to-branch, no true penetration; every input "
                    "has timestamp <= t"),
        default="oracle (backward compat; select with --features "
                "deployable)")
    man_out["build"] = dict(
        built_utc=datetime.now(timezone.utc).isoformat(),
        n_sidecars=len(done),
        n_decision_times_with_sidecar=int(sum(r[2] for r in done)),
        n_sources_with_context_note=len(notes),
        context_notes=notes,
        builder="partB_v0data/build_v031.py",
        command=" ".join(sys.argv[1:]),
        holdout_accessed=False)
    man_path = out_root / "MANIFEST.json"
    for rel, sha, _n in done:
        # rel = worlds/<chunk>/<world>__<source>_feat_dep.npz
        parts = rel.split("/")
        chunk_id, fname = parts[1], parts[2]
        world, rest = fname.rsplit("__", 1)
        source = rest.removesuffix("_feat_dep.npz")
        wkey = f"{chunk_id}/{world}__{source}"
        assert wkey in man_out["worlds"], f"sidecar without entry: {rel}"
        man_out["worlds"][wkey]["files"][rel] = sha
    B.atomic_write_json(man_path, man_out)

    n_sums = write_sha256sums(out_root)
    summary = dict(
        schema="wd01_partB_v0data_v031_build/1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        v03_root=str(v03_root),
        v03_manifest_sha256=link_rec["v03_manifest_sha256"],
        link=dict(n_files=link_rec["n_files"],
                  n_hardlinks=link_rec["n_hardlinks"],
                  n_copies=link_rec["n_copies"],
                  mismatches=link_rec["mismatches"]),
        n_sources=len(man_out["worlds"]),
        n_sidecars=len(done),
        context_notes=notes,
        n_sha256sums_entries=n_sums,
        n_holdout_listed=len(holdout),
        holdout_accessed=False,
        command=" ".join(sys.argv[1:]))
    B.atomic_write_json(out_root / "results" / "build_v031_last.json",
                        summary)
    write_readme(out_root, summary, n_sums)

    print("V031 BUILD SUMMARY", flush=True)
    print(f"  root: {out_root}", flush=True)
    print(f"  parent: {v03_root} (MANIFEST sha256 "
          f"{link_rec['v03_manifest_sha256'][:12]}...)", flush=True)
    print(f"  link: {link_rec['n_files']} files verified "
          f"({link_rec['n_hardlinks']} hard links, "
          f"{link_rec['n_copies']} copies), 0 mismatches", flush=True)
    print(f"  sidecars: {len(done)} deployable feat_dep.npz written and "
          f"context-verified against the stored oracle context", flush=True)
    if notes:
        print(f"  context notes: {len(notes)} sources shipped the "
              f"physics-correct q (stored v0.3 context was built without "
              f"the shared pre-t0 insertion history); see MANIFEST "
              f"build.context_notes", flush=True)
    print(f"  SHA256SUMS: {n_sums} entries", flush=True)
    print("  holdout_accessed=false", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
