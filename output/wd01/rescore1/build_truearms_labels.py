"""WD-01-PB-RESCORE1 label-root builder (ZERO SIMULATION).

Builds the new label root
  E:/research-dense/output/wd01_partB_truearms_labels_v1
per output/wd01/REGISTRATION_PB_RESCORE1.md, with per TRAIN/CAL world of
chunks 2-3 the five sources:

  base             damping fleet base arm     -- COPY from the frozen
                   v0.3.2 root (labels+td only), hash-verified
  native           the mislabelled "__native" switching arm -- COPY from
                   the frozen v0.3.2 root (labels+td only), hash-verified
  pi               the "__pi" switching arm   -- BUILT from the raw runs
                   with the v0.3.2 label code (absent from v0.3.2)
  truearms_native  TRUE fleet native (J3/WM-73) -- BUILT from the raw runs
  truearms_pi      TRUE fleet PI (J3/WM-73)     -- BUILT from the raw runs

Safety properties (registered constraints):
  * COPY only, never hard-link: every copied file is shutil.copy2'd and
    asserted NOT to be a hardlink (different st_ino), and its sha256 must
    match the parent MANIFEST;
  * HOLDOUT is never opened: the split comes from each chunk's
    WORLD_PLAN.json BEFORE any run file is accessed; HOLDOUT worlds are
    skipped by name; tools/holdout_guard.py +
    protocols/holdout_exclusions.json are applied on top (excluded and
    disclosed worlds are never selected here -- they are HOLDOUT);
    an audit hook records every path this process opens and the
    close-out asserts none of them contains a HOLDOUT world name;
  * label code = partB_v0data build_v0data/v0lib (physics departed_ids,
    bases_root aware; insertions.jsonl and dispatcher_rows.jsonl are
    never read);
  * no D: paths anywhere (WD_FORBID_D clean).

Usage:
  python build_truearms_labels.py                 # both chunks, resumable
  python build_truearms_labels.py --chunk chunk2
  python build_truearms_labels.py --worlds w8480m2600p02t300
  python build_truearms_labels.py --backup        # copy root -> backups,
                                                  # verify every sha256
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import types
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (WD01 / "partB_v0data", WD01 / "partA_data", WD01 / "part4"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v0lib as V  # noqa: E402  (v0.3.2 label code, read-only reuse)
import build_v0data as B  # noqa: E402  (build_source, verify_done, utils)

sys.path.insert(0, str(WD01))                 # output/wd01 (wdpaths)
import wdpaths  # noqa: E402

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
BACKUPS = Path(wdpaths.BACKUPS_ROOT)

PLAN_OF_CHUNK = {
    "chunk2": RD / "output" / "campaign_stage1_v2" / "results_plan_chunk2"
              / "WORLD_PLAN.json",
    "chunk3": RD / "output" / "campaign_stage1_v3" / "results_plan_chunk3"
              / "WORLD_PLAN.json",
}
RUNS_OF_CHUNK = {
    "chunk2": RD / "runs" / "campaign_stage1_chunk2",
    "chunk3": RD / "runs" / "campaign_stage1_chunk3",
}
TRUEARMS_OF_CHUNK = {
    "chunk2": RD / "runs" / "campaign_stage1_chunk2_truearms",
    "chunk3": RD / "runs" / "campaign_stage1_chunk3_truearms",
}
PARENT_ROOT = RD / "output" / "wd01_partB_v0data_v032"
DEFAULT_OUT = RD / "output" / "wd01_partB_truearms_labels_v1"
DEFAULT_BACKUP = BACKUPS / "wd01" / "partB_truearms_labels_v1"
DEFAULT_SPEED_MODE = 27                       # chunks 2-3 fleet speed mode

SCHEMA = "wd01_partB_truearms_labels/1"
REGISTRATION = "output/wd01/REGISTRATION_PB_RESCORE1.md"
REGISTRATION_LF_SHA256_PREFIX = "1cd9e39600e4267e"

# canonical policy labels per source of THIS root (the report's names)
POLICY_CANONICAL = {
    "base": "damping_fleet",
    "native": "switching_mix",
    "pi": "switching_mix",
    "truearms_native": "true_native_fleet",
    "truearms_pi": "true_pi_fleet",
}
TRUEARMS_PROSE = {
    "truearms_native": (
        "TRUE fleet native (all CAVs on native IDM; campaign J3/WM-73 "
        "truearms re-run of the chunk branch; fleet-wide native policy, "
        "no per-CAV commands; registration b9fe37e5...)"),
    "truearms_pi": (
        "TRUE fleet PI (all CAVs on the fleet PI external-U controller; "
        "campaign J3/WM-73 truearms re-run of the chunk branch; "
        "fleet-wide PI policy; registration b9fe37e5...)"),
}
BRANCH_SOURCES_TO_BUILD = ("pi", "truearms_native", "truearms_pi")
COPIED_SOURCES = ("base", "native")
T0W = time.time()

# ------------------------------------------------------- holdout guard ----


class OpenedPathRecorder:
    """Audit hook recording every path this process opens/lists; the
    close-out asserts no recorded path contains a HOLDOUT world name."""

    def __init__(self):
        self.paths: set = set()
        self.installed = False

    def install(self):
        def hook(event, args):
            if event in ("open", "os.listdir", "os.scandir", "os.mkdir",
                         "os.rename", "os.remove"):
                a = args[0] if args else None
                if isinstance(a, (str, bytes, os.PathLike)):
                    try:
                        self.paths.add(os.fspath(a))
                    except Exception:                     # pragma: no cover
                        pass
        sys.addaudithook(hook)
        self.installed = True

    def assert_no_holdout(self, holdout_names: set) -> dict:
        hit = []
        low = {h.lower() for h in holdout_names}
        for p in sorted(self.paths):
            pl = p.lower()
            for h in low:
                if h in pl:
                    hit.append(p)
                    break
        if hit:
            raise AssertionError(
                f"HOLDOUT world path reached during the build: {hit[:5]}")
        return dict(n_paths_recorded=len(self.paths),
                    n_holdout_hits=0, holdout_accessed=False)


REC = OpenedPathRecorder()


def load_holdout_exclusions():
    """tools/holdout_guard.py exclusions (authoritative owner list)."""
    tools = RD / "tools"
    sys.path.insert(0, str(tools))
    import holdout_guard
    data = holdout_guard.load_holdout_exclusions()
    return data, holdout_guard


def select_worlds(plan_worlds: list, exclusions: dict) -> tuple[list, list]:
    """TRAIN/CAL worlds only, split read from the PLAN before any run
    file access; HOLDOUT worlds are skipped by name.  Applies
    holdout_guard.assert_final_holdout_eligible on the selection
    (defence in depth: excluded roots are refused even if a plan ever
    carried a non-HOLDOUT split for them)."""
    guard = load_holdout_exclusions()[1]
    selected, skipped = [], []
    for w in plan_worlds:
        split = str(w["split"]).upper()
        if split == "HOLDOUT":
            skipped.append(dict(world=w["world"], split=split,
                                role=w.get("role", "")))
            continue
        selected.append(w)
    guard.assert_final_holdout_eligible([w["world"] for w in selected],
                                        exclusions)
    names = {w["world"] for w in selected}
    assert not (names & {s["world"] for s in skipped}), \
        "a world is both selected and skipped"
    return selected, skipped


# ------------------------------------------------------------- copying ----


def _assert_not_hardlink(src: Path, dst: Path) -> None:
    """COPY-only contract: dst must not share an inode with src."""
    s1, s2 = os.stat(src), os.stat(dst)
    if (s1.st_dev, s1.st_ino) == (s2.st_dev, s2.st_ino):
        raise AssertionError(
            f"COPY-ONLY violation: {dst} is a hardlink of {src}")


def copy_source_files(parent_root: Path, parent_entry: dict, out_root: Path,
                      world: str, source: str,
                      kinds=("labels.parquet", "td.parquet")) -> dict:
    """Copy the parent root's label/td files for one source into out_root
    (COPY only, sha256-verified against the parent MANIFEST).  Returns
    the new-root files dict {rel: sha}."""
    chunk_id = parent_entry["chunk_id"]
    files = {}
    for kind in kinds:
        rel = f"worlds/{chunk_id}/{world}__{source}_{kind}"
        want = parent_entry["files"].get(rel)
        if want is None:
            raise FileNotFoundError(f"{rel} not listed in the parent "
                                    f"MANIFEST entry of {world}__{source}")
        src = parent_root / rel
        dst = out_root / rel
        if dst.is_file() and V.sha256_file(dst) == want:
            _assert_not_hardlink(src, dst)
            files[rel] = want
            continue
        assert src.is_file(), f"missing in parent root: {src}"
        got_src = V.sha256_file(src)
        assert got_src == want, f"parent file hash mismatch: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)                    # COPY, never os.link
        got = V.sha256_file(dst)
        assert got == want, f"copy hash mismatch: {dst}"
        _assert_not_hardlink(src, dst)
        files[rel] = got
    return files


# ------------------------------------------------------------ building ----


def _prime_t_pre_cache(base_rt_cache: dict, wname: str, base_arm: Path,
                       t0: float) -> None:
    """Prime the shared pre-t0 insertion cache with a LIGHT streaming pass
    over the base physics departed_ids (the same derivation build_source
    would do per branch -- done once per world instead of per branch).
    The cache only needs an object with .t_ins."""
    if wname in base_rt_cache:
        return
    t_pre = B.t_ins_pre_from_base_physics(base_arm, t0)
    base_rt_cache[wname] = types.SimpleNamespace(t_ins=t_pre)


def build_world_job(job: dict) -> dict:
    """Build the three missing branch sources of one TRAIN/CAL world.
    Runs in a worker process; returns the MANIFEST entries + skip info."""
    out_root = Path(job["out_root"])
    chunk_id = job["chunk_id"]
    runs_root = Path(job["runs_root"])
    truearms_root = Path(job["truearms_root"])
    w = job["world"]
    wname = w["world"]
    t0 = float(w["t0_s"])
    split = str(w["split"]).upper()
    assert split in ("TRAIN", "CALIBRATION"), \
        f"refusing non-TRAIN/CAL world {wname!r} (split {split!r})"

    for name in job["holdout_names"]:
        assert name != wname and name not in wname

    # runtime extension of the v0.3.2 policy prose (v0lib is read-only)
    V.SOURCE_POLICY.update(TRUEARMS_PROSE)

    geom = V.cached_geometry()
    base_arm = runs_root / f"{wname}__base"
    # v0lib.build_run_trace derives the schedule arm by string-replacing
    # "__native" with "__base", which is a no-op for the __pi /
    # truearms arm names; pre-seed its schedule cache with the SAME base
    # schedule every chunk branch restores from (read-only; the shared
    # v0.3.2 code is not modified).
    base_ck = str(base_arm / "run")
    sched_cache: dict = {}
    base_rt_cache: dict = {}
    _prime_t_pre_cache(base_rt_cache, wname, base_arm, t0)
    # seed after the loop with one shared load (cache key = the exact
    # string build_run_trace computes for each branch arm)
    shared_sched = None
    for source in BRANCH_SOURCES_TO_BUILD:
        if source.startswith("truearms_"):
            arm_name = f"{wname}__{source.split('_', 1)[1]}"
            arm = truearms_root / arm_name
        else:
            arm_name = f"{wname}__{source}"
            arm = runs_root / arm_name
        ck = str(arm / "run")
        if shared_sched is None:
            shared_sched = V.load_schedule(base_arm / "run")
        sched_cache[ck] = shared_sched
    entries, skipped_incomplete = {}, []
    for source in BRANCH_SOURCES_TO_BUILD:
        if source.startswith("truearms_"):
            arm_name = f"{wname}__{source.split('_', 1)[1]}"
            arm = truearms_root / arm_name
        else:
            arm = runs_root / f"{wname}__{source}"
        ok, receipt, reason = V.receipt_status(arm)
        if not ok:
            skipped_incomplete.append(dict(world=wname, source=source,
                                           run=str(arm), reason=reason))
            continue
        # resume: existing done marker with intact hashes
        done = B.verify_done(out_root, out_root / "worlds" / chunk_id,
                             wname, source)
        if done is not None:
            e = dict(done)
            e["world"], e["source"] = wname, source
            e["split"] = split
            e["chunk_id"] = chunk_id
            e["resumed"] = True
            e["policy_canonical"] = POLICY_CANONICAL[source]
            entries[f"{chunk_id}/{wname}__{source}"] = e
            continue
        t1 = time.time()
        entry = B.build_source(
            w, source, arm, runs_root, out_root / "worlds" / chunk_id,
            geom, sched_cache, base_rt_cache, chunk_id=chunk_id,
            default_speed_mode=DEFAULT_SPEED_MODE, receipt=receipt,
            bases_root=runs_root)
        entry["policy_canonical"] = POLICY_CANONICAL[source]
        entries[f"{chunk_id}/{wname}__{source}"] = entry
        print(f"[{time.time() - T0W:7.1f}s] built "
              f"{chunk_id}/{wname}__{source}: "
              f"labels={entry['rows']['n_label_rows']} "
              f"({time.time() - t1:.1f}s + trace "
              f"{entry['trace_stats']['wall_s']}s)", flush=True)
    return dict(chunk_id=chunk_id, world=wname, entries=entries,
                skipped_incomplete=skipped_incomplete)


# ----------------------------------------------------------- manifest ----


def copied_entry(parent_entry: dict, files: dict, parent_sha: str) -> dict:
    e = dict(parent_entry)
    e["files"] = dict(files)
    e["copied_from_v032"] = True
    e["parent_root"] = str(PARENT_ROOT)
    e["parent_manifest_sha256"] = parent_sha
    e["copied_kinds"] = ["labels.parquet", "td.parquet"]
    e["policy_canonical"] = POLICY_CANONICAL[e["source"]]
    e.pop("resumed", None)
    return e


def sha_of(path: Path) -> str:
    return V.sha256_file(path)


def write_sha256sums(out_root: Path) -> int:
    rels = ["MANIFEST.json"] + \
        [p.relative_to(out_root).as_posix()
         for p in sorted((out_root / "worlds").rglob("*")) if p.is_file()]
    lines = [f"{sha_of(out_root / rel)}  {rel}" for rel in rels]
    (out_root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    return len(lines)


def load_manifest_state(out_root: Path) -> dict:
    man_path = out_root / "MANIFEST.json"
    if man_path.exists():
        return json.loads(man_path.read_text(encoding="utf-8"))
    return dict(
        schema=SCHEMA, zero_simulation=True,
        registration=dict(file=REGISTRATION,
                          lf_sha256_prefix=REGISTRATION_LF_SHA256_PREFIX),
        tau_s=list(V.TAUS), commit_window_s=V.COMMIT_S,
        decide_every_s=V.DECIDE_EVERY_S,
        base_decision_min_t_s=V.BASE_DECISION_MIN_T,
        note=("WD-01-PB-RESCORE1 label root: per TRAIN/CAL world of "
              "chunks 2-3, damping bases + switching arms (copied "
              "hash-verified from the frozen v0.3.2 root) + switching pi "
              "+ TRUE fleet native/PI (built from the raw runs with the "
              "v0.3.2 label code).  COPY only; HOLDOUT never opened."),
        chunks={}, worlds={})


def git_commit() -> str:
    from partB_v0.outputs import git_commit as g
    return g(str(Path(__file__).resolve().parents[3]))


def finalize(out_root: Path, man: dict) -> dict:
    for cid, sec in man["chunks"].items():
        sec["world_keys"] = sorted(
            k for k in man["worlds"] if k.startswith(f"{cid}/"))
        sec["n_sources"] = len(sec["world_keys"])
    man["build"] = dict(
        built_utc=datetime.now(timezone.utc).isoformat(),
        builder="output/wd01/rescore1/build_truearms_labels.py",
        label_code_commit=git_commit(),
        label_code=("partB_v0data build_v0data.build_source / v0lib "
                    "(physics departed_ids pre-t0 history, bases_root "
                    "aware; insertions.jsonl and dispatcher_rows.jsonl "
                    "never read)"),
        copy_only=True,
        holdout=man.get("holdout", {}),
        chunks=sorted(man["chunks"]),
        n_sources=len(man["worlds"]),
        command=" ".join(sys.argv))
    man_path = out_root / "MANIFEST.json"
    B.atomic_write_json(man_path, man)
    n = write_sha256sums(out_root)
    return n


# --------------------------------------------------------------- main ----


def run_build(args) -> int:
    B.lower_priority()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "results").mkdir(exist_ok=True)
    REC.install()
    exclusions, guard = load_holdout_exclusions()
    excluded = sorted(guard.excluded_roots(exclusions))
    disclosed = list(guard.disclosed_roots(exclusions))

    parent_man = json.loads(
        (PARENT_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    parent_sha = sha_of(PARENT_ROOT / "MANIFEST.json")
    parent_files_verified = 0

    man = load_manifest_state(out_root)
    man.setdefault("policies", dict(
        canonical_by_source=dict(POLICY_CANONICAL),
        prose_by_source={**V.SOURCE_POLICY, **TRUEARMS_PROSE},
        note=("canonical policy labels of this root: damping bases are "
              "the campaign incumbent fleet-uniform damping arm; "
              "'native'/'pi' are the mislabelled random per-CAV "
              "SWITCHING arms (v0.3.2 fix #3); truearms_native/"
              "truearms_pi are the TRUE fleet native/PI references "
              "(J3/WM-73).")))
    holdout_sec = man.setdefault("holdout", dict(
        accessed=False,
        guard="tools/holdout_guard.py + protocols/holdout_exclusions.json",
        n_excluded_worlds=len(excluded),
        excluded_worlds=excluded,
        kept_disclosed=disclosed,
        note=("w8477m2600p02t480 is kept_disclosed for final HOLDOUT "
              "evaluations, but it is a HOLDOUT world and this analysis "
              "is TRAIN/CAL only -- it is skipped by name from the plan "
              "split BEFORE any file access, like every HOLDOUT world.")))

    chunks = [args.chunk] if args.chunk else sorted(PLAN_OF_CHUNK)
    want = set(args.worlds.split(",")) if args.worlds else None
    holdout_names: set = set()

    for chunk_id in chunks:
        plan_path = PLAN_OF_CHUNK[chunk_id]
        runs_root = RUNS_OF_CHUNK[chunk_id]
        truearms_root = TRUEARMS_OF_CHUNK[chunk_id]
        plan, worlds = V.load_plan(plan_path)
        selected, skipped_holdout = select_worlds(worlds, exclusions)
        if want:
            selected = [w for w in selected if w["world"] in want]
        sec = man["chunks"].setdefault(chunk_id, dict(
            chunk_id=chunk_id,
            default_speed_mode=DEFAULT_SPEED_MODE,
            plan=str(plan_path), plan_sha256=sha_of(plan_path),
            runs_root=str(runs_root), truearms_root=str(truearms_root),
            parent_root=str(PARENT_ROOT),
            parent_manifest_sha256=parent_sha,
            skipped_holdout=[], skipped_incomplete=[], world_keys=[]))
        known_ho = {s["world"] for s in sec.get("skipped_holdout", [])}
        for s in skipped_holdout:
            if s["world"] not in known_ho:
                sec["skipped_holdout"].append(s)
        holdout_names = {s["world"] for s in sec["skipped_holdout"]} \
            | {s["world"] for c in man["chunks"].values()
               for s in c.get("skipped_holdout", [])}

        jobs = []
        for w in selected:
            wname = w["world"]
            world_entries = {}
            for source in COPIED_SOURCES:
                key = f"{chunk_id}/{wname}__{source}"
                pe = parent_man["worlds"].get(key)
                if pe is None:
                    raise KeyError(f"{key} missing from the v0.3.2 "
                                   f"MANIFEST")
                if pe["split"] != w["split"]:
                    raise AssertionError(
                        f"split disagreement for {key}: plan "
                        f"{w['split']} vs parent {pe['split']}")
                files = copy_source_files(PARENT_ROOT, pe, out_root,
                                          wname, source)
                parent_files_verified += len(files)
                world_entries[key] = copied_entry(pe, files, parent_sha)
            for key, e in world_entries.items():
                man["worlds"][key] = e
            jobs.append(dict(
                out_root=str(out_root), chunk_id=chunk_id,
                runs_root=str(runs_root), truearms_root=str(truearms_root),
                world=w, holdout_names=sorted(holdout_names),
                label=f"{chunk_id}/{wname}"))

        print(f"[{time.time() - T0W:7.1f}s] {chunk_id}: {len(jobs)} "
              f"TRAIN/CAL worlds ({len(skipped_holdout)} HOLDOUT skipped "
              f"by name), {len(COPIED_SOURCES)} copied + "
              f"{len(BRANCH_SOURCES_TO_BUILD)} built sources per world, "
              f"{args.workers} workers", flush=True)
        n_done = 0
        if args.workers > 1:
            with Pool(args.workers) as pool:
                for res in pool.imap_unordered(build_world_job, jobs):
                    n_done += 1
                    _merge(man, sec, res, out_root, n_done, len(jobs))
        else:
            for res in map(build_world_job, jobs):
                n_done += 1
                _merge(man, sec, res, out_root, n_done, len(jobs))

    rec = REC.assert_no_holdout(holdout_names_all(man))
    holdout_sec.update(rec)
    holdout_sec["opened_path_note"] = (
        "audit-hook record over every path this builder process opened/"
        "listed; zero hits for any HOLDOUT world name")
    n_sums = finalize(out_root, man)
    print(f"DONE: {len(man['worlds'])} sources, SHA256SUMS {n_sums} "
          f"entries, copied files verified this run: "
          f"{parent_files_verified}, holdout_accessed=false", flush=True)
    return 0


def holdout_names_all(man: dict) -> set:
    return {s["world"] for c in man["chunks"].values()
            for s in c.get("skipped_holdout", [])}


def _merge(man: dict, sec: dict, res: dict, out_root: Path,
           n_done: int, n_jobs: int) -> None:
    for key, e in res["entries"].items():
        man["worlds"][key] = e
    for s in res["skipped_incomplete"]:
        if s not in sec["skipped_incomplete"]:
            sec["skipped_incomplete"].append(s)
    sec["world_keys"] = sorted(
        k for k in man["worlds"] if k.startswith(f"{res['chunk_id']}/"))
    sec["n_sources"] = len(sec["world_keys"])
    sec["updated_utc"] = datetime.now(timezone.utc).isoformat()
    finalize(out_root, man)
    print(f"[{time.time() - T0W:7.1f}s] progress: {n_done}/{n_jobs} worlds "
          f"of {res['chunk_id']} ({len(man['worlds'])} sources in "
          f"MANIFEST)", flush=True)


# -------------------------------------------------------------- backup ----


def run_backup(out_root: Path, backup_root: Path) -> int:
    """Copy the finished root to the backup (COPY only) and verify every
    SHA256SUMS entry in the backup."""
    sums = out_root / "SHA256SUMS"
    assert sums.is_file(), f"{sums} missing -- finalize the root first"
    entries = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, rel = line.split("  ", 1)
            entries.append((sha, rel))
    backup_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for sha, rel in entries:
        src, dst = out_root / rel, backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or sha_of(dst) != sha:
            assert src.is_file(), f"missing in source root: {src}"
            shutil.copy2(src, dst)
            _assert_not_hardlink(src, dst)
        got = sha_of(dst)
        assert got == sha, f"backup hash mismatch: {dst}"
        n += 1
    # no extra files in the backup
    extra = [p.relative_to(backup_root).as_posix()
             for p in backup_root.rglob("*") if p.is_file()
             and p.relative_to(backup_root).as_posix()
             not in {rel for _s, rel in entries}]
    assert not extra, f"unexpected extra files in backup: {extra[:5]}"
    rec = dict(backup_root=str(backup_root), source_root=str(out_root),
               n_files=n, mismatches=0, extra_files=0,
               holdout_accessed=False,
               verified_utc=datetime.now(timezone.utc).isoformat())
    dest = HERE.parent / "partB_rescore1" / "backup_verify_truearms_labels_v1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    B.atomic_write_json(dest, rec)
    print(f"BACKUP OK: {n} files verified in {backup_root} "
          f"(copy-only, 0 mismatches, 0 extra)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--chunk", default=None, choices=sorted(PLAN_OF_CHUNK))
    ap.add_argument("--worlds", default="")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--backup", action="store_true",
                    help="copy the finished root to the backups root and "
                         "verify every SHA256SUMS entry there")
    ap.add_argument("--backup-root", default=str(DEFAULT_BACKUP))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if args.backup:
        return run_backup(Path(args.out), Path(args.backup_root))
    return run_build(args)


if __name__ == "__main__":
    sys.exit(main())
