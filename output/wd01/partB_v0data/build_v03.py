"""WD-01 Part B B1: build the VERSIONED v0.3 dataset root incrementally.

    v0.3 = v0.2 (chunk1 + chunk2, copied and hash-verified) \
         + chunk3 (TRAIN/CAL worlds whose base AND native runs are complete \
           per run receipts, cross-checked against the campaign INDEX.json) \
         + chunk1_fix27 (speedMode-27 re-runs of the chunk-1 p40 worlds, \
           superseding the old chunk-1 p40 sources).

ZERO SIMULATION: only campaign run logs, plans and the frozen network XML
are read.  HOLDOUT worlds are never opened in any chunk (plan-level skip,
in-code guard in build_v0data.build_source, and the copy-stage file guard
below); every run prints holdout_accessed=false.

Stages (all idempotent -- re-running picks up newly completed worlds and
never rebuilds existing ones):
  copy      v0.2 -> v0.3, every file sha256-verified before AND after the
            copy against the v0.2 MANIFEST (marker .v02_copy_verified.json;
            skipped when present).  Never writes into the v0.2 root.
  chunk3    build_v0data.py --chunk <chunk3-chunk> over the chunk-3
            plan/runs; incomplete worlds are listed as pending.
  fix27     only when --fix27-plan/--fix27-runs are given: VERIFY the fix27
            plan against the chunk-1 WM-44 plan (world id, split, t0,
            penetration==0.4; refusal on ANY mismatch, before anything is
            built), then build under chunk id chunk1_fix27.  The fix27
            BASE arm dirs live in a SEPARATE root (--fix27-bases-root,
            the verify27 mode-27 bases at
            runs/campaign_stage1_verify27/<world>__base, *.lock_refused_*
            dirs ignored); only the native branches come from
            --fix27-runs.  The fix27 plan carries no base_arm entries;
            the base run name is <world>__base.  Every (world, source)
            whose run receipt is complete is built as soon as it exists
            (the verify27 bases land long before the branch runs), so a
            re-run picks up newly landed natives; a world counts as
            complete -- and leaves missing_fix27_worlds -- only once BOTH
            its sources are built.
  supersede mark every chunk-1 p40 TRAIN/CAL source in the MANIFEST with
            superseded_by='chunk1_fix27' (owner rule: even when its fix27
            re-run is still missing; the missing worlds are reported).
            Files are KEPT on disk; the trainer's default
            --exclude-superseded drops the rows.
  finalize  build summary (results/build_v03_last.json), SHA256SUMS over
            MANIFEST + worlds/, README_DATA_NEVER_DELETE.md.

Usage:
  python build_v03.py                       # copy + supersede + chunk3
  python build_v03.py --fix27-plan <P> --fix27-runs <R>   # + fix27 stage
  python build_v03.py --chunk3-chunk chunk3_test --plan <P2> --runs <R2> \
      --worlds <w> --v02-root <tiny> --v02-manifest-sha ""   # test build
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_v0data as B  # noqa: E402  (builder machinery, reused verbatim)
import v0lib as V  # noqa: E402

sys.path.insert(0, str(HERE.parent))         # output/wd01 (wdpaths)
import wdpaths  # noqa: E402

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
DEFAULT_V02 = str(RD / "output" / "wd01_partB_v0data")
DEFAULT_OUT = str(RD / "output" / "wd01_partB_v0data_v03")
V02_MANIFEST_SHA = ("f7fd64116b7c3f0024c5168fe829eb58988a31816cb28e14ebca30"
                    "f366cc5dfc")
DEFAULT_PLAN = str(RD / "output" / "campaign_stage1_v3" /
                   "results_plan_chunk3" / "WORLD_PLAN.json")
DEFAULT_RUNS = str(RD / "runs" / "campaign_stage1_chunk3")
DEFAULT_CHUNK1_PLAN = str(RD / "output" / "campaign_stage1_chunk1" /
                          "results_plan_wm44" / "WORLD_PLAN.json")
SUPERSEDED_BY = "chunk1_fix27"
COPY_MARKER = ".v02_copy_verified.json"
T0W = time.time()


def log(msg):
    print(f"[{time.time() - T0W:7.1f}s] {msg}", flush=True)


def _atomic_write_json(path: Path, obj):
    B.atomic_write_json(path, obj)


# ------------------------------------------------------------- stage: copy

def copy_v02(v02_root: Path, out_root: Path, expect_sha: str) -> dict:
    """One-time verified copy of the v0.2 root into the v0.3 root.

    Every data file is hashed before AND after the copy against the v0.2
    MANIFEST; done markers are validated against their embedded hashes.
    Refuses (exit via raised SystemExit) if the v0.2 MANIFEST sha does not
    match the expected constant or a HOLDOUT world file is listed."""
    marker = out_root / COPY_MARKER
    if marker.exists():
        rec = json.loads(marker.read_text(encoding="utf-8"))
        log(f"copy stage: already verified ({rec['n_files']} files, v0.2 "
            f"MANIFEST sha {rec['v02_manifest_sha256'][:12]}...); skipping")
        return rec
    man_path = v02_root / "MANIFEST.json"
    v02_sha = V.sha256_file(man_path)
    if expect_sha and v02_sha != expect_sha:
        raise SystemExit(
            f"REFUSAL: {man_path} sha256 {v02_sha} != expected v0.2 "
            f"MANIFEST sha {expect_sha} (pass --v02-manifest-sha '' only "
            f"for synthetic test roots)")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    holdouts = {s["world"] for sec in man.get("chunks", {}).values()
                for s in sec.get("skipped_holdout", [])}
    copies: list[tuple[Path, Path, str]] = []      # (src, dst, sha)
    copies.append((man_path, out_root / "MANIFEST.json", v02_sha))
    for key, e in sorted(man["worlds"].items()):
        chunk_rel = e["chunk_id"]
        for rel, sha in sorted(e["files"].items()):
            if any(f"/{w}__" in rel or rel.startswith(f"worlds/{w}__")
                   for w in holdouts):
                raise SystemExit(f"REFUSAL: HOLDOUT file listed: {rel}")
            copies.append((v02_root / rel, out_root / rel, sha))
        marker_name = f"{key.split('/')[1]}_done.json"
        if any(w in marker_name for w in holdouts):
            raise SystemExit(f"REFUSAL: HOLDOUT marker listed: {marker_name}")
        src_m = v02_root / "worlds" / chunk_rel / marker_name
        d = json.loads(src_m.read_text(encoding="utf-8"))
        assert d.get("files") == e["files"], \
            f"done marker hashes disagree with MANIFEST: {key}"
        copies.append((src_m, out_root / "worlds" / chunk_rel / marker_name,
                       ""))
    log(f"copy stage: v0.2 MANIFEST sha OK ({v02_sha[:12]}...); "
        f"{len(man['worlds'])} sources, {len(copies)} files, "
        f"{len(holdouts)} HOLDOUT worlds listed (no files)")
    n_ok = 0
    for src, dst, sha in copies:
        assert src.is_file(), f"missing in v0.2 root: {src}"
        sha_before = V.sha256_file(src)
        if sha:
            assert sha_before == sha, f"source hash mismatch: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert V.sha256_file(dst) == sha_before, f"copy hash mismatch: {dst}"
        n_ok += 1
        if n_ok % 200 == 0:
            log(f"  copied+verified {n_ok}/{len(copies)}")
    rec = dict(
        v02_root=str(v02_root), v02_manifest_sha256=v02_sha,
        n_sources=len(man["worlds"]), n_files=n_ok, mismatches=0,
        n_holdout_listed=len(holdouts), holdout_accessed=False,
        copied_utc=datetime.now(timezone.utc).isoformat())
    _atomic_write_json(marker, rec)
    print(f"copy: {n_ok}/{len(copies)} files sha256-verified before AND "
          f"after copy; 0 mismatches; holdout_accessed=false", flush=True)
    return rec


# -------------------------------------------------- stage: chunk builds

def _native_index_status(runs_root: Path, world: str):
    """INDEX.json branch status for <world>__native (None if unrecorded)."""
    idx = runs_root / "INDEX.json"
    if not idx.exists():
        return None, "INDEX.json missing (receipts only)"
    try:
        d = json.loads(idx.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"INDEX.json unreadable ({e})"
    wsec = (d.get("worlds") or {}).get(world)
    if not wsec:
        return None, "world not in INDEX.json"
    for b in wsec.get("branches", []):
        if b.get("run") == f"{world}__native":
            return b.get("status"), ""
    return None, "native branch not in INDEX.json"


def eligible_worlds(plan_path: Path, runs_root: Path,
                    bases_root: Path | None = None):
    """(eligible world names, pending [{world, reason}]) among non-HOLDOUT
    plan worlds: base AND native receipts complete AND the campaign
    INDEX.json does not contradict (status None/'complete' ok).  Base arm
    dirs are resolved against bases_root when given (chunk1_fix27: the
    verify27 mode-27 bases live outside the branch runs root).  HOLDOUT
    worlds are skipped WITHOUT opening anything (they never enter the
    receipt loop)."""
    _, worlds = V.load_plan(plan_path)
    eligible, pending = [], []
    for w in worlds:
        if str(w["split"]).upper() == "HOLDOUT":
            continue                      # listed by the builder, never read
        wname = w["world"]
        _, arms = V.world_sources(w, runs_root, bases_root)
        missing = []
        for arm in ("base", "native"):
            ok, _, reason = V.receipt_status(arms[arm])
            if not ok:
                missing.append(f"{arm}: {reason}")
        if missing:
            pending.append(dict(world=wname, reason="; ".join(missing)))
            continue
        status, _note = _native_index_status(runs_root, wname)
        if status is not None and status != "complete":
            pending.append(dict(world=wname,
                                reason=f"INDEX native status={status!r}"))
            continue
        eligible.append(wname)
    return eligible, pending


def already_built_worlds(manifest: dict, chunk_id: str) -> set:
    return {k.split("/", 1)[1].rsplit("__", 1)[0]
            for k in manifest.get("worlds", {})
            if k.startswith(f"{chunk_id}/")}


def complete_worlds(manifest: dict, chunk_id: str) -> set:
    """Worlds of this chunk whose base AND native sources are all built.
    Bases may land before the native branches (the verify27 bases are
    complete long before the fix27 runs); such worlds are still pending."""
    have: dict = {}
    for k in manifest.get("worlds", {}):
        if k.startswith(f"{chunk_id}/"):
            w_, s_ = k.split("/", 1)[1].rsplit("__", 1)
            have.setdefault(w_, set()).add(s_)
    return {w_ for w_, ss in have.items() if {"base", "native"} <= ss}


def build_chunk(chunk_id: str, plan_path: Path, runs_root: Path,
                out_root: Path, default_mode: int, worlds_filter: set | None,
                sources: str, bases_root: Path | None = None) -> dict:
    """Run the chunk-agnostic builder for ONE chunk incrementally and
    return its counts.  Worlds already built are always re-passed (their
    done markers make it a hash-verified no-op); new eligible worlds
    honour the optional --worlds filter; incomplete worlds stay pending."""
    manifest_path = out_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) \
        if manifest_path.exists() else dict(worlds={})
    eligible, pending = eligible_worlds(plan_path, runs_root, bases_root)
    already = already_built_worlds(manifest, chunk_id)
    want_new = eligible if worlds_filter is None \
        else sorted(set(eligible) & worlds_filter)
    # keep every already-built world in the pass so its manifest entry is
    # re-verified instead of dropped (build_v0data pops this chunk's keys)
    worlds_arg = sorted(already | set(want_new))
    argv = ["--chunk", chunk_id, "--plan", str(plan_path),
            "--runs", str(runs_root), "--default-speed-mode",
            str(default_mode), "--out", str(out_root),
            "--sources", sources]
    if bases_root is not None:
        argv += ["--bases-root", str(bases_root)]
    if worlds_arg:
        argv += ["--worlds", ",".join(worlds_arg)]
    log(f"{chunk_id} stage: {len(eligible)} eligible, "
        f"{len(already)} already built, building/resuming "
        f"{len(worlds_arg)} worlds, {len(pending)} pending")
    rc = B.main(argv)
    assert rc == 0, f"builder failed for {chunk_id} (rc={rc})"
    man = json.loads(manifest_path.read_text("utf-8"))
    sec = man["chunks"][chunk_id]
    built_now = man.get("build", {}).get("n_sources_built_this_run", 0)
    pending_names = {p["world"] for p in pending}
    pending_names -= complete_worlds(man, chunk_id)
    return dict(
        chunk=chunk_id, plan=str(plan_path), runs_root=str(runs_root),
        bases_root=str(bases_root) if bases_root is not None else None,
        n_worlds_plan=len(V.load_plan(plan_path)[1]),
        n_worlds_eligible=len(eligible), n_worlds_built_now=len(
            already_built_worlds(man, chunk_id) - already),
        n_sources_built_this_run=built_now,
        n_sources_complete=len(sec.get("world_keys", [])),
        n_pending_worlds=len(pending_names),
        pending_worlds=sorted(pending_names),
        pending_reasons={p["world"]: p["reason"] for p in pending
                         if p["world"] in pending_names},
        skipped_holdout=[s["world"] for s in sec.get("skipped_holdout", [])],
        n_skipped_incomplete=len(sec.get("skipped_incomplete", [])),
        default_speed_mode=default_mode)


# --------------------------------------------------- stage: fix27 verify

def verify_fix27_plan(fix27_plan: Path, chunk1_plan: Path):
    """REFUSE (raise SystemExit) unless every non-HOLDOUT fix27-plan world
    matches the chunk-1 WM-44 plan on world id, split, t0 and
    penetration==0.4.  HOLDOUT worlds in the fix27 plan are skipped and
    never opened."""
    _, w1 = V.load_plan(chunk1_plan)
    c1 = {w["world"]: w for w in w1}
    _, wf = V.load_plan(fix27_plan)
    bad = []
    checked = 0
    for w in wf:
        if str(w["split"]).upper() == "HOLDOUT":
            continue
        checked += 1
        name = w["world"]
        ref = c1.get(name)
        if ref is None:
            bad.append(f"{name}: not in chunk-1 plan")
            continue
        if str(ref["split"]).upper() != str(w["split"]).upper():
            bad.append(f"{name}: split {w['split']!r} != chunk-1 "
                       f"{ref['split']!r}")
        if abs(float(ref["t0_s"]) - float(w["t0_s"])) > 1e-6:
            bad.append(f"{name}: t0 {w['t0_s']} != chunk-1 {ref['t0_s']}")
        if abs(float(ref["penetration"]) - 0.4) > 1e-9 or \
                abs(float(w["penetration"]) - 0.4) > 1e-9:
            bad.append(f"{name}: penetration {w['penetration']} (chunk-1 "
                       f"{ref['penetration']}) != 0.4")
    if bad:
        raise SystemExit(
            "REFUSAL: fix27 plan disagrees with the chunk-1 WM-44 plan "
            "(nothing was built):\n  " + "\n  ".join(bad))
    return checked


def chunk1_p40_worlds(chunk1_plan: Path):
    """(sorted p40 TRAIN/CAL world names, p40 HOLDOUT count) per WM-44."""
    _, w1 = V.load_plan(chunk1_plan)
    p40 = sorted(w["world"] for w in w1
                 if abs(float(w["penetration"]) - 0.4) < 1e-9
                 and str(w["split"]).upper() != "HOLDOUT")
    n_hold = sum(1 for w in w1 if abs(float(w["penetration"]) - 0.4) < 1e-9
                 and str(w["split"]).upper() == "HOLDOUT")
    return p40, n_hold


# ------------------------------------------------- stage: supersession

def apply_supersession(out_root: Path) -> list:
    """Mark every chunk-1 p40 TRAIN/CAL MANIFEST source with
    superseded_by (files kept on disk); mirror the flag into the copied
    done markers so a later builder resume keeps it.  Returns the keys."""
    manifest_path = out_root / "MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"REFUSAL: {manifest_path} missing (copy stage "
                         f"must run first)")
    man = json.loads(manifest_path.read_text("utf-8"))
    marked = []
    for key, e in sorted(man.get("worlds", {}).items()):
        if (e.get("chunk_id") == "chunk1" and e.get("p40")
                and str(e.get("split", "")).upper()
                in ("TRAIN", "CALIBRATION")
                and not e.get("superseded_by")):
            e["superseded_by"] = SUPERSEDED_BY
            marked.append(key)
    for key in marked:
        marker = out_root / "worlds" / "chunk1" / \
            f"{key.split('/')[1]}_done.json"
        if marker.exists():
            d = json.loads(marker.read_text(encoding="utf-8"))
            if d.get("superseded_by") != SUPERSEDED_BY:
                d["superseded_by"] = SUPERSEDED_BY
                _atomic_write_json(marker, d)
    if marked:
        sec = man.setdefault("chunks", {}).setdefault("chunk1", {})
        sec["superseded_rule"] = (
            f"all chunk-1 p40 TRAIN/CAL sources marked "
            f"superseded_by='{SUPERSEDED_BY}' (speedMode-27 re-runs); "
            f"files kept on disk; trainer default --exclude-superseded "
            f"drops them")
        _atomic_write_json(manifest_path, man)
    return marked


def superseded_keys(manifest: dict) -> list:
    return sorted(k for k, e in manifest.get("worlds", {}).items()
                  if e.get("superseded_by"))


# ------------------------------------------------------------ finalize

def write_sha256sums(out_root: Path) -> int:
    rels = ["MANIFEST.json"] + \
        [p.relative_to(out_root).as_posix()
         for p in sorted((out_root / "worlds").rglob("*")) if p.is_file()]
    lines = [f"{V.sha256_file(out_root / rel)}  {rel}" for rel in rels]
    (out_root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    return len(lines)


def write_readme(out_root: Path, summary: dict) -> None:
    man = json.loads((out_root / "MANIFEST.json").read_text("utf-8"))
    sha = V.sha256_file(out_root / "MANIFEST.json")
    n_sums = summary["n_sha256sums_entries"]
    chunks = man.get("chunks", {})
    lines = [
        "# DATA — NEVER DELETE: WD-01 Part B V0 value dataset v0.3 "
        "(versioned root)", "",
        "This directory is the v0.3 Part B V0 dataset: v0.2 (chunk 1 +",
        "chunk 2) copied and hash-verified, plus chunk 3 (TRAIN/CAL worlds",
        "built incrementally as runs complete) and chunk1_fix27 (speedMode",
        "27 re-runs of the chunk-1 p40 worlds).  It is the training input",
        "for Part B V0 v0.3 bundles.", "",
        "- **Do not delete, move or rewrite** anything here.",
        "- **Integrity:**",
        f"  - `MANIFEST.json`: sha256 {sha} (raw bytes; changes only when",
        "    the incremental build completes more worlds).",
        f"  - `SHA256SUMS`: {n_sums} entries (MANIFEST + every worlds/ file,",
        "    including kept superseded files).",
        "  - Verify with: `tr -d '\\r' < SHA256SUMS | sha256sum -c --quiet`",
        "- **Provenance:**",
        f"  - v0.2 source: {summary['v02_root']} (MANIFEST sha256",
        f"    {summary['v02_manifest_sha256'][:12]}..., copied and",
        "    hash-verified before AND after; 0 mismatches).",
        "  - Chunk 1/2 plan/runs paths inside the MANIFEST are historical",
        "    D: junction strings; resolve them with wdpaths.remap().",
        "  - Supersession: every chunk-1 p40 TRAIN/CAL source is marked",
        f"    superseded_by='{SUPERSEDED_BY}' ({summary['n_superseded']} "
        "sources;", "    files kept; the trainer's default",
        "    --exclude-superseded drops them).",
        f"  - chunk1_fix27 worlds missing so far: "
        f"{len(summary['missing_fix27_worlds'])} (reported in",
        "    results/build_v03_last.json and the build stdout).",
        "- **HOLDOUT** worlds are never built and their run files are",
        "  never opened (listed per chunk in MANIFEST skipped_holdout;",
        "  guarded in code).",
        "- **Backup:** E:/research-dense-backups/wd01/partB_v0data_v03/",
        "- See DATA_CARD_V0.md and REPORT_V0DATA.md.", ""]
    (out_root / "README_DATA_NEVER_DELETE.md").write_text(
        "\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--v02-root", default=DEFAULT_V02)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--v02-manifest-sha", default=V02_MANIFEST_SHA,
                    help="expected sha256 of the v0.2 MANIFEST (refusal "
                         "guard; pass '' for synthetic test roots)")
    ap.add_argument("--chunk3-chunk", default="chunk3")
    ap.add_argument("--plan", default=DEFAULT_PLAN,
                    help="chunk-3 WORLD_PLAN.json")
    ap.add_argument("--runs", default=DEFAULT_RUNS,
                    help="chunk-3 runs root")
    ap.add_argument("--default-speed-mode", type=int, default=27)
    ap.add_argument("--skip-chunk3", action="store_true")
    ap.add_argument("--fix27-plan", default=None,
                    help="fix27 WORLD_PLAN.json (stage runs only when this "
                         "and --fix27-runs are given)")
    ap.add_argument("--fix27-runs", default=None)
    ap.add_argument("--fix27-bases-root", default=None,
                    help="root holding the fix27 BASE arm dirs (the "
                         "verify27 mode-27 bases); default: --fix27-runs")
    ap.add_argument("--fix27-chunk", default=SUPERSEDED_BY)
    ap.add_argument("--chunk1-plan", default=DEFAULT_CHUNK1_PLAN,
                    help="chunk-1 WM-44 plan the fix27 plan is verified "
                         "against")
    ap.add_argument("--fix27-default-speed-mode", type=int, default=27)
    ap.add_argument("--sources", default="base,native")
    ap.add_argument("--worlds", default="",
                    help="optional comma filter on NEW worlds built this "
                         "run (already-built worlds are always resumed)")
    args = ap.parse_args(argv)

    B.lower_priority()
    v02_root, out_root = Path(args.v02_root), Path(args.out)
    worlds_filter = ({w.strip() for w in args.worlds.split(",") if w.strip()}
                     if args.worlds.strip() else None)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "results").mkdir(exist_ok=True)

    # fix27 plan verification happens FIRST: refusal must precede any build
    fix27_checked = None
    if args.fix27_plan and args.fix27_runs:
        fix27_checked = verify_fix27_plan(Path(args.fix27_plan),
                                          Path(args.chunk1_plan))
        log(f"fix27 plan verified against chunk-1 plan: "
            f"{fix27_checked} non-HOLDOUT worlds, id/split/t0/p40 all equal")

    copy_rec = copy_v02(v02_root, out_root, args.v02_manifest_sha)

    stages = {}
    if not args.skip_chunk3:
        stages[args.chunk3_chunk] = build_chunk(
            args.chunk3_chunk, Path(args.plan), Path(args.runs), out_root,
            args.default_speed_mode, worlds_filter, args.sources)
    if args.fix27_plan and args.fix27_runs:
        stages[args.fix27_chunk] = build_chunk(
            args.fix27_chunk, Path(args.fix27_plan), Path(args.fix27_runs),
            out_root, args.fix27_default_speed_mode, worlds_filter,
            args.sources,
            bases_root=(Path(args.fix27_bases_root)
                        if args.fix27_bases_root else None))

    marked_now = apply_supersession(out_root)
    man = json.loads((out_root / "MANIFEST.json").read_text("utf-8"))
    sup_keys = superseded_keys(man)
    sup_worlds = {man["worlds"][k]["world"] for k in sup_keys}

    # missing fix27: superseded chunk-1 p40 worlds without a COMPLETE
    # chunk1_fix27 source pair yet (a world with only its fix27 base built
    # is still missing its native half)
    fix27_complete = complete_worlds(man, args.fix27_chunk) \
        if args.fix27_plan and args.fix27_runs else set()
    _, n_p40_holdout = chunk1_p40_worlds(Path(args.chunk1_plan)) \
        if args.fix27_plan and args.fix27_runs else (None, None)
    missing_fix27 = sorted(sup_worlds - fix27_complete)

    n_sums = write_sha256sums(out_root)
    summary = dict(
        schema="wd01_partB_v0data_v03_build/1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        v02_root=str(v02_root), v02_manifest_sha256=copy_rec[
            "v02_manifest_sha256"],
        copy=dict(n_files=copy_rec["n_files"],
                  mismatches=copy_rec["mismatches"],
                  copied_utc=copy_rec.get("copied_utc")),
        stages=stages,
        n_superseded=len(sup_keys),
        superseded_marked_this_run=marked_now,
        superseded_keys=sup_keys,
        missing_fix27_worlds=missing_fix27,
        n_p40_holdout_chunk1=n_p40_holdout,
        n_sha256sums_entries=n_sums,
        holdout_accessed=False,
        command=" ".join(sys.argv[1:]))
    _atomic_write_json(out_root / "results" / "build_v03_last.json", summary)
    write_readme(out_root, summary)

    print("V03 BUILD SUMMARY", flush=True)
    print(f"  root: {out_root}", flush=True)
    print(f"  copy: {copy_rec['n_files']} files verified, "
          f"{copy_rec['mismatches']} mismatches", flush=True)
    for name, st in stages.items():
        print(f"  {name}: sources complete {st['n_sources_complete']} "
              f"(built this run {st['n_sources_built_this_run']}); worlds "
              f"pending {st['n_pending_worlds']}; HOLDOUT listed "
              f"{len(st['skipped_holdout'])} (never opened)", flush=True)
    print(f"  superseded: {len(sup_keys)} chunk-1 p40 sources marked "
          f"superseded_by='{SUPERSEDED_BY}' "
          f"({len(marked_now)} newly); files kept on disk", flush=True)
    print(f"  missing fix27 worlds: {len(missing_fix27)}"
          f"{': ' + ', '.join(missing_fix27) if missing_fix27 else ''}",
          flush=True)
    print(f"  SHA256SUMS: {n_sums} entries", flush=True)
    print("  holdout_accessed=false", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
