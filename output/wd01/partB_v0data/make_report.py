"""Generate REPORT_V0DATA.md (commands, wall-clock, checks, git log).

v0.2: consolidated chunk-1 + chunk-2 dataset report.

Usage:  python make_report.py [--out DIR] [--build-log PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                              timeout=60).stdout.strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--version", default="v0.2",
                    help="dataset version for the title (v0.2 | v0.3)")
    ap.add_argument("--build-log",
                    default=str(HERE / "results_build_log_chunk2.txt"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    build = man["build"]

    log_lines = []
    log_path = Path(args.build_log)
    if log_path.exists():
        log_lines = log_path.read_text(encoding="utf-8").splitlines()
    per_source_s = []
    for ln in log_lines:
        m = re.search(r"\(([\d.]+)s \+ trace ([\d.]+)s\)", ln)
        if m:
            per_source_s.append(float(m.group(1)))
    total_wall = None
    for ln in log_lines:
        m = re.match(r"\[ *([\d.]+)s\]", ln)
        if m:
            total_wall = float(m.group(1))

    tdj = out / "results" / "td_consistency.json"
    td_summary = json.loads(tdj.read_text(encoding="utf-8"))["summary"] \
        if tdj.exists() else None
    saj = out / "results" / "sanity_v02.json"
    sanity = json.loads(saj.read_text(encoding="utf-8")) \
        if saj.exists() else None
    vej = out / "results" / "verify_v0data.json"
    verify = json.loads(vej.read_text(encoding="utf-8")) \
        if vej.exists() else None

    n_sources = len(man["worlds"])
    n_dt = sum(e["rows"]["n_decision_times"] for e in man["worlds"].values())
    n_lab = sum(e["rows"]["n_label_rows"] for e in man["worlds"].values())
    n_td = sum(e["rows"]["n_td_rows"] for e in man["worlds"].values())

    repo = HERE.parents[2]
    git_log = sh(["git", "log", "--oneline", "-5"], cwd=repo)
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    head = sh(["git", "rev-parse", "HEAD"], cwd=repo)

    chunk_lines = []
    for chunk_id, sec in sorted(man.get("chunks", {}).items()):
        keys = sec.get("world_keys", [])
        census = {}
        for k in keys:
            s_ = man["worlds"][k].get("speed_mode_source", "n/a")
            census[s_] = census.get(s_, 0) + 1
        chunk_lines.append(
            f"- [{chunk_id}] sources: {len(keys)}; plan: `{sec.get('plan')}`"
            f" (sha {str(sec.get('plan_sha256'))[:12]}...); runs root: "
            f"`{sec.get('runs_root')}`; default speed mode: "
            f"{sec.get('default_speed_mode')}; speed-mode resolution: "
            f"{census}; HOLDOUT listed: "
            f"{len(sec.get('skipped_holdout', []))}; incomplete skipped: "
            f"{len(sec.get('skipped_incomplete', []))}")

    sanity_lines = []
    if sanity:
        fb = sanity["chunk2_fallback_sources"]
        sanity_lines += [
            f"- chunk-2 contract: {sanity['chunk2_rows_checked']} parquet "
            f"rows checked, {sanity['chunk2_contract_violations']} "
            "violations of chunk_id=='chunk2' / cav_speed_mode==27 / "
            "pre_fix==False",
            f"- chunk-2 speed-mode fallback (config default 27) used on "
            f"{fb} of 112 sources (receipts record null; physics rows "
            "carry mixed 31/27 pre/post-restore values)",
            "",
            "Sanity table (TRAIN, native source; total delay label, veh·s "
            "per demand; descriptive only, NOT interpreted):", "",
            "| demand | tau | c1 mean | c1 median | c2 mean | c2 median |",
            "| --- | --- | --- | --- | --- | --- |"]
        for r in sanity["sanity_train_native"]:
            fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
            sanity_lines.append(
                f"| {r['demand_veh_h']} | {r['tau']} | {fmt(r['c1_mean'])} |"
                f" {fmt(r['c1_median'])} | {fmt(r['c2_mean'])} | "
                f"{fmt(r['c2_median'])} |")

    if args.version == "v0.3":
        objective = [
            "Versioned V0 (native-policy value) dataset v0.3 = the immutable",
            "v0.2 (chunk 1 + chunk 2, copied and hash-verified) + chunk 3",
            "(WM-54, built incrementally from the campaign logs as runs",
            "complete) + chunk1_fix27 (speedMode-27 re-runs of the chunk-1",
            "p40 worlds; the old chunk-1 p40 sources are marked",
            "superseded_by='chunk1_fix27' and excluded from training by",
            "default).  ZERO SIMULATION: read-only over the campaign run",
            "logs; HOLDOUT worlds never opened (in-code guard;",
            "holdout_accessed=false).",
        ]
        v03_additions = [
            "v0.3 additions:", "",
            "- `build_v03.py` - versioned-root builder: verified v0.2 copy,",
            "  incremental chunk-3 build, fix27 build with chunk-1 plan",
            "  verification (refusal on id/split/t0/p40 mismatch), p40",
            "  supersession, SHA256SUMS + build summary",
            "  (results/build_v03_last.json)",
            "- `../make_v0_bundle.py` - verified PLAIN training bundle zip",
            "  from a dataset root + code tree, with a clean-unzip CPU",
            "  --smoke self-test (superseded sources' files omitted;",
            "  MANIFEST entries kept)",
            "- speed-mode resolution tiers extended: receipt (config or",
            "  chunk-3-style top-level key) -> CAV physics rows -> all",
            "  physics rows -> per-chunk config default",
            "- `verify_v0data.py` structural checks generalized to every",
            "  chunk in the MANIFEST",
            "- `partB_v0/data.py`: clear error when a superseded source is",
            "  explicitly included but its files are bundle-omitted",
            "",]
    else:
        objective = [
            "Consolidated V0 (native-policy value) dataset v0.2 = chunk 1",
            " (WM-44, pre-fix fleet, copied from the wd01n worktree with every",
            "file sha256-verified against the old MANIFEST before and after the",
            "copy) + chunk 2 (WM-47, post-fix fleet, newly built with the",
            "chunk-agnostic builder from the chunk-2 campaign logs).  ZERO",
            "SIMULATION: read-only over the campaign run logs; HOLDOUT worlds",
            "never opened (in-code guard; holdout_accessed=false).",
        ]
        v03_additions = []

    lines = [
        f"# WD-01 Part B round 0 - REPORT: V0 native-value dataset "
        f"{args.version} (partB_v0data)", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Repo: `{repo}` branch `{branch}` HEAD `{head[:12]}`", "",
        "## Objective", "",
        *objective, "",
        "## Deliverables", "",
        "- `v0lib.py` - streaming trace builder + prefix-sum contract-delay",
        "  labels (equivalence to part4 delay library tested), entry",
        "  waiting, window exposure, insertion derivation",
        "- `featurize_v0.py` - V0 featurizer on the part4/partB skeleton",
        "- `build_v0data.py` - CHUNK-AGNOSTIC resumable per-world builder",
        "  (--chunk; worlds in worlds/<chunk>/; cav_speed_mode resolved",
        "  receipt -> observed-cav -> observed -> per-chunk config;",
        "  pre_fix/p40 flags; HOLDOUT refused in code)", "",
        *v03_additions,
        "v0.2 additions:", "",
        "- `copy_chunk1_v02.py` - verified chunk-1 migration (hash before +",
        "  after, done markers validated, HOLDOUT file guard)",
        "- `verify_v0data.py` - re-hashes EVERY MANIFEST-listed file;",
        "  PASS/FAIL counts (results/verify_v0data.json)",
        "- `sanity_v02.py` - chunk-2 row contract + speed-mode fallback",
        "  census + TRAIN/native sanity table (results/sanity_v02.json)",
        "- `backup_v02.py` - dataset backup + SHA256SUMS + verification",
        "- `td_report.py`, `make_data_card.py`, `make_report.py`,",
        "  `tests/`, `MANIFEST.json`, `DATA_CARD_V0.md`,",
        "  `worlds/<chunk>/` npz + parquet per source (git-ignored)", "",
        "## Commands", "",
        "```powershell",
        "python copy_chunk1_v02.py      # verified chunk-1 copy (done)",
        "python build_v0data.py --chunk chunk2 `",
        "    --plan RESEARCH_DENSE_ROOT/output/campaign_stage1_v2/"
        "results_plan_chunk2/WORLD_PLAN.json `",
        "    --runs RESEARCH_DENSE_ROOT/runs/campaign_stage1_chunk2 `",
        "    --default-speed-mode 27",
        "python verify_v0data.py        # re-hash every MANIFEST file",
        "python td_report.py            # TD identity report",
        "python sanity_v02.py           # chunk-2 contract + sanity table",
        "python make_data_card.py       # DATA_CARD_V0.md",
        "python make_report.py          # this file",
        "python backup_v02.py           # backup + SHA256SUMS + verify",
        "python -m pytest tests -q",
        "```", "",
        "## Build result", "",
        f"- sources: {n_sources} across chunks "
        f"{sorted(man.get('chunks', {}))}; decision times {n_dt}; label "
        f"rows {n_lab}; TD tuples {n_td}",
        *chunk_lines,
        f"- wall-clock chunk-2 build: "
        + (f"{total_wall:.0f} s single process below-normal priority"
           if total_wall is not None else "n/a"),
        "- per-source median build: "
        + (f"{sorted(per_source_s)[len(per_source_s)//2]:.1f} s"
           if per_source_s else "n/a"),
        f"- TD consistency: "
        + (f"{td_summary['n_tuples_checked']} tuples, "
           f"{td_summary['n_violations']} violations, worst "
           f"{td_summary['worst_diff']:.2e} veh·s" if td_summary else "n/a"),
        "- file verification: "
        + (f"PASS={verify['n_files_pass']} FAIL={verify['n_files_fail']}"
           if verify else "run verify_v0data.py"), "",
        *sanity_lines, "",
        "## Verification (pytest)", "",
        "The suite covers: label computation vs the part4 delay library",
        "(synthetic windows exact to 1e-9; real windows to 1e-6),",
        "TD-tuple consistency on ALL built sources, chunk/fix-flag",
        "columns on every row (per-chunk fleet modes: chunk1 -> {31},",
        "chunk2 -> {27}), resumability, featurizer shapes and the",
        "no-vType guarantee, entry-waiting truncation semantics, HOLDOUT",
        "and incomplete-receipt handling, and the in-code HOLDOUT guard.",
        "Final pytest run of this session: see the session report.", "",
        "## Known limitations / notes", "",
        "- The base source is the campaign incumbent arm (fleet-uniform",
        "  damping), included per DESIGN §4 Step 0; `policy`/`source`",
        "  columns let training filter to native-only.  Post-t0 base and",
        "  native share the CRN realization - treat cross-source rows as",
        "  correlated.",
        "- Two fleets in one file: chunk 1 pre-fix (31), chunk 2 post-fix",
        "  (27); filter on pre_fix / chunk_id before training.",
        "- Chunk-1 native receipts are missing for w8469m2400p40t420 and",
        "  w8473m2700p40t540 (base only) - inherited from the chunk-1",
        "  build; re-run the chunk-1 builder if their branch receipts",
        "  appear.",
        "- Branch insertions.jsonl logs are truncated for some worlds;",
        "  realized insertions are derived from the physics departed_ids",
        "  (cross-checked exact against the logs where present).", "",
        "## Git log", "", "```", git_log, "```", "",
    ]
    (out / "REPORT_V0DATA.md").write_text("\n".join(lines), encoding="utf-8")
    print("REPORT_V0DATA.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
