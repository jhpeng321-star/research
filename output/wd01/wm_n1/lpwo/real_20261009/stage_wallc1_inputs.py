"""Coordinator staging: copy permitted TRAIN metadata + canonical guard into the
workspace so the guarded LPWO builder can run workspace-local.

Copies only small metadata (guard files, pack INDEX/manifest, per-run
switch_schedule/branch_record/wall_events). Never copies pack arrays. Read-only
on sources. No CAL/HOLDOUT/sealed/chunk-5/J4B/INC1 source is touched.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

W = Path(r"E:/rd-wt/ws-wallc1")
INPUTS = W / "_out" / "wallc1_inputs"
RUNS = Path(r"E:/research-dense/runs")
RESEARCH = Path(r"E:/research-dense")

PACKS = {
    "pack_v3": Path(r"D:/rd-staging/wmn1_feat/pack_v3/INDEX.json"),
    "pack_dspeed": Path(r"E:/rd-wt/dspeed-data/_out/pack_dspeed/INDEX.json"),
    "pack_damping": Path(r"E:/rd-wt/damping-snap/_out/pack_damping/INDEX.json"),
}

# run metadata files the builder / timeline reader may open (only if present)
RUN_FILES = ("switch_schedule.json", "branch_record.json", "wall_events.jsonl")


def copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    shutil.copy2(src, dst)
    return True


def main() -> int:
    INPUTS.mkdir(parents=True, exist_ok=True)
    # canonical guard files (guard requires a workspace path)
    for name, src in (
        ("canonical_exclusions.json", RESEARCH / "protocols" / "holdout_exclusions.json"),
        ("canonical_guard.py", RESEARCH / "tools" / "holdout_guard.py"),
    ):
        copy_file(src, INPUTS / name)

    # pack INDEX + manifest (metadata only)
    v3 = json.loads(PACKS["pack_v3"].read_text(encoding="utf-8"))
    for pack, idx in PACKS.items():
        dst_dir = INPUTS / pack
        copy_file(idx, dst_dir / "INDEX.json")
        man = idx.parent / "manifest.json"
        if man.is_file():
            copy_file(man, dst_dir / "manifest.json")

    rows = {r["key"]: r for r in v3["runs"]}
    lpo = W / "_out" / "lpo"
    windows = []
    for name in ("windows_train_noPI.json", "windows_val.json"):
        windows += json.loads((lpo / name).read_text(encoding="utf-8"))["windows"]

    need = {}
    for w in windows:
        if w["pack"] != "pack_v3":
            continue
        row = rows.get(w["key"])
        if row is None:
            continue
        run = row["key"].split("/", 1)[1]
        need[(row["root_rel"], run)] = row

    n_runs = n_files = 0
    missing_files = []
    for (root_rel, run), row in sorted(need.items()):
        src_dir = RUNS / Path(root_rel).name / run / "run"
        dst_dir = INPUTS / root_rel / run / "run"
        got = 0
        for fname in RUN_FILES:
            if copy_file(src_dir / fname, dst_dir / fname):
                got += 1
        if got:
            n_runs += 1
            n_files += got
        else:
            missing_files.append(f"{root_rel}/{run}")
    report = {
        "packs": {k: str(v) for k, v in PACKS.items()},
        "n_windows": len(windows),
        "n_distinct_pack_v3_runs": len(need),
        "n_runs_with_metadata": n_runs,
        "n_metadata_files": n_files,
        "n_runs_without_metadata": len(missing_files),
        "runs_without_metadata_sample": sorted(missing_files)[:10],
    }
    (W / "_out" / "WALL_C1_STAGING.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
