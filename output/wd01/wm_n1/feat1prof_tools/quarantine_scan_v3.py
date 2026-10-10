"""FEAT1 quarantine scan v3 after the 09:28 BrokenProcessPool crash.

Walks every leaf shard dir under ``shards_v2`` (``<batch>/<world>/<arm>``),
verifies the v1 ``DONE.json`` (every listed output sha256, all 12 v1 arrays
present) and the ``IDEA_A_DONE.json`` receipt (every Idea A array sha256 +
bytes).  Every dir that is not fully verified -- and nothing else -- is
MOVED (never deleted, never overwritten) to
``D:/rd-staging/wmn1_feat/quarantine_v3/<utc>/<batch>/<world>/<arm>``.
A ``QUARANTINE.json`` manifest records scanned/ok/moved counts and every
entry with its reason.  Complete dirs stay untouched (read-only hashing).

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):

  python -X utf8 feat1prof_tools/quarantine_scan_v3.py \
      --scan-root D:/rd-staging/wmn1_feat/shards_v2 \
      --quarantine-root D:/rd-staging/wmn1_feat/quarantine_v3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

NPY_FILES = ("veh_step", "veh_mask", "slot_index", "plan", "decision_t",
             "wall", "wall_mask", "counter", "accel_next", "target_mask",
             "lane_next", "arrivals")
IDEA_ARRAYS = ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
               "event_mask")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dir(dirp: Path, pool: ThreadPoolExecutor) -> tuple[str, dict]:
    """Return (verdict, details); verdict is 'ok' or a quarantine reason."""
    done_path = dirp / "DONE.json"
    if not done_path.is_file():
        files = [p.name for p in dirp.iterdir()]
        if not files:
            return "empty dir (no files, no DONE.json)", {"files": []}
        return ("incomplete dir: no DONE.json", {"files": sorted(files)})
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"unreadable DONE.json: {exc}", {}
    outputs = done.get("outputs") or {}
    missing = [n for n in NPY_FILES if n not in outputs]
    if missing:
        return f"DONE.json outputs missing arrays: {missing}", {}
    bad = []
    futures = {n: pool.submit(_sha, dirp / f"{n}.npy") for n in NPY_FILES}
    futures["slot_vid.json"] = pool.submit(_sha, dirp / "slot_vid.json")
    futures["meta.json"] = pool.submit(_sha, dirp / "meta.json")
    for name, fut in futures.items():
        p = dirp / (f"{name}.npy" if name in NPY_FILES else name)
        if not p.is_file():
            bad.append(f"{name}: missing file")
        elif fut.result() != outputs.get(name):
            bad.append(f"{name}: sha mismatch")
    if "outcomes.parquet" in outputs:
        p = dirp / "outcomes.parquet"
        if not p.is_file() or _sha(p) != outputs["outcomes.parquet"]:
            bad.append("outcomes.parquet: sha mismatch")
    if bad:
        return "DONE.json verification failed", {"bad": bad}
    idea_path = dirp / "IDEA_A_DONE.json"
    if not idea_path.is_file():
        return ("complete v1 shard but no IDEA_A_DONE.json "
                "(Idea-A-incomplete)", {})
    try:
        idea = json.loads(idea_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"unreadable IDEA_A_DONE.json: {exc}", {}
    arrays = idea.get("arrays") or {}
    bad_idea = []
    for name in IDEA_ARRAYS:
        p = dirp / f"{name}.npy"
        rec = arrays.get(name) or {}
        if not p.is_file():
            bad_idea.append(f"{name}: missing file")
            continue
        if rec.get("sha256") != _sha(p):
            bad_idea.append(f"{name}: sha mismatch")
        elif p.stat().st_size != rec.get("bytes"):
            bad_idea.append(f"{name}: bytes mismatch")
    if bad_idea:
        return "IDEA_A_DONE.json verification failed", {"bad": bad_idea}
    return "ok", {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", required=True)
    ap.add_argument("--quarantine-root", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    scan_root = Path(a.scan_root)
    q_root = Path(a.quarantine_root) / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arm_dirs = []
    for batch in sorted(p for p in scan_root.iterdir() if p.is_dir()):
        for world in sorted(p for p in batch.iterdir() if p.is_dir()):
            for arm in sorted(p for p in world.iterdir() if p.is_dir()):
                arm_dirs.append(arm)
    print(f"quarantine-scan-v3: {len(arm_dirs)} arm dirs under {scan_root}",
          flush=True)

    entries = []
    n_ok = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.threads)) as pool:
        for i, dirp in enumerate(arm_dirs):
            verdict, details = verify_dir(dirp, pool)
            rel = dirp.relative_to(scan_root)
            key = "/".join(rel.parts)
            if verdict == "ok":
                n_ok += 1
            else:
                entry = dict(key=key, src=str(dirp), reason=verdict,
                             details=details, moved_to=None)
                if not a.dry_run:
                    dest = q_root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        dest = dest.with_name(
                            dest.name + "_" + datetime.now(
                                timezone.utc).strftime("%H%M%S%f"))
                    shutil.move(str(dirp), str(dest))
                    entry["moved_to"] = str(dest)
                entries.append(entry)
            if (i + 1) % 250 == 0:
                print(f"  scanned {i + 1}/{len(arm_dirs)} ok={n_ok} "
                      f"moved={len(entries)} ({time.time() - started:.0f}s)",
                      flush=True)

    report = dict(
        schema="wm_n1_feat1_quarantine_v3/1",
        created_utc=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        scan_root=str(scan_root),
        quarantine_root=str(q_root),
        dry_run=bool(a.dry_run),
        n_scanned=len(arm_dirs), n_ok=n_ok, n_moved=len(entries),
        entries=entries)
    if not a.dry_run:
        q_root.mkdir(parents=True, exist_ok=True)
        (q_root / "QUARANTINE.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("created_utc", "n_scanned", "n_ok", "n_moved")},
                     indent=1))
    for e in entries:
        print(f"  [moved] {e['key']}: {e['reason']}"
              + (f" -> {e['moved_to']}" if e["moved_to"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
