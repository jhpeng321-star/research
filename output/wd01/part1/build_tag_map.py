"""T1 step 1: explicit tag -> regenerated run_dir map for the 88 track tags.

Resolution: runs/regenerated_v1_retry1/<cvNN>/<job>/run takes precedence when
it exists; else runs/regenerated_v1/<cvNN>/<job>/run.

Integrity anchors recorded per tag:
  - sha256(physics_steps.jsonl) and sha256(telemetry.jsonl) actually used;
  - sha256 of the restored dataset jsonl (output/worldmodel_dataset_v2/data/)
    == worldmodel_dataset_v2/MANIFEST.json output_sha256 (original, pre-MIG)
    == evidence PLAN expected_output_sha256  -> dataset bytes preserved;
  - resolved run dir == PLAN regenerated_run_dir (retry1 entry when its dir
    exists), and (split, arm, seed, loading) agree with the v2 MANIFEST.

Writes results/tag_run_map.json.  Read-only on the research-dense source
repo (wdpaths.RESEARCH_DENSE_ROOT, default E:/research-dense).
"""
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import wd1lib as L


def strip_run(p):
    s = str(p).replace("\\", "/").rstrip("/")
    return s[: -len("/run")] if s.endswith("/run") else s


def main():
    t0 = time.time()
    man = L.load_json(L.WM / "build_tracks_manifest.json")
    v2man = L.load_json(L.SRC / "output" / "worldmodel_dataset_v2" / "MANIFEST.json")
    v2 = {r["run_tag"]: r for r in v2man["runs"]}
    plan = {}
    for name in ("PLAN_retry1.json", "PLAN.json"):
        p = L.load_json(L.EVID / name)
        for e in p["runs"]:
            tag = e["run_tag"]
            if (Path(e["regenerated_run_dir"]) / "run").exists():
                if tag not in plan or "retry1" in name:
                    plan[tag] = ("retry1" if "retry1" in name else "regen1", e)
    rows, problems, warnings = [], [], []
    for tag in sorted(man_by := {r["tag"]: r for r in man["runs"]}):
        rec = man_by[tag]
        m = re.match(r"^cv(\d+)_(.+)$", tag)
        if not m:
            problems.append(f"{tag}: tag not cvNN_job")
            continue
        cands = []
        for base, pref in ((L.RETRY, "retry1"), (L.REGEN, "regen1")):
            jd = base / f"cv{m.group(1)}" / m.group(2)
            if (jd / "run").is_dir():
                cands.append((pref, jd / "run"))
        if not cands:
            problems.append(f"{tag}: no regenerated run dir")
            continue
        pref, rdir = cands[0]
        phys, tele = rdir / "physics_steps.jsonl", rdir / "telemetry.jsonl"
        if not phys.exists() or not tele.exists():
            problems.append(f"{tag}: missing physics/telemetry jsonl")
            continue
        row = dict(tag=tag, split=rec["split"], arm=rec["arm"],
                   action_idx=rec["action_idx"], seed=rec["seed"],
                   loading=rec["loading"], pref=pref, run_dir=str(rdir),
                   n_pairs_manifest=rec["n_pairs"],
                   physics_sha256=L.sha256_file(phys),
                   telemetry_sha256=L.sha256_file(tele))
        ds = L.SRC / "output" / "worldmodel_dataset_v2" / "data" / \
            rec["split"] / f"{tag}.jsonl"
        if not ds.exists():
            problems.append(f"{tag}: dataset jsonl missing")
            row.update(dataset_sha256=None, manifest_sha_match=None,
                       plan_dir_match=None, plan_sha_match=None, meta_match=None)
        else:
            dsha = L.sha256_file(ds)
            row["dataset_sha256"] = dsha
            row["manifest_sha_match"] = (dsha == v2[tag]["output_sha256"])
            row["plan_dir_match"] = (strip_run(rdir) ==
                                     strip_run(plan[tag][1]["regenerated_run_dir"]))
            row["plan_sha_match"] = (dsha == plan[tag][1]["expected_output_sha256"])
            v2r = v2[tag]
            row["meta_match"] = all((
                rec["split"] == v2r["split"], rec["arm"] == v2r["arm"],
                rec["seed"] == v2r["seed"]))
            row["loading_match"] = ((rec["loading"] or 0) ==
                                    (v2r.get("loading_kg") or 0))
            row["pref_plan"] = plan[tag][0]
            if not row["manifest_sha_match"]:
                problems.append(f"{tag}: dataset sha != v2 MANIFEST")
            if not row["plan_dir_match"]:
                problems.append(f"{tag}: run dir != PLAN")
            if not row["plan_sha_match"]:
                problems.append(f"{tag}: dataset sha != PLAN expected")
            if not row["meta_match"]:
                problems.append(f"{tag}: split/arm/seed != v2 MANIFEST")
            if not row["loading_match"]:
                warnings.append(
                    f"{tag}: loading build={rec['loading']} "
                    f"v2={v2r.get('loading_kg')} (informational)")
            if pref == "retry1" and plan[tag][0] != "retry1":
                problems.append(f"{tag}: retry1 resolved but PLAN prefers regen1")
        rows.append(row)
    out = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
               n_tags=len(rows),
               n_retry1=sum(1 for r in rows if r["pref"] == "retry1"),
               n_problems=len(problems), problems=problems,
               warnings=warnings, rows=rows)
    L.dump_json(L.RESULTS / "tag_run_map.json", out)
    print(f"tags={len(rows)} retry1={out['n_retry1']} "
          f"problems={len(problems)} warnings={len(warnings)} "
          f"({time.time() - t0:.0f}s)")
    for p in problems[:20]:
        print("  PROBLEM:", p)
    return 0 if not problems and len(rows) == 88 else 1


if __name__ == "__main__":
    sys.exit(main())
