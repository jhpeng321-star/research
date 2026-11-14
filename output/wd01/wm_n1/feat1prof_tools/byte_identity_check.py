"""FEAT1 byte-identity check: re-featurize TRAIN runs with --idea-fast code.

For each named TRAIN run this re-featurizes into an E: tmp directory (1
process, BELOW_NORMAL, free-virtual >= 8 GB gate) with the fast Idea A
baseline, then asserts every written file is sha256-identical to the
completed shard under the live reference root (D:/rd-staging/wmn1_feat/
shards_v2 -- READ-ONLY): the 12 v1 arrays + slot_vid.json + meta.json
(+ outcomes.parquet) via the reference DONE.json ``outputs`` map, and the 5
Idea A arrays via IDEA_A_DONE.json ``arrays``.  Also asserts the reference
``input_sha256`` equals the local physics sha (same source run).  TRAIN
only; HOLDOUT/CAL/chunk-5 refused by the driver's own split resolution.

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):

  python -X utf8 feat1prof_tools/byte_identity_check.py \
      --ref-root D:/rd-staging/wmn1_feat/shards_v2 \
      --out-root E:/rd-wt/feat1prof/output/wd01/wm_n1/_feat1prof_tmp/ident \
      --spec chunk1 w8443m2600p10t480 w8443m2600p10t480__switching1 \
             runs/campaign_stage1_chunk1 480 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


def _gate() -> None:
    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.apply_env_d_policy()
    U.ensure_wm_n1_on_path()
    U.lower_priority_below_normal()
    from corpus import memory as MEM

    sample = MEM.free_memory()
    if float(sample.get("free_virtual_gb", 0.0)) < 8.0:
        raise SystemExit(f"free virtual memory below 8 GB: {sample}")
    print(f"gate: {sample}")


def _policy_of(batch: str, world: str, run: str) -> str:
    """The inventory's plan-level policy label (metadata; byte-identity)."""
    import pandas as pd

    inv = WM_N1 / "inventory" / "inventory.parquet"
    df = pd.read_parquet(inv)
    sel = df[(df["batch"].astype(str) == batch)
             & (df["world"].astype(str) == world)
             & (df["run"].astype(str) == run)]
    if len(sel) != 1:
        raise SystemExit(f"inventory lookup {batch}/{world}/{run}: "
                         f"{len(sel)} rows")
    return str(sel.iloc[0].get("policy") or run.split("__")[-1])


def check_run(spec, data_root: Path, ref_root: Path, out_root: Path,
              *, idea_fast: bool = True) -> dict:
    from corpus import _util as U
    from corpus import featurize_all as FA
    from corpus.idea_a import write_idea_a_channels
    from features import writer as W

    batch, world, run, root_rel, t0 = spec
    row = dict(batch=batch, chunk=batch, world=world, split="TRAIN",
               root_rel=root_rel, t0_s=t0, arm=run.split("__")[-1],
               policy=_policy_of(batch, world, run), policy_source="probe",
               is_fleet=False, schedule_arm="", duration_s=0.0, present=True,
               receipt_status="ok", is_complete=True,
               physics_rows_written=0, physics_file_bytes=0,
               saved_state_id=None, is_smoke=False, mode="probe", run=run)
    job = FA._resolve_run(row, data_root, out_root, force=False)
    if job["split"] != "TRAIN":
        raise SystemExit(f"{run}: refusing non-TRAIN split {job['split']!r}")
    if FA.sealed_reason(job["chunk"], world):
        raise SystemExit(f"{run}: sealed world refused")

    from corpus.j4_wall_fix import install as install_j4_wall_fix

    install_j4_wall_fix()
    from features.featurizer import build_shard

    shard = build_shard(job["chunk"], world, job["arm"], job["run_dir"],
                        split=job["split"], plan_row=job["plan_row"],
                        base_history_run_dir=job.get("base_history_run_dir"),
                        batch=batch, policy=job.get("policy"))
    rec = W.write_shard(shard, str(out_root), force=True)
    if rec["status"] != "written":
        raise SystemExit(f"{run}: v1 write status {rec['status']!r}")
    shard_dir = W.shard_dir(out_root, batch, world, job["arm"])
    idea = write_idea_a_channels(shard, shard_dir, force=True,
                                 fast=idea_fast)
    if idea["status"] != "written":
        raise SystemExit(f"{run}: idea write status {idea['status']!r}")

    ref_dir = W.shard_dir(ref_root, batch, world, job["arm"])
    ref_done = json.loads((ref_dir / "DONE.json").read_text(
        encoding="utf-8"))
    ref_idea = json.loads((ref_dir / "IDEA_A_DONE.json").read_text(
        encoding="utf-8"))

    checks = {}
    input_sha = _sha(Path(job["run_dir"]) / "run" / "physics_steps.jsonl")
    checks["input_physics"] = (input_sha == ref_done["input_sha256"])
    for name in NPY_FILES + ("slot_vid.json", "meta.json"):
        if name not in ref_done["outputs"]:
            continue
        p = shard_dir / f"{name}.npy" if name in NPY_FILES \
            else shard_dir / name
        checks[name] = (_sha(p) == ref_done["outputs"][name])
    if "outcomes.parquet" in ref_done["outputs"]:
        checks["outcomes.parquet"] = (
            _sha(shard_dir / "outcomes.parquet")
            == ref_done["outputs"]["outcomes.parquet"])
    for name in IDEA_ARRAYS:
        checks[f"idea_a/{name}"] = (
            _sha(shard_dir / f"{name}.npy")
            == ref_idea["arrays"][name]["sha256"])
    ok = all(checks.values())
    return dict(run=run, key=U.run_key(batch, world, job["arm"]), ok=ok,
                n_checks=len(checks),
                failed=[k for k, v in checks.items() if not v],
                input_sha256=input_sha, checks=checks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="E:/research-dense")
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--spec", nargs=5, action="append", required=True,
                    metavar=("BATCH", "WORLD", "RUN", "ROOT_REL", "T0_S"))
    ap.add_argument("--idea-slow", dest="idea_slow", action="store_true",
                    help="rebuild the Idea A channels through the SLOW "
                         "reference path (fast=False); proves the live "
                         "--idea-fast shards are byte-identical")
    a = ap.parse_args(argv)
    _gate()
    out_root = Path(a.out_root).resolve()
    if str(out_root).upper().startswith("D:"):
        raise SystemExit("refusing a D: output root")
    out_root.mkdir(parents=True, exist_ok=True)
    reps = [check_run((s[0], s[1], s[2], s[3], float(s[4])),
                      Path(a.data_root), Path(a.ref_root), out_root,
                      idea_fast=not a.idea_slow)
            for s in a.spec]
    print(json.dumps(reps, indent=1))
    n_ok = sum(1 for r in reps if r["ok"])
    n_checks = sum(r["n_checks"] for r in reps)
    print(f"BYTE-IDENTITY: {n_ok}/{len(reps)} runs ok, "
          f"{n_checks} sha comparisons")
    return 0 if n_ok == len(reps) else 1


if __name__ == "__main__":
    sys.exit(main())
