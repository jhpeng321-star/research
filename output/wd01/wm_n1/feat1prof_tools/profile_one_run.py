"""FEAT1 offline profiling harness (FEAT1_PROFILE evidence; zero SUMO).

Re-featurizes ONE already-completed TRAIN run into an E: tmp directory
(1 process, BELOW_NORMAL, free-virtual-memory >= 8 GB gate) and reports the
v1 pass vs Idea A pass wall-time breakdown, the physics_steps.jsonl
read/parse time, and the shard write time.  With ``--cprofile PASS`` the
named pass runs under cProfile and a pstats dump is written next to the tmp
shard root.  Read-only for the live data roots; never writes outside
``--out-root`` (must NOT be under D:/rd-staging).

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):

  python -X utf8 feat1prof_tools/profile_one_run.py \
      --batch chunk1 --world w8447m2700p10t240 \
      --run w8447m2700p10t240__native --root-rel runs/campaign_stage1_chunk1 \
      --t0 240 --data-root E:/research-dense \
      --out-root E:/rd-wt/feat1prof/output/wd01/wm_n1/_feat1prof_tmp \
      [--cprofile v1|idea|both]
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))


def _gate() -> None:
    import os

    os.environ.setdefault("WD_FORBID_D", "1")
    if str(os.environ.get("WD_FORBID_D", "")).strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1 for the profiling harness")
    from corpus import _util as U

    U.apply_env_d_policy()
    U.ensure_wm_n1_on_path()
    U.lower_priority_below_normal()
    from corpus import memory as MEM

    sample = MEM.free_memory()
    if float(sample.get("free_virtual_gb", 0.0)) < 8.0:
        raise SystemExit(f"free virtual memory below 8 GB: {sample}")
    if int(U.current_priority_class()) != 0x4000 and os.name == "nt":
        print(f"note: priority class = {U.current_priority_class():#x} "
              "(BELOW_NORMAL = 0x4000)")


def _policy_of(batch: str, world: str, run: str) -> str:
    import pandas as pd

    df = pd.read_parquet(WM_N1 / "inventory" / "inventory.parquet")
    sel = df[(df["batch"].astype(str) == batch)
             & (df["world"].astype(str) == world)
             & (df["run"].astype(str) == run)]
    if len(sel) != 1:
        raise SystemExit(f"inventory lookup {batch}/{world}/{run}: "
                         f"{len(sel)} rows")
    return str(sel.iloc[0].get("policy") or run.split("__")[-1])


def _resolve(batch: str, world: str, run: str, root_rel: str, t0: float,
             data_root: str, out_root: str) -> dict:
    from corpus import featurize_all as FA

    row = dict(batch=batch, chunk=batch, world=world, split="TRAIN",
               root_rel=root_rel, t0_s=t0, arm=run.split("__")[-1],
               policy=_policy_of(batch, world, run), policy_source="probe",
               is_fleet=False, schedule_arm="", duration_s=0.0, present=True,
               receipt_status="ok", is_complete=True,
               physics_rows_written=0, physics_file_bytes=0,
               saved_state_id=None, is_smoke=False, mode="probe",
               run=run)
    out_root = Path(out_root).resolve()
    if str(out_root).upper().startswith("D:"):
        raise SystemExit("refusing a D: profiling output root")
    job = FA._resolve_run(row, Path(data_root), out_root, force=False)
    if job["split"] != "TRAIN":
        raise SystemExit(f"refusing non-TRAIN split {job['split']!r} "
                         "(FEAT1_PROFILE constraint)")
    return job


def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def run_passes(job: dict, out_root: Path, *, cprofile: str,
               fast_idea: bool = False) -> dict:
    from corpus.j4_wall_fix import install as install_j4_wall_fix

    install_j4_wall_fix()
    from features import writer as W
    from features.featurizer import build_shard
    from features import idea_a as IA
    from corpus.idea_a import write_idea_a_channels

    plan_row = job["plan_row"]
    physics = Path(job["run_dir"]) / "run" / "physics_steps.jsonl"

    rep: dict = {"run": job["run"], "batch": job["batch"],
                 "world": job["world"], "arm": job["arm"],
                 "split": job["split"], "physics_bytes": physics.stat().st_size,
                 "priority_class_hex": hex(_prio())}

    # -- raw read + json parse split --------------------------------------
    raw, t_read = _timed(lambda: physics.read_bytes())
    rep["read_raw_s"] = t_read
    n_rows = 0

    def _parse_only(buf: bytes = raw) -> int:
        dec = json.JSONDecoder()
        n = 0
        for line in buf.split(b"\n"):
            s = line.strip()
            if not s:
                continue
            dec.raw_decode(s.decode("utf-8"))
            n += 1
        return n

    n_rows, t_json = _timed(_parse_only)
    rep["json_decode_only_s"] = t_json
    rep["jsonl_rows"] = n_rows
    rep["json_decode_mb_per_s"] = (len(raw) / 1e6 / t_json) if t_json else 0.0

    # -- v1 pass (build_shard) --------------------------------------------
    shard = None
    t_build = 0.0
    prof1 = cProfile.Profile() if cprofile in ("v1", "both") else None

    def _v1():
        return build_shard(job["chunk"], job["world"], job["arm"],
                           job["run_dir"], split=job["split"],
                           plan_row=plan_row,
                           base_history_run_dir=job.get("base_history_run_dir"),
                           batch=job.get("batch"), policy=job.get("policy"))

    t0w = time.perf_counter()
    if prof1 is not None:
        prof1.enable()
    shard, t_build = _timed(_v1)
    if prof1 is not None:
        prof1.disable()
        t_build = time.perf_counter() - t0w
    rep["v1_build_s"] = t_build
    rep["v1_n_ticks"] = int(shard["meta"]["n_ticks"])
    rep["v1_n_vehicles"] = int(shard["meta"]["n_vehicles"])

    if prof1 is not None:
        _dump(prof1, out_root / f"pstats_{job['world']}_{job['arm']}_v1.pstats")

    # -- v1 write ----------------------------------------------------------
    rec, t_write = _timed(lambda: W.write_shard(shard, str(out_root),
                                                force=True))
    rep["v1_write_s"] = t_write
    rep["v1_write_bytes"] = int(rec["record"]["total_bytes"])

    # -- Idea A pass -------------------------------------------------------
    params = IA.load_nominal_params()
    prof2 = cProfile.Profile() if cprofile in ("idea", "both") else None

    def _idea_build():
        return IA.idea_a_channels_from_shard(shard, params=params,
                                             fast=fast_idea)

    t0w = time.perf_counter()
    if prof2 is not None:
        prof2.enable()
    built, t_idea_build = _timed(_idea_build)
    if prof2 is not None:
        prof2.disable()
        t_idea_build = time.perf_counter() - t0w
    rep["idea_build_s"] = t_idea_build
    rep["idea_fast"] = bool(fast_idea)

    if fast_idea:
        # same-process reference timing of the untouched slow path
        _slow, t_slow = _timed(lambda: IA.idea_a_channels_from_shard(
            shard, params=params, fast=False))
        rep["idea_slow_reference_s_same_process"] = t_slow
        for name in ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                     "event_mask"):
            assert np.array_equal(np.asarray(built[name]),
                                  np.asarray(_slow[name])), name
        rep["idea_fast_equals_slow_same_process"] = True

    def _idea_write():
        shard_dir = W.shard_dir(out_root, job["batch"], job["world"],
                                job["arm"])
        import numpy as np

        out = {}
        for name in ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                     "event_mask"):
            arr = np.ascontiguousarray(built[name])
            mm = np.lib.format.open_memmap(shard_dir / f"{name}.npy",
                                           mode="w+", dtype=arr.dtype,
                                           shape=arr.shape)
            mm[...] = arr
            mm.flush()
            del mm
            out[name] = W.sha256_file(shard_dir / f"{name}.npy")
        return out

    (shas, t_idea_write) = _timed(_idea_write)
    rep["idea_write_s"] = t_idea_write
    rep["idea_array_shas"] = shas
    rep["idea_total_s"] = t_idea_build + t_idea_write
    rep["run_total_s"] = t_build + t_write + t_idea_build + t_idea_write

    # full wrapper (parity check against the fused driver path)
    (_wr, t_wrap) = _timed(lambda: write_idea_a_channels(
        shard, W.shard_dir(out_root, job["batch"], job["world"], job["arm"]),
        force=True, fast=fast_idea))
    rep["idea_wrapper_s_incl_build_write_sha"] = t_wrap

    if prof2 is not None:
        suffix = "idea_fast" if fast_idea else "idea"
        _dump(prof2, out_root / f"pstats_{job['world']}_{job['arm']}_"
              f"{suffix}.pstats")
    return rep


def _prio() -> int:
    from corpus import _util as U

    return U.current_priority_class()


def _dump(prof: cProfile.Profile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    st = pstats.Stats(prof, stream=buf)
    st.strip_dirs().sort_stats("cumulative").print_stats(28)
    st2 = pstats.Stats(prof, stream=buf)
    st2.strip_dirs().sort_stats("tottime").print_stats(28)
    Path(path).write_text(buf.getvalue(), encoding="utf-8")
    print(f"pstats -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root-rel", required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--data-root", default="E:/research-dense")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--cprofile", choices=("none", "v1", "idea", "both"),
                    default="none")
    ap.add_argument("--idea-fast", dest="idea_fast", action="store_true",
                    help="time the byte-identical fast Idea A baseline")
    a = ap.parse_args(argv)
    _gate()
    job = _resolve(a.batch, a.world, a.run, a.root_rel, a.t0, a.data_root,
                   a.out_root)
    rep = run_passes(job, Path(a.out_root).resolve(), cprofile=a.cprofile,
                     fast_idea=a.idea_fast)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
