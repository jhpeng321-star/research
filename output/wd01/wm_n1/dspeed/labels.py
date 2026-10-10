"""A9 step 1: build per-run v0.4a label parquets for the trajectory arms.

Conventions are ``partB_v0data/build_v04a_cov.label_rows_for_source`` (the
labels-only coverage builder): the shared damping-base trace supplies the
schedule and the t0-row ``n_vehicles_present``; the branch trace is the
trajectory run (origin ``t0``, first physics row ``t0 + 0.5``); decision
grid ``t0`` then ``t0+10 .. t0+570`` step 10; ``v0lib.window_label`` with
``skip_first_tick=False``; every label ``source`` is the run arm
(``arm0`` .. ``arm3``) and every ``policy`` is ``trajectory``.

TRAIN only: the world set is exactly the campaign ``worlds.json`` set
(holdout-guarded, TRAIN-filtered at campaign registration).  No
CAL/HOLDOUT/chunk-5/J4B/INC1 world is reachable.  Re-runnable: a done marker
whose recorded label SHA still verifies is reused; adding later runs is one
more invocation.

Usage::

    python -m dspeed.labels --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import common as D  # noqa: E402

HERE = Path(__file__).resolve().parent
_WM_N1 = HERE.parent
if str(_WM_N1) not in sys.path:
    sys.path.insert(0, str(_WM_N1))
_PARTB = _WM_N1.parent / "partB_v0data"
if str(_PARTB) not in sys.path:
    sys.path.insert(0, str(_PARTB))

import build_v0data as B  # noqa: E402
import v0lib as V  # noqa: E402

LABEL_COLS = B.LABEL_COLS + ["pi_native", "pi_damping", "pi_pi", "role",
                             "aux", "arm_kind"]
SCHEMA = "wm_n1_dspeed_labels/1"
_T0W = time.time()


def _log(msg):
    print(f"[{time.time() - _T0W:7.1f}s] {msg}", flush=True)


def _has_snapshot(rt, t: float) -> bool:
    k = int(round((t - rt.t_origin) / V.DT))
    return 0 <= k <= rt.n_ticks and rt.tick_last[k] > rt.tick_first[k]


def _label_rows_for_run(run: dict, rt_base, sched_cache) -> pd.DataFrame:
    world, source = run["world"], run["arm"]
    t0 = float(run["t0_s"])
    t_pre = V.build_t_ins_pre(rt_base, t0)
    rt = V.build_run_trace(world, "native", run["run_dir"], t0,
                           t_ins_pre=t_pre, sched_cache=sched_cache,
                           base_arm_dir=run["base_dir"],
                           physics_path=run["physics"])
    ts_all = np.concatenate([[t0], V.decision_grid(t0, rt.data_end,
                                                   "native")])
    keep_t = [float(t) for t in ts_all
              if _has_snapshot(rt_base if abs(float(t) - t0) < 1e-9 else rt,
                               float(t))]
    ts = np.asarray(keep_t, np.float64)
    tau_avail = rt.data_end - ts
    speed_mode = int(run["receipt"].get("cav_speed_mode") or 27)
    meta = D.world_metadata(world)
    rows = []
    for i, t in enumerate(ts):
        t = float(t)
        ta = float(tau_avail[i])
        at_t0 = abs(t - t0) < 1e-9
        base = dict(
            world=world, source=source, chunk_id=run["chunk"],
            split="TRAIN", demand_veh_h=int(meta["demand_veh_h"]),
            penetration=float(meta["penetration"]),
            p40=bool(abs(meta["penetration"] - 0.4) < 1e-9),
            pre_fix=bool(speed_mode == 31), t0_s=t0, policy="trajectory",
            t=t, tau_available=ta, cav_speed_mode=speed_mode,
            pi_native=0.0, pi_damping=0.0, pi_pi=0.0,
            role="primary", aux=False, arm_kind="trajectory")
        for tau in V.TAUS:
            if tau > ta + 1e-9:
                continue
            row = V.window_label(rt, t, tau, skip_first_tick=False)
            if at_t0:
                snap = rt_base.snapshot_at(t0)
                row["n_vehicles_present"] = int(0 if snap is None
                                                else len(snap["vid"]))
            rows.append({**base, "tau": float(tau), **row})
    return pd.DataFrame(rows, columns=LABEL_COLS)


def _world_task(task):
    root, chunk, world, base_dir, runs = task
    root = Path(root)
    sched_cache: dict = {}
    ok, receipt, reason = V.receipt_status(Path(base_dir))
    if not ok:
        return [dict(world=world, refused=True,
                     reason=f"base run incomplete: {reason}")]
    rt_base = V.build_run_trace(world, "base", Path(base_dir),
                                float(runs[0]["t0_s"]), t_ins_pre={},
                                sched_cache=sched_cache)
    out = []
    for run in runs:
        t1 = time.time()
        try:
            frame = _label_rows_for_run(run, rt_base, sched_cache)
        except Exception as ex:                       # noqa: BLE001
            out.append(dict(world=world, arm=run["arm"], refused=True,
                            reason=f"{type(ex).__name__}: {ex}"))
            _log(f"REFUSED {chunk}/{world}__{run['arm']}: {ex}")
            continue
        out_dir = root / "worlds" / chunk
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / f"{world}__{run['arm']}_labels.parquet"
        tmp = Path(str(f) + ".tmp.parquet")
        frame.to_parquet(tmp, index=False)
        tmp.replace(f)
        rec = dict(world=world, source=run["arm"], chunk_id=chunk,
                   arm_index=run["arm_index"], split="TRAIN",
                   policy="trajectory", t0_s=run["t0_s"],
                   run_dir=str(run["run_dir"]),
                   physics_gz_sha256=D.sha256_file(run["physics"]),
                   n_decision_times=int(frame["t"].nunique()),
                   n_label_rows=int(len(frame)),
                   total_delay_s=float(frame["total_delay_ref_s"].sum()),
                   files={f"worlds/{chunk}/{f.name}": D.sha256_file(f)},
                   built_utc=datetime.now(timezone.utc).isoformat())
        D.atomic_write_json(out_dir / f"{world}__{run['arm']}_done.json", rec)
        out.append(rec)
        _log(f"built {chunk}/{world}__{run['arm']}: rows={len(frame)} "
             f"({time.time() - t1:.1f}s)")
    return out


def verify_done(root: Path, chunk: str, world: str, arm: str):
    p = root / "worlds" / chunk / f"{world}__{arm}_done.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                 # noqa: BLE001
        return None
    rel = f"worlds/{chunk}/{world}__{arm}_labels.parquet"
    sha = (d.get("files") or {}).get(rel)
    f = root / rel
    if not isinstance(sha, str) or not f.is_file() \
            or D.sha256_file(f) != sha:
        return None
    return d


def build(workers: int = 4, limit_worlds: set | None = None) -> dict:
    root = D.LABELS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    (root / "worlds").mkdir(exist_ok=True)
    runs = D.enumerate_runs()
    if limit_worlds:
        runs = [r for r in runs if r["world"] in limit_worlds]

    tasks: dict = {}
    for r in runs:
        tasks.setdefault((r["chunk"], r["world"]), dict(
            base_dir=r["base_dir"], runs=[]))
        tasks[(r["chunk"], r["world"])]["runs"].append(r)

    task_list = []
    results: list = []
    for (chunk, world), v in sorted(tasks.items()):
        todo = []
        for r in v["runs"]:
            d = verify_done(root, chunk, world, r["arm"])
            if d is not None:
                d["resumed"] = True
                results.append(d)
            else:
                todo.append(r)
        if todo:
            task_list.append((str(root), chunk, world, str(v["base_dir"]),
                              todo))

    _log(f"labels: {len(runs)} runs, {len(task_list)} worlds to build "
         f"({len(results)} resumed), workers={workers}")
    if workers > 1 and task_list:
        with Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_world_task,
                                                        task_list, 1), 1):
                results.extend(res)
                if i % 10 == 0 or i == len(task_list):
                    _log(f"  worlds {i}/{len(task_list)}")
    else:
        for i, t in enumerate(task_list, 1):
            results.extend(_world_task(t))
            _log(f"  worlds {i}/{len(task_list)}")

    entries = [r for r in results if not r.get("refused")]
    refused = [r for r in results if r.get("refused")]
    manifest = dict(
        schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        note=("WM-N1-DSPEED trajectory label parquets (v0.4a window_label "
              "conventions; build_v04a_cov labels-only); TRAIN only"),
        tau_s=list(V.TAUS), n_runs=len(entries), n_refused=len(refused),
        runs={f"{e['chunk_id']}/{e['world']}__{e['source']}": e
              for e in entries}, refused=refused)
    D.atomic_write_json(root / "MANIFEST.json", manifest)
    rels = ["MANIFEST.json"] + [
        p.relative_to(root).as_posix()
        for p in sorted((root / "worlds").rglob("*")) if p.is_file()]
    lines = [f"{D.sha256_file(root / rel)}  {rel}" for rel in rels]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                    encoding="utf-8")
    return dict(n_built=len(entries), n_refused=len(refused),
                root=str(root), sha256=_sha_of(root / "MANIFEST.json"))


def _sha_of(p):
    return D.sha256_file(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dspeed.labels")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-worlds", default="")
    args = ap.parse_args(argv)
    limit = {w for w in args.limit_worlds.split(",") if w}
    rep = build(workers=args.workers, limit_worlds=limit or None)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
