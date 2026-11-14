"""HORIZON-CHECK part B: decision accuracy vs horizon, SUMO saved-state branches vs WM imagination.

One SUMO episode per (seed, penetration), native base (IDENTICAL start states for every
controller under test).  At each decision time the SUMO state is saved and, per controller,
the arms ALL / GROUP / ONLY:i (i = up to --max-solo members of the close group G) plus one
shared all-native reference B are run for 120 s each in a fresh SUMO process; the metrics are
read off the SAME branch at 30 / 60 / 120 s (a 30 s branch is the prefix of the 120 s one:
deterministic SUMO + common random numbers).  The WM imagines the same arms from the env's
oracle state (``wms.control.coop_imagine`` semantics, entries off), metrics at the same
horizons, for each WM checkpoint.

Per arm and horizon we store SUMO metrics on (a) all vehicles (the branching metric, incl.
vehicles inserted during the branch; + source waiting) and (b) the t0 cohort only (what the
WM can see without entries), the RSU-field zone aggregates (all / cohort), and for B the
per-vehicle HDV |ds| of the imagined B vs SUMO B.

    python scripts/horizon_check/branch_horizon.py --seed 4121 --penetration 0.2 --out <dir> \
        --controllers random damping <policy.pt> --wm C4=<ckpt> --wm C4R=<ckpt>

SUMO seeds via the C4 seed guard (4107-4148).  CPU only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from traffic_shepherd.c4 import branching as BR

HORIZONS = (30, 60, 120)
COHORT_KEYS = ("reward", "delay", "throughput", "hard_brake", "smooth", "cost_ttc")


# ===================================================================== SUMO side
def zone_aggregates(field: np.ndarray) -> dict:
    """field [N, 6] (density, mean_v, std_v, queue, occupied, cav_share) -> zone scalars."""
    cnt = field[:, 0] * 0.05
    n = float(cnt.sum())
    return {"count": n, "mean_speed": float((cnt * field[:, 1]).sum() / n) if n > 0 else None,
            "queued_cells": float(field[:, 3].sum())}


def sumo_field(env, cohort=None) -> np.ndarray:
    from wms.control.rsu_field import build_rsu_field
    st = env.state()
    st = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st.items()}
    if cohort is not None:
        f = env.frame
        ids = env.builder.vehicle_ids
        keep = np.array([vi >= 0 and ids[vi] in cohort for vi in f.vid_index.tolist()], bool)
        st = dict(st, present=st["present"].bool() & torch.as_tensor(keep)[None])
    return build_rsu_field(st)[0].numpy()


def present_vids(env) -> set:
    f = env.frame
    ids = env.builder.vehicle_ids
    return {ids[int(f.vid_index[s])] for s in np.flatnonzero(f.present).tolist()}


class HorizonTracker(BR.BranchTracker):
    """BranchTracker + t0-cohort team metrics + snapshots at every horizon."""

    def __init__(self, env, cohort: set, *, record: bool):
        super().__init__(env, record_trajectory=record)
        self.cohort = cohort
        self.cohort_sums = {k: 0.0 for k in COHORT_KEYS}
        self.cohort_fair = []
        self.snaps = {}

    def update(self, info: dict) -> None:
        super().update(info)
        env = self.env
        raw = env.history.raw(3)
        ids = env.builder.vehicle_ids
        for j in (0, 1):
            kin0, m0, id0 = raw["kin"][j], raw["mask"][j], raw["vid_index"][j]
            kin1, m1, id1 = raw["kin"][j + 1], raw["mask"][j + 1], raw["vid_index"][j + 1]
            state, nxt, accel, _ = BR.step_pair(kin0, m0, id0, kin1, m1, id1)
            n0 = m0.shape[0]
            coh = np.zeros(state["present"].shape[0], bool)
            coh[:n0] = np.asarray(m0, bool) & np.array(
                [vi >= 0 and ids[int(vi)] in self.cohort for vi in id0.tolist()], bool)
            st = dict(state, present=state["present"] & coh)
            nx = dict(nxt, present=nxt["present"] & coh)
            terms = self.reward_fn(st, nx, np.where(coh, accel, 0.0))
            for k in COHORT_KEYS:
                self.cohort_sums[k] += terms[k]
            self.cohort_fair.append(terms["fairness_delay"])

    def snapshot(self, h: int) -> None:
        m = self.finish()
        keep = ("team_delay_vs", "road_delay_vs", "source_wait_vs", "reward", "throughput_soft",
                "hard_brake", "smooth", "cost_ttc", "fairness_step_mean_s", "fairness_cvar90_s",
                "merge_crossings", "arrivals", "collisions", "teleports", "n_vehicles")
        self.snaps[h] = {
            "all": {k: m[k] for k in keep},
            "cohort": {**{k: float(v) for k, v in self.cohort_sums.items()},
                       "fairness_step_mean_s": float(np.mean(self.cohort_fair))},
            "field_all": sumo_field(self.env),
            "field_cohort": sumo_field(self.env, self.cohort),
            "n_present": len(present_vids(self.env)),
            "n_present_cohort": len(present_vids(self.env) & self.cohort),
        }


SUMO_CPU = {"s": 0.0}


def sumo_child_cpu() -> float:
    """CPU seconds of this process's live SUMO children (measured before each relaunch)."""
    try:
        import psutil
        return sum(sum(c.cpu_times()[:2]) for c in psutil.Process().children(recursive=True)
                   if c.name().lower().startswith("sumo"))
    except Exception:
        return 0.0


def run_arm(env, controller, members, fn, cohort, *, record: bool) -> tuple[dict, list]:
    tracker = HorizonTracker(env, cohort, record=record)
    for step in range(1, max(HORIZONS) + 1):
        accel, kwargs = controller.command(env, env.state(), members, fn)
        _, info = env.step(accel, **kwargs)
        if info["done"]:
            raise RuntimeError("episode ended inside a branch")
        tracker.update(info)
        if step in HORIZONS:
            tracker.snapshot(step)
    SUMO_CPU["s"] += sumo_child_cpu()
    return tracker.snaps, tracker.trajectory


# ===================================================================== WM side
class TimelineCandidate:
    """Damping/PI in imagination: constant action timeline through
    ``wms.train.cav_drive.TimelineControllers`` (restored verbatim from deploy.py at
    1ea2409^; removed from deploy.py on c4/int although coop_imagine still imports it)."""

    def __init__(self, model, controller: str, n_control: int, dependencies: tuple):
        self.model, self.controller = model, controller
        self.n_control, self.dependencies = int(n_control), dependencies
        self.inner = None

    def __call__(self, state):
        if self.inner is None:
            from wms.data.action_timeline import CONTROLLERS as TIMELINE
            from wms.train.cav_drive import TimelineControllers
            b, k = state["s_m"].shape
            ids = torch.full((b, self.n_control, k), TIMELINE.index(self.controller), dtype=torch.long)
            zeros = torch.zeros(b, self.n_control, k)
            self.inner = TimelineControllers(
                self.model, {"controller": ids, "acceleration_mps2": zeros,
                             "target_speed_mps": zeros}, dependencies=self.dependencies)
        return self.inner(state)


_DEPS: list = []


def classic_dependencies() -> tuple:
    """Frozen controller sources with ONE shared (stateless) CIRCLES ONNX session."""
    if not _DEPS:
        from wms.train.cav_drive import load_controller_dependencies
        deps = list(load_controller_dependencies(np.array([1, 2])))
        shared = deps[1](deps[0])
        deps[1] = lambda _config, _shared=shared: _shared
        _DEPS.append(tuple(deps))
    return _DEPS[0]


@torch.no_grad()
def imagine_multi(ckpt, env, arms, controller, fn_snap, *, n_samples=2, seed=0,
                  driver_mode: str = "sample"):
    """``coop_imagine.imagined_cooperation`` with metrics at every horizon in HORIZONS, plus
    the imagined B arm's RSU field and per-vehicle positions (compact slot -> vid)."""
    from wms.control.contract import RewardWeights
    from wms.control.coop_imagine import (METRIC_MAP, SIGN, CallableCavController, _member_slots,
                                          _model)
    from wms.control.deploy import OracleSource, compact_inputs
    from wms.control.reward import team_reward
    from wms.control.rsu_field import build_rsu_field

    started = time.monotonic()
    model = _model(ckpt, "cpu")
    raw, _ = OracleSource(model).inputs(env)
    inputs, keep = compact_inputs(raw)
    state = inputs.pop("state")
    k_full = int(raw["state"]["s_m"].shape[1])
    names = [a["name"] for a in arms]
    n_arm, n = len(arms), int(n_samples)
    batch = n_arm * n
    cav_orig = state["is_cav"][0].bool()
    enc_state = dict(state, is_cav=torch.zeros_like(state["is_cav"]))
    hist = {key: inputs.get(key) for key in ("history", "history_mask", "driver_history",
                                             "driver_history_mask")}
    for h, m in (("history", "history_mask"), ("driver_history", "driver_history_mask")):
        if hist[h] is not None and hist[m] is None:
            hist[m] = torch.ones_like(hist[h][..., 0], dtype=torch.bool)
    model.eval()
    encoded, embedding, params = model.encode_drivers(enc_state, **hist)
    gen = torch.Generator().manual_seed(int(seed))
    drivers = {}
    if driver_mode == "internal":           # let rollout encode drivers itself (scorer path)
        params = embedding = None
    if params is not None:
        pn = tuple(params)
        values = torch.stack([params[p] for p in pn], -1)
        draws = torch.randn((n, *values.shape[1:]), generator=gen).to(values)
        if driver_mode == "mean":
            draws = torch.zeros_like(draws)
        scale = encoded["uncertainty"] if encoded is not None else values.abs() * 0.1
        samples = values + draws * scale
        samples = torch.maximum(torch.minimum(samples, values * 1.5), values * 0.5)
        drivers["driver_params"] = {p: samples.repeat(n_arm, 1, 1)[..., i] for i, p in enumerate(pn)}
    if embedding is not None:
        draws = torch.randn((n, *embedding.shape[1:]), generator=gen).to(embedding)
        drivers["driver_embedding"] = (embedding + 0.05 * draws).repeat(n_arm, 1, 1)

    def rep(x):
        return x.repeat(batch, *([1] * (x.ndim - 1))) if torch.is_tensor(x) and x.ndim else x
    bstate = {key: rep(value) for key, value in state.items()}
    member = torch.stack([_member_slots(env, a["members"], keep) for a in arms])
    bstate["is_cav"] = member.repeat_interleave(n, 0) & bstate["present"].bool()
    bhist = {key: rep(value) for key, value in hist.items()}
    steps = 2 * max(HORIZONS)
    if controller.named is None:
        fns = [copy.deepcopy(fn_snap) for _ in range(batch)]
        ctrl = CallableCavController(fns, keep, k_full, cav_orig[None].repeat(batch, 1))
        dispatch = ctrl
    elif controller.named == "native":
        dispatch = model.hold_teacher
    else:
        name = "damping" if controller.named == "damping" else "pi-external-u"
        dispatch = TimelineCandidate(model, name, max(HORIZONS) + 2, classic_dependencies())
    roll = model.rollout(dict(bstate), n_steps=steps, cav_teacher=None, cav_controller=dispatch,
                         **bhist, **drivers)
    weights = RewardWeights()
    sums = {key: torch.zeros(batch, dtype=torch.float64) for key in METRIC_MAP}
    current = dict(bstate)
    out_h = {}
    b = names.index("B")
    f = env.frame
    ids = env.builder.vehicle_ids
    slot_vid = [ids[int(f.vid_index[s])] for s in keep.tolist()]
    for step in range(steps):
        accel = roll["accel"][:, step]
        nxt = {**current, "s_m": roll["positions"][:, step], "v_mps": roll["speeds"][:, step],
               "lane": roll["lanes"][:, step], "present": roll["masks"][:, step], "a_mps2": accel}
        nxt.pop("veh_kin", None)
        terms = team_reward(current, nxt, accel, weights)
        for key in METRIC_MAP:
            sums[key] += terms[key].double().cpu()
        current = nxt
        if (step + 1) % 2 == 0 and (step + 1) // 2 in HORIZONS:
            h = (step + 1) // 2
            per = {METRIC_MAP[k]: v.reshape(n_arm, n).clone() for k, v in sums.items()}
            fld = build_rsu_field({"s_m": nxt["s_m"], "v_mps": nxt["v_mps"], "lane": nxt["lane"],
                                   "present": nxt["present"], "is_cav": cav_orig[None].repeat(batch, 1)})
            rows_b = slice(b * n, (b + 1) * n)
            out_h[h] = {
                "metrics": {arm: {m: float(per[m][i].mean()) for m in per}
                            for i, arm in enumerate(names)},
                "D": {arm: {m: float((SIGN[m] * (per[m][i] - per[m][b])).mean()) for m in per}
                      for i, arm in enumerate(names)},
                "field_B": fld[rows_b].mean(0).numpy(),
                "zone_by_arm": {arm: zone_aggregates(fld[i * n:(i + 1) * n].mean(0).numpy())
                                for i, arm in enumerate(names)},
                "B_s": {vid: float(nxt["s_m"][rows_b, j].mean()) for j, vid in enumerate(slot_vid)
                        if bool(nxt["present"][b * n, j])},
            }
    return {"h": out_h, "n_samples": n, "slots": int(keep.numel()),
            "wall_s": time.monotonic() - started, "checkpoint": str(ckpt)}


# ===================================================================== driver
def parse_wms(specs, default_samples: int = 2) -> dict:
    """``TAG=path[,mode[,n_samples]]`` -> {tag: (path, mode, n)}.  mode: ``mean`` (driver
    point estimate, = the scorer's internal encoding; default), ``internal`` or ``sample``
    (coop_imagine's N(0, uncertainty) driver draws, clamped to [0.5, 1.5] x estimate)."""
    out = {}
    for spec in specs:
        tag, rest = spec.split("=", 1)
        parts = rest.split(",")
        mode = parts[1] if len(parts) > 1 else "mean"
        n = int(parts[2]) if len(parts) > 2 else (1 if mode in ("mean", "internal")
                                                   else default_samples)
        out[tag] = (parts[0], mode, n)
    return out


def jsonable(x):
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), 3).tolist()
    return BR._jsonable(x) if not isinstance(x, dict) else {str(k): jsonable(v) for k, v in x.items()}


def main() -> int:
    from traffic_shepherd.c4.env import CavSumoEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--penetration", type=float, required=True)
    ap.add_argument("--demand", type=int, default=2700)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--times", default="360,480")
    ap.add_argument("--controllers", nargs="+", default=["random", "damping"])
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--wm", action="append", default=[], help="TAG=ckpt[,mode[,n_samples]]")
    ap.add_argument("--wm-samples", type=int, default=2)
    ap.add_argument("--max-solo", type=int, default=3)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = BR.checked(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    from traffic_shepherd.c4.seeds import check_case
    check_case(args.seed, args.demand)          # C4 seed guard (4107-4148); env.reset re-checks
    wms = parse_wms(args.wm, args.wm_samples)
    labels = args.labels or args.controllers
    times = [int(t) for t in args.times.split(",")]
    rng = np.random.default_rng(args.rng_seed + args.seed)
    (out / "MANIFEST.json").write_text(json.dumps({"schema": "c4_horizon_check_B/1",
                                                   "args": vars(args), "horizons_s": HORIZONS},
                                                  indent=1), encoding="utf-8")
    env = CavSumoEnv(k_slots=192, controller="direct", extra_sumo_cli=BR.SAVE_STATE_CLI,
                     scenario_root=out / "scenarios", physics_log=False)
    native = BR.BranchController("native")
    ctrls = [BR.BranchController(c, seed=args.rng_seed + args.seed, name=lab)
             for c, lab in zip(args.controllers, labels)]
    end = times[-1] + max(HORIZONS) + 1
    sumo_s = 0.0
    t_start = time.monotonic()
    env.reset(seed=args.seed, demand_vph=args.demand, penetration=args.penetration,
              warmup_s=args.warmup, horizon_s=end - args.warmup, controller="direct")
    sumo_s += time.monotonic() - t_start
    fp = open(out / "points.jsonl", "w", encoding="utf-8")
    try:
        for k, t_k in enumerate(times):
            t0 = time.monotonic()
            while env.now < t_k - 1e-9:
                BR.base_step(env, native, None)
            sumo_s += time.monotonic() - t0
            SUMO_CPU["s"] += sumo_child_cpu()      # base episode up to t_k (process reused)
            path = out / "states" / f"s{args.seed}_t{t_k}.xml"
            path.parent.mkdir(parents=True, exist_ok=True)
            BR.save_state(env, path)
            snap = BR.snapshot_env(env)
            cavs = BR.cav_positions(env)
            group = BR.select_group(cavs)
            pool = list(group) if group else sorted(cavs)
            solo = sorted(rng.choice(pool, size=min(args.max_solo, len(pool)), replace=False).tolist()) \
                if pool else []
            cohort = present_vids(env)
            field0_all = sumo_field(env)
            record = {"seed": args.seed, "penetration": args.penetration, "demand": args.demand,
                      "t_s": t_k, "n_cavs": len(cavs), "group": group, "solo": solo,
                      "n_cohort": len(cohort), "field_t0": field0_all,
                      "zone_t0": zone_aggregates(field0_all), "controllers": {}}
            # arms per controller (B shared)
            arm_sets = {}
            for c in ctrls:
                arms = [{"name": "B", "members": frozenset()},
                        {"name": "ALL", "members": None}]
                if group:
                    arms.append({"name": "GROUP", "members": frozenset(group)})
                arms += [{"name": f"ONLY:{v}", "members": frozenset([v])} for v in solo]
                arm_sets[c.name] = arms
            # imagination first (env sits at t_k)
            fn_snaps = {c.name: c.snapshot() for c in ctrls}
            imag = {}
            for c in ctrls:
                for tag, (ckpt, mode, n_s) in wms.items():
                    try:
                        imag[(c.name, tag)] = imagine_multi(ckpt, env, arm_sets[c.name], c,
                                                            fn_snaps[c.name], n_samples=n_s,
                                                            seed=0, driver_mode=mode)
                    except Exception as exc:  # keep the episode alive
                        import traceback
                        imag[(c.name, tag)] = {"error": f"{type(exc).__name__}: {exc} "
                                                        f"{traceback.format_exc()[-600:]}"}
                    print(f"seed {args.seed} t {t_k} imagine {c.name} {tag}: "
                          f"{imag[(c.name, tag)].get('wall_s', -1):.0f} s", flush=True)
            # SUMO branches
            sumo_runs = {}
            t_b = time.monotonic()
            BR.load_state(env, path, snap)
            BR.prepare_branch_mode(env)
            snaps_b, traj_b = run_arm(env, native, frozenset(), None, cohort, record=True)
            sumo_runs["B"] = snaps_b
            for c in ctrls:
                for arm in arm_sets[c.name][1:]:
                    BR.load_state(env, path, snap)
                    BR.prepare_branch_mode(env)
                    fn = None if fn_snaps[c.name] is None else copy.deepcopy(fn_snaps[c.name])
                    s_, _ = run_arm(env, c, arm["members"], fn, cohort, record=False)
                    sumo_runs[(c.name, arm["name"])] = s_
            sumo_s += time.monotonic() - t_b
            # per-vehicle |ds| of imagined B vs SUMO B (HDVs of the t0 cohort)
            traj = dict(traj_b)
            cav_ids = set(cavs)
            ds_b = {}
            for (cname, tag), im in imag.items():
                if "error" in im:
                    continue
                for h in HORIZONS:
                    row = traj.get(round(t_k + h, 3), {})
                    d = [abs(im["h"][h]["B_s"][v] - row[v][0]) for v in im["h"][h]["B_s"]
                         if v in row and v not in cav_ids]
                    ds_b.setdefault(tag, {}).setdefault(cname, {})[h] = (
                        float(np.median(d)) if d else None)
            record["B_hdv_abs_ds_median_m"] = ds_b
            record["sumo_B"] = {h: {kk: vv for kk, vv in snaps_b[h].items()} for h in HORIZONS}
            for h in HORIZONS:
                record["sumo_B"][h]["zone_all"] = zone_aggregates(snaps_b[h]["field_all"])
                record["sumo_B"][h]["zone_cohort"] = zone_aggregates(snaps_b[h]["field_cohort"])
            for c in ctrls:
                blk = {"arms": [a["name"] for a in arm_sets[c.name]], "sumo": {}, "imagined": {}}
                for arm in arm_sets[c.name][1:]:
                    sr = sumo_runs[(c.name, arm["name"])]
                    blk["sumo"][arm["name"]] = {h: {"all": sr[h]["all"], "cohort": sr[h]["cohort"],
                                                    "zone_all": zone_aggregates(sr[h]["field_all"]),
                                                    "zone_cohort": zone_aggregates(sr[h]["field_cohort"])}
                                                for h in HORIZONS}
                for tag in wms:
                    im = imag[(c.name, tag)]
                    if "error" in im:
                        blk["imagined"][tag] = im
                        continue
                    blk["imagined"][tag] = {
                        "wall_s": im["wall_s"], "slots": im["slots"],
                        "h": {h: {"metrics": im["h"][h]["metrics"], "D": im["h"][h]["D"],
                                  "zone_by_arm": im["h"][h]["zone_by_arm"],
                                  "field_B": im["h"][h]["field_B"]} for h in HORIZONS}}
                record["controllers"][c.name] = blk
            record["sumo_wall_s_point"] = time.monotonic() - t_b
            fp.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
            fp.flush()
            print(f"seed {args.seed} t {t_k}: G={group} solo={solo} SUMO branches "
                  f"{time.monotonic() - t_b:.0f} s", flush=True)
            if k + 1 < len(times):
                BR.load_state(env, path, snap)
    finally:
        fp.close()
        env.close()
    (out / "DONE.json").write_text(json.dumps({"sumo_active_s": sumo_s,
                                               "sumo_process_cpu_s_approx": SUMO_CPU["s"],
                                               "total_s": time.monotonic() - t_start}),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
