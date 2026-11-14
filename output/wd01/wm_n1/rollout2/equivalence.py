"""P3 G1 zero-residual equivalence: batched Idea A rollout vs the scalar
Part A v2g engine (owner pre-flight gate G1, PREFLIGHT_GATES_IDEA_A.md).

With ``residual_enabled=False`` and ``arrivals_enabled=False``, the P3
rollout in ``integrator="v2g"`` mode must reproduce the READ-ONLY
``partA_v2g.engine`` rollout on the same window, per vehicle per tick,
within 1e-6 (the port is exact, so the observed error is 0.0).  The
integrator switch exists because the v2g engine uses the ballistic rule
``s + v_new*dt`` (engine.py:661) while the Idea A default is the D1
trapezoid (see rollout2/validity.py: the per-step difference is
``0.5*(v - v_new)*dt``); ``"v2g"`` mode compares like for like.

Window selection (TRAIN split checked FIRST; HOLDOUT/CAL/chunk-5 are
never touched): J4-package TRAIN runs (``campaign_walls_pilot_chunk4``,
splits from the J4 chunk-4 ``WORLD_PLAN.json`` via the wm_n1 inventory)
are preferred, then chunk-4 TRAIN ``native_base`` runs.  The J4 package
has no ``native_base`` arm (its runs are the wall arms ``W-*``), so ANY
complete non-smoke arm is accepted there — the zero-residual G1 window
table is built with the registered eval_a2 machinery
(``partA_v2g.eval_a`` NativeLog + ``tracked_state``, read-only) from the
run's own physics rows regardless of arm; only the generic chunk-4
fallback still requires ``native_base``.  ``--n-windows`` (default 2)
runs that many distinct ``(world, t0)`` TRAIN windows and aggregates the
per-window max errors.  When no run data is available under
``--data-root`` (the wmn1p3 workspace carries no physics logs), the tool
falls back to a deterministic synthetic J4-world scene and LABELS the
report ``source=synthetic_fallback``.

Usage (from ``output/wd01/wm_n1``):

    python -m rollout2.equivalence                 # 2 auto J4 TRAIN windows
    python rollout2/equivalence.py --data-root E:/research-dense --n-steps 240
    python rollout2/equivalence.py --vehicles-json TAB.json
    python rollout2/equivalence.py --synthetic --n-steps 240

Exit code 0 = PASS (max position/speed error <= tolerance), 1 = FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_WM_N1 = _HERE.parent
_WD01 = _WM_N1.parent
for _p in (str(_WM_N1), str(_WD01)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rollout2.engine import IdeaABundle, Rollout2Config, Rollout2Engine  # noqa: E402
from rollout2.state import Rollout2State, WorldState  # noqa: E402
from partA_v2g.engine import PartAV2GEngine, load_params  # noqa: E402  read-only

TOLERANCE = 1e-6
DT = 0.5
# slot width is a format parameter (amendment 5): 128 = v2, 192 = v3; the
# synthetic defaults here keep the v2 width while every real rollout takes K
# from its tensors (WorldState.slots).
K_SLOTS = 128


def assert_not_d(path) -> None:
    """Refuse drive-D paths (WD_FORBID_D convention; the D: junctions
    are retired)."""
    s = str(path)
    if len(s) >= 2 and s[1] == ":" and s[0] in "Dd":
        raise PermissionError(
            f"forbid-D: {s!r} (D: junctions are retired)")


# ------------------------------------------------------------- movers
class NativeMover:
    """The native command contract: never issues a command (NaN target,
    reason 1 = 'selected_native')."""

    def init_state(self, batch, slots=K_SLOTS, *, device=None):
        return {"batch": batch}

    def step(self, state, inp, actions, *, wall_speed_mps=None):
        from ctrl.mover import CavStepOutput

        target = torch.full_like(inp.s_m, float("nan"), dtype=torch.float64)
        return CavStepOutput(
            target_speed_mps=target,
            accel_cmd_mps2=torch.full_like(inp.s_m, float("nan"),
                                           dtype=torch.float64),
            reason_code=torch.ones_like(inp.lane), state=state)


class ReplayMover:
    """Replays recorded per-control-tick commands {t: {vid: target}}."""

    def __init__(self, commands: Dict[float, Dict], names: List[str]) -> None:
        self.by_slot = {}
        for t, cmds in commands.items():
            slots = {}
            for key, val in cmds.items():
                slots[names.index(key) if key in names else int(key)] = \
                    float(val)
            self.by_slot[round(float(t), 6)] = slots
        self._last: Dict[int, float] = {}

    def init_state(self, batch, slots=K_SLOTS, *, device=None):
        return {}

    def step(self, state, inp, actions, *, wall_speed_mps=None):
        from ctrl.mover import CavStepOutput

        t = round(float(inp.control_t_s if inp.control_t_s is not None
                        else inp.t_s), 6)
        for slot, val in self.by_slot.get(t, {}).items():
            self._last[slot] = val
        target = torch.full_like(inp.s_m, float("nan"), dtype=torch.float64)
        for slot, val in self._last.items():
            target[:, slot] = val
        reason = torch.where(torch.isfinite(target),
                             torch.zeros_like(inp.lane),
                             torch.ones_like(inp.lane))
        return CavStepOutput(
            target_speed_mps=target,
            accel_cmd_mps2=torch.full_like(inp.s_m, float("nan"),
                                           dtype=torch.float64),
            reason_code=reason, state=state)


# ------------------------------------------------------- world helpers
def world_from_table(vehicles: Dict[str, dict], *,
                     dtype=torch.float64) -> Tuple[WorldState, List[str]]:
    names = sorted(vehicles)
    b_n, k_n = 1, max(16, len(names))
    s = torch.zeros((b_n, k_n), dtype=dtype)
    v = torch.zeros((b_n, k_n), dtype=dtype)
    a = torch.zeros((b_n, k_n), dtype=dtype)
    lane = torch.full((b_n, k_n), -1, dtype=torch.int64)
    length = torch.full((b_n, k_n), 5.0, dtype=dtype)
    present = torch.zeros((b_n, k_n), dtype=torch.bool)
    is_cav = torch.zeros((b_n, k_n), dtype=torch.bool)
    for i, vid in enumerate(names):
        rec = vehicles[vid]
        s[0, i] = rec["s"]
        v[0, i] = rec.get("v", 0.0)
        a[0, i] = rec.get("a", 0.0)
        lane[0, i] = int(rec["lane"])
        length[0, i] = rec.get("length_m", 5.0)
        present[0, i] = True
        is_cav[0, i] = bool(rec.get("is_cav", False))
    return WorldState(
        s_m=s, v_mps=v, a_mps2=a, lane=lane, length_m=length,
        present=present, is_cav=is_cav,
        wall_group=torch.full((b_n, k_n), -1, dtype=torch.int64),
        slot_id=torch.arange(k_n).expand(b_n, k_n).contiguous()), names


def run_batched(params: dict, vehicles: Dict[str, dict], *, t0: float,
                n_steps: int, commands: Optional[Dict] = None,
                integrator: str = "v2g"):
    world, names = world_from_table(vehicles)
    if commands:
        mover = ReplayMover(commands, names)
    else:
        # REAL P1 CavMover (native action -> no command, exactly like the
        # recorded-native G1 window).  ``NativeMover`` remains only a
        # test/no-command interface stand-in.
        from ctrl import CavMover, CtrlConfig

        mover = CavMover(CtrlConfig())
    bundle = IdeaABundle(ctrl=mover, hdv=None, arrival=None, corr=None)
    cfg = Rollout2Config(params=params, integrator=integrator,
                         residual_enabled=False, arrivals_enabled=False,
                         t0_s=t0)
    cfg.state = Rollout2State(world.batch, world.slots)
    engine = Rollout2Engine(bundle, cfg, seed=0)
    out = engine.run(world, None, n_steps=n_steps)
    return out, names


def run_scalar(params: dict, vehicles: Dict[str, dict], *, t0: float,
               n_steps: int, commands: Optional[Dict] = None,
               lane_changes_enabled: bool = True) -> PartAV2GEngine:
    hook = None
    if commands:
        _names = sorted(vehicles)

        def hook(t_now, _eng, _n=_names, _c=commands):
            cmds = _c.get(round(float(t_now), 6), {})
            return {(_n[int(k)] if k not in _n else k): float(x)
                    for k, x in cmds.items()}
    eng = PartAV2GEngine(params, t0=t0, vehicles=vehicles,
                         control_hook=hook,
                         lane_changes_enabled=lane_changes_enabled)
    eng.run(int(n_steps))
    return eng


def compare(scalar_eng: PartAV2GEngine, rollouts, names: List[str],
            n_steps: int) -> Tuple[float, float, int, Dict[str, float]]:
    """Per vehicle per tick max |ds|, |dv| over the ticks where the
    scalar snapshot still carries the vehicle (pre-exit)."""
    pos = rollouts.positions[0, :, 0]
    spd = rollouts.speeds[0, :, 0]
    lane = rollouts.lanes[0, :, 0]
    worst: Dict[str, float] = {}
    max_s = max_v = 0.0
    n = 0
    for j in range(n_steps + 1):
        snap = scalar_eng.snapshots.get(j, {})
        for i, vid in enumerate(names):
            ref = snap.get(vid)
            if ref is None:
                continue
            ds = abs(float(pos[j, i]) - float(ref[1]))
            dv = abs(float(spd[j, i]) - float(ref[2]))
            assert int(lane[j, i]) == int(ref[0]), \
                f"lane mismatch {vid} step {j}"
            max_s = max(max_s, ds)
            max_v = max(max_v, dv)
            worst[vid] = max(worst.get(vid, 0.0), ds, dv)
            n += 1
    return max_s, max_v, n, worst


# -------------------------------------------------- J4 TRAIN selection
def select_j4_train_windows(inventory: Path, data_root: Path, *,
                            limit: int = 1, world: str = "") -> List[dict]:
    """Up to ``limit`` usable J4 TRAIN windows (TRAIN split FIRST; then
    the J4 walls-pilot package ``campaign_walls_pilot_chunk4`` — any
    complete non-smoke arm, since the J4 package has no ``native_base``
    arm — before generic chunk-4 ``native_base``; HOLDOUT/CAL are never
    selected).  Windows are distinct ``(world, t0_s)`` pairs."""
    import pandas as pd

    assert_not_d(inventory)
    assert_not_d(data_root)
    df = pd.read_parquet(inventory)
    if "run" in df:
        df = df[df["run"].notna()]
    if "is_smoke" in df:
        df = df[~df["is_smoke"].astype(bool)]
    df = df[df["is_complete"].astype(bool)] if "is_complete" in df else df
    j4 = df[df["root_rel"].str.contains("walls_pilot", na=False)]
    c4 = df[(df["root_rel"].str.contains("chunk4", na=False))
            & ~df["root_rel"].str.contains("walls_pilot", na=False)]
    if "arm" in c4:                     # fallback preference only
        c4 = c4[c4["arm"] == "native_base"]
    out: List[dict] = []
    seen = set()
    for frame in (j4, c4):
        for split in ("TRAIN",):   # TRAIN split checked first; nothing else
            rows = frame[frame["split"] == split] if "split" in frame \
                else frame
            for _, row in rows.iterrows():
                phys = (data_root / str(row["root_rel"]) / str(row["run"])
                        / "run" / "physics_steps.jsonl")
                if not phys.exists():
                    continue
                if world and str(row["world"]) != world:
                    continue
                key = (str(row["world"]), float(row.get("t0_s", 0.0) or 0.0))
                if key in seen:      # one window per distinct (world, t0)
                    continue
                seen.add(key)
                out.append(dict(world=str(row["world"]),
                                root_rel=str(row["root_rel"]),
                                run=str(row["run"]),
                                split=str(row.get("split", "TRAIN")),
                                physics=phys,
                                t0_s=float(row.get("t0_s", 0.0) or 0.0)))
                if len(out) >= max(1, int(limit)):
                    return out
    return out


def window_table_from_run(entry: dict, w: float,
                          params: dict) -> Tuple[Dict[str, dict], set]:
    """The registered eval_a2 window initial state (zone-restricted
    tracked vehicles, CAV flags from the run's own physics rows)."""
    from partA_v2g import eval_a as EA

    EA.MAPPER = EA.ModelMapper()
    log = EA.NativeLog(dict(world=entry["world"], t0_s=entry["t0_s"],
                            native_run=entry["physics"]))
    if not (w > log.t0 and w + 1e-9 <= log.t_end):
        raise ValueError(f"window {w} outside log [{log.t0}, {log.t_end}]")
    truth_w = log.truth_at(w)
    j0 = int(round((w - log.t0) / DT))
    vehicles, _free = EA.tracked_state(truth_w, params,
                                       inj_at=log.inj.get(j0))
    assert vehicles, f"empty zone state at w={w}"
    for vid, rec in vehicles.items():
        rec["is_cav"] = vid in log.cav_ids
    return vehicles, set(log.cav_ids)


def synthetic_j4_scene(seed: int = 0) -> Dict[str, dict]:
    """Deterministic synthetic J4-world fallback scene (documented in
    the report; NOT a data window)."""
    rng = np.random.default_rng(seed)
    rows = []
    s = 2340.0
    for i in range(4):                     # main_right platoon
        rows.append((1, s, 28.5 + 0.4 * rng.random(), i == 0))
        s += 28.0 + 6.0 * rng.random()
    rows.append((0, 2400.0, 29.0, False))  # main_left pair
    rows.append((0, 2320.0, 28.0, False))
    rows.append((2, 392.0, 21.5, False))   # accel lane, merge zone
    rows.append((2, 428.0, 19.5, False))
    out = {}
    for i, (ln, x, vv, cav) in enumerate(rows):
        out[f"veh{i:03d}"] = dict(lane=ln, s=x, v=vv, is_cav=cav)
    return out


# ----------------------------------------------------------------- main
def run_equivalence(params: dict, vehicles: Dict[str, dict], *, t0: float,
                    n_steps: int, commands: Optional[Dict] = None,
                    integrator: str = "v2g") -> dict:
    t_start = time.perf_counter()
    scalar_eng = run_scalar(params, vehicles, t0=t0, n_steps=n_steps,
                            commands=commands)
    rollouts, names = run_batched(params, vehicles, t0=t0,
                                  n_steps=n_steps, commands=commands,
                                  integrator=integrator)
    max_s, max_v, n_pairs, worst = compare(scalar_eng, rollouts, names,
                                           n_steps)
    ok = max(max_s, max_v) <= TOLERANCE
    return dict(
        status="pass" if ok else "fail",
        integrator=integrator,
        residual="0", arrivals="off",
        n_vehicles=len(names), n_cav=sum(
            1 for v in vehicles.values() if v.get("is_cav")),
        n_steps=int(n_steps), t0_s=float(t0),
        n_compared_pairs=int(n_pairs),
        max_abs_position_err=float(max_s),
        max_abs_speed_err=float(max_v),
        tolerance=TOLERANCE,
        worst_vehicles=dict(sorted(worst.items(),
                                   key=lambda kv: -kv[1])[:5]),
        scalar_collisions=int(scalar_eng.collisions),
        scalar_lane_changes=dict(scalar_eng.lc_counts),
        wall_s=round(time.perf_counter() - t_start, 2),
    )


def _default_data_root() -> Path:
    """Data root for the G1 gate: ``WM_N1_DATA_ROOT`` env, else the inventory's
    recorded ``data_root``, else the workspace root."""
    env = os.environ.get("WM_N1_DATA_ROOT") or os.environ.get(
        "PREFLIGHT_DATA_ROOT")
    if env:
        return Path(env)
    summary = _WM_N1 / "inventory" / "summary.json"
    if summary.is_file():
        try:
            root = json.loads(summary.read_text(encoding="utf-8")).get(
                "data_root")
            if root:
                return Path(root)
        except (OSError, ValueError):
            pass
    return _WD01.parents[1]


def zero_residual_report(*, data_root=None, n_steps: int = 240,
                         window_offset_s: float = 30.0,
                         world: Optional[str] = None) -> dict:
    """G1 pre-flight entry point (owner ``PREFLIGHT_GATES_IDEA_A.md``).

    Runs the P3 batched Idea A rollout with the HDV residual forced to zero and
    arrivals off, in ``integrator="v2g"`` mode, against the read-only
    ``partA_v2g.engine`` scalar rollout on a real J4/chunk-4 TRAIN window
    (TRAIN split checked first; HOLDOUT/CAL/chunk-5 are never read).  Returns
    the gate-shaped report.  A labelled synthetic fallback is used only when no
    TRAIN run data is available under the data root.
    """
    inv = _WM_N1 / "inventory" / "inventory.parquet"
    root = Path(data_root) if data_root else _default_data_root()
    params = load_params()
    source, note, entry = "synthetic_fallback", None, None
    try:
        assert_not_d(root)
        candidates = select_j4_train_windows(inv, root, limit=1,
                                             world=world or "")
        if not candidates:
            raise FileNotFoundError(
                f"no usable J4/chunk-4 TRAIN native_base/wall-arm run under "
                f"{root}")
        entry = candidates[0]
        t0 = entry["t0_s"] + float(window_offset_s)
        vehicles, _cavs = window_table_from_run(entry, t0, params)
        source = "j4_train_inventory"
    except (FileNotFoundError, ValueError, ImportError, OSError,
            PermissionError) as exc:
        vehicles = synthetic_j4_scene(seed=0)
        t0 = 30.0
        note = (f"J4 TRAIN run data unavailable ({type(exc).__name__}: {exc}); "
                "deterministic synthetic J4-world scene, documented fallback "
                "- NOT a data window")
    result = run_equivalence(params, vehicles, t0=t0, n_steps=int(n_steps),
                             integrator="v2g")
    windows = [f"{entry['world']}@{t0:.1f}" if entry is not None
               else f"synthetic_j4_world@{t0:.1f}"]
    return {
        "ok": result["status"] == "pass",
        "windows": windows,
        "max_position_error_m": float(result["max_abs_position_err"]),
        "max_speed_error_mps": float(result["max_abs_speed_err"]),
        "integrator": "v2g",
        "tolerance": TOLERANCE,
        "tolerance_explained": None,
        "source": source,
        "note": note,
        "n_compared_pairs": int(result["n_compared_pairs"]),
        "n_steps": int(n_steps),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="G1 zero-residual equivalence (rollout2 vs partA_v2g)")
    ap.add_argument("--inventory", type=Path,
                    default=_WM_N1 / "inventory" / "inventory.parquet")
    ap.add_argument("--data-root", type=Path, default=_WD01.parents[1],
                    help="root containing the runs/ tree (default: the "
                         "workspace root; no external data is read)")
    ap.add_argument("--world", type=str, default="",
                    help="restrict the J4 TRAIN selection to this world")
    ap.add_argument("--window-offset-s", type=float, default=30.0,
                    help="window start = t0 + offset (eval_a2 convention)")
    ap.add_argument("--n-steps", type=int, default=240,
                    help="rollout length in 0.5 s ticks (240 = H120)")
    ap.add_argument("--vehicles-json", type=Path, default=None,
                    help="explicit window table {t0, vehicles, commands?}")
    ap.add_argument("--commands-json", type=Path, default=None,
                    help="recorded commands {t: {vid: target_mps}}")
    ap.add_argument("--n-windows", type=int, default=2,
                    help="number of distinct J4 TRAIN windows to run "
                         "(default 2; data mode only)")
    ap.add_argument("--synthetic", action="store_true",
                    help="force the labelled synthetic J4-world fallback")
    ap.add_argument("--integrator", choices=("v2g", "d1"), default="v2g")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the JSON report here")
    args = ap.parse_args(argv)

    params = load_params()
    report: dict = dict(gate="G1 zero-residual",
                        comparator="partA_v2g.engine (read-only)",
                        param_set="partA_v2g/params_nominal.json",
                        source=None, world=None, note=None)
    commands = None
    if args.commands_json:
        assert_not_d(args.commands_json)
        raw = json.loads(args.commands_json.read_text(encoding="utf-8"))
        commands = {float(t): {vid: float(x) for vid, x in cmds.items()}
                    for t, cmds in raw.items()}
    if args.vehicles_json:
        assert_not_d(args.vehicles_json)
        tab = json.loads(args.vehicles_json.read_text(encoding="utf-8"))
        vehicles = tab["vehicles"]
        t0 = float(tab.get("t0", args.window_offset_s))
        n_steps = int(tab.get("n_steps", args.n_steps))
        commands = commands or tab.get("commands")
        report.update(source="explicit_table", world=tab.get("world"))
        result = run_equivalence(params, vehicles, t0=t0, n_steps=n_steps,
                                 commands=commands, integrator=args.integrator)
        report.update(result)
        report["windows"] = [dict(world=report["world"], **{
            k: result[k] for k in ("status", "max_abs_position_err",
                                   "max_abs_speed_err", "n_compared_pairs")})]
    elif not args.synthetic:
        entries = select_j4_train_windows(
            args.inventory, args.data_root, limit=args.n_windows,
            world=args.world)
        if not entries:
            raise FileNotFoundError(
                f"no usable J4 TRAIN native_base/wall-arm run under "
                f"--data-root {str(args.data_root)!r}")
        n_steps = args.n_steps
        window_rows = []
        agg_s = agg_v = 0.0
        for entry in entries:
            t0 = entry["t0_s"] + args.window_offset_s
            row = dict(world=entry["world"], run=entry["run"],
                       split=entry["split"], t0_s=entry["t0_s"],
                       window_t0_s=t0, physics=str(entry["physics"]))
            try:
                vehicles, _cavs = window_table_from_run(entry, t0, params)
                res = run_equivalence(params, vehicles, t0=t0,
                                      n_steps=n_steps, commands=commands,
                                      integrator=args.integrator)
                row.update({k: res[k] for k in
                            ("status", "n_vehicles", "n_cav",
                             "max_abs_position_err", "max_abs_speed_err",
                             "n_compared_pairs", "worst_vehicles",
                             "scalar_collisions", "scalar_lane_changes",
                             "wall_s")})
            except (ValueError, KeyError, OSError) as exc:
                row.update(status="error", error=f"{type(exc).__name__}: "
                                                  f"{exc}")
            window_rows.append(row)
            if row["status"] == "pass":
                agg_s = max(agg_s, float(row["max_abs_position_err"]))
                agg_v = max(agg_v, float(row["max_abs_speed_err"]))
        ok_all = all(r["status"] == "pass" for r in window_rows)
        report.update(
            source="j4_train_inventory",
            world=",".join(r["world"] for r in window_rows),
            run=",".join(r["run"] for r in window_rows),
            split="TRAIN",
            n_windows=len(window_rows),
            windows=window_rows,
            max_abs_position_err=agg_s,
            max_abs_speed_err=agg_v,
            n_compared_pairs=int(sum(r.get("n_compared_pairs", 0)
                                     for r in window_rows)),
            tolerance=TOLERANCE,
            status="pass" if ok_all else "fail",
            integrator=args.integrator,
            residual="0", arrivals="off",
            n_steps=int(n_steps),
            wall_s=round(sum(r.get("wall_s", 0.0) for r in window_rows), 2)
            if all("wall_s" in r for r in window_rows) else None)
    else:
        vehicles = synthetic_j4_scene(seed=0)
        t0 = 30.0
        n_steps = args.n_steps
        report.update(
            source="synthetic_fallback", world="synthetic_j4_world",
            note="--synthetic requested; deterministic synthetic J4-world "
                 "scene, documented fallback - NOT a data window")
        result = run_equivalence(params, vehicles, t0=t0, n_steps=n_steps,
                                 commands=commands, integrator=args.integrator)
        report.update(result)
        report["windows"] = [dict(world="synthetic_j4_world", **{
            k: result[k] for k in ("status", "max_abs_position_err",
                                   "max_abs_speed_err", "n_compared_pairs")})]
    if report["source"] == "synthetic_fallback" and \
            report["status"] == "pass" and not commands:
        # supplementary like-for-like evidence on the labelled fallback
        # scene: (i) replayed CAV commands (the execution-rule port),
        # (ii) the documented d1-vs-v2g integrator difference.
        veh_cmd = synthetic_j4_scene(seed=0)
        n_cmd = sorted(veh_cmd)
        veh_cmd[n_cmd[0]]["is_cav"] = True
        veh_cmd[n_cmd[3]]["is_cav"] = True
        sched = {round(t0 + 0.5 * k, 6): {0: 20.0 if (k // 2) % 2 == 0
                                          else 29.0, 3: 25.0}
                 for k in range(0, n_steps, 2)}
        cmd_res = run_equivalence(params, veh_cmd, t0=t0,
                                  n_steps=n_steps, commands=sched,
                                  integrator="v2g")
        report["commanded_check"] = dict(
            status=cmd_res["status"],
            max_abs_position_err=cmd_res["max_abs_position_err"],
            max_abs_speed_err=cmd_res["max_abs_speed_err"],
            n_cav=cmd_res["n_cav"])
        out_v2g, _ = run_batched(params, vehicles, t0=t0,
                                 n_steps=n_steps, integrator="v2g")
        out_d1, _ = run_batched(params, vehicles, t0=t0,
                                n_steps=n_steps, integrator="d1")
        ds = (out_d1.positions[0, :, 0, :]
              - out_v2g.positions[0, :, 0, :]).abs()
        m = out_v2g.masks[0, :, 0, :].to(torch.bool)
        report["integrator_difference"] = dict(
            rule="s_d1 - s_v2g = 0.5*(v - v_new)*dt per step",
            max_single_step_abs_m=float(
                (out_d1.positions[0, 1, 0, :] - out_v2g.positions[0, 1, 0,
                                                                :]).abs()
                [m[1]].max()),
            window_mean_abs_m=float(ds[m].mean()),
            window_max_abs_m=float(ds[m].max()))
    ok = report["status"] == "pass"
    print(json.dumps(report, indent=1))
    print("G1", "PASS" if ok else "FAIL",
          f"max_pos_err={report['max_abs_position_err']:.3e} "
          f"max_speed_err={report['max_abs_speed_err']:.3e} "
          f"(tol {TOLERANCE:g}) source={report['source']}")
    if args.out:
        assert_not_d(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1),
                            encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
