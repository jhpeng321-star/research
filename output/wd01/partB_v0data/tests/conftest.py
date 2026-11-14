"""Shared fixtures for the partB_v0data tests."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
for p in (HERE, HERE.parent / "partA_data", HERE.parent / "part4"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import v0lib as V  # noqa: E402
from delay import core  # noqa: E402
from delay.geometry import make_geometry  # noqa: E402

T0 = 240.0                       # synthetic branch point
DATA_END = 420.0                 # synthetic data end (short runs)
DT = 0.5

L_PRE_R = "1191554636_0"         # physical lane 1 (main_right), pre-merge
L_PRE_L = "1191554636_1"         # physical lane 0
L_MERGE_R = "1487606179_1"
J_MAIN = ":13638027693_0_0"      # junction mainline pre->merge (boundary s)
L_RAMP = "11553048#1_0"          # physical lane 2 (accel)
J_RAMP = ":1780739017_0_0"       # junction ramp->merge (boundary s)
L_MERGE_A = "1487606179_0"


def _row(t, vehicles, departed=(), cav_ids=(), colliding=(), speed_mode=31):
    return dict(time_s=t, live_ids=[v[0] for v in vehicles],
                departed_ids=list(departed), arrived_ids=[],
                colliding_ids=list(colliding), cav_ids=list(cav_ids),
                speed_modes={}, cav_actions={},
                boundary_crossings={"mainline_entry": 0, "mainline_exit": 0,
                                    "ramp_entry": 0},
                vehicles={v[0]: dict(road_id=v[1], lane_id=v[1],
                                     lane_position_m=v[2],
                                     speed_mps=v[3],
                                     acceleration_mps2=v[4],
                                     type_id="normal_car",
                                     speed_mode=speed_mode,
                                     lane_changed=False)
                          for v in vehicles})


def synth_physics_rows(speed_mode=31):
    """Deterministic tiny trajectory set (real lane ids so pdl maps them):
    a: mainline right lane at 20 m/s, holds at s=2600;
    b: inserted at t=101 behind a at 10 m/s;
    c: lane change main_right -> main_left at t=100 (within the edge);
    d: crosses the mainline junction at t=100 (boundary-continuous);
    e: merges from the ramp at t=90 (boundary-continuous).
    f is scheduled at 295 s but never inserted (queued at the horizon).
    """
    g = V.wm30_geometry()
    b_main = float(g.segments[1][1].s0)      # main_right pre->merge boundary
    b_ramp = float(g.segments[2][1].s0)      # accel ramp->merge boundary
    rows = []
    n = int(round(DATA_END / DT))
    seen: set = set()
    for k in range(1, n + 1):
        t = k * DT
        veh, dep = [], []
        if t <= 250.0:                      # a exits the log at t=250
            s = min(100.0 + 20.0 * (t - DT), 2600.0)
            veh.append(("a", L_PRE_R, s, 20.0, 0.0))
        if t >= 101.0:                      # b inserted at t=101
            s2 = min(60.0 + 10.0 * (t - 100.5), 2600.0)
            veh.append(("b", L_PRE_R, s2, 10.0, 0.0))
        if 50.0 <= t <= 200.0:              # c changes lane at t=100
            lane = L_PRE_R if t < 100.0 else L_PRE_L
            veh.append(("c", lane, min(200.0 + 15.0 * (t - DT), 2500.0),
                        15.0, 0.0))
        if 40.0 <= t <= 240.0:              # d crosses the junction at t=100
            if t < 100.0:
                veh.append(("d", L_PRE_R, b_main - 25.0 * (100.0 - t),
                            25.0, 0.0))
            elif t == 100.0:
                veh.append(("d", J_MAIN, 0.0, 25.0, 0.0))
            else:
                veh.append(("d", L_MERGE_R, 25.0 * (t - 100.0), 25.0, 0.0))
        if 70.0 <= t <= 240.0:              # e merges from the ramp at t=90
            if t < 90.0:
                veh.append(("e", L_RAMP, b_ramp - 18.0 * (90.0 - t),
                            18.0, 0.0))
            elif t == 90.0:
                veh.append(("e", J_RAMP, 0.0, 18.0, 0.0))
            else:
                veh.append(("e", L_MERGE_A, 18.0 * (t - 90.0), 18.0, 0.0))
        if t == 101.0:
            veh[0] = veh[0][:4] + (-3.5,)   # occasional hard brake
        for v in veh:                        # departures = first appearance
            if v[0] not in seen:
                seen.add(v[0])
                dep.append(v[0])
        rows.append(_row(t, veh, departed=dep, cav_ids=("a", "c"),
                         speed_mode=speed_mode))
    return rows


def write_mini_arm(arm_dir: Path, rows, t0: float, status="complete",
                   with_schedule=True, receipt_speed_mode=None):
    (arm_dir / "run" / "scenario").mkdir(parents=True, exist_ok=True)
    with open(arm_dir / "run" / "physics_steps.jsonl", "w",
              encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if with_schedule:
        vs = "".join(
            f'<vehicle id="{vid}" type="normal_car" route="{rt}" '
            f'depart="{dep}" departLane="best" departSpeed="max"/>'
            for vid, rt, dep in
            [("a", "mainline_route", 1.0), ("b", "mainline_route", 99.0),
             ("c", "mainline_route", 40.0), ("d", "mainline_route", 90.0),
             ("e", "ramp_route", 55.0), ("f", "mainline_route", 295.0)])
        (arm_dir / "run" / "scenario" / "requests.rou.xml").write_text(
            '<routes><vType id="normal_car" speedFactor="1.0"/>' + vs +
            "</routes>", encoding="utf-8")
        ins = [("a", 1.0, 0.5, 1), ("b", 99.0, 101.0, 0),
               ("c", 40.0, 40.5, 1), ("d", 90.0, 90.5, 0),
               ("e", 55.0, 60.0, 0)]
        with open(arm_dir / "run" / "insertions.jsonl", "w",
                  encoding="utf-8") as fh:
            for vid, sd, ti, cav in ins:
                fh.write(json.dumps(dict(vehicle_id=vid,
                                         scheduled_depart_s=sd,
                                         inserted_s=ti,
                                         entrance="mainline" if vid != "e"
                                         else "ramp", depart_speed_mps=20.0,
                                         is_cav=bool(cav),
                                         type_id="normal_car")) + "\n")
    receipt = dict(run=arm_dir.name, world=arm_dir.name.split("__")[0],
                   t0_s=t0, exit_status=status, schema="test/1")
    if receipt_speed_mode is not None:
        receipt["evaluate_summary"] = dict(
            config=dict(speed_mode=receipt_speed_mode))
    (arm_dir / "RUN_RECEIPT.json").write_text(json.dumps(receipt),
                                              encoding="utf-8")


@pytest.fixture(scope="session")
def synth_world(tmp_path_factory):
    """A tiny fake campaign: runs_root with one TRAIN world (base+native),
    one error TRAIN world, one HOLDOUT world; plus the rebuilt base trace
    and the delay-library geometry."""
    root = tmp_path_factory.mktemp("campaign")
    runs = root / "runs"
    rows = synth_physics_rows()
    rows27 = synth_physics_rows(speed_mode=27)
    write_mini_arm(runs / "wsyntm000p02t240__base", rows, T0)
    write_mini_arm(runs / "wsyntm000p02t240__native", rows, T0)
    write_mini_arm(runs / "wsynterrp02t240__base", rows[:4], T0,
                   status="error")
    write_mini_arm(runs / "wsynterrp02t240__native", rows[:4], T0,
                   status="error")
    write_mini_arm(runs / "wsynholdp02t240__base", rows[:4], T0)
    write_mini_arm(runs / "wsynholdp02t240__native", rows[:4], T0)
    # chunk-B material: post-fix world (receipt records speed_mode 27)
    write_mini_arm(runs / "wsynfixm2700p40t240__base", rows27, T0,
                   receipt_speed_mode=27)
    write_mini_arm(runs / "wsynfixm2700p40t240__native", rows27, T0,
                   receipt_speed_mode=27)

    def world(name, split, pen=0.02):
        return dict(world=name, seed=1, loading_veh_h=2700,
                    penetration=pen, split=split, role="exposed", t0_s=T0,
                    base_arm=dict(arm="base", run=f"{name}__base",
                                  controller_name="damping"),
                    branches=[dict(arm="native", run=f"{name}__native",
                                   suffix_controller="native")])

    plan = dict(schema="test",
                worlds=[world("wsyntm000p02t240", "TRAIN"),
                        world("wsynterrp02t240", "TRAIN"),
                        world("wsynfixm2700p40t240", "TRAIN", pen=0.4),
                        world("wsynholdp02t240", "HOLDOUT")],
                constants={})
    plan_path = root / "WORLD_PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return dict(root=root, runs=runs, plan=plan_path, t0=T0,
                data_end=DATA_END)


@pytest.fixture(scope="session")
def synth_base_trace(synth_world):
    return V.build_run_trace("wsyntm000p02t240", "base",
                             synth_world["runs"] / "wsyntm000p02t240__base",
                             synth_world["t0"], t_ins_pre={})


@pytest.fixture(scope="session")
def geom():
    return V.wm30_geometry()


@pytest.fixture(scope="session")
def built_available():
    """Real built V0 data present (MANIFEST + worlds/<chunk>/) from the
    default out."""
    man = HERE / "MANIFEST.json"
    have = man.exists() and (HERE / "worlds").exists() and \
        len(list((HERE / "worlds").glob("*/*_done.json"))) > 0
    if not have:
        pytest.skip("built V0 data not present in this worktree "
                    "(worlds/ is git-ignored; the real v0.x roots are "
                    "validated by verify_v0data.py / td_report.py)")
    return str(HERE)


# ------------------------------------------- v0.3 fixtures and guards ---

WATCH_EVENTS = frozenset({
    "open", "os.listdir", "os.scandir", "os.mkdir", "os.rename",
    "os.replace", "os.remove", "os.rmdir", "shutil.copyfile",
    "shutil.copytree", "os.truncate"})


class HoldoutAccessWatch:
    """Audit-hook recorder: records every filesystem event whose path
    argument mentions a watched HOLDOUT world name.  sys.addaudithook
    cannot be unregistered, so install it immediately before the call
    under test and read .hits right after (later hits are irrelevant)."""

    def __init__(self, world_names):
        self.names = list(world_names)
        self.hits = []

    def __call__(self, event, args):
        if event not in WATCH_EVENTS:
            return
        for a in args:
            try:
                s = os.fspath(a)
            except TypeError:
                continue
            if isinstance(s, str) and any(w in s for w in self.names):
                self.hits.append((event, s))

    def install(self):
        sys.addaudithook(self)
        return self


def make_mini_v02_root(root: Path) -> dict:
    """A tiny v0.2-shaped dataset root: chunk-1 built from synthetic runs
    with 2 p40 TRAIN/CAL worlds, 1 non-p40 TRAIN world, 1 non-p40
    CALIBRATION world and 1 p40 HOLDOUT world (whose run dirs exist but
    must never be opened).  Returns paths plus the p40 / holdout world
    names."""
    import build_v0data as B
    runs = root / "runs"
    rows31 = synth_physics_rows(speed_mode=31)
    p40a, p40b, p02a, p02b, p40h = ("wsynp40am2400p40t240",
                                    "wsynp40bm2600p40t300",
                                    "wsynp02am2700p02t360",
                                    "wsynp02bm2400p02t480",
                                    "wsynp40hm2000p40t480")
    for w in (p40a, p40b, p02a, p02b, p40h):
        write_mini_arm(runs / f"{w}__base", rows31, T0)
        write_mini_arm(runs / f"{w}__native", rows31, T0)

    def world(name, split, pen):
        return dict(world=name, seed=1, loading_veh_h=2700,
                    penetration=pen, split=split, role="exposed", t0_s=T0,
                    base_arm=dict(arm="base", run=f"{name}__base",
                                  controller_name="damping"),
                    branches=[dict(arm="native", run=f"{name}__native",
                                   suffix_controller="native")])

    plan = dict(schema="test", constants={}, worlds=[
        world(p40a, "TRAIN", 0.4), world(p40b, "CALIBRATION", 0.4),
        world(p02a, "TRAIN", 0.02), world(p02b, "CALIBRATION", 0.02),
        world(p40h, "HOLDOUT", 0.4)])
    plan_path = root / "WORLD_PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    rc = B.main(["--plan", str(plan_path), "--runs", str(runs),
                 "--out", str(root)])
    assert rc == 0
    (root / "DATA_CARD_V0.md").write_text(
        "# mini v0.2 data card (synthetic test fixture)\n",
        encoding="utf-8")
    return dict(root=root, runs=runs, plan=plan_path,
                p40=(p40a, p40b), p02=(p02a, p02b), holdout=p40h, t0=T0)


def rewrite_arm_to_speed_mode(src_arm: Path, dst_arm: Path, mode: int,
                              top_level_receipt_mode: int | None):
    """Copy a synthetic run dir with every physics-row speed_mode
    rewritten to `mode` (the fix27 fixture: chunk-1 p40 runs re-run at
    speedMode 27) and a fresh receipt (chunk-3-style top-level
    cav_speed_mode, chunk-2-style config null)."""
    (dst_arm / "run" / "scenario").mkdir(parents=True, exist_ok=True)
    with open(src_arm / "run" / "physics_steps.jsonl", encoding="utf-8") \
            as src, \
            open(dst_arm / "run" / "physics_steps.jsonl", "w",
                 encoding="utf-8") as dst:
        for line in src:
            r = json.loads(line)
            for vv in (r.get("vehicles") or {}).values():
                vv["speed_mode"] = mode
            dst.write(json.dumps(r) + "\n")
    for name in ("insertions.jsonl",):
        p = src_arm / "run" / name
        if p.exists():
            shutil.copy2(p, dst_arm / "run" / name)
    sreq = src_arm / "run" / "scenario" / "requests.rou.xml"
    if sreq.exists():
        shutil.copy2(sreq, dst_arm / "run" / "scenario" / "requests.rou.xml")
    old = json.loads((src_arm / "RUN_RECEIPT.json").read_text("utf-8"))
    receipt = dict(old, exit_status="complete")
    receipt["evaluate_summary"] = dict(config=dict(speed_mode=None))
    if top_level_receipt_mode is not None:
        receipt["cav_speed_mode"] = top_level_receipt_mode
    (dst_arm / "RUN_RECEIPT.json").write_text(json.dumps(receipt),
                                              encoding="utf-8")


@pytest.fixture(scope="session")
def mini_v02(tmp_path_factory):
    return make_mini_v02_root(tmp_path_factory.mktemp("mini_v02"))
