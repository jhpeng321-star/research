"""Synthetic WD-01-PB-V04A world: a damping base + branch arms
(native / pi / switching) with schedules and command logs, built on the
partB_v0data synthetic physics (CAVs a, c)."""
from __future__ import annotations

import json
from pathlib import Path

from conftest import T0, synth_physics_rows, write_mini_arm

WORLD = "wsyn4m2700p02t240"


def _cmds(path: Path, rows: list, kind="command"):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _command_rows(vid, t_from, t_to, owner):
    return [dict(kind="command", timestamp_s=float(t), vehicle_id=vid,
                 owner=owner, issued_target_speed_mps=None,
                 fallback_reason=None) for t in range(int(t_from),
                                                      int(t_to))]


def write_schedule(arm: Path, arm_name: str, events: list, t0=T0):
    d = dict(run=arm.name, arm=arm_name, branch_seed=1,
             schedule=dict(schema="wm44_branch_types/1", arm=arm_name,
                           seed=1, t0_s=float(t0), horizon_s=1080.0,
                           events=[dict(vehicle_id=v, time_s=float(t),
                                        action=a) for v, t, a in events]))
    (arm / "run" / "switch_schedule.json").write_text(json.dumps(d),
                                                      encoding="utf-8")


def make_world(root: Path, world=WORLD, *, native_owner="native",
               dispatcher=True):
    """Returns dict of arm dirs.  native_owner='damping' makes the native
    arm's command log CONTRADICT its schedule (verification must refuse)."""
    runs = root / "runs"
    rows = synth_physics_rows(speed_mode=27)
    arms = {}
    base = runs / f"{world}__base"
    write_mini_arm(base, rows, T0)
    (base / "run" / "manifest.json").write_text(
        json.dumps(dict(controller="damping")), encoding="utf-8")
    _cmds(base / "run" / "telemetry.jsonl",
          _command_rows("a", 1, 250, "damping")
          + _command_rows("c", 40, 200, "damping")
          + _command_rows("a", 300, 301, "native"))
    arms["base"] = base

    def branch(name, arm_name, events, cmds):
        d = runs / f"{world}__{name}"
        write_mini_arm(d, rows, T0)
        write_schedule(d, arm_name, events)
        _cmds(d / "run" / "suffix_commands.jsonl", cmds)
        if dispatcher:      # must never be opened by any v0.4a code
            (d / "run" / "dispatcher_rows.jsonl").write_text(
                "{not json\n", encoding="utf-8")
        arms[name] = d
        return d

    branch("native", "native", [("a", T0, "native"), ("c", T0, "native")],
           _command_rows("a", 241, 250, native_owner)
           + _command_rows("c", 241, 200 + 41, "native"))
    branch("pi", "fleet-pi", [("a", T0, "pi"), ("c", T0, "pi")],
           _command_rows("a", 241, 250, "damping"))
    # switching: a native -> damping at 245.3 -> pi at 248.9
    branch("switching1", "switching",
           [("a", T0, "native"), ("a", 245.3, "damping"),
            ("a", 248.9, "pi"), ("c", T0, "damping"), ("c", 300.0, "pi")],
           _command_rows("a", 241, 246, "native")[:5]
           + _command_rows("a", 246, 249, "damping")
           + _command_rows("a", 249, 250, "damping"))
    return arms
