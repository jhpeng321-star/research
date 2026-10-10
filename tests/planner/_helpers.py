"""Shared helpers for the planner tests (no simulation imports)."""

from types import SimpleNamespace

from traffic_shepherd.planner import MockWorldModel, PlanCandidate, WallSpec
from traffic_shepherd.planner import plan_key  # noqa: F401  (re-exported)

__all__ = [
    "MockWorldModel",
    "PlanCandidate",
    "RecordingProcessFactory",
    "WallSpec",
    "make_samples",
    "make_world",
    "state_from_plan",
    "wall",
    "plan",
    "plan_key",
]


class _FakeConnection:
    """Identity-keyed dummy connection (SimpleNamespace would compare by
    value, colliding distinct branches that share a label)."""

    def __init__(self, label):
        self.label = label


class RecordingProcessFactory:
    """PL-10d fake launcher (ONE fresh process per restored branch).

    Hands out a throwaway connection object per ``start_process`` call and
    records the start/close order. Closing a connection that never started,
    or closing any connection twice, asserts -- the world model must retire
    every branch process exactly once, branch exceptions included."""

    def __init__(self):
        self.started = []
        self.closed = []
        self.labels = []

    def start_process(self, *, label=None):
        connection = _FakeConnection(label)
        self.started.append(connection)
        self.labels.append(label)
        return connection

    def close_process(self, connection):
        assert any(connection is c for c in self.started), "closed a process that never started"
        assert not any(connection is c for c in self.closed), "a process was closed twice"
        self.closed.append(connection)


def make_samples(k=3, observed=True, time_s=0.0):
    """K StateSample stand-ins; ``observed`` is True/False or a True count."""
    if observed is True:
        observed = k
    if observed is False:
        observed = 0
    return [
        SimpleNamespace(
            hybrid=None,
            arrivals={"seed": i},
            meta={
                "sample_id": f"s{i}",
                "time_s": time_s,
                "merge_observed": i < observed,
                "coverage": 1.0,
            },
        )
        for i in range(k)
    ]


def make_world(cav_ids=("c0", "c1"), n_members=1, part_a=10, native_delay=100.0):
    return MockWorldModel(
        cav_ids,
        n_members=n_members,
        part_a_horizon_s=part_a,
        native_delay_s=native_delay,
    )


def state_from_plan(plan, *, phase_time=0.0):
    """A controller_state consistent with ``plan`` (members locked)."""
    state = {
        cav: {
            "action": action,
            "wall_id": None,
            "wall_phase": None,
            "time_in_phase_s": phase_time,
            "wall_speed_mps": None,
            "locked": False,
        }
        for cav, action in plan.per_cav.items()
    }
    for wall in plan.walls:
        for member in wall.members:
            state[member] = {
                "action": "wall",
                "wall_id": wall.wall_id,
                "wall_phase": wall.phase,
                "time_in_phase_s": phase_time,
                "wall_speed_mps": wall.speed_mps,
                "locked": True,
            }
    return state


def wall(wall_id, members, speed=16.0, phase="hold", hold_s=60.0):
    return WallSpec(
        wall_id=wall_id, members=tuple(members), speed_mps=speed, phase=phase, hold_s=hold_s
    )


def plan(per_cav, walls=()):
    return PlanCandidate(per_cav=per_cav, walls=tuple(walls))


# ------------------------------------------- PL-04e2 campaign base-run fixture
def make_base_run(tmp_path, *, world="w8468chunk1train", t0_s=240.0, seed=8468,
                  damping_cavs=("mainline.6", "mainline.7"),
                  late_cav="mainline.99", other_owner="mainline.5",
                  cav_ids=None):
    """A minimal campaign-format base run dir (<world>__base) that passes
    ``closed_loop.resolve_base_run``: saved_state (precision-17 + rngState
    state file + a wm41_saved_state/1 manifest with digests), a
    scenario-exact campaign scenario, damping command telemetry, insertions
    and the AUTHORITATIVE roster source (PL-04m): a
    ``run/physics_steps.jsonl`` whose every row carries the SAME constant
    ``cav_ids`` list. ``cav_ids`` defaults to the damping CAVs plus
    ``late_cav`` -- the roster includes ``late_cav`` even though it is
    inserted AFTER t0 (PL-04m: the full designated set, not only the
    <= t0 CAVs) and even though it has no command row; ``other_owner`` got
    a non-damping command row and is NOT in ``cav_ids``.

    READ-ONLY rule preserved: the tests only read it back (and copy the
    scenario into a run dir)."""
    import hashlib
    import json

    base = tmp_path / f"{world}__base"
    scenario_dir = base / "run" / "scenario"
    state_dir = base / "saved_state"
    scenario_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    (scenario_dir / "requests.rou.xml").write_text(
        '<routes>\n'
        '  <vType id="normal_car" carFollowModel="IDM" speedDev="0"/>\n'
        '  <route id="mainline_route" edges="e0 e1"/>\n'
        + "".join(
            f'  <vehicle id="v{i}" type="normal_car" route="mainline_route" depart="{i}.0"/>\n'
            for i in range(3)
        )
        + "</routes>\n",
        encoding="utf-8",
    )
    # a campaign-shaped cfg whose net-file value gets rewritten (value-agnostic)
    (scenario_dir / "scenario.sumocfg").write_text(
        "<configuration>\n  <input>\n"
        '    <net-file value="legacy-mount/assets/net.xml.gz"/>\n'
        '    <route-files value="requests.rou.xml"/>\n'
        "  </input>\n</configuration>\n",
        encoding="utf-8",
    )

    with (base / "run" / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
        for cav in damping_cavs:
            stream.write(
                json.dumps(
                    {
                        "kind": "command",
                        "vehicle_id": cav,
                        "timestamp_s": 10.0,
                        "owner": "damping",
                        "issued_target_speed_mps": 25.0,
                        "fallback_reason": None,
                        "released_to_native": False,
                    }
                )
                + "\n"
            )
        stream.write(
            json.dumps(
                {
                    "kind": "command",
                    "vehicle_id": other_owner,
                    "timestamp_s": 11.0,
                    "owner": "pi",
                    "issued_target_speed_mps": 24.0,
                    "fallback_reason": None,
                    "released_to_native": False,
                }
            )
            + "\n"
        )
    inserted = {cav: 5.0 for cav in damping_cavs}
    inserted[other_owner] = 6.0
    inserted[late_cav] = t0_s + 1.0  # designated? no commands -> just present late
    with (base / "run" / "insertions.jsonl").open("w", encoding="utf-8") as stream:
        for vehicle_id, inserted_s in sorted(inserted.items()):
            stream.write(
                json.dumps(
                    {
                        "vehicle_id": vehicle_id,
                        "scheduled_depart_s": inserted_s,
                        "inserted_s": inserted_s,
                        "entrance": "mainline",
                        "is_cav": False,
                        "type_id": "normal_car",
                    }
                )
                + "\n"
            )

    # PL-04m: the AUTHORITATIVE roster source -- every physics row carries
    # the SAME constant full designated cav_ids list (incl. the post-t0
    # late_cav; the roster is never filtered to inserted_s <= t0)
    roster_ids = list(cav_ids) if cav_ids is not None else sorted(
        set(damping_cavs) | {late_cav}
    )
    with (base / "run" / "physics_steps.jsonl").open("w", encoding="utf-8") as stream:
        for row_time in (0.5, float(t0_s)):
            stream.write(
                json.dumps(
                    {
                        "time_s": row_time,
                        "live_ids": [],
                        "departed_ids": [],
                        "arrived_ids": [],
                        "colliding_ids": [],
                        "cav_ids": list(roster_ids),
                        "speed_modes": {},
                        "cav_actions": {},
                    }
                )
                + "\n"
            )

    state_text = (
        '<state><rngState seed="%d"/>'
        '<vehicle id="%s" pos="1197.84675455360253" speed="28.149877900994554"/>'
        "</state>" % (seed, damping_cavs[0])
    )
    (state_dir / "state.xml").write_text(state_text, encoding="utf-8")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema": "wm41_saved_state/1",
        "seed": seed,
        "branch_point_t0_s": float(t0_s),
        "state_options": {
            "save_state_precision_decimals": 17,
            "rng_state_in_file": True,
            "required_cli": ["--save-state.precision", "17", "--save-state.rng", "true"],
        },
        "sumo_version": [1, 27, 1],
        "sumo_state_file": "state.xml",
        "sumo_state_sha256": sha(state_dir / "state.xml"),
        "scenario_digests": {
            "scenario/requests.rou.xml": sha(scenario_dir / "requests.rou.xml"),
            "scenario/scenario.sumocfg": sha(scenario_dir / "scenario.sumocfg"),
        },
        "harness_components": ["ownership", "circles_controller"],
        "harness_state": {
            "roster": {"cav_ids": list(damping_cavs)},
            # T-B: the campaign manifest's carried python state (a
            # base-run master restores it; refuses a manifest without it)
            "ownership": {
                "cav_speed_mode": 27,
                "mode_configured": [],
                "owned": {},
                "last": {},
                "time_s": float(t0_s) - 1.0,
            },
            "loop_state": {"previous_live": [], "retired_cavs": [], "collision_previous": []},
        },
        "physics_row_at_t0": {"time_s": float(t0_s)},
    }
    (state_dir / "STATE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return base
