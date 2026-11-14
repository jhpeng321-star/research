"""E3: batched wall-v2 member command law == WallGroupSession.tick."""
import numpy as np
import torch

import emulators
import refload

refload.load_reference()

from traffic_shepherd.sensing import SensorFrame  # noqa: E402
from traffic_shepherd.wall_group import WallGroupSession  # noqa: E402

MEMBERS = ("w0", "w1", "w2")


def _frames(speeds, spans, now):
    """SensorFrame-like frames on distinct lanes at merge-relative spans."""
    out = {}
    for i, vid in enumerate(MEMBERS):
        out[vid] = SensorFrame(
            vid, now, float(speeds[i]), 0.0, 31.29, i,
            distance_to_merge_m=float(-spans[i]),  # s_m = -distance
            front=None, rear=None, front_coverage_m=0.0, rear_coverage_m=0.0)
    return out


def test_wall_member_command_law_equivalence():
    """Commands per tick through assembly and hold == batched _pace."""
    rng = np.random.default_rng(23)
    wall = WallGroupSession("wall", members=frozenset(MEMBERS),
                            wall_speed_mps=20.0)
    speeds = np.array([26.0, 27.5, 25.0])
    spans = np.array([0.0, 3.0, 6.0])  # within the assembly span tolerance
    max_diff = 0.0
    for tick in range(1, 40):
        now = float(tick)
        frames = _frames(speeds, spans, now)
        was_assembling = wall.phase == "assembly"
        commands = wall.tick(frames, now)
        # wall-v2.1: ASSEMBLY uses the closure target (gap to the front);
        # legitimate change of this test (owner decision 2026-09-30).
        gap = emulators.wall_gap_to_front(
            torch.tensor(spans, dtype=torch.float64),
            torch.zeros(3, dtype=torch.int64),
            closure=torch.full((3,), was_assembling))
        accel = emulators.wall_pace(
            torch.full((3,), 20.0, dtype=torch.float64),
            torch.tensor(speeds, dtype=torch.float64),
            gap_to_front_m=gap).numpy()
        assert set(commands) == set(MEMBERS), f"tick {tick}: {wall.phase}"
        for i, vid in enumerate(MEMBERS):
            max_diff = max(max_diff, abs(commands[vid] - float(accel[i])))
        # scripted convergence: wall law drives speeds toward 20 m/s
        speeds = speeds + 0.5 * accel
        spans = spans * 0.95
        if wall.phase == "hold":
            break
    assert wall.phase == "hold", "assembly never completed in scenario"
    # hold keeps producing the same law
    for tick in range(40, 50):
        frames = _frames(speeds, spans, float(tick))
        commands = wall.tick(frames, float(tick))
        accel = emulators.wall_pace(
            torch.full((3,), 20.0, dtype=torch.float64),
            torch.tensor(speeds, dtype=torch.float64)).numpy()
        for i, vid in enumerate(MEMBERS):
            max_diff = max(max_diff, abs(commands[vid] - float(accel[i])))
    assert max_diff <= 1e-12, f"max |pace diff| = {max_diff}"


def test_wall_release_and_abort_paths():
    """Released/aborted walls issue no commands; law matches while active."""
    wall = WallGroupSession("wall", members=frozenset(MEMBERS),
                            wall_speed_mps=20.0)
    speeds = [20.0, 20.0, 20.0]
    spans = [0.0, 1.0, 2.0]
    commands = wall.tick(_frames(speeds, spans, 1.0), 1.0)
    assert commands and wall.phase == "hold"  # already aligned -> assembled
    assert wall.release(2.0) is True
    assert wall.tick(_frames(speeds, spans, 3.0), 3.0) == {}

    aborting = WallGroupSession("wall2", members=frozenset(MEMBERS),
                                wall_speed_mps=20.0)
    # span beyond the abort cap -> safety abort, no commands ever
    commands = aborting.tick(_frames(speeds, [0.0, 100.0, 200.0], 1.0), 1.0)
    assert commands == {}
    assert aborting.phase == "handback"
    assert aborting.abort_reason == "member_span_exceeded"

    # direct pacing-law values (independent anchors)
    pace = emulators.wall_pace(torch.tensor([20.0, 5.0, 23.0]),
                               torch.tensor([18.0, 9.0, 21.0]))
    assert torch.equal(pace, torch.tensor([1.5, -3.0, 1.5]))
