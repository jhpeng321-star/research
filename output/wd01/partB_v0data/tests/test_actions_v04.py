"""v0.4 per-CAV action reconstruction tests (brief item 1).

Synthetic: schedule/command merging, command-priority ties, unknown mask,
fleet-run policy + verification.  REAL TRAIN data (read-only, split checked
BEFORE any run file is opened):

* chunk2 w8481m2700p02t480__native  -- damping 556 / native 487 command
  owners, every command reproduced by the reconstruction;
* chunk4 w8623m2600p02t540__native_base -- fleet-native: every CAV native
  at every decision time.

Plus the functional open-guard: no code path opens the dispatcher action
log or insertions.jsonl during reconstruction.
"""
from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import actions_v04 as A  # noqa: E402

RD = Path("E:/research-dense")
CHUNK2_PLAN = RD / "output/campaign_stage1_v2/results_plan_chunk2/" \
                 "WORLD_PLAN.json"
CHUNK2_ARM = RD / "runs/campaign_stage1_chunk2/w8481m2700p02t480__native"
CHUNK4_PLAN = RD / "output/campaign_stage1_v3/results_plan_chunk4/" \
                 "WORLD_PLAN.json"
CHUNK4_ARM = RD / "runs/campaign_stage1_chunk4/w8623m2600p02t540__native_base"


# ------------------------------------------------------------- fixtures --

def write_switching_arm(arm_dir: Path, *, schedule: list[dict],
                        commands: list[dict], controller: str | None = None):
    (arm_dir / "run").mkdir(parents=True, exist_ok=True)
    sched = dict(schema="wm44_branch_types/1", arm="switching", seed=1,
                 t0_s=480.0, horizon_s=1080.0, interval_lo_s=20.0,
                 interval_hi_s=120.0, events=schedule, intervals_s=[])
    (arm_dir / "run" / "switch_schedule.json").write_text(
        json.dumps(dict(run=arm_dir.name, arm="switching",
                        branch_seed=1, schedule=sched,
                        schedule_sha256="0" * 64)),
        encoding="utf-8")
    with open(arm_dir / "run" / "suffix_commands.jsonl", "w",
              encoding="utf-8") as fh:
        for r in commands:
            fh.write(json.dumps(r) + "\n")
    if controller is not None:
        (arm_dir / "run" / "manifest.json").write_text(
            json.dumps(dict(schema="traffic_shepherd_evaluation/1",
                            controller=controller)), encoding="utf-8")


@pytest.fixture()
def switching_arm(tmp_path):
    """C0 damping at t0, switches to native at 500; commands confirm and
    one realised native override at 520 (fallback) while scheduled
    damping, back to damping after the 540 switch."""
    arm = tmp_path / "wsyntp02t480__native"
    schedule = [
        dict(vehicle_id="cav0", time_s=480.0, action="damping"),
        dict(vehicle_id="cav0", time_s=500.0, action="native"),
        dict(vehicle_id="cav0", time_s=540.0, action="pi-external-u"),
        dict(vehicle_id="cav1", time_s=480.0, action="native"),
        dict(vehicle_id="cav1", time_s=560.0, action="damping"),
    ]
    commands = [
        dict(kind="command", timestamp_s=480.0, vehicle_id="cav0",
             owner="damping", issued_target_speed_mps=25.0,
             fallback_reason=None),
        dict(kind="command", timestamp_s=481.0, vehicle_id="cav0",
             owner="damping", issued_target_speed_mps=25.0,
             fallback_reason=None),
        dict(kind="pi_owner_receipt", timestamp_s=481.0, vehicle_id="cav0",
             status="written_owner_receipt"),
        dict(kind="command", timestamp_s=500.0, vehicle_id="cav0",
             owner="native", issued_target_speed_mps=None,
             fallback_reason="selected_native"),
        dict(kind="command", timestamp_s=520.0, vehicle_id="cav0",
             owner="native", issued_target_speed_mps=None,
             fallback_reason="missing_or_uncovered_leader"),
        dict(kind="command", timestamp_s=540.0, vehicle_id="cav0",
             owner="damping", issued_target_speed_mps=24.0, fallback_reason=None),
        dict(kind="command", timestamp_s=480.0, vehicle_id="cav1",
             owner="native", issued_target_speed_mps=None,
             fallback_reason="selected_native"),
        dict(kind="command", timestamp_s=560.0, vehicle_id="cav1",
             owner="damping", issued_target_speed_mps=24.0, fallback_reason=None),
    ]
    write_switching_arm(arm, schedule=schedule, commands=commands)
    return arm


# ------------------------------------------------------------ synthetic --

def test_schedule_holds_between_events_and_commands_win_ties(switching_arm):
    tl = A.ActionTimeline.from_run_dir(switching_arm)
    assert tl.fleet_policy is None            # no manifest: per-CAV only
    # before any evidence: unknown (never silently native)
    assert tl.action_at("cav0", 479.9) == "unknown"
    assert tl.action_at("cav9", 600.0) == "unknown"
    # schedule assignment holds between commands
    assert tl.action_at("cav0", 490.0) == "damping"
    assert tl.action_at("cav0", 499.9) == "damping"
    assert tl.action_at("cav0", 500.0) == "native"     # command at the tie
    assert tl.action_at("cav0", 519.9) == "native"     # persists
    assert tl.action_at("cav0", 530.0) == "native"
    # at 540 the schedule says pi but the realised command is damping:
    # the realised owner wins the tie (and afterwards)
    assert tl.action_at("cav0", 540.0) == "damping"
    assert tl.action_at("cav0", 545.0) == "damping"
    assert tl.action_at("cav1", 550.0) == "native"
    assert tl.action_at("cav1", 560.0) == "damping"
    assert tl.command_owner_counts() == {"damping": 4, "native": 3}
    ids, unk = tl.action_ids(["cav0", "cav1", "cavX"], 545.0)
    assert ids.tolist() == [A.ACTIONS_V04.index("damping"),
                            A.ACTIONS_V04.index("native"),
                            A.ACTIONS_V04.index("unknown")]
    assert unk.tolist() == [0.0, 0.0, 1.0]      # explicit mask channel


def test_every_logged_command_is_reproduced_exactly(switching_arm):
    """The reconstruction matches the realised owner of EVERY command row
    at its own (vid, t) -- the property the acceptance check requires."""
    tl = A.ActionTimeline.from_run_dir(switching_arm)
    with open(Path(switching_arm) / "run" / "suffix_commands.jsonl",
              encoding="utf-8") as fh:
        n = 0
        for line in fh:
            r = json.loads(line)
            if r.get("kind") != "command":
                continue
            assert tl.action_at(r["vehicle_id"],
                                r["timestamp_s"]) == \
                A.canonical_action(r["owner"])
            n += 1
    assert n == 7


def test_unknown_owner_and_action_spellings_fail_loudly(switching_arm):
    with pytest.raises(A.ActionReconstructionError, match="spelling"):
        A.canonical_action("super-cruise")
    arm = Path(switching_arm)
    cmd = arm / "run" / "suffix_commands.jsonl"
    cmd.write_text(json.dumps(dict(kind="command", timestamp_s=480.0,
                                   vehicle_id="cav0", owner="cruise"))
                   + "\n", encoding="utf-8")
    with pytest.raises(A.ActionReconstructionError):
        A.ActionTimeline.from_run_dir(arm)


def test_fleet_run_from_manifest_is_uniform_and_verified(tmp_path):
    arm = tmp_path / "wsyntp02t300__native_base"
    commands = [dict(kind="command", timestamp_s=float(t),
                     vehicle_id=f"cav{i % 3}", owner="native",
                     issued_target_speed_mps=None,
                     fallback_reason="selected_native")
                for i, t in enumerate(range(10, 300, 7))]
    write_switching_arm(arm, schedule=[], commands=commands,
                        controller="native")
    tl = A.ActionTimeline.from_run_dir(arm)
    assert tl.fleet_policy == "native"
    for t in (5.0, 100.0, 900.0):
        for vid in ("cav0", "cav1", "never_seen"):
            assert tl.action_at(vid, t) == "native"
    rec = tl.verify_fleet_policy()
    assert rec["ok"] and rec["owner_counts"] == {"native": len(commands)}


def test_fleet_damping_base_tolerates_native_fallback_but_nothing_else(
        tmp_path):
    arm = tmp_path / "wsyntp02t480__base"
    rows = []
    for t in range(0, 600, 2):
        rows.append(dict(kind="command", timestamp_s=float(t),
                         vehicle_id="cav0",
                         owner="damping" if t % 4 else "native",
                         issued_target_speed_mps=25.0,
                         fallback_reason=None if t % 4 else
                         "missing_or_uncovered_leader"))
    write_switching_arm(arm, schedule=[], commands=rows,
                        controller="damping")
    tl = A.ActionTimeline.from_run_dir(arm)
    assert tl.fleet_policy == "damping"
    assert tl.action_at("cav0", 137.5) == "damping"   # uniform, not per-tick
    tl.verify_fleet_policy()                          # native fallback OK
    # a PI command inside a damping fleet is contract drift -> loud
    rows.append(dict(kind="command", timestamp_s=601.0, vehicle_id="cav0",
                     owner="pi-external-u", issued_target_speed_mps=20.0,
                     fallback_reason=None))
    write_switching_arm(arm, schedule=[], commands=rows,
                        controller="damping")
    tl2 = A.ActionTimeline.from_run_dir(arm)
    with pytest.raises(A.ActionReconstructionError, match="fleet policy"):
        tl2.verify_fleet_policy()


def test_no_evidence_at_all_is_unknown_not_native(tmp_path):
    arm = tmp_path / "wsyntp02t480__mystery"
    write_switching_arm(arm, schedule=[], commands=[])
    tl = A.ActionTimeline.from_run_dir(arm)
    assert tl.fleet_policy is None
    assert tl.action_at("cav0", 600.0) == "unknown"
    ids, unk = tl.action_ids(["cav0"], 600.0)
    assert ids[0] == A.ACTIONS_V04.index("unknown") and unk[0] == 1.0


def test_reconstruction_never_opens_dispatcher_or_insertions(
        switching_arm, monkeypatch):
    real_open = builtins.open

    def guarded_open(file, *a, **k):
        s = str(file)
        if "dispatcher_rows" in s or "insertions.jsonl" in s:
            raise AssertionError(f"action reconstruction opened {s}")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    tl = A.ActionTimeline.from_run_dir(switching_arm)
    for t in (480.0, 500.0, 545.0, 600.0):
        tl.action_at("cav0", t)


# -------------------------------------------------- real chunk2 TRAIN arm --

def _split_of(plan_path: Path, world: str) -> str:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return next(w["split"] for w in plan["worlds"] if w["world"] == world)


@pytest.fixture(scope="module")
def chunk2_timeline():
    if not CHUNK2_ARM.is_dir() or not CHUNK2_PLAN.is_file():
        pytest.skip("chunk2 runs/plan not present")
    # split check BEFORE opening anything under the run dir
    excl = json.loads((RD / "protocols/holdout_exclusions.json")
                      .read_text(encoding="utf-8"))
    assert _split_of(CHUNK2_PLAN, "w8481m2700p02t480") == "TRAIN"
    assert "w8481m2700p02t480" not in {e["world"] for e in excl["excluded"]}
    return A.ActionTimeline.from_run_dir(CHUNK2_ARM)


def test_chunk2_native_owner_counts(chunk2_timeline):
    counts = chunk2_timeline.command_owner_counts()
    assert counts == {"damping": 556, "native": 487}


def test_chunk2_native_every_command_reproduced(chunk2_timeline):
    tl = chunk2_timeline
    n = 0
    with open(CHUNK2_ARM / "run" / "suffix_commands.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") != "command":
                continue
            got = tl.action_at(r["vehicle_id"], r["timestamp_s"])
            assert got == A.canonical_action(r["owner"]), r
            n += 1
    assert n == 1043


def test_chunk2_native_schedule_events_reproduced(chunk2_timeline):
    """Where no command row sits at the exact (vid, t), the scheduled
    action is the reconstruction at that tick."""
    tl = chunk2_timeline
    sched = json.loads((CHUNK2_ARM / "run" / "switch_schedule.json")
                       .read_text(encoding="utf-8"))["schedule"]
    cmd_at = set()
    with open(CHUNK2_ARM / "run" / "suffix_commands.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") == "command":
                cmd_at.add((r["vehicle_id"], round(float(r["timestamp_s"]),
                                                   6)))
    checked = 0
    for ev in sched["events"]:
        key = (ev["vehicle_id"], round(float(ev["time_s"]), 6))
        if key in cmd_at:
            continue                      # realised command wins the tie
        got = tl.action_at(ev["vehicle_id"], float(ev["time_s"]))
        assert got == A.canonical_action(ev["action"]), ev
        checked += 1
    assert checked > 0


def test_chunk2_native_decision_grid_actions(chunk2_timeline):
    tl = chunk2_timeline
    t0 = 480.0
    for t in np.arange(t0 + 10.0, t0 + 570.0 + 1e-9, 60.0):
        for vid in ("mainline.1038", "mainline.1173"):
            assert tl.action_at(vid, float(t)) in \
                ("native", "damping", "pi")


# -------------------------------------------------- real chunk4 TRAIN arm --

@pytest.fixture(scope="module")
def chunk4_timeline():
    if not CHUNK4_ARM.is_dir() or not CHUNK4_PLAN.is_file():
        pytest.skip("chunk4 runs/plan not present")
    excl = json.loads((RD / "protocols/holdout_exclusions.json")
                      .read_text(encoding="utf-8"))
    # split check BEFORE opening anything under the run dir
    assert _split_of(CHUNK4_PLAN, "w8623m2600p02t540") == "TRAIN"
    assert "w8623m2600p02t540" not in {e["world"] for e in excl["excluded"]}
    return A.ActionTimeline.from_run_dir(CHUNK4_ARM)


def test_chunk4_native_base_is_all_native(chunk4_timeline):
    tl = chunk4_timeline
    assert tl.fleet_policy == "native"        # manifest controller
    rec = tl.verify_fleet_policy()
    assert rec["ok"]
    assert set(rec["owner_counts"]) == {"native"}
    for t in (10.0, 300.0, 540.0, 890.0):
        for vid in ("mainline.9", "mainline.0", "ramp.3"):
            assert tl.action_at(vid, t) == "native"


def test_chunk4_native_base_fleet_policy_on_fresh_timeline():
    """from_run_dir resolves the fleet policy from the run manifest on a
    fresh timeline object too (no shared state)."""
    if not CHUNK4_ARM.is_dir():
        pytest.skip("chunk4 runs not present")
    tl = A.ActionTimeline.from_run_dir(CHUNK4_ARM)
    assert tl.fleet_policy == "native"
    assert tl.verify_fleet_policy()["ok"]
