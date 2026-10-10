"""partA_act unit tests (A2 step 2).

All data access is TRAIN-only (the ActWorld guard refuses HOLDOUT /
CALIBRATION) and read-only; ZERO SIMULATION.  The data tests use one
declared TRAIN world (w8411m2700p02t240, t0=240) and are skipped
automatically if the chunk-1 data root is unavailable.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import act_lib as L  # noqa: E402
import act_engine as AE  # noqa: E402

WORLD = "w8411m2700p02t240"
T0 = 240.0
ARMS = ["native", "pi", "switching1"]

torch.set_num_threads(1)


def _world_available():
    try:
        return WORLD in L.train_worlds() and \
            L.refload.CHUNK1_RUNS.exists()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _world_available(), reason="chunk-1 TRAIN data not available")


@pytest.fixture(scope="module")
def world():
    w = L.ActWorld(WORLD, T0, arms=ARMS)
    for a in ARMS:
        if a not in w.arms:
            pytest.skip(f"arm {a} lacks a complete receipt")
    return w


# ------------------------------------------------------------- execution

def test_execution_rule_bounds():
    """The measured rule never exceeds the kinematic ceiling, never goes
    below 0, and honours the target when it is reachable."""
    v = torch.tensor([20.0, 20.0, 20.0, 20.0])
    T = torch.tensor([25.0, 15.0, 21.0, 0.0])
    gap = torch.full((4,), 250.0)
    dv = torch.zeros(4)
    a = torch.full((4,), 2.6)
    b = torch.full((4,), 4.5)
    tau = torch.full((4,), 1.1)
    s0 = torch.full((4,), 2.5)
    v0 = torch.full((4,), 31.29)
    out = AE.execution_speed(T, v, gap, dv, a, b, tau, s0, v0)
    # reachable target (21.0 at +0.5*2.6) is honoured exactly
    assert float(out[2]) == pytest.approx(21.0, abs=1e-12)
    # unreachable upward: the IDM free term caps below the kinematic
    # ceiling (interaction term nonzero even at gap 250)
    import p3lib as P3
    acc_free = float(P3.idm_acc(torch.tensor([20.0]), torch.tensor([250.0]),
                                torch.tensor([0.0]), torch.tensor([2.6]),
                                torch.tensor([4.5]), torch.tensor([1.1]),
                                torch.tensor([2.5]),
                                torch.tensor([31.29]))[0])
    assert float(out[0]) == pytest.approx(20.0 + acc_free * AE.DT)
    assert float(out[0]) <= 20.0 + 2.6 * AE.DT + 1e-12
    # unreachable downward: capped at v - b*DT
    assert float(out[1]) == pytest.approx(20.0 - 4.5 * AE.DT)
    assert float(out[3]) == pytest.approx(20.0 - 4.5 * AE.DT)
    assert (out >= 0).all()
    # IDM safe-speed term binds when the leader is close and slow
    gap2 = torch.tensor([6.0])
    dv2 = torch.tensor([8.0])
    out2 = AE.execution_speed(torch.tensor([30.0]), torch.tensor([20.0]),
                              gap2, dv2, a[:1], b[:1], tau[:1], s0[:1],
                              v0[:1])
    assert float(out2[0]) < 20.0 - 2.0    # hard braking below the kin cap


def test_native_acts_as_human(world):
    """'native' CAVs are exactly idm_merge humans: a rollout whose whole
    schedule is native and the CAVs-as-native (force_native) rollout are
    identical, so the action machinery adds nothing under native.
    (chunk-1 note: even the native arm's dispatcher assigns damping to
    some post-t0 entrant CAVs - visible in the dispatcher log - so the
    all-native comparison needs the overridden schedule.)"""
    ra = AE.ActionRollout(world, "native", member_seed=3)
    ra.actions = {}
    ra.run(60)
    rb = AE.ActionRollout(world, "native", member_seed=3,
                          force_native=True).run(60)
    assert ra.n_commands == 0 and rb.n_commands == 0
    for j in range(0, 61):
        sa, sb = ra.snapshots[j], rb.snapshots[j]
        assert np.array_equal(sa["names"], sb["names"])
        assert np.allclose(sa["v"], sb["v"], atol=0, rtol=0)
        assert np.allclose(sa["s"], sb["s"], atol=0, rtol=0)
        assert np.array_equal(sa["lane"], sb["lane"])
    assert ra.collisions == rb.collisions


# ---------------------------------------------------- command reproduction

def test_emulator_hook_reproduces_logged_commands(world):
    """The emulator INSIDE the engine (ideal-sensing frames of the
    engine's teacher-forced state = the exact logged world state)
    reproduces the logged issued targets and fallback classification of
    the pi branch over the whole 30 s window (E4 contract: |diff| <
    0.01 m/s, identical fallback)."""
    arm = "pi"
    truth = world.truth_rows(arm)
    r = AE.ActionRollout(world, arm, member_seed=0)
    issued_hist = {}
    orig = r._control

    def spy(tick):
        orig(tick)
        for k, v in r.cmd.items():
            issued_hist[(round(float(tick), 6), k)] = v
    r._control = spy
    r.run_teacher(truth, n_ticks=L.H30_TICKS)
    cmds = world.commands(arm)
    matched = bad = missing = 0
    for (tick, vid), (owner, issued, fallback) in sorted(cmds.items()):
        if (tick, vid) not in issued_hist:
            if issued is None:
                continue          # CAV not live at that tick: no row
            missing += 1
            continue
        emu = issued_hist[(tick, vid)]
        if issued is None:
            ok = emu != emu       # fallback: emulator issued nothing
        else:
            ok = emu == emu and abs(emu - issued) <= 0.01
        if ok:
            matched += 1
        else:
            bad += 1
    assert missing == 0, f"{missing} logged commands never issued"
    assert matched + bad > 0
    assert bad == 0, f"{bad} command mismatches"
    assert r.anomalies.get("missing_action_row", 0) == 0


def test_switch_schedule_drives_actions():
    """The engine applies exactly the dispatcher action schedule per
    control tick for every LIVE CAV - the DISPATCHER ROWS are the
    applied action truth (E4 convention: stamp t+0.5 carries the
    decision at tick t; a CAV without a row is unassigned and runs
    native; rows for CAVs the engine has already exited are ignored,
    exactly as E4's physics-row liveness gate ignores them).  Uses a
    declared TRAIN world whose window contains a live action switch
    (w8414m2000p02t240: mainline.140 pi -> damping at t=266)."""
    name = "w8414m2000p02t240"
    t0w = float(L.train_worlds()[name]["t0_s"])
    arm = "switching1"
    w = L.ActWorld(name, t0w, arms=[arm])
    truth = w.truth_rows(arm)
    r = AE.ActionRollout(w, arm, member_seed=0)
    per_tick = {}
    live_at = {}
    orig = r._control

    def spy(tick):
        live_at[round(float(tick), 6)] = {str(x) for x in r.names}
        orig(tick)
        per_tick[round(float(tick), 6)] = dict(r.last_actions)
    r._control = spy
    r.run_teacher(truth, n_ticks=L.H30_TICKS)
    sched = w.actions(arm)
    n_checked = n_dead = 0
    for (tick, name_), action in sched.items():
        tick6 = round(tick, 6)
        if name_ not in live_at.get(tick6, set()):
            n_dead += 1
            continue
        applied = per_tick.get(tick6, {}).get(name_)
        assert applied == action, \
            f"{name_}@{tick}: schedule {action} but engine applied {applied}"
        n_checked += 1
    assert n_checked > 20
    per_vid = {}
    for (tick, name_), action in sched.items():
        if name_ in live_at.get(round(tick, 6), set()):
            per_vid.setdefault(name_, set()).add(action)
    switchers = {v: a for v, a in per_vid.items() if len(a) > 1}
    assert switchers, "expected an in-window action switch"
    print(f"switching1: {n_checked} live (tick, CAV) action checks applied "
          f"exactly ({n_dead} rows for already-exited CAVs ignored); "
          f"switching CAVs: {sorted(switchers)}")


def test_crn_pairing(world):
    """Common random numbers: human vTypes are identical across arms for
    the same member seed, and the birth vehicle SETS are identical
    across arms (same arrival identity; per-branch insertion states may
    differ by branch seed - that part of each branch's world is kept,
    not drawn).  The naive baseline is therefore rolled PER ARM on its
    own branch world."""
    rolls = {arm: AE.ActionRollout(world, arm, member_seed=5,
                                   force_native=(arm == "pi"))
             for arm in ("native", "pi", "switching1")}
    v0 = rolls["native"].vtype
    for r in rolls.values():
        assert np.array_equal(r.vtype, v0)
    bn = world.births("native")
    bp = world.births("pi")
    vids_n = {x[0] for rows in bn.values() for x in rows}
    vids_p = {x[0] for rows in bp.values() for x in rows}
    assert vids_n == vids_p


def test_pi_command_changes_cav_speed(world):
    """Action conditioning actually bites: a commanded CAV's speed
    differs between the pi rollout and the naive rollout, and follows
    the executed target in the direction of the command."""
    arm = "pi"
    truth = world.truth_rows(arm)
    ra = AE.ActionRollout(world, arm, member_seed=1).run(60)
    rb = AE.ActionRollout(world, arm, member_seed=1,
                          force_native=True).run(60)
    assert ra.n_commands > 0
    sa, sb = ra.snap_dicts(), rb.snap_dicts()
    diffs = 0
    for j in range(1, 61):
        for name, st in sa[j].items():
            if name in sb[j] and abs(st[2] - sb[j][name][2]) > 1e-9:
                diffs += 1
    assert diffs > 0, "commanded CAVs never diverge from the naive roll"


def test_holdout_refused():
    with pytest.raises(ValueError):
        L.ActWorld("w8407m2400p02t360", 360.0)


def test_phys_delay_matches_p3k_convention():
    """The A2 contract-delay helper agrees with part3k's phys_delay_sums
    on a synthetic window (same step model, per-vehicle records)."""
    truth = {j: {} for j in range(7)}
    truth[0]["a"] = (1, 100.0, 20.0, 0.0)
    for j in range(1, 7):
        truth[j]["a"] = (1, 100.0 + 10.0 * j, 20.0, 0.0)
    snaps = {j: {"a": (1, 100.0 + 5.0 * j, 10.0)} for j in range(7)}
    snaps[0]["a"] = (1, 100.0, 20.0)
    dp, dt_, np_, nt_ = L.phys_delay(snaps, {}, truth, 6)
    # truth: constant 20 m/s -> delay DT - 10/limit per step
    assert dt_ == pytest.approx(6 * (AE.DT - 10.0 / 31.29), rel=1e-6)
    # pred: 5 m per 0.5 s -> delay 0.5 - 5/31.29 per step
    assert dp == pytest.approx(6 * (AE.DT - 5.0 / 31.29), rel=1e-6)
    assert np_ == nt_ == 6
