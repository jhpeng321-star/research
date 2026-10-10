"""CAV action path via the VERIFIED partA_ctrl emulators (torch + the
frozen ONNX model, read-only): damping command parity, wall-v2 pacing +
release, PI engagement."""
import sys
from pathlib import Path

import numpy as np
import pytest

WD01 = Path(__file__).resolve().parents[2]
PKG = WD01 / "partA_v2g"
for _p in (str(WD01), str(PKG), str(WD01 / "partA_ctrl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")
import emulators  # noqa: E402  (partA_ctrl)
import refload  # noqa: E402

from partA_v2g.engine import PartAV2GEngine, load_params  # noqa: E402
from partA_v2g import cavctrl  # noqa: E402

P = load_params()
MODEL = refload.MODEL_ONNX


def _vehicles():
    return {
        "cav": dict(lane=1, s=2000.0, v=28.0, is_cav=True),
        "lead": dict(lane=1, s=2050.0, v=20.0),
    }


def test_damping_command_matches_emulator_decide():
    """The first control tick's commanded target equals the partA_ctrl
    DampingEmulator.decide on the same ideal-sensor frame, and the engine
    executes it through the measured execution rule."""
    e = PartAV2GEngine(P, t0=480.0, vehicles=_vehicles())
    loop = cavctrl.CavControllerLoop(t0=480.0, params=P,
                                     per_cav_actions={"cav": "damping"},
                                     walls=[], cav_state_hint={})
    cmd = loop(480.0, e)
    # hand frame: same lane/segment, leader gap 2050-5-2000 = 45 m
    frame = dict(now_s=480.0, ts_s=480.0, speed_mps=28.0,
                 allowed_speed_mps=31.29,
                 front_coverage_m=100.0, rear_coverage_m=100.0,
                 front={"gap_m": 45.0, "relative_speed_mps": -8.0},
                 rear=None)
    batch = emulators.FrameBatch.stack([frame])
    emu = emulators.DampingEmulator(MODEL)
    target, code, _ = emu.decide(batch)
    assert int(code[0]) == 0                     # a command was issued
    assert cmd["cav"] == pytest.approx(float(target[0]), rel=1e-12)
    # execution rule parity for the produced step speed
    e.cmd = cmd
    T = np.array([cmd["cav"]])
    v_new = e.execution_speed(T, np.array([28.0]), np.array([45.0]),
                              np.array([8.0]), np.array([31.29]))
    kin = np.clip(T, 28.0 - 3.0 * 0.5, 28.0 + 1.5 * 0.5)
    assert 0.0 <= float(v_new[0]) <= float(max(kin[0], 28.0)) + 1e-9


def test_damping_cav_follows_command_in_rollout():
    def hook(t, engine):
        return {"cav": 10.0}                     # hard command: slow to 10

    e = PartAV2GEngine(P, t0=480.0, vehicles=_vehicles(),
                       control_hook=hook)
    e.run(20)
    # 10 s of commanding 10 m/s: the CAV must approach 10 m/s
    assert e.snapshots[20]["cav"][2] == pytest.approx(10.0, abs=0.6)


def test_wall_pace_and_release():
    """Wall members pace toward the wall speed (the wall_pace law through
    the direct_target bridge + the execution rule), the wall assembles,
    and after hold_s the members run free again."""
    vehicles = {
        "w0": dict(lane=0, s=2000.0, v=28.0, is_cav=True),
        "w1": dict(lane=1, s=2000.0, v=28.0, is_cav=True),
    }
    from predict.contract import WallSpec
    from partA_v2g.cavctrl import CavControllerLoop
    w = WallSpec(wall_id="W1", members=("w0", "w1"), speed_mps=6.0,
                 phase="assemble", release_at_s=None, hold_s=10.0)
    loop = CavControllerLoop(t0=480.0, params=P, per_cav_actions={},
                             walls=[w], cav_state_hint={})
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles,
                       control_hook=loop)
    # first tick parity: pace a = clip((6-28)/1, -3, 1.5) = -3 ->
    # direct_target T = 25.0; execution kin clip: 26.5
    e.run(1)
    assert e.snapshots[1]["w0"][2] == pytest.approx(26.5, abs=1e-9)
    assert e.snapshots[1]["w1"][2] == pytest.approx(26.5, abs=1e-9)
    e.run(25)
    # both members decelerate hard toward the wall speed
    assert e.snapshots[6]["w0"][2] < 20.0
    assert e.snapshots[6]["w1"][2] < 20.0
    # assembled once both are near 6 m/s (before the 10 s release)
    assert loop.walls.assembled_at["W1"] is not None
    assert loop.walls.assembled_at["W1"] <= 10.0
    # released after 10 s: they accelerate again on free road
    assert e.snapshots[26]["w0"][2] > 6.0


def test_pi_engaged_and_invalidates_on_native():
    vehicles = _vehicles()
    loop = cavctrl.CavControllerLoop(t0=480.0, params=P,
                                     per_cav_actions={"cav": "pi-external-u"},
                                     walls=[], cav_state_hint={})
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles,
                       control_hook=loop)
    cmd = loop(480.0, e)
    assert "cav" in cmd                            # the PI issues a command
    assert loop._pi_store["cav"][2] is True        # active
    # off-engagement tick: lifecycle invalidation (adapter semantics)
    loop2 = cavctrl.CavControllerLoop(t0=480.0, params=P,
                                      per_cav_actions={"cav": "native"},
                                      walls=[], cav_state_hint={})
    loop2._pi_store["cav"] = [20.0, 479.0, True]
    loop2(481.0, e)
    assert loop2._pi_store["cav"][2] is False      # invalidated


def test_native_only_import_is_torch_free():
    """Importing cavctrl alone loads no torch: the emulator stack is
    lazy (native-only rollouts never pay for it)."""
    import subprocess
    import sys as _sys
    code = ("import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, "
            "r'%s'); import cavctrl; assert 'torch' not in sys.modules; "
            "print('torch-free OK')" % (PKG, WD01 / "partA_ctrl"))
    out = subprocess.run([_sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "torch-free OK" in out.stdout
