"""S-SUMOES-RUN CPU tests: real-evaluator accounting/dispatch (fake worker), cap-relative
composition, head setting, canary identity and CRN guards. No SUMO."""
import json
import math
import sys
import textwrap

import pytest
import torch

from tools.sumoes.config import SumoEsConfig
from tools.sumoes.driver import AcceptAllScreener, MechanicalFailure, SumoEsDriver
from tools.sumoes.remote import arm
from tools.sumoes.stub import StubSimulator
from tools.sumoes.sumo_eval import SumoRuntimeEvaluator, theta_sha

CONFIG = SumoEsConfig(enabled=True)


def design():
    from tools.sumoes.make_design import build
    return build(CONFIG)


FAKE_WORKER = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    own = Path(sys.argv[1]).parents[2]
    for job in json.loads(Path(sys.argv[1]).read_text()):
        out = own / "jobs" / job["id"]
        out.mkdir(parents=True)
        bad = job["seed"] == 969999
        j = 1000.0 + job["seed"] % 7 + (0.5 if job["arm"] == "ours" else 0.0)
        result = dict(ok=True, actual_sim_s=900.0, J600=j, lane_audit=dict(changeLane=0),
                      metrics=dict(collisions=0, teleports=0, dropped_vehicle_steps=0,
                                   initial_state_sha256="h%d" % job["seed"]),
                      head_stats=dict(rows=10, sensitivity_rms=2.0, calls=1))
        (out / "result.json").write_text(json.dumps(result))
''')


def make_evaluator(tmp_path, cap=650_000):
    own = tmp_path / "own"
    worker = own / "pytools/tools/sumoes/remote/worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text(FAKE_WORKER)
    return SumoRuntimeEvaluator(own, cap_sim_s=cap, workers=4, python=sys.executable,
                                poll_s=0.05, launcher=[], design=design())


def test_evaluator_precharges_durably_and_collects(tmp_path):
    ev = make_evaluator(tmp_path)
    theta = torch.zeros(137, dtype=torch.float64)
    requests = [dict(theta=theta + i * 1e-3, seed=969440 + i, update=0, tag=f"t{i}")
                for i in range(6)]
    results = ev.evaluate_many(requests)
    rows = ev.ledger_rows()
    assert [r["id"] for r in rows] == [f"t{i}" for i in range(6)]
    assert all(r["sim_s"] == 900 for r in rows)
    assert [r["job"] for r in results] == [f"t{i}" for i in range(6)]
    assert all(r["safety_clean"] and r["actual_sim_s"] == 900 for r in results)
    assert results[0]["initial_state_sha256"] == "h969440"
    files = sorted((ev.own / "thetas").glob("*.json"))
    assert len(files) == 6
    data = json.loads(files[0].read_text())
    assert data["sha256"] == theta_sha(data["theta"])
    with pytest.raises(MechanicalFailure, match="already"):
        ev.evaluate_many(requests[:1])


def test_evaluator_refuses_cap_overflow_before_any_worker(tmp_path):
    ev = make_evaluator(tmp_path, cap=1800)
    theta = torch.zeros(137, dtype=torch.float64)
    with pytest.raises(MechanicalFailure, match="cap"):
        ev.evaluate_many([dict(theta=theta, seed=969440 + i, update=0, tag=f"c{i}")
                          for i in range(3)])
    assert ev.ledger_rows() == []
    assert not any((ev.own / "jobs").iterdir())


def test_evaluator_stop_marker_refuses(tmp_path):
    ev = make_evaluator(tmp_path)
    (ev.own / "STOP").write_text("x")
    with pytest.raises(MechanicalFailure, match="STOP"):
        ev.evaluate_many([dict(theta=torch.zeros(137), seed=969440, update=0, tag="s")])
    assert ev.ledger_rows() == []


class FakeVehicle:
    def __init__(self, accel=1.2):
        self.accel = accel

    def getAccel(self, vid):
        return self.accel

    def getSpeed(self, vid):
        return 10.


class FakeApi:
    vehicle = FakeVehicle()


def test_caprel_composition_zero_is_native_and_envelope(monkeypatch):
    from wms.control import nativeinit
    natives = {"a": -0.37, "b": 0.8, "c": -4.0}
    monkeypatch.setattr(nativeinit, "sumo_native_accel", lambda api, vid, dt: natives[vid])
    from tools.sumoes import caprel
    monkeypatch.setattr(caprel, 'live_cap', lambda api, vid, dt: 1.2)
    zero = arm.caprel_commands(FakeApi(), {"a": 0.0, "b": -0.0, "c": 0.0}, -3.0, 1.2)
    assert zero == natives
    out = arm.caprel_commands(FakeApi(), {"a": 5.0, "b": -9.0, "c": 0.5}, -3.0, 1.2)
    assert out["a"] == 1.2                       # upper = min(hi, getAccel)
    assert out["b"] == -3.0                      # lower = min(lo, native)
    assert out["c"] == -3.5                      # native below lo stays reachable
    small = arm.caprel_commands(FakeApi(), {"a": 0.1}, -3.0, 1.2)
    assert math.isclose(small["a"], -0.27)
    with pytest.raises(ValueError):
        arm.caprel_commands(FakeApi(), {"a": float("nan")}, -3.0, 1.2)


def test_set_head_writes_137_scalars_and_zero_is_exact_zero_residual():
    policy = torch.nn.Module()
    policy.residual_head = torch.nn.Linear(136, 1)

    class Exe:
        residual_on_native = True

    exe = Exe()
    exe.policy = policy
    theta = torch.arange(137, dtype=torch.float64) / 1000
    arm.set_head(exe, theta)
    assert torch.equal(policy.residual_head.weight.double().flatten(),
                       theta[:136].float().double())
    assert float(policy.residual_head.bias) == pytest.approx(0.136)
    arm.set_head(exe, torch.zeros(137))
    features = torch.randn(5, 3, 136)
    residual = 1.2 * torch.tanh(policy.residual_head(features).squeeze(-1))
    assert torch.equal(residual.abs(), torch.zeros_like(residual))
    with pytest.raises(ValueError):
        arm.set_head(exe, torch.zeros(136))
    exe.residual_on_native = False
    with pytest.raises(ValueError):
        arm.set_head(exe, torch.zeros(137))


def test_head_sensitivity_hook_at_zero_head():
    stats = arm.HeadSensitivity(radius=1.2)
    head = torch.nn.Linear(136, 1)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    head.register_forward_hook(stats)
    features = torch.ones(2, 4, 136)
    head(features)
    summary = stats.summary()
    assert summary["rows"] == 8
    assert summary["sensitivity_rms"] == pytest.approx(1.2 * math.sqrt(137))


class ArmStub:
    """Stub with evaluate_many + arms: ctrl0 equals ours@0 unless ``break_zero``."""

    def __init__(self, break_zero=False, break_hash=False):
        self.sim = StubSimulator()
        self.break_zero, self.break_hash, self.jobs = break_zero, break_hash, []

    def evaluate_many(self, requests):
        out = []
        for r in requests:
            res = self.sim.episode(r["theta"], r["seed"], r["update"])
            res["job"] = r["tag"]
            res["head_stats"] = dict(rows=4, sensitivity_rms=3.0)
            if r.get("arm") == "ctrl0" and self.break_zero:
                res["J600"] += 1.0
            if self.break_hash and "_u00_panel" in "_" + r["tag"]:
                res["initial_state_sha256"] = "other"
            self.jobs.append(r["tag"])
            out.append(res)
        return out


def test_canary_arm_check_baseline_and_calibration(tmp_path):
    ev = ArmStub()
    driver = SumoEsDriver(config=CONFIG, evaluator=ev, sensitivity=20.0,
                          screener=AcceptAllScreener(3), canary_arm_check=True,
                          state_path=tmp_path / "state.json")
    state = driver.initial_state()
    receipt = driver.baseline(state)
    assert ev.jobs[:2] == ["base_s969536_ctrl0", "base_s969536_ours0"]
    assert len(ev.jobs) == 9 and receipt["budget"]["charged_sim_s"] == 9 * 900
    assert receipt["canary"]["identical"] is True
    assert driver.sensitivity == pytest.approx(3.0)
    assert (tmp_path / "state.json").exists()
    update = driver.run_update(state)
    assert len(ev.jobs) == 9 + 44
    assert update["budget"]["charged_sim_s"] == (9 + 44) * 900
    resumed, rstate = SumoEsDriver.resume(tmp_path / "state.json", config=CONFIG,
                                          evaluator=ev, screener=AcceptAllScreener(3))
    assert rstate.update_count == 1 and resumed.ledger.charged_sim_s == 53 * 900
    assert torch.equal(rstate.theta, state.theta)


def test_canary_mismatch_stops(tmp_path):
    driver = SumoEsDriver(config=CONFIG, evaluator=ArmStub(break_zero=True),
                          screener=AcceptAllScreener(3), canary_arm_check=True)
    with pytest.raises(MechanicalFailure, match="zero identity"):
        driver.baseline(driver.initial_state())


def test_crn_world_hash_mismatch_stops():
    driver = SumoEsDriver(config=CONFIG, evaluator=ArmStub(break_hash=True),
                          screener=AcceptAllScreener(3))
    state = driver.initial_state()
    driver.baseline(state)
    with pytest.raises(MechanicalFailure, match="CRN"):
        driver.run_update(state)


def test_file_screener_binds_request_and_result(tmp_path):
    import hashlib
    from tools.sumoes.file_screen import FileScreener, request_payload, write_request
    driver = SumoEsDriver(config=CONFIG, evaluator=ArmStub(), screener=None)
    state = driver.initial_state()
    driver.baseline(state)
    sigma = state.sigma(driver.sensitivity)
    directions = state.directions(0, 8)
    digest = write_request(tmp_path, request_payload(0, state.theta, sigma, driver.sensitivity,
                                                     directions))
    (tmp_path / "screen_u00.json").write_text(json.dumps(dict(
        update=0, request_sha256=digest, kept=[5, 1], predicted_gain=[3.0, 2.0],
        rejected=[0, 2, 3, 4, 6, 7])))
    driver.screener = FileScreener(tmp_path)
    receipt = driver.run_update(state)
    assert receipt["screen"]["kept"] == [5, 1]
    assert len(receipt["pair_J"]) == 2
    text = (tmp_path / "request_u00.json").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == digest
    with pytest.raises(ValueError, match="different content"):
        write_request(tmp_path, request_payload(0, state.theta + 1, sigma, 1.0, directions))
    bad = FileScreener(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        bad.screen(state.theta + 1.0, sigma, directions, 0)
