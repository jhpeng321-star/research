"""Offline collection regressions: no SUMO, no launchers, no deployment."""
import json
from types import SimpleNamespace

import pytest
import torch

from traffic_shepherd.c4.probe import (FRACTION, HORIZONS, correction_report, distribution,
                                      fit_cqr, select, split_world, world_options)
from traffic_shepherd.c4.sumocredit import PairExtras


def calibration():
    worlds = [f"train-world-{i}" for i in range(50)]
    rows = [{"world": w, "split": "TRAIN", "wm_sha256": "accepted",
             "paired_benefit": {str(h): [-8, -6, -4, -2, 0, 2, 4, 6] for h in HORIZONS},
             "realized_benefit": {str(h): 9 for h in HORIZONS}} for w in worlds]
    return fit_cqr(rows, "accepted")


def candidate(name, values, kind="single:hold"):
    return {"id": name, "action_type": kind,
            "paired_benefit": {str(h): values for h in HORIZONS}}


def test_cqr_world_split_finite_sample_correction_and_stale_rejection():
    doc = calibration()
    assert doc["ready"] and not set(doc["fit_worlds"]) & set(doc["calibration_worlds"])
    assert doc["scale"]["300"] > 0
    assert all(split_world(w) == "fit" for w in doc["fit_worlds"])
    chosen = select([candidate("x", [-8, -6, -4, -2, 0, 2, 4, 6])], doc, "accepted", "fresh")
    assert chosen["prediction"]["300"]["upper"] == pytest.approx(9)
    for sha, world in (("stale", "fresh"), ("accepted", doc["calibration_worlds"][0])):
        with pytest.raises(ValueError, match="stale"):
            select([], doc, sha, world)
    assert fit_cqr([], "accepted")["ready"] is False
    assert select([], None, "accepted", "fresh") is None


def test_select_negative_mean_uncertain_highest_ucb_not_highest_variance_or_mean():
    xs = [candidate("certain", [-100] * 8), candidate("positive", [20] * 8),
          candidate("a", [-20, -15, -10, -9, -8, -7, -6, 1]),
          candidate("b", [-100, -90, -80, -70, -60, -50, 8, 10])]
    picked = select(xs, calibration(), "accepted", "fresh")
    assert picked["id"] == "b" and picked["prediction"]["300"]["mean"] < 0
    # Additive WM level shifts cancel before selection.
    for v in ([0] * 7, [float("nan")] * 8):
        with pytest.raises(ValueError):
            distribution(v)


def test_exact_half_allocation_guard_exclusion_and_unchanged_reserve():
    cfg = {"branch_horizon": 300, "long_group": {"enabled": True},
           "long_single_hold": {"enabled": True}, "probe": {"enabled": True, "calibration": "cqr.json"}}
    assert world_options(cfg, index=0, cycle=2) == {}
    for cycle in range(3, 7):
        slots = [world_options(cfg, index=i, cycle=cycle)["probe_slot"] for i in range(13)]
        assert len(slots) / (2 * 13) == FRACTION
        assert world_options(cfg, index=0, cycle=cycle, guard=True) == {}
    original = PairExtras(long_group_horizon=300, long_single_horizon=300)
    enabled = PairExtras(long_group_horizon=300, long_single_horizon=300,
                         probe_slot="single", probe_calibration="cqr.json", probe_role="collection")
    assert original.sim_s_per_point(300) == enabled.sim_s_per_point(300) == 600
    assert 13 * enabled.sim_s_per_point(300) == 7800 < 8000
    assert enabled.marks(300) == {60, 120, 300}
    assert PairExtras().probe_slot == ""
    with pytest.raises(ValueError, match="collection"):
        PairExtras(long_group_horizon=300, long_single_horizon=300, probe_slot="single",
                   probe_calibration="cqr.json", probe_role="guard")


@pytest.mark.parametrize("slot", ["single", "group"])
def test_retargets_not_appends_and_falls_back_without_evidence(monkeypatch, slot):
    import traffic_shepherd.c4.probe_runtime as runtime
    extras = PairExtras(long_group_horizon=300, long_single_horizon=300,
                        long_group_time=360, long_single_time=360, probe_slot=slot,
                        probe_calibration="cqr.json", probe_role="collection", credit_imagine=False)
    calls = []
    monkeypatch.setattr(runtime, "choose", lambda *a: {"id": "p"})
    monkeypatch.setattr(runtime, "execute", lambda *a: {"extra_sim_s": 300})
    monkeypatch.setattr(extras, "_credit", lambda *a, **k: calls.append("group") or {"extra_sim_s": 300})
    monkeypatch.setattr(extras, "_single_hold", lambda *a: calls.append("single") or {"extra_sim_s": 300})
    prefix = {(a, h): {"collisions": 0} for a in ("B", "POLICY") for h in HORIZONS}
    result = extras.run_point(None, {"t_s": 360}, prefix, policy_factory=None,
                              seed=968180, world="fresh", branch_h=300)
    assert result["sim_s"] == 600 and calls == ["group" if slot == "single" else "single"]
    calls.clear()
    monkeypatch.setattr(runtime, "choose", lambda *a: None)
    result = extras.run_point(None, {"t_s": 360}, prefix, policy_factory=None,
                              seed=968180, world="fresh", branch_h=300)
    assert result["sim_s"] == 600 and calls == ["group", "single"]


def test_collision_stops_immediately_and_blocks_later_probe(monkeypatch):
    from traffic_shepherd.c4.probe_runtime import execute
    extras = PairExtras()
    class Env:
        k = 1
        def state(self):
            return {}
        def step(self, *a, **kw):
            return {}, {"interval": {"collision_involvements": 1, "starting_teleports": 2}}
    env = Env()
    monkeypatch.setattr(extras, "_load", lambda *a: None)
    # Collision occurs on the first native tick after the hold: no executor call.
    factory = SimpleNamespace(components=(None, None, None), settings=None, wm_sha256="accepted")
    import wms.control.policy as policy
    monkeypatch.setattr(policy, "PolicyExecutor", lambda *a, **kw: None)
    c = {"id": "p", "duration_s": 0, "commands": {}, "action_type": "single:hold"}
    record = execute(extras, env, {"t_s": 360}, {}, factory, c, 968180, "fresh")
    assert extras._probe_stopped and record["extra_sim_s"] == 1
    assert record["collisions"] == 1 and record["teleports"] == 2 and not record["complete"]


def test_probe_execution_native_release_prefix_ledgers_and_separate_pair_export(tmp_path, monkeypatch):
    import traffic_shepherd.c4.probe_runtime as runtime
    import traffic_shepherd.c4.genrand as genrand
    import wms.control.policy as policy
    extras = PairExtras()
    actions = []
    class Env:
        k = 1
        def state(self):
            return {}
        def step(self, accel, **kw):
            actions.append(accel)
            return {}, {"interval": {"speed_deficit_vehicle_seconds": 2., "source_wait_vehicle_seconds": 1.}}
    env = Env()
    monkeypatch.setattr(extras, "_load", lambda *a: None)
    monkeypatch.setattr(extras, "_members_step", lambda *a: lambda *a: ("learned", {}))
    monkeypatch.setattr(policy, "PolicyExecutor", lambda *a, **kw: None)
    written = []
    monkeypatch.setattr(genrand, "write_branch_episode", lambda path, **kw: written.append((path, kw)))
    native = tmp_path / "native"
    native.mkdir()
    (native / "summary.json").write_text('{"kind":"B"}')
    point = {"t_s": 360, "native_meta": {"branch": {"pair_key": "world@t360", "kind": "B"}},
             "probe_root": tmp_path / "probe_branches", "native_episode": native,
             "base_recorder": None, "i_k": 120}
    prefix = {("B", h): {"J": h * 5., "road": h * 4., "source": h,
                         "collisions": 0, "teleports": 0} for h in HORIZONS}
    factory = SimpleNamespace(components=(None, None, None), settings=None, wm_sha256="accepted")
    c = {"id": "p", "duration_s": 30, "commands": {}, "action_type": "single:hold"}
    result = runtime.execute(extras, env, point, prefix, factory, c, 968180, "world")
    assert actions == ["learned"] * 30 + [None] * 270
    assert result["realized_benefit"] == {str(h): h * 2. for h in HORIZONS}
    assert result["extra_sim_s"] == 300 and result["valid"]
    assert written[0][1]["meta"]["branch"]["kind"] == "PROBE"
    assert written[0][1]["meta"]["branch"]["pair_key"] == "world@t360"
    assert written[0][0].parent == tmp_path / "probe_branches"
    assert (tmp_path / "probe_branches/t0360_native/summary.json").exists()


def test_frozen_commands_follow_vehicle_ids_and_existing_lane_mask():
    from traffic_shepherd.c4.probe_runtime import FrozenAction
    actor = FrozenAction(SimpleNamespace(), {"cav": (-.4, 1)}, ["new", "cav"])
    obs = SimpleNamespace(cav_mask=torch.ones(1, 2, dtype=torch.bool), own=torch.zeros(1, 2, 1))
    accel, _ = actor(obs, None)
    assert accel.tolist() == [[0., pytest.approx(-.4)]]
    out = actor(obs, None, lane_mask=torch.tensor([[[1, 0, 0], [1, 0, 0]]], dtype=torch.bool))
    assert out.lane_choice.tolist() == [[0, 0]]


def test_report_and_probe_file_separate_from_credit_gate_and_policy(tmp_path):
    extras = PairExtras(probe_slot="single", probe_role="collection", probe_calibration="cqr.json",
                        long_group_horizon=300, long_single_horizon=300)
    row = {"action_type": "single:hold", "prediction": {str(h): {"mean": -2} for h in HORIZONS},
           "realized_benefit": {str(h): 3 for h in HORIZONS}}
    report = correction_report([row])
    assert all(v["mean_realized_minus_predicted"] == 5 for v in report["by_action_type"])
    receipt = extras.write(tmp_path, [{"probe": row, "sim_s": 300, "wall_s": 1}])
    assert receipt["files"] == {"probe": 1}
    assert not (tmp_path / "SUMOCREDIT_POINTS.jsonl").exists()
    assert json.loads((tmp_path / "PROBE_REPORT.json").read_text())["question"] == "WM pessimism corrected?"


def test_score_batches_eight_crn_futures_once_and_extracts_prefixes(monkeypatch):
    import wms.control.confident_gate as cg
    import wms.control.reward as reward
    from traffic_shepherd.c4.probe_runtime import score
    calls = []
    class Model:
        physics = SimpleNamespace(dt_s=.5)
        def rollout(self, initial, **kwargs):
            calls.append(kwargs["n_steps"])
            # Reference and action have common future levels: they cancel.
            level = torch.arange(8.).repeat(2)
            level[8:] -= 1
            shape = (16, 600, 1)
            return {"positions": level[:, None, None].expand(shape), "speeds": torch.zeros(shape),
                    "lanes": torch.zeros(shape, dtype=torch.long), "masks": torch.ones(shape, dtype=torch.bool),
                    "accel": torch.zeros(shape)}
    class Imagination:
        def __init__(self, models, *a, **kw):
            self.models = models
        def arms_J(self, state, arms, **kw):
            assert self.n_samples == 8 and len(arms) == 2
            self.models[0].rollout({"s_m": torch.zeros(16, 1)}, n_steps=600)
    monkeypatch.setattr(cg, "ArmImagination", Imagination)
    monkeypatch.setattr(reward, "team_reward", lambda cur, nxt, *a: {"delay": nxt["s_m"].sum(-1)})
    factory = SimpleNamespace(models=[Model()], components=(None, None, None))
    c = {"id": "p", "commands": {"v": (0., 0)}, "duration_s": 30}
    result = score(factory, {}, {}, ["v"], [c], 42)
    assert calls == [600]
    assert result[0]["paired_benefit"] == {str(h): [h * 2.] * 8 for h in HORIZONS}


def test_prepare_and_enable_identical_twins_only_at_unstarted_cycle3(tmp_path):
    from tools.probe.prepare import prepare
    configs = []
    for name, coop in (("main", True), ("twin", False)):
        arm = tmp_path / name
        arm.mkdir()
        root = arm / "loop"
        root.mkdir()
        config = {"loop_root": str(root), "fastloop": {"two_box": True},
                  "policy": {"coop": coop}, "collect": {"worlds_per_cycle": 13, "branch_horizon": 300,
                  "long_group": {"enabled": True}, "long_single_hold": {"enabled": True}}}
        path = arm / "CONFIG.json"
        path.write_text(json.dumps(config))
        state = root / "LOOP14_STATE.json"
        state.write_text(json.dumps({"accepted_wm": {"sha256": "accepted"}, "cycles": [{"cycle": 2}]}))
        (arm / "probe_cqr.json").write_text(json.dumps(calibration()))
        original = path.read_bytes()
        prepare(arm)
        assert path.read_bytes() == original
        receipt = prepare(arm, True)
        assert receipt["extra_sim_s"] == 0
        updated = json.loads(path.read_text())
        assert updated["policy"]["coop"] == coop
        configs.append({k: v for k, v in updated["collect"]["probe"].items() if k != "calibration"})
        state.write_text(json.dumps({"accepted_wm": {"sha256": "accepted"}, "cycles": [{"cycle": 3}]}))
        with pytest.raises(ValueError, match="cycle 3"):
            prepare(arm, True)
    assert configs[0] == configs[1] == {"enabled": True, "from_cycle": 3}


def test_world_artifact_exports_probes_as_separate_hashed_training_source(tmp_path, monkeypatch):
    from tools.fastloop.world import build, verify
    import wms.data.c4_episodes as episodes
    import wms.control.real_credit14 as credit
    work = tmp_path / "work"
    raw = work / "collect/branches/policy/world"
    raw.mkdir(parents=True)
    probes = work / "collect/probe_branches/world"
    probes.mkdir(parents=True)
    row = {"action_type": "single:hold", "prediction": {str(h): {"mean": -2} for h in HORIZONS},
           "realized_benefit": {str(h): 3 for h in HORIZONS}}
    (raw / "PROBE_POINTS.jsonl").write_text(json.dumps(row) + "\n")
    (raw / "PROBE_SELECTION.jsonl").write_text('{"selected":true}\n')
    calls = []
    def convert(roots, out, **kwargs):
        calls.append(roots)
        out.mkdir()
        (out / "C4_EPISODES_MANIFEST.json").write_text('{"episodes":[]}')
    def gather(root, credit_path, gate_path):
        credit_path.write_text("")
        gate_path.write_text("")
    monkeypatch.setattr(episodes, "convert", convert)
    monkeypatch.setattr(credit, "gather", gather)
    build(work, 0, 0.)
    assert calls == [[work / "collect/branches"], [work / "collect/probe_branches"]]
    hashes = verify(work / "artifact")
    assert "probes/C4_EPISODES_MANIFEST.json" in hashes
    assert "PROBE_POINTS.jsonl" in hashes and "PROBE_SELECTION.jsonl" in hashes
