"""F3: observation contracts, native CAV parity, strict diagnostic geometry."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.data.recorded_prefix import causal_scene
from wms.model.merge_pivot import geometry
from wms.physics.config import PhysicsConfig
from wms.train.cav_drive import TimelineControllers
from wms.train.losses.merge_pivot import build_recorded_geometry

from tests.test_merge_pivot_training import _fixture


def test_junction_lane_carry_is_causal_identity_scoped_and_keeps_observed_front():
    kin = np.zeros((4, 1, 7), np.float32)
    kin[:, 0, 0] = [0, 0, 30, 0]
    kin[:, 0, 6] = 5
    kin[0, 0, 2] = kin[2, 0, 2] = 1
    present = np.ones((4, 1), bool)
    cont = np.array([[True], [True], [False]])
    adapted = causal_scene(kin, present, cont)
    assert adapted[1, 0, 2] == 1 and adapted[1, 0, 0] == 0
    assert not adapted[3, 0, 1:4].any()  # new junction-born identity is unknown
    kin[2, 0, 0] = 100
    np.testing.assert_array_equal(causal_scene(kin, present, cont)[:2], adapted[:2])


def test_full_scene_replay_keeps_later_arriving_cav_and_current_length():
    model, inp, targets = _fixture()
    inp.present[:, :, 2] = False  # outside the t0 supervision cohort
    base = build_recorded_geometry(model, inp, targets)
    full_present = inp.present.clone()
    full_present[:, 1:, 2] = True
    full_cav = inp.is_cav.clone()
    full_cav[:, :, 2] = True
    states = inp.states.clone()
    states[:, :, 2, 6] = 7
    inp.merge_pivot_scene = [dict(states=states[0].numpy(), present=full_present[0].numpy(),
        is_cav=full_cav[0].numpy(), continuity=np.ones((4, 3), bool),
        action_timeline=dict(controller=np.zeros((3, 3), np.int64),
                             acceleration_mps2=np.zeros((3, 3), np.float32),
                             target_speed_mps=np.zeros((3, 3), np.float32), control_dt_s=1.))]
    full = build_recorded_geometry(model, inp, targets)
    assert not base["has_f"][0, 1, 0] and full["has_f"][0, 1, 0]
    assert full["index_f"][0, 1, 0] == 2
    assert full["length"][0, 1, 2] == 7


def test_real_c4_loader_separates_cohort_and_scene_without_future_interpolation(monkeypatch, tmp_path):
    from wms.data import c4_episodes as c4
    kin = np.zeros((6, 2, 7), np.float32)
    kin[..., 6] = 5
    kin[:, 0, 3] = 1
    kin[:, 1, 2] = 1
    kin[:, 1, 0] = [0, 1, 2, 2, 30, 40]
    kin[3, 1, 1:4] = 0
    occupied = np.ones((6, 2), bool)
    occupied[:3, 1] = False  # enters after window t0
    kin[:3, 1, 6] = 0
    arrays = dict(veh_kin=kin, veh_mask=occupied, target_mask=np.ones((6, 2), bool),
        is_cav=np.ones((6, 2), bool), vid=np.tile([10, 20], (6, 1)),
        time_s=np.arange(6) * .5, control_time_s=np.arange(3),
        controller=np.zeros((3, 2), np.int64), issued_acceleration_mps2=np.zeros((3, 2)),
        issued_target_speed_mps=np.zeros((3, 2)))
    ep = SimpleNamespace(array=arrays.__getitem__, run_start=lambda tick: 0,
                         run_bounds=lambda tick: (0, 6), has=lambda name: name in arrays)
    entry = dict(world="w32m1900p20t1", arm="native", episode="toy", controller="native")
    loader = c4.C4WindowLoader(tmp_path, [entry], [(0, 2)], horizon_steps=3,
        history_s=None, history_stride_s=1., entries_v2=False, junction_fix="ffill",
        merge_pivot_history=True)
    monkeypatch.setattr(loader, "episode", lambda e: ep)
    first = loader(0)
    scene = first["merge_pivot_scene"]
    assert not first["present"][1, 1] and scene["present"][1, 1]
    assert scene["states"][1, 1, 6] == 5 and scene["is_cav"][1, 1]
    # Junction-born neighbour has no known lane, and remains a strict scene veto.
    assert not scene["states"][1, 1, 1:4].any()
    kin[4, 1, 0] = 1000
    second = loader(0)
    np.testing.assert_array_equal(first["states"][:2], second["states"][:2])


def test_runtime_captures_full_scene_comparison_geometry_including_entrant_length():
    from wms.train.losses.base import LossContext
    from wms.train.losses.merge_pivot import MergePivotConfig, MergePivotTerm
    from tests.test_merge_pivot_training import _prediction
    model, inp, targets = _fixture()
    term = MergePivotTerm(MergePivotConfig(weight=1.))
    term.prepare(targets)
    prep = targets.prepared["merge_pivot"]
    pred = _prediction(inp, model)
    for key in ("gap_f", "gap_l", "v_f", "v_l", "has_f", "has_l"):
        pred["merge_pivot_geometry_" + key] = prep["geometry"][key].clone()
    # Simulate a zero-length recorded padding column. Runtime's captured geometry
    # must remain authoritative; reconstruction with target metadata is forbidden.
    targets.states[..., 6] = 0
    model.pivot_decision_geometry = lambda *a, **kw: pytest.fail("reconstructed runtime geometry")
    support, _ = term._comparison(LossContext(pred=pred, current=inp.initial,
        targets=targets, c0=0, c1=4), prep)
    assert support[0, :3, 0].all()


@pytest.mark.parametrize("stop", [False, True])
def test_native_timeline_uses_same_encoded_plant_and_integration_as_runtime(stop):
    model, inp, targets = _fixture()
    from dataclasses import replace
    model.physics = replace(model.physics, stop_time_integration=stop)
    from wms.model.merge_pivot import advance_longitudinal
    from wms.train.inputs import state_from_kin
    state = state_from_kin(inp.states[:, 0], inp.present[:, 0])
    state["is_cav"] = inp.is_cav[:, 0]
    accel = torch.full_like(state["v_mps"], -30. if stop else 1.)
    # Deliberately different native default plant; encoded forward must win.
    model.physics_accel = lambda state: torch.zeros_like(state["v_mps"])
    state["merge_pivot_native_accel"] = accel
    timeline = dict(controller=torch.zeros(1, 3, 3, dtype=torch.long),
                    acceleration_mps2=torch.zeros(1, 3, 3),
                    target_speed_mps=torch.zeros(1, 3, 3), control_dt_s=1.)
    actual = TimelineControllers(model, timeline)(state)
    expected = advance_longitudinal(inp.states[:, 0], accel, .5, stop)
    torch.testing.assert_close(actual[..., 0], expected[..., 0])
    torch.testing.assert_close(actual[..., 2], expected[..., 4])


def test_geometry_explains_all_neighbor_overlap_without_padding_or_lane_commit():
    kin = torch.tensor([[[20., 0, 0, 1, 0, 0, 5],
                         [26., 0, 1, 0, 0, 0, 2],
                         [40., 0, 1, 0, 0, 0, 30]]])
    geo = geometry(kin, torch.ones(1, 3, dtype=torch.bool), PhysicsConfig())
    assert geo["overlap_index"][0, 0] == 2
    assert geo["index_l"][0, 0] == 1
    assert not geo["eligible"][0, 0] and geo["gap_l"][0, 0] == 4
    kin[0, 2, 6] = 20  # exact touching, no positive interval intersection
    assert geometry(kin, torch.ones(1, 3, dtype=torch.bool), PhysicsConfig())["eligible"][0, 0]


def test_audit_paths_and_eval_seeds_fail_closed():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("r9_support_audit",
        Path(__file__).resolve().parents[1] / "tools/r9_support_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    for path in ("../outside", "sealed/test", "xholdoutx/test", "heldout/test",
                 "cal_split/test", "chunk5/test", "j4b/test", "inc1/test"):
        with pytest.raises(ValueError):
            audit.local_path(path)
    for seed in (964000, 964199):
        with pytest.raises(ValueError):
            audit.check_seed(dict(seed=seed))
    audit.check_seed(dict(seed=963004))


def test_audit_includes_unknown_and_cav_and_reports_exact_zero_mass():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("r9_support_audit_small",
        Path(__file__).resolve().parents[1] / "tools/r9_support_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    # Small real runtime replay model; event never raises its earlier maximum.
    model, _, _ = _fixture()
    kin = np.zeros((5, 2, 7), np.float32)
    kin[..., 6] = 5
    kin[:, 0, 0] = [-1, 1, 2, 3, 4]
    kin[:, 1, 0] = 30
    kin[:4, 0, 3] = 1
    kin[4, 0, 2] = 1
    kin[:, 1, 2] = 1
    arrays = dict(veh_kin=kin, veh_mask=np.ones((5, 2), bool),
                  target_mask=np.ones((4, 2), bool), is_cav=np.zeros((5, 2), bool),
                  vid=np.broadcast_to(np.array([10, 11]), (5, 2)),
                  time_s=np.arange(5) * .5, control_time_s=np.arange(2),
                  controller=np.zeros((2, 2), np.int64),
                  issued_acceleration_mps2=np.zeros((2, 2), np.float32),
                  issued_target_speed_mps=np.zeros((2, 2), np.float32))
    rows, counts = audit.audit_episode(model, SimpleNamespace(array=arrays.__getitem__),
        dict(episode="synthetic", warmup_s=1.), "test", window_s=1.)
    assert len(rows) == 1 and counts["hdv_events"] == 1
    assert rows[0]["unknown_age_before"] and not rows[0]["unknown_age_after"]
    assert counts["after_unknown_age_events"] == 0
