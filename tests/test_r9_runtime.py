"""The fitter and audit must replay identical causal scenes and event identities."""
import copy
import json

import numpy as np
import pytest
import torch

from tools import merge_fit_v2 as legacy
from tools import r9_exact as exact
from tools import r9_fit_exact as heads
from tools import r9_runtime as runtime
from tools import r9_support_audit as audit


@pytest.fixture(scope="module")
def frozen():
    return exact.load_frozen()


def write_episode(path):
    n, k = 9, 3
    kin = np.zeros((n, k, 7), np.float32)
    kin[..., 0] = np.array([-3., 25., -25.]) + np.arange(n)[:, None] * 2
    kin[:, 0, 3], kin[:, 1:, 2] = 1., 1.
    kin[2:, 0, 3], kin[2:, 0, 2] = 0., 1.  # merge in the first zone-crossing step
    kin[..., 4], kin[..., 6] = 4., 5.
    # 8 m/s carries the ramp vehicle across the 2.93 m internal lane in front of the merge
    # edge, so the WM motion enters the zone from the observed upstream prestate (row 1).
    kin[:, 0, 4] = 8.
    kin[3, 1, 1:4] = 0.  # already observed target enters a junction
    cav = np.zeros((n, k), bool)
    cav[:, 2] = True
    arrays = dict(veh_kin=kin, veh_mask=np.ones((n, k), bool), is_cav=cav,
                  vid=np.tile(np.arange(k), (n, 1)), time_s=np.arange(n) * .5,
                  target_mask=np.ones((n - 1, k), bool), control_time_s=np.arange(4),
                  controller=np.zeros((4, k), np.int64),
                  issued_acceleration_mps2=np.zeros((4, k), np.float32),
                  issued_target_speed_mps=np.zeros((4, k), np.float32))
    for name, value in arrays.items():
        np.save(path / (name + ".npy"), value)
    meta = dict(seed=963001, world="synthetic", episode="synthetic", source_name="test",
                warmup_s=0., manifest_role="train")
    (path / "episode.json").write_text(json.dumps(meta), encoding="utf-8")
    return {**meta, "path": path}


def test_shared_fitter_geometry_matches_actual_audit_replay(frozen, tmp_path):
    ep = write_episode(tmp_path)
    scene = runtime.load_scene(tmp_path)
    assert scene["kin"][3, 1, 2] == 1 and not scene["raw"][3, 1, 1:4].any()
    cache = runtime.prepare(frozen, scene)
    model = runtime.runtime_model(frozen)
    geo = audit.runtime_replay(model, torch.tensor(scene["kin"])[None],
                               torch.tensor(scene["present"])[None],
                               torch.tensor(scene["is_cav"])[None],
                               torch.tensor(scene["continuity"])[None], scene["timeline"])
    for key in geo:
        torch.testing.assert_close(cache["geometry"][key], geo[key], atol=1e-5, rtol=1e-5)
    # This event was absent from the old fitter because its recorded pre-s was negative.
    sequences, _ = heads.build_dataset([ep], frozen, allow_unverified_physics=True)
    events = [s for s in sequences if s["event"]]
    assert len(events) == 1 and events[0]["event_row"] == 2 and events[0]["entry_row"] == 1
    seq = events[0]
    assert seq["physical"][-1] == bool(geo["eligible"][0, 1, 0])
    assert not seq["unknown_age"]
    # All checkpoint tensors survive migration, including nonzero complete heads.
    for name, value in frozen.model.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name


def test_shared_replay_uses_current_native_controls_but_no_future_motion(frozen, tmp_path):
    write_episode(tmp_path)
    scene = runtime.load_scene(tmp_path)
    baseline = runtime.prepare(frozen, scene)
    changed = copy.deepcopy(scene)
    changed["kin"][5:, :, 0] += 1000
    other = runtime.prepare(frozen, changed)
    labels = [key for key in baseline["geometry"] if key.startswith("label_")]
    assert labels and any(not torch.equal(baseline["geometry"][key][:, :5],
                                          other["geometry"][key][:, :5]) for key in labels)
    for key in baseline["geometry"]:
        if key not in labels:  # label_* is the recorded successor, never a model input
            torch.testing.assert_close(baseline["geometry"][key][:, :5],
                                       other["geometry"][key][:, :5], atol=0, rtol=0)
    assert baseline["metadata"]["CAV_controls"] == "current_timeline"
    assert torch.equal(baseline["proposals"][..., 1:4], torch.tensor(scene["kin"][:-1, :, 1:4]))


def test_runtime_risk_cohort_does_not_drop_first_step_zone_entry(tmp_path):
    ep = write_episode(tmp_path)
    with pytest.raises(ValueError, match="no recorded ramp sequences"):
        legacy.build_dataset([ep])
    risk = np.zeros((8, 3), bool)
    risk[1, 0] = True  # causal motion reached the zone in this interval
    corrected, _ = legacy.build_dataset([ep], runtime_risk=risk)
    assert len(corrected) == 1 and corrected[0]["event_row"] == 2
    assert corrected[0]["entry_row"] == 1


def test_successor_diagnostic_checks_all_targets_and_preserves_prechange_lanes():
    from wms.model.merge_pivot import geometry
    from wms.physics.config import PhysicsConfig

    raw = np.array([[[20, 0, 0, 1, 0, 0, 5], [26, 0, 1, 0, 0, 0, 2],
                     [40, 0, 1, 0, 0, 0, 30]],
                    [[20, 0, 1, 0, 0, 0, 5], [26, 0, 1, 0, 0, 0, 2],
                     [40, 1, 0, 0, 0, 0, 30]]], dtype=np.float32)
    geo = {key: value[:, None] for key, value in geometry(
        torch.tensor(raw[:1]), torch.ones(1, 3, dtype=torch.bool), PhysicsConfig()).items()}
    params = dict(s_f=2., T_f=1., b_f=2., s_l=2., T_l=1., b_l=2.)
    details = audit.event_details(geo, raw, np.tile(np.arange(3), (2, 1)),
                                   torch.zeros(1, 2, 3, dtype=torch.bool), 0, 0, params)
    assert details["leader"]["slot"] == 1
    assert details["diagnostic_recorded_motion_pre_lanes_intersections"][0]["slot"] == 2
    # the only intersecting body is AHEAD of the ego and leaves the target lane in the same
    # step: SUMO's front-first sequential lane changes have already vacated it.
    assert details["veto_cause"] == "same_step_vacated_leader"
    assert len(details["all_pre_target_neighbours"]) == 2


def test_attached_runtime_geometry_cannot_regain_likelihood_after_risk_exit(frozen, tmp_path):
    from tools import merge_fit_stage2a as fitting

    ep = write_episode(tmp_path)
    kin = np.load(tmp_path / "veh_kin.npy", allow_pickle=False)
    kin[:7, 0, 3], kin[:7, 0, 2] = 1., 0.
    kin[2, 0, 0] = 80.  # observed route/zone exit, then an inconsistent return
    kin[3:7, 0, 0] = np.arange(4) + 10
    np.save(tmp_path / "veh_kin.npy", kin)
    seqs, _ = heads.build_dataset([ep], frozen, allow_unverified_physics=True)
    seq = next(s for s in seqs if s["event"])
    assert seq["event_row"] == 7 and seq["absorbed_before_event"]
    assert not any(seq["lifecycle_active"][2 - seq["entry_row"]:])
    initial = {k: float(v) for k, v in frozen.core.params().items()}
    initial["sigma"] = float(frozen.core.alpha_scale())
    for kind in ("F1", "P1-P3", "M0"):
        law = fitting.make_model(kind, initial, frozen)
        assert torch.isposinf(law.nll(fitting.pack([seq]))).all()
