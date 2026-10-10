"""Synthetic CPU contracts for the optional controller-driven training path."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from output.wd01.wm_n1.wms import train_controllers
from output.wd01.wm_n1.wms.plan_steps import project
from output.wd01.wm_n1.wms.tests.test_wms import small_model, state
from output.wd01.wm_n1.wms.train import (
    TrainDataRefusal, TrajectoryDataset, WmSTrainConfig, WmSTrainer,
)
from output.wd01.wm_n1.wms.train_controllers import TimelineControllers
from output.wd01.wm_n1.wms.train_guards import TrainingWindowGuard


def _current():
    current = state(batch=1)
    current["veh_kin"] = current["veh_kin"].clone()
    current["veh_kin"][..., 0] = torch.tensor([[0., 30., 60., 90.]])
    current["veh_kin"][..., 6] = 5.
    current["s_m"] = current["veh_kin"][..., 0]
    current["length_m"] = current["veh_kin"][..., 6]
    current["is_cav"][:, 0] = True
    return current


def _timeline(actions, *, acceleration=0., target=0.):
    ids = torch.as_tensor(actions, dtype=torch.long).unsqueeze(0)
    return {"controller": ids,
            "acceleration_mps2": torch.full(ids.shape, acceleration),
            "target_speed_mps": torch.full(ids.shape, target),
            "control_dt_s": 1.0}


def _dependencies(*, reject_pi=False):
    """Inject the complete dependency interface; never import external PI files."""
    audit = SimpleNamespace(circles=[], events=[], conversions=[])
    cfg = SimpleNamespace(sensor_range_m=100., acceleration_max_mps2=3.,
                          acceleration_min_mps2=-4.)

    class Circles:
        def __init__(self, config):
            assert config is cfg
            self.calls = []
            audit.circles.append(self)

        def decide(self, frame, *, now_s):
            self.calls.append((frame, now_s))
            # A changed second command makes both sample/hold halves observable.
            delta = .8 if len(self.calls) == 1 else -.4
            return SimpleNamespace(target_speed_mps=frame.speed_mps + delta)

    class Neighbor:
        def __init__(self, gap_m, relative_speed_mps):
            self.gap_m, self.relative_speed_mps = gap_m, relative_speed_mps

    class PI:
        def __init__(self, *, sample_period_s, parameters):
            assert sample_period_s == 1.0
            self.pending = None
            self.speeds = {}

        def reset(self, vid, *, measured_ego_speed_mps):
            self.speeds[vid] = measured_ego_speed_mps
            audit.events.append(("reset", vid, measured_ego_speed_mps))

        def propose(self, vid, **kwargs):
            assert vid in self.speeds and self.pending is None
            self.pending = SimpleNamespace(vehicle_id=vid,
                                           command_mps=kwargs["ego_speed_mps"] + .6)
            audit.events.append(("propose", vid, kwargs))
            return self.pending

        def commit(self, proposal, *, writer_ack):
            assert writer_ack is True and proposal is self.pending
            audit.events.append(("commit", proposal.vehicle_id, writer_ack))
            self.pending = None

        def discard(self, proposal):
            assert proposal is self.pending
            audit.events.append(("discard", proposal.vehicle_id))
            self.pending = None

        def invalidate(self, *, vehicle_ids, native_ids):
            assert vehicle_ids == native_ids
            for vid in vehicle_ids:
                self.speeds.pop(vid, None)
                audit.events.append(("invalidate", vid))

    def direct_target(frame, acceleration, config):
        assert config is cfg
        audit.conversions.append((frame.vehicle_id, acceleration))
        return ((None, None) if reject_pi else
                (SimpleNamespace(), frame.speed_mps + acceleration))

    dependencies = (cfg, Circles, direct_target, Neighbor, SimpleNamespace,
                    PI, lambda *args: args, SimpleNamespace)
    return dependencies, audit


def _model(monkeypatch, *, native_acceleration=4.):
    model = small_model().eval()
    monkeypatch.setattr(model, "physics_accel", lambda current, *args, **kwargs:
                        torch.full_like(current["v_mps"], native_acceleration))
    return model


def test_damping_holds_both_physics_halves_across_odd_chunk_boundary(monkeypatch):
    model = _model(monkeypatch)
    current = _current()
    timeline = _timeline([[1, 0, 0, 0], [1, 0, 0, 0]])
    dependencies, audit = _dependencies()
    callback = TimelineControllers(model, timeline, dependencies=dependencies)
    first = model.rollout(current, n_steps=1, cav_controller=callback)
    rest = model.rollout(first["state"], n_steps=3, k0=1,
                         history=first["history"], history_mask=first["history_mask"],
                         cav_controller=callback)
    acceleration = torch.cat((first["accel"], rest["accel"]), dim=1)[0, :, 0]
    torch.testing.assert_close(acceleration, torch.tensor([.8, .8, -.4, -.4]))
    assert callback.step_index == 4
    assert len(audit.circles) == 1
    assert [now for _, now in audit.circles[0].calls] == [0., 1.]
    assert audit.circles[0].calls[0][0].front.gap_m == 25.

    full_dependencies, _ = _dependencies()
    full = model.rollout(current, n_steps=4, cav_controller=TimelineControllers(
        model, timeline, dependencies=full_dependencies))
    torch.testing.assert_close(rest["positions"], full["positions"][:, 1:])
    torch.testing.assert_close(rest["speeds"], full["speeds"][:, 1:])


def test_pi_switch_out_invalidates_and_reentry_resets_before_commit(monkeypatch):
    model = _model(monkeypatch)
    dependencies, audit = _dependencies()
    callback = TimelineControllers(model, _timeline([
        [2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]]), dependencies=dependencies)
    roll = model.rollout(_current(), n_steps=6, cav_controller=callback)
    torch.testing.assert_close(roll["accel"][0, :, 0],
                               torch.tensor([.6, .6, 4., 4., .6, .6]))
    assert [event[0] for event in audit.events] == [
        "reset", "propose", "commit", "invalidate", "reset", "propose", "commit"]
    resets = [event[2] for event in audit.events if event[0] == "reset"]
    assert resets == pytest.approx([12., 16.6])
    proposals = [event[2] for event in audit.events if event[0] == "propose"]
    assert [proposal["timestamp_s"] for proposal in proposals] == [0., 2.]
    assert [delta for _, delta in audit.conversions] == pytest.approx([.6, .6])
    assert callback.pi_active == {"0:0"}


def test_pi_rejected_writer_discards_and_invalidates_instead_of_committing(monkeypatch):
    model = _model(monkeypatch)
    dependencies, audit = _dependencies(reject_pi=True)
    callback = TimelineControllers(model, _timeline([[2, 0, 0, 0]]),
                                   dependencies=dependencies)
    output = callback(_current())
    assert [event[0] for event in audit.events] == ["reset", "propose", "discard", "invalidate"]
    assert callback.pi_active == set() and callback.active == {}
    assert output[0, 0, 3].item() == 4.


def test_pi_loses_leader_invalidates_then_resets_on_covered_reentry(monkeypatch):
    model = _model(monkeypatch)
    dependencies, audit = _dependencies()
    callback = TimelineControllers(model, _timeline([[2, 0, 0, 0]] * 3),
                                   dependencies=dependencies)
    current = _current()
    callback(current)
    callback(current)
    current["present"][:, 1] = False
    assert callback(current)[0, 0, 3].item() == 4.
    callback(current)
    current["present"][:, 1] = True
    callback(current)
    assert [event[0] for event in audit.events] == [
        "reset", "propose", "commit", "invalidate", "reset", "propose", "commit"]


@pytest.mark.parametrize("hdv_action", [0, 1, 2, 3])
def test_native_and_hdv_have_no_timeline_override(monkeypatch, hdv_action):
    model = _model(monkeypatch)
    dependencies, audit = _dependencies()
    callback = TimelineControllers(model, _timeline([[0, hdv_action, 0, 3]],
                                   acceleration=-3., target=1.), dependencies=dependencies)
    current = _current()
    # An absent CAV must not acquire an override either.
    current["is_cav"][:, 3] = True
    output = callback(current)
    torch.testing.assert_close(output[..., 3], torch.full((1, 4), 4.))
    torch.testing.assert_close(output[..., 2], current["v_mps"] + 2.)
    assert callback.active == {} and callback.pi_active == set()
    assert audit.circles == [] and audit.events == []


@pytest.mark.parametrize("acceleration,target,expected", [
    (.4, 40., [.4, .4]),       # recorded acceleration, not the unused zero default
    (2., 12.75, [1.5, 0.]),   # target limits both halves, without overshooting
    (-1., 40., [-1., -1.]),
])
def test_direct_uses_recorded_acceleration_and_target(monkeypatch, acceleration, target, expected):
    model = _model(monkeypatch)
    callback = TimelineControllers(model, _timeline([[3, 3, 0, 0]],
                                   acceleration=acceleration, target=target))
    roll = model.rollout(_current(), n_steps=2, cav_controller=callback)
    torch.testing.assert_close(roll["accel"][0, :, 0], torch.tensor(expected))
    assert set(callback.active) == {"0:0"}
    assert callback.active["0:0"] == pytest.approx(acceleration)
    assert callback.step_index == 2


def test_safe_native_plant_can_brake_below_recorded_request(monkeypatch):
    model = _model(monkeypatch, native_acceleration=-2.)
    callback = TimelineControllers(model, _timeline([[3, 0, 0, 0]],
                                   acceleration=1., target=40.))
    assert callback(_current())[0, 0, 3].item() == -2.


def test_mixed_base_and_entrant_cavs_tick_callback_once_per_physics_step(monkeypatch):
    model = _model(monkeypatch)
    dependencies, audit = _dependencies()
    callback = TimelineControllers(model, _timeline([[1, 1, 0, 0]] * 2),
                                   dependencies=dependencies)
    current = _current()
    current["is_cav"][:, 1] = True
    current["entrant"] = torch.tensor([[False, True, False, False]])
    model.step(current, cav_controller=callback)
    assert callback.step_index == 1, "one physics step must not consume both timeline halves"
    assert len(audit.circles) == 2
    assert all(len(controller.calls) == 1 for controller in audit.circles)


class _Corpus:
    """Two disjoint runs; position mapping intentionally differs from item order."""
    def __init__(self):
        self._items = [{"tick": 6, "world": "clean"}, {"tick": 0, "world": "suspect"}]
        self._order = [1, 0]
        self.rows = np.zeros((12, 4, 14), dtype=np.float32)
        self.mask = np.ones((12, 4), dtype=bool)
        self.rows[:, 0, 7] = 1.
        self.reads = []

    def _run_end(self, tick):
        return 6 if tick < 6 else 12

    def _array(self, name):
        self.reads.append(name)
        return {"veh_step": self.rows, "veh_mask": self.mask}[name]


@pytest.mark.parametrize("tick", range(5))
@pytest.mark.parametrize("slot", [0, 3], ids=["t0-cav", "entrant-cav"])
@pytest.mark.parametrize("channel", [10, 11, 13], ids=["pi", "wall", "wall-owner"])
def test_guard_drops_forbidden_action_at_any_tick_and_cav(tick, slot, channel):
    corpus = _Corpus()
    if slot == 3:
        corpus.mask[:tick, slot] = False
        corpus.rows[tick:, slot, 7] = 1.
    corpus.rows[tick, slot, channel] = 1.
    kept, receipt = TrainingWindowGuard(horizon_steps=4, leave_pi_wall_out=True)(corpus, [0, 1])
    assert kept == [1]
    assert receipt == {"leave_pi_wall_out": True, "controller_windows_dropped": 1,
                       "n_kept": 1, "held_world_guard": False}


def test_guard_ignores_hdv_indicators_and_nonfinite_hdv_channels():
    corpus = _Corpus()
    corpus.rows[:, 1, [10, 11, 13]] = 1.
    corpus.rows[:, 2, 8:14] = np.nan
    kept, receipt = TrainingWindowGuard(horizon_steps=4, leave_pi_wall_out=True)(corpus, [0, 1])
    assert kept == [0, 1] and receipt["controller_windows_dropped"] == 0


def test_guard_only_scans_requested_window_and_does_not_cross_runs():
    corpus = _Corpus()
    corpus.rows[5, 0, 10] = 1.  # outside the first requested horizon
    guard = TrainingWindowGuard(horizon_steps=4, leave_pi_wall_out=True)
    assert guard(corpus, [0])[0] == [0]
    corpus.rows[5, 0, 10] = 0.
    corpus.rows[6, 0, 10] = 1.  # next run must not contaminate the first
    assert TrainingWindowGuard(horizon_steps=99, leave_pi_wall_out=True)(corpus, [0])[0] == [0]


@pytest.mark.parametrize("channel", range(8, 14))
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_guard_refuses_nonfinite_cav_ownership(channel, invalid):
    corpus = _Corpus()
    corpus.rows[4, 0, channel] = invalid
    with pytest.raises(TrainDataRefusal, match="nonfinite controller ownership"):
        TrainingWindowGuard(horizon_steps=4, leave_pi_wall_out=True)(corpus, [0])


def test_guard_refuses_held_world_before_any_row_access():
    corpus = _Corpus()
    calls = []

    class Heldout:
        def refuse_worlds(self, worlds, *, where):
            calls.append((worlds, where))
            raise TrainDataRefusal("held world is sealed")

    with pytest.raises(TrainDataRefusal, match="held world"):
        TrainingWindowGuard(horizon_steps=4, heldout=Heldout(),
                            leave_pi_wall_out=True)(corpus, [0])
    assert calls == [(["suspect"], "original TRAIN windows")]
    assert corpus.reads == []


def test_guard_disabled_controller_filter_is_metadata_only():
    corpus = _Corpus()
    corpus.rows[..., 8:14] = np.nan
    kept, receipt = TrainingWindowGuard(horizon_steps=4)(corpus, [1, 0])
    assert kept == [1, 0] and receipt["controller_windows_dropped"] == 0
    assert corpus.reads == []


def _write_log(tmp_path, rows):
    log = tmp_path / "smoke.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return log


def test_project_preserves_explicit_steps_and_uses_matching_stage_timings(tmp_path):
    log = _write_log(tmp_path, [
        {"stage_s": 1., "seconds": 1000.},
        {"stage_s": 2., "seconds": 1000.},
        {"stage_s": 1., "seconds": 2.},
        {"stage_s": 2., "seconds": 4.},
        {"stage_s": 1., "seconds": 4.},
        {"stage_s": 2., "seconds": 8.},
    ])
    receipt = project(log, [100, 200], stages_s=[1., 2.])
    assert receipt["stage_steps"] == [100, 200]
    assert receipt["per_stage"]["1.0"]["median_step_s"] == 3.
    assert receipt["per_stage"]["2.0"]["median_step_s"] == 6.
    assert receipt["projected_hours"] == pytest.approx(1500. / 3600.)
    assert receipt["estimated_hours"] == receipt["projected_hours"]


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -float("inf"), -1., 0.])
def test_project_refuses_nonfinite_or_nonpositive_timing(tmp_path, seconds):
    log = _write_log(tmp_path, [{"stage_s": 1., "seconds": seconds}])
    with pytest.raises(ValueError, match="timing must be finite and positive"):
        project(log, [100], stages_s=[1.])


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -1.])
def test_project_bad_retained_timing_cannot_be_hidden_by_valid_median(tmp_path, seconds):
    log = _write_log(tmp_path, [
        {"stage_s": 1., "seconds": 100.},  # deliberately dropped warmup
        {"stage_s": 1., "seconds": seconds},
        {"stage_s": 1., "seconds": 1.},
        {"stage_s": 1., "seconds": 1.},
    ])
    with pytest.raises(ValueError, match="timing must be finite and positive"):
        project(log, [100], stages_s=[1.])


def test_project_refuses_missing_stage_instead_of_reusing_other_timing(tmp_path):
    log = _write_log(tmp_path, [{"stage_s": 1., "seconds": 2.}])
    with pytest.raises(ValueError, match="no records for stage 2.0"):
        project(log, [100, 200], stages_s=[1., 2.])


def test_trainer_persists_callback_across_chunks_and_logs_six_gradient_shares(
        monkeypatch, tmp_path):
    model = _model(monkeypatch)
    current = _current()
    states = np.repeat(current["veh_kin"].numpy()[:, None], 5, axis=1)
    states[..., 0] += np.arange(5, dtype=np.float32)[None, :, None] * 6.
    dataset = TrajectoryDataset.from_arrays({
        "states": states,
        "present": np.repeat(current["present"].numpy()[:, None], 5, axis=1),
        "is_cav": np.repeat(current["is_cav"].numpy()[:, None], 5, axis=1),
        "controller_timeline": np.array([[[1, 0, 0, 0]] * 2]),
        "controller_timeline_schema": "wms_action_timeline/1",
    })
    callbacks, audits = [], []

    def make_callback(model, timeline):
        dependencies, audit = _dependencies()
        callback = TimelineControllers(model, timeline, dependencies=dependencies)
        callbacks.append(callback)
        audits.append(audit)
        return callback

    monkeypatch.setattr(train_controllers, "TimelineControllers", make_callback)
    cfg = WmSTrainConfig(
        device="cpu", threads=1, batch_size=1, num_workers=0,
        unroll_curriculum_s=(1., 2.), stage_steps=(1, 1), truncation_steps=1,
        cav_drive="closed_loop", loss_share_logging=True,
        traffic_loss=True, traffic_bin_s=.5, traffic_ramp_start_s=0.,
        traffic_ramp_end_s=.5, conservation_enabled=True, conservation_weight=.1,
        stability_weight=.1, stability_margin=0., out_dir=str(tmp_path),
    )
    trainer = WmSTrainer(model, cfg)
    receipt = trainer.fit(dataset)
    assert receipt["device"] == "cpu" and receipt["steps"] == 2
    assert [callback.step_index for callback in callbacks] == [2, 4]
    assert [[now for _, now in audit.circles[0].calls] for audit in audits] == [[0.], [0., 1.]]
    records = [json.loads(line) for line in (tmp_path / "loss_share.jsonl").read_text(
        encoding="utf-8").splitlines()]
    training = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert [row["stage_s"] for row in records] == [1., 2.]
    assert [row["step"] for row in records] == [1, 2]
    terms = {"position", "speed", "lane", "traffic", "stability", "conservation"}
    for row, logged in zip(records, training):
        assert row == {key: logged[key] for key in row}
        assert set(row["loss_share"]) == set(row["loss_grad_norm"]) == terms
        norms = row["loss_grad_norm"]
        assert all(np.isfinite(value) and value >= 0. for value in norms.values())
        assert sum(norms.values()) > 0.
        assert all(np.isfinite(value) and 0. <= value <= 1.
                   for value in row["loss_share"].values())
        assert sum(row["loss_share"].values()) == pytest.approx(1.)
        for name in terms:
            assert row["loss_share"][name] == pytest.approx(norms[name] / sum(norms.values()))
