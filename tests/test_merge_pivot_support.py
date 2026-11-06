"""Which vetoed recorded merges are P1-unrepresentable, and how they leave the likelihood."""
import numpy as np
import torch

from wms.data.collate import collate
from wms.data.dataset import TrajectoryDataset
from wms.model import ModelSpec, WmSModel
from wms.physics.config import PhysicsConfig
from wms.train.inputs import build_targets, prepare_inputs
from wms.train.losses.merge_pivot import MergePivotConfig, MergePivotTerm
from wms.train.losses.merge_pivot_support import classify_vetoed_events, sequence_rows
from wms.train.settings import TrainSettings

TARGET = 1


def geo_for(*others, ego_x=40.0, ego_len=5.0, eligible=False):
    """One decision tick, slot 0 = vetoed ego; ``others`` = dict(x, lane, length, junction, linked)."""
    k = 1 + len(others)
    zeros = lambda dtype=torch.float32: torch.zeros(1, 1, k, dtype=dtype)  # noqa: E731
    geo = {"eligible": torch.zeros(1, 1, k, dtype=torch.bool), "pre_target": zeros(torch.bool),
           "label_succ_x": zeros(), "label_succ_length": zeros(),
           "label_succ_lane": zeros(torch.long), "label_linked": zeros(torch.bool),
           "label_pre_junction": zeros(torch.bool)}
    geo["eligible"][0, 0, 0] = eligible
    geo["label_succ_x"][0, 0, 0], geo["label_succ_length"][0, 0, 0] = ego_x, ego_len
    geo["label_succ_lane"][0, 0, 0] = TARGET
    for j, o in enumerate(others, 1):
        geo["pre_target"][0, 0, j] = True
        geo["label_succ_x"][0, 0, j] = o["x"]
        geo["label_succ_length"][0, 0, j] = o.get("length", 5.0)
        geo["label_succ_lane"][0, 0, j] = o.get("lane", TARGET)
        geo["label_linked"][0, 0, j] = o.get("linked", True)
        geo["label_pre_junction"][0, 0, j] = o.get("junction", False)
    event = torch.zeros(1, 1, k, dtype=torch.bool)
    event[0, 0, 0] = True
    return geo, event


def classify(*others, **kwargs):
    geo, event = geo_for(*others, **kwargs)
    return {key: bool(val[0, 0, 0]) for key, val in
            classify_vetoed_events(geo, event, target_lane=TARGET).items()}


def test_leader_ahead_that_leaves_the_target_lane_is_explained():
    out = classify(dict(x=44.0, lane=0))  # [39, 44] intersects the ego [35, 40]
    assert out == dict(vetoed=True, recorded_clear=False, vacated=True, junction=False, explained=True)


def test_blockers_that_stay_or_are_behind_or_mixed_are_never_explained():
    assert not classify(dict(x=44.0, lane=TARGET))["explained"]       # still blocks: real conflict
    assert not classify(dict(x=37.0, lane=0))["explained"]            # leaves but is BEHIND the ego
    assert not classify(dict(x=44.0, lane=0), dict(x=38.0, lane=TARGET))["explained"]  # one stays
    out = classify(dict(x=44.0, lane=0, linked=False))
    assert out["recorded_clear"] and not out["explained"]             # unobserved successor: no claim


def test_clear_recorded_successor_is_the_models_own_motion_and_is_not_explained():
    out = classify(dict(x=80.0, lane=TARGET))  # far away in the recorded successor
    assert out["vetoed"] and out["recorded_clear"] and not out["explained"]
    assert not classify(dict(x=44.0, lane=0), eligible=True)["vetoed"]


def test_junction_internal_target_vehicle_behind_the_ego_is_explained():
    out = classify(dict(x=38.0, lane=TARGET, junction=True))
    assert out["junction"] and out["explained"] and not out["vacated"]
    assert not classify(dict(x=38.0, lane=TARGET, junction=False))["explained"]
    assert classify(dict(x=44.0, lane=0), dict(x=38.0, lane=TARGET, junction=True))["explained"]


def test_touching_bodies_do_not_intersect():
    assert classify(dict(x=45.0, lane=0))["recorded_clear"]  # [40, 45] only touches [35, 40]


def test_excluded_event_removes_its_whole_risk_sequence_not_just_the_event():
    n, k = 6, 2
    risk = torch.zeros(1, n, k, dtype=torch.bool)
    risk[0, 1:5, 0] = True      # slot 0: risk rows 1..4, event at 4
    risk[0, :, 1] = True        # slot 1 untouched
    reset = torch.zeros(1, n, k, dtype=torch.bool)
    excluded = torch.zeros_like(risk)
    excluded[0, 4, 0] = True
    rows = sequence_rows(excluded, risk, reset)
    assert rows[0, :, 0].tolist() == [False, True, True, True, True, False]
    assert not rows[0, :, 1].any()
    reset[0, 3, 0] = True       # a new identity starts at row 3: only ITS rows (3, 4) belong
    assert sequence_rows(excluded, risk, reset)[0, :, 0].tolist() == [False, False, False, True,
                                                                      True, False]


def _term_for(leader_next_lane, perturb=False, full=False):
    """Ramp car (slot 0) waiting beside a target-lane leader that overlaps it before the move."""
    states = np.zeros((6, 3, 7), np.float32)
    states[..., 6] = 5.
    states[:, 0, 0], states[:, 0, 3] = 40., 1.          # ramp lane, in the zone
    states[:, 1, 0], states[:, 1, 2] = 43., 1.          # lane 1 leader, overlaps [38, 43]
    states[:, 2, 0], states[:, 2, 2] = -30., 1.         # far away
    states[4:, 0, 3], states[4:, 0, 2] = 0., 1.         # merges in the first window step (3 -> 4)
    states[4:, 1, 2], states[4:, 1, leader_next_lane + 1] = 0., 1.
    states[4:, 1, 0] = 44.
    arrays = dict(states=states[3:], present=np.ones((3, 3), bool), history_kin=states[:4],
                  history_mask=np.ones((4, 3), bool), history_target_mask=np.ones((3, 3), bool),
                  history_is_cav=np.zeros((4, 3), bool))
    torch.manual_seed(32)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", lane_model="hazard_v5",
                               merge_decision="first_passage"), physics=PhysicsConfig()).eval()
    if perturb:  # the cohort must not depend on the trained boundary parameters
        with torch.no_grad():
            for p in model.merge_pivot.parameters():
                p.add_(0.7)
    batch = collate([TrajectoryDataset.from_arrays(arrays, merge_pivot_history=True)[0]])
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5",
                             merge_decision="first_passage")
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), 2)
    targets = build_targets(inp, batch, settings, .5, torch.device("cpu"), physics=model.physics)
    targets.prepared["merge_pivot_source"] = model, inp
    term = MergePivotTerm(MergePivotConfig(tf_weight=1.))
    term.prepare(targets)
    if full:
        return term, targets, model, inp
    return targets.prepared["merge_pivot"]


def test_prepare_excludes_every_vetoed_event_and_counts_it_by_class():
    # Owner decision 2026-10-22: ALL recorded events vetoed under the WM's one-step geometry leave
    # the loss with their whole chain (counted), not only the "explained" classes.
    gone = _term_for(leader_next_lane=0)
    assert gone["support_counts"]["vetoed_events"] == 1
    assert gone["support_counts"]["excluded_events"] == 1
    assert gone["support_counts"]["excluded_cars"] == 1
    assert gone["support_counts"]["excluded_event_vetoed_explained"] == 1
    assert not gone["event"].any() and not gone["risk"][0, :, 0].any()
    stays = _term_for(leader_next_lane=TARGET)
    assert stays["support_counts"]["vetoed_events"] == 1
    assert stays["support_counts"]["excluded_events"] == 1
    assert stays["support_counts"]["excluded_event_vetoed_other"] == 1
    assert not stays["event"].any() and not stays["risk"][0, :, 0].any()


def test_all_excluded_batch_has_finite_zero_loss_and_is_logged():
    from wms.train.losses.base import LossContext
    from wms.train.losses.merge_pivot import merge_pivot_metrics

    term, targets, model, inp = _term_for(leader_next_lane=TARGET, full=True)
    prep = targets.prepared["merge_pivot"]
    assert prep["scored_cars"] == 0 and prep["n_cars"] == 0
    value = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=2)).value
    assert torch.isfinite(value) and float(value.detach()) == 0.0
    value.backward()  # graph exists, gradients are finite zeros
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.merge_pivot.parameters())
    metrics = merge_pivot_metrics(prep, .5)
    assert metrics["merge_pivot_support_empty_retained_batch"] == 1
    assert metrics["merge_pivot_support_original_cars"] == 1
    assert metrics["merge_pivot_support_retained_cars"] == 0
    assert metrics["merge_pivot_support_excluded_cars"] == 1
    assert metrics["merge_pivot_support_excluded_event_cars"] == 1
    assert metrics["merge_pivot_support_excluded_censor_cars"] == 0
    assert metrics["merge_pivot_support_exclusion_version"] == 1
    assert metrics["merge_pivot_tf_scored_cars"] == 0
    assert metrics["merge_pivot_tf_zero_mass_events"] == 0 and metrics["merge_pivot_tf_events"] == 0


def test_cohort_membership_cannot_change_with_the_trained_boundary_parameters():
    base = _term_for(leader_next_lane=0)
    moved = _term_for(leader_next_lane=0, perturb=True)
    for key in ("risk", "event", "known", "reset"):
        assert torch.equal(base[key], moved[key])
    for key, value in base["support_counts"].items():
        assert moved["support_counts"][key] == value
    for key in ("eligible", "physical_eligible", "ramp", "has_f", "has_l", "gap_f", "gap_l"):
        assert torch.equal(base["geometry"][key], moved["geometry"][key])


def _scene_arrays(leader_next_lane):
    """The `_term_for` scene as a whole recorded episode (rows 0..5), for the audit consumer."""
    states = np.zeros((6, 3, 7), np.float32)
    states[..., 6] = 5.
    states[:, 0, 0], states[:, 0, 3] = 40., 1.
    states[:, 1, 0], states[:, 1, 2] = 43., 1.
    states[:, 2, 0], states[:, 2, 2] = -30., 1.
    states[4:, 0, 3], states[4:, 0, 2] = 0., 1.
    states[4:, 1, 2], states[4:, 1, leader_next_lane + 1] = 0., 1.
    states[4:, 1, 0] = 44.
    return dict(veh_kin=states, veh_mask=np.ones((6, 3), bool), target_mask=np.ones((5, 3), bool),
                is_cav=np.zeros((6, 3), bool), vid=np.broadcast_to(np.array([10, 11, 12]), (6, 3)),
                time_s=np.arange(6) * .5, control_time_s=np.arange(3),
                controller=np.zeros((3, 3), np.int64),
                issued_acceleration_mps2=np.zeros((3, 3), np.float32),
                issued_target_speed_mps=np.zeros((3, 3), np.float32))


def test_training_loss_and_audit_apply_the_same_exclusion_to_the_same_scene():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "r9_support_audit_parity", Path(__file__).resolve().parents[1] / "tools/r9_support_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    for lane in (0, TARGET):    # an explained and an unexplained veto: both leave scoring
        term, targets, model, inp = _term_for(leader_next_lane=lane, full=True)
        train = targets.prepared["merge_pivot"]["support_counts"]
        arrays = _scene_arrays(lane)
        rows, counts = audit.audit_episode(model, SimpleNamespace(array=arrays.__getitem__),
                                           dict(episode="synthetic", warmup_s=0.), "test",
                                           window_s=10.)
        assert train["excluded_cars"] == counts["excluded_cars"] == 1
        assert train["excluded_events"] == counts["counted_exclusion_events"] == 1
        assert train["excluded_event_cars"] == counts["excluded_event_cars"] == 1
        for name in ("event_vetoed_explained", "event_vetoed_wm_motion", "event_vetoed_other"):
            assert train["excluded_" + name] == counts["excluded_" + name]
        assert counts["after_zero_mass_events"] == 0 and rows[0]["support_verdict"] == "excluded_counted"
