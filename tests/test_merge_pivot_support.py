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


def _term_for(leader_next_lane):
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
    batch = collate([TrajectoryDataset.from_arrays(arrays, merge_pivot_history=True)[0]])
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5",
                             merge_decision="first_passage")
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), 2)
    targets = build_targets(inp, batch, settings, .5, torch.device("cpu"), physics=model.physics)
    targets.prepared["merge_pivot_source"] = model, inp
    term = MergePivotTerm(MergePivotConfig(tf_weight=1.))
    term.prepare(targets)
    return targets.prepared["merge_pivot"]


def test_prepare_excludes_a_vacated_leader_event_and_keeps_a_real_conflict():
    gone = _term_for(leader_next_lane=0)
    assert gone["support_counts"]["vetoed_events"] == 1
    assert gone["support_counts"]["excluded_events"] == 1
    assert not gone["event"].any() and not gone["risk"][0, :, 0].any()
    stays = _term_for(leader_next_lane=TARGET)
    assert stays["support_counts"]["vetoed_events"] == 1
    assert stays["support_counts"]["excluded_events"] == 0
    assert stays["support_counts"]["unexplained_vetoed_events"] == 1
    assert stays["event"].any()  # stays a zero-mass failure, visible to the gate
