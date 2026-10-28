"""Birth commands, arm isolation, observable classes and frozen OFF contract."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control.entrant import EntrantControllers, RowControllers, inferred_timeline
from wms.data.action_timeline import CONTROLLERS
from wms.model.entries import EntranceHistory
from wms.physics.config import PhysicsConfig
from wms.train.inputs import prepare_inputs
from wms.train.settings import TrainSettings
from wms.train.scoring_random import scoring_rng


def batch():
    kin = torch.zeros(1, 3, 1, 7)
    kin[..., 1] = 1; kin[..., 4] = 10; kin[..., 6] = 5
    return dict(states=kin, present=torch.ones(1, 3, 1, dtype=torch.bool),
                is_cav=torch.ones(1, 3, 1, dtype=torch.bool),
                entry_feat=torch.tensor(EntranceHistory.defaults().to_features())[None],
                entry_queue=torch.tensor([[1., 0., .1, .1, 1., 1.]]),
                action_timeline=dict(controller=torch.zeros(1, 2, 1, dtype=torch.long),
                    acceleration_mps2=torch.zeros(1, 2, 1),
                    target_speed_mps=torch.zeros(1, 2, 1), control_dt_s=1.))


def fixture():
    model = SimpleNamespace(physics=PhysicsConfig(), long_history=False,
        rollout_mode="driver_params", merge_decision="first_passage")
    settings = TrainSettings(entries_v2=True, entry_pad=2, entry_pad_auto=True,
        entry_queue_state=True, cav_drive="closed_loop")
    return model, settings


def test_off_keeps_original_tensors_timeline_and_queue_class_contract():
    model, settings = fixture()
    a = prepare_inputs(batch(), model, settings, torch.device("cpu"), 2)
    b = prepare_inputs(batch(), model, replace(settings, entrant_control=False), torch.device("cpu"), 2,
                       entrant_controller=lambda _: (_ for _ in ()).throw(AssertionError("called")))
    for k in a.initial:
        assert torch.equal(a.initial[k], b.initial[k])
    for k in a.action_timeline:
        x, y = a.action_timeline[k], b.action_timeline[k]
        assert torch.equal(x, y) if torch.is_tensor(x) else x == y
    assert a.entries.queues[0].cav_share is None
    assert type(a.cav_controller) is type(b.cav_controller)


def test_on_uses_observable_queue_class_and_persistent_feedback_for_births():
    model, settings = fixture()
    seen = []
    def policy(st):
        seen.append(st["s_m"].clone())
        return torch.full((*st["s_m"].shape, 4), -2.)
    inp = prepare_inputs(batch(), model, replace(settings, entrant_control=True),
                         torch.device("cpu"), 2, entrant_controller=policy)
    assert np.array_equal(inp.entries.queues[0].cav_share, [1., 1.])
    inp.cav_controller.recorded = lambda st: torch.zeros((*st["s_m"].shape, 4))
    st = dict(s_m=torch.tensor([[10., 20.]]), present=torch.tensor([[True, True]]),
              is_cav=torch.tensor([[True, True]]), entrant=torch.tensor([[False, True]]))
    out = inp.cav_controller(st)
    assert torch.equal(out[0, 0], torch.zeros(4))
    assert torch.equal(out[0, 1], torch.full((4,), -2.))
    st["s_m"] += 1
    inp.cav_controller(st)
    assert len(seen) == 2 and seen[1][0, 1] == 21


def test_direct_arm_without_algorithm_is_refused_not_copied():
    timeline = {"controller": torch.tensor([[[CONTROLLERS.index("direct")]]])}
    with pytest.raises(ValueError, match="explicit feedback"):
        inferred_timeline(timeline, 1)


def test_row_callbacks_receive_only_own_scene_and_keep_identity_history():
    seen = [[], []]
    def cb(i):
        def run(st):
            seen[i].append(st["s_m"].clone())
            return torch.full((*st["s_m"].shape, 4), float(i))
        return run
    controller = RowControllers([cb(0), cb(1)])
    st = dict(s_m=torch.tensor([[1., 2.], [3., 4.]]),
              present=torch.ones(2, 2, dtype=torch.bool))
    out = controller(st)
    assert out[0].sum() == 0 and out[1].sum() == 8
    assert torch.equal(seen[0][0], st["s_m"][:1])
    assert torch.equal(seen[1][0], st["s_m"][1:])


def test_scoring_rng_replays_identical_draws_and_restores_caller():
    torch.manual_seed(42)
    before = torch.random.get_rng_state().clone()
    with scoring_rng(140014, "cpu"):
        a = torch.randn(100)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.randn(71)
    with scoring_rng(140014, "cpu"):
        b = torch.randn(100)
    assert torch.equal(a, b)


def test_config_flag_roundtrip():
    from wms.config.load import load_train_config
    cfg = load_train_config("configs/c4_wm_loop14_r16_ft.toml", ["data.entrant_control=true"])
    assert TrainSettings.from_config(cfg).entrant_control is True
    from wms.control.dreamer import DreamerConfig
    assert DreamerConfig().entrant_control is False


def test_observable_insertion_features_are_causal_and_legacy_unchanged():
    from wms.data.pack_history import entry_features
    kin = np.zeros((5, 2, 7), np.float32)
    kin[..., 1] = 1; kin[..., 4] = 10; kin[..., 6] = 5
    mask = np.array([[1, 0], [1, 1], [1, 1], [1, 1], [1, 1]], bool)
    cav = np.array([[0, 0], [0, 1], [0, 1], [0, 1], [0, 0]], bool)
    arrays = dict(veh_kin=kin, veh_mask=mask, target_mask=mask[:-1] & mask[1:],
                  is_cav=cav, counter=np.ones((5, 2, 30), np.float32))
    class Corpus:
        def run_start(self, tick): return 0
        def array(self, name):
            assert name in arrays  # no sidecar/type/truth accesses
            return arrays[name]
    corpus = Corpus()
    old = entry_features(corpus, {"tick": 2})
    new = entry_features(corpus, {"tick": 2}, observable_cav=True)
    assert EntranceHistory.from_features(old).cav_share.tolist() == [0., 0.]
    assert EntranceHistory.from_features(new).cav_share.tolist() == [1., 0.]
    arrays['is_cav'][3:] = True  # future observations cannot affect either vector
    assert np.array_equal(old, entry_features(corpus, {"tick": 2}))
    assert np.array_equal(new, entry_features(corpus, {"tick": 2}, observable_cav=True))


def test_inserted_cav_gets_feedback_and_virgin_slot_is_not_reused():
    model, settings = fixture()
    inp = prepare_inputs(batch(), model, replace(settings, entrant_control=True),
                         torch.device('cpu'), 2,
                         entrant_controller=lambda st: torch.full((*st['s_m'].shape, 4), -2.))
    state = {k: v.clone() for k, v in inp.initial.items()}
    state['entrant'] = torch.zeros_like(state['present'])
    births = inp.entries.insert(state)
    assert (births & state['is_cav']).any()
    state['entrant'] |= births
    inp.cav_controller.recorded = lambda st: torch.zeros((*st['s_m'].shape, 4))
    out = inp.cav_controller(state)
    assert (out[state['entrant'] & state['is_cav']] == -2.).all()
    occupied = births.clone()
    state['present'][occupied] = False
    inp.entries.pending[:] = 1
    later = inp.entries.insert(state)
    assert not (later & occupied).any()


def test_wrapper_preserves_merge_ownership_and_actual_t0_command_history():
    from wms.model.cav import WM_MERGE_KEY
    seen = []
    def recorded(st):
        st[WM_MERGE_KEY] = torch.tensor([[True, False]])
        return torch.ones(1, 2, 4)
    def feedback(st):
        seen.append(st['cmd_accel_mps2'].clone())
        st[WM_MERGE_KEY] = torch.tensor([[False, True]])
        return torch.full((1, 2, 4), -2.)
    cb = EntrantControllers(recorded, feedback)
    st = dict(s_m=torch.zeros(1, 2), present=torch.ones(1, 2, dtype=torch.bool),
              is_cav=torch.ones(1, 2, dtype=torch.bool), entrant=torch.tensor([[False, True]]))
    cb(st); cb(st)
    assert st[WM_MERGE_KEY].all()
    assert seen[0].sum() == 0 and seen[1][0, 0] == 1


def test_generic_random_feedback_uses_recorded_algorithm_and_branch_seed():
    from traffic_shepherd.c4.genrand import GenericRandomPolicy
    from wms.control.entrant import generic_random_feedback
    algorithm = GenericRandomPolicy(123)
    branch = dict(policy_params=algorithm.params(), arm_seed=456)
    cb = generic_random_feedback(branch)
    state = dict(s_m=torch.tensor([[-100., -70.]]), v_mps=torch.full((1, 2), 10.),
                 a_mps2=torch.zeros(1, 2), length_m=torch.full((1, 2), 5.),
                 lane=torch.zeros(1, 2, dtype=torch.long),
                 present=torch.ones(1, 2, dtype=torch.bool), is_cav=torch.ones(1, 2, dtype=torch.bool))
    algorithm.reseed(456)
    expected, _ = algorithm(state, with_lanes=True)
    actual = cb(state)
    # Safety may lower requests; the algorithm's deterministic requested acceleration
    # must still be the bound (and the unobstructed leader's delivered command).
    assert np.all(actual[..., 3].numpy() <= expected + 1e-6)
    assert actual[0, 1, 3].item() == pytest.approx(expected[0, 1], abs=1e-6)
