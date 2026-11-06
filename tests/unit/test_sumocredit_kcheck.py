"""S-SUMOCREDIT test-time plan check (K learned-policy candidates): synthetic CPU inputs only."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from .test_sumocredit import FakeEnv, FakeSource, StubImagination, _state
from wms.control import confident_gate as cg


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _step_table(comparison):
    """raw <= 8: 0/14 wins (p 0.0625); raw >= 9: 6/6 wins (p 0.875)."""
    raw = list(range(-5, 15))
    return cg.ReliabilityTable.fit(raw, [int(r > 8) for r in raw],
                                   {"split": "TRAIN", "unit": "group", "reference": "hold", "horizon_s": 20.0,
                                    "state_source": "belief", "comparison": comparison})


def _kgate(J, *, mean=True, K=3):
    imag = StubImagination(J)
    return cg.ConfidentGate(imag, _step_table("reference"), unit="group", state_source="belief",
                            n_candidates=K, mean_table=_step_table("mean") if mean else None), imag


# arms: REF; unit [2]: mean, s1, s2; unit [1, 0]: mean, s1, s2 (J costs; raw = J_A - J_B)
J = [[[100.0], [90.0], [80.0], [95.0], [99.0], [92.0], [101.0]]]


def test_k_check_picks_best_eligible_candidate_and_needs_the_mean_table():
    st = _state([0, 100, 1000], [1, 1, 1], [True, True, True])
    gate, imag = _kgate(J)
    d = gate.decide(st, now_s=1.0, ids=["a", "b", "c"])
    assert len(imag.members) == 7 and imag.members[1][1] is None          # candidate 0 = mean plan
    assert imag.members[2][1].correlated is False and imag.members[3][1].correlated is True
    # unit [2]: mean raw 10 eligible, s1 raw 20 beats the mean by 10 (p 0.875) -> s1 replaces it
    assert d.choice[0] == 1 and d.candidates[0]["chosen_vs_mean"] == 10.0
    assert d.candidates[0]["eligible"] == [True, True, False]
    # unit [1, 0]: mean raw 1, s1 raw 8, s2 raw -1: nothing calibrated >= 0.8 -> abstain
    assert d.choice[1] is None and d.open == [True, False]
    s = gate.summary()
    assert s["abstain_rate"] == 0.5 and s["units_chose_sample"] == 1 and s["chosen_vs_mean_mean"] == 10.0
    assert s["decision_ms"]["n"] == 1 and s["n_candidates"] == 3
    gate, _ = _kgate(J, mean=False)                     # no mean table: a sample never replaces it
    assert gate.decide(st, now_s=1.0, ids=["a", "b", "c"]).choice == [0, None]
    J3 = [[[100.0], [95.0], [80.0], [95.0], [99.0], [92.0], [101.0]]]
    gate, _ = _kgate(J3)                                # s1 eligible vs REF and vs mean, mean is not
    assert gate.decide(st, now_s=1.0, ids=["a", "b", "c"]).choice[0] == 1


def test_k_check_refuses_a_reference_table_as_mean_table():
    with pytest.raises(ValueError):
        cg.ConfidentGate(StubImagination([[[0.0]]]), _step_table("reference"), unit="group",
                         state_source="belief", n_candidates=2, mean_table=_step_table("reference"))


def test_id_noise_is_keyed_by_vehicle_and_replayed_on_env_slots():
    s = cg.IdNoiseStream(7, 1)
    s.begin(3)
    a = s.draw(0, ["x", None, "y"])
    b = s.draw(0, ["y", "x"])
    assert a.accel[0, 0] == b.accel[0, 1] and a.accel[0, 2] == b.accel[0, 0] and a.accel[0, 1] == 0
    assert torch.equal(a.lane_gumbel[0, 0], b.lane_gumbel[0, 1])
    assert not torch.equal(s.draw(1, ["x"]).accel, s.draw(0, ["x"]).accel)
    c = cg.IdNoiseStream(7, 2, correlated=True)
    c.begin(3)
    d = c.draw(0, ["x", "y", None])
    assert d.accel[0, 0] == d.accel[0, 1] and d.accel[0, 2] == 0
    replay = cg.SumoNoise(s, FakeEnv(["y", "h1", "x"]))
    out = replay((1, 3))
    assert out.accel[0, 0] == a.accel[0, 2] and out.accel[0, 2] == a.accel[0, 0] and out.accel[0, 1] == 0
    assert replay.step == 1
    with pytest.raises(ValueError):
        cg.IdNoiseStream(0, 0)


def test_gated_executor_runs_the_chosen_sample():
    st = _state([0, 100, 1000], [1, 1, 1], [True, True, True])
    gate, imag = _kgate(J)
    imag.reference = "native"                                            # abstain = native release
    env = FakeEnv(["a", "b", "c"])
    plan = (np.array([[1.0, 1.0, 1.0]]), np.array([[-1, -1, -1]]))
    made = []

    def sample_fn(stream, env):
        made.append(stream.index)
        return lambda env, state: (np.array([[2.0, 2.0, 2.0]]), np.array([[0, 0, 0]]))
    gp = cg.GatedSumoPolicy(lambda env, state: plan, gate, FakeSource(st), decision_period_s=5.0,
                            sample_fn=sample_fn)
    accel, lanes = gp(env, st)
    assert made == [1] and accel[0, 2] == 2.0 and lanes[0, 2] == 0     # "c" runs sample 1
    assert np.isnan(accel[0, :2]).all() and lanes[0, :2].tolist() == [-1, -1]   # abstain: native
    assert gp.summary()["cav_ticks_sample"] == 1
    with pytest.raises(ValueError):
        cg.GatedSumoPolicy(lambda e, s: plan, gate, FakeSource(st), decision_period_s=5.0)


def test_sampled_arm_imagination_is_paired():
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    torch.manual_seed(5)
    model = WmSModel(ModelSpec(rollout_mode="hybrid")).eval().requires_grad_(False)
    policy, encoder = CavPolicy(lane_head=False).eval(), RsuGroupEncoder().eval()
    st = _state([-300, -250, -200, -150, -100, -60], [1, 1, 0, 1, 2, 1],
                [True, False, False, True, False, True])
    ids = ["c0", None, None, "c3", None, "c5"]
    stream = cg.IdNoiseStream(11, 1)
    stream.begin(0)
    imag = cg.ArmImagination([model], policy, encoder, reference="native", horizon_s=2.0)
    Js = imag.arms_J(st, [None, (None, stream), (None, stream)], ids=ids)
    assert torch.equal(Js[:, 1], Js[:, 2]) and torch.isfinite(Js).all()
    with pytest.raises(ValueError):
        imag.arms_J(st, [(None, stream)])                                   # ids required
