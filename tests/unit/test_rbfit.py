"""Inference/observable-contract regressions for physical fitting."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy", reason="offline physical fitting requires optional SciPy")

MODULE = Path(__file__).resolve().parents[2] / "tools" / "rbfit" / "idm_fit.py"
spec = importlib.util.spec_from_file_location("rbfit_idm", MODULE)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)
sys.path.insert(0, str(MODULE.parent))
import lane_fit  # noqa: E402
import idm_mixture  # noqa: E402


def test_reject_non_fit_before_access(tmp_path):
    with pytest.raises(ValueError, match="Only registered FIT"):
        fit.native_episode(tmp_path, 969432)


def test_forward_acceleration_and_identity_reuse():
    # A reused packed slot is not the same driver; a cut-in leader is not the
    # leader that generated the previous interval's acceleration.
    kin = np.zeros((3, 3, 7))
    kin[:, :, 1] = 1
    kin[:, :, 6] = 5
    kin[:, :, 0] = [[0, 30, 60], [5, 36, 66], [11, 42, 72]]
    kin[:, :, 4] = [[10, 12, 12], [12, 12, 12], [14, 12, 12]]
    data = dict(veh_kin=kin, veh_mask=np.ones((3, 3), dtype=bool),
                is_cav=np.zeros((3, 3), dtype=bool),
                vid=np.array([[1, 2, 3], [1, 2, 3], [4, 2, 3]]),
                time_s=np.array([299.5, 300, 300.5]))
    rows, hist, _ = fit.extract_rows(data)
    ego = rows[rows[:, 1] == 1]
    assert len(ego) == 1
    assert fit.observed(ego)[0] == 4  # forward dv/dt, not stored channel 5
    assert ego[0, 5] == 25  # subtract LEADER length
    assert hist[1] == 0.5
    assert 4 not in hist  # post-300 identity cannot enter a causal posterior
    data["vid"][1, 1] = 99
    rows, _, _ = fit.extract_rows(data)
    assert not np.any(rows[:, 1] == 1)


def test_recovers_identifiable_synthetic_physical_parameters():
    rng = np.random.default_rng(33)
    physical = np.array([33.0, 1.3, 2.2, 3.8, 2.8])
    rows = np.zeros((600, 10))
    rows[:, 3] = rng.uniform(2, 34, len(rows))
    rows[:, 5] = rng.uniform(8, 120, len(rows))
    rows[:, 6] = rng.uniform(0, 35, len(rows))
    rows[:, 9] = 0.5
    rows[:, 4] = fit.next_speed(rows[:, 3], rows[:, 5], rows[:, 6], physical)
    _, z, cov, _, _, success = fit.posterior_job(
        ("synthetic", rows, np.log(fit.INITIAL), np.full(5, 4.0), 0.01))
    assert success
    np.testing.assert_allclose(np.exp(z), physical, rtol=0.001)
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.diag(cov).max() < 4


def test_receding_leader_uses_finite_free_speed():
    p = np.array([30, 1, 2.6, 4.5, 2.5])
    free = fit.next_speed(np.array([0, 20, 30]), np.full(3, np.inf), np.zeros(3), p)
    assert np.isfinite(free).all()
    assert 0 < free[0] <= 1.3
    assert 20 < free[1] <= 21.3
    assert abs(free[2] - 30) < 1e-6


def test_lane_nonidentification_retains_population_uncertainty():
    # Physically different parameter candidates with identical observed
    # decisions cannot yield a spurious identified driver subtype.
    counts = np.tile(np.array([2, 3, 1, 94]), (8, 3, 1))
    weights, _, _, posterior, trace = lane_fit.fit_mixture(counts)
    np.testing.assert_allclose(weights, np.ones(3)/3)
    np.testing.assert_allclose(posterior, np.tile(weights, (8,1)))
    assert np.diff(trace).min(initial=0) >= -1e-8


def test_zero_history_lane_posterior_equals_population():
    counts = np.array([[[9, 1, 1, 89], [0, 20, 10, 70]],
                       [[0, 20, 10, 70], [9, 1, 1, 89]],
                       [[0, 0, 0, 0], [0, 0, 0, 0]]])
    weights, p0, p1, posterior, _ = lane_fit.fit_mixture(counts)
    np.testing.assert_allclose(posterior[-1], weights)
    assert p1 > p0
    assert posterior[0, 0] > posterior[0, 1]
    assert posterior[1, 1] > posterior[1, 0]


def test_runtime_sampler_causal_binding_fallback_and_identity(tmp_path):
    import torch
    from wms.physics.motion_fit import MotionFit

    pop = dict(schema="wm_phys.idm_population.v1", parameter_order=list(fit.NAMES),
               log_mean=np.log(fit.INITIAL).tolist(), log_cov=(np.eye(5)*0.1).tolist(), delta=4,
               distribution="diagonal_lognormal")
    special = dict(log_mean=np.log(fit.INITIAL*1.1).tolist(), log_cov=(np.eye(5)*0.02).tolist())
    post = dict(schema="wm_phys.idm_posteriors.v1", parameter_order=list(fit.NAMES),
                conditioning_end_s=300, drivers={"969400:42": special})
    fit.write_json(tmp_path/"idm_population.json", pop)
    fit.write_json(tmp_path/"idm_posteriors.json", post)
    reader = MotionFit(tmp_path)
    ids = torch.tensor([42, 99, 42])
    med = reader.idm(969400, ids, t0_s=300)
    np.testing.assert_allclose(med["v0"].numpy(), [34.419, 31.29, 34.419])
    draw = reader.idm(969400, ids, t0_s=300, sample=True,
                      generator=torch.Generator().manual_seed(100))
    assert draw["v0"][0] == draw["v0"][2]
    assert all((value > 0).all() for value in draw.values())
    with pytest.raises(ValueError, match="conditioning time"):
        reader.idm(969400, ids, t0_s=299.5)
    prior = reader.idm(969400, ids, t0_s=299.5, use_posterior=False)
    np.testing.assert_allclose(prior["v0"].numpy(), [31.29]*3)


def test_finite_physical_mixture_ml_and_uninformative_driver():
    # Different full driver histories support different PHYSICAL profiles;
    # an unobserved driver's posterior is exactly the learned population.
    sse=np.array([[1,100],[100,1],[0,0]],dtype=float)
    n=np.array([20,20,0])
    weights,_,posterior,trace=idm_mixture.mixture_em(sse,n,0.5)
    np.testing.assert_allclose(weights,[0.5,0.5])
    np.testing.assert_allclose(posterior[-1],weights)
    assert posterior[0,0]>0.99 and posterior[1,1]>0.99
    assert np.diff([r["negative_log_likelihood"] for r in trace]).max(initial=0)<1e-7


def test_finite_sampler_never_invents_off_support_parameters(tmp_path):
    import gzip
    import json
    import torch
    from wms.physics.motion_fit import MotionFit

    support=np.stack([fit.INITIAL,fit.INITIAL*1.2])
    pop=dict(schema="wm_phys.idm_population.v1",parameter_order=list(fit.NAMES),
             distribution="finite_support_mixture",support=support.tolist(),weights=[0.5,0.5],
             **idm_mixture.moments(support,np.array([0.5,0.5])),delta=4)
    post=dict(schema="wm_phys.idm_posteriors.v1",parameter_order=list(fit.NAMES),
              conditioning_end_s=300,drivers={"969400:42":dict(**idm_mixture.moments(support,np.array([0,1])),
                                                        support_indices=[1],weights=[1.0])})
    fit.write_json(tmp_path/"idm_population.json",pop)
    (tmp_path/"idm_posteriors.json.gz").write_bytes(gzip.compress(json.dumps(post).encode()))
    reader=MotionFit(tmp_path)
    sample=reader.idm(969400,torch.tensor([42]*20),t0_s=300,sample=True)
    np.testing.assert_allclose(sample["v0"].numpy(),[37.548]*20)


@pytest.mark.parametrize("v,gap,vl,p,dt,expected",[
    (10,22.5,10,[50/3.6,1,2.6,4.5,2.5],0.5,10.505369042766489),
    (0,32.5,50/3.6,[50/3.6,1,2.6,4.5,2.5],0.5,1.292304647309232),
    (10,10.5,0,[50/3.6,1,2.6,4.5,2.5],0.5,5.5),
    (10,22.5,10,[50/3.6*1.2,1,2.6,4.5,2.5],0.5,10.684841865711107),
    (9,12,6,[40/3.6,1.5,1.5,2,2],0.5,7.198219766344515),
    (10,17.5,8,[50/3.6,1,2.6,4.5,2.5],1.0,9.76782360599685),
])
def test_independent_shared_sumo_reference_vectors(v,gap,vl,p,dt,expected):
    # Authored reference vectors T14-17,T19-20 in rb-shared/idmspec (SUMO
    # 1.27.1 source-derived). Convert net gap to physical bumper gap ONCE.
    assert float(fit.next_speed(v,gap,vl,np.array(p),dt))==pytest.approx(expected,abs=1e-10)
