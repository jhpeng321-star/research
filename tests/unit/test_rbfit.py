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
import calibrate_shrinkage  # noqa: E402


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


def test_shrinkage_crps_polynomial_matches_pairwise_definition():
    rng = np.random.default_rng(73)
    values, y = rng.normal(size=(17, 7)), rng.normal(size=17)
    prior, posterior = rng.dirichlet(np.ones(7), size=2)
    c = calibrate_shrinkage.crps_coefficients(values, y, prior, posterior)
    for r in (0, .19, .73, 1):
        w = (1-r)*prior+r*posterior
        direct = abs(values-y[:, None])@w - .5*np.einsum(
            'i,nij,j->n', w, abs(values[:, :, None]-values[:, None, :]), w)
        np.testing.assert_allclose(c[:, 0]+r*c[:, 1]+r*r*c[:, 2], direct, atol=1e-14)
    assert calibrate_shrinkage.posterior_weight(0, 0) == 0
    assert calibrate_shrinkage.posterior_weight(100, 1) == 0
    assert calibrate_shrinkage.posterior_weight(10, .5) < calibrate_shrinkage.posterior_weight(100, .5)


def test_unc_bridge_preserves_cavs_auxiliary_coordinates_and_identity(tmp_path):
    import hashlib
    import torch
    from types import SimpleNamespace
    from wms.physics.motion_fit import MotionFit, MotionFitForUNC, ENCODER_ORDER

    support = np.stack([fit.INITIAL, fit.INITIAL*1.2])
    pop = dict(schema='wm_phys.idm_population.v1', parameter_order=list(fit.NAMES),
        distribution='finite_support_mixture', support=support.tolist(), weights=[1., 0.],
        **idm_mixture.moments(support, np.array([1., 0.])), delta=4)
    post = dict(schema='wm_phys.idm_posteriors.v1', parameter_order=list(fit.NAMES),
        conditioning_end_s=300, drivers={'969400:42': dict(
            **idm_mixture.moments(support, np.array([0., 1.])),
            support_indices=[1], weights=[1.], n_observations=0)})
    fit.write_json(tmp_path/'idm_population.json', pop)
    fit.write_json(tmp_path/'idm_posteriors.json', post)
    policy = dict(schema='wm_phys.idm_shrinkage.v1', conditioning_end_s=300,
        idm_population_sha256=hashlib.sha256((tmp_path/'idm_population.json').read_bytes()).hexdigest(),
        population_only=False, prior_count=20.)
    fit.write_json(tmp_path/'policy.json', policy)
    reader = MotionFit(tmp_path, shrinkage_path=tmp_path/'policy.json')
    ids = torch.tensor([[42, 7, -1, 42]])
    present, cav = torch.tensor([[1, 1, 0, 1]]), torch.tensor([[0, 1, 0, 0]])
    params = {name: torch.full((1, 4), float(j+2)) for j, name in enumerate(ENCODER_ORDER)}
    embedding = torch.tensor([123.])
    model = SimpleNamespace(training=False, encode_drivers=lambda state, **kw: ({'keep': 9}, embedding, params))
    proxy = MotionFitForUNC(model, reader, [969400], ids, t0_s=300, n_particles=5)
    encoded, emb, point = proxy.encode_drivers(dict(present=present, is_cav=cav))
    samples = encoded['posterior_samples']
    assert samples.shape == (5, 1, 4, 9) and encoded['keep'] == 9
    assert emb is embedding and point is params and proxy.training is False
    baseline = torch.stack([params[name] for name in ENCODER_ORDER], -1)
    torch.testing.assert_close(samples[:, :, 1:3], baseline[:, 1:3].expand(5, 1, 2, 9))
    torch.testing.assert_close(samples[..., 5:], baseline[..., 5:].expand(5, 1, 4, 4))
    torch.testing.assert_close(samples[:, 0, 0], samples[:, 0, 3])
    # No evidence must override even this deliberately overconfident raw posterior.
    np.testing.assert_allclose(samples[:, 0, 0, 0].numpy(), np.full(5, fit.INITIAL[0]))
    permutation = [3, 1, 2, 0]
    permuted = reader.unc_posterior_samples([969400], ids[:, permutation], present[:, permutation],
        cav[:, permutation], {k: v[:, permutation] for k, v in params.items()}, n_samples=5, t0_s=300)
    torch.testing.assert_close(samples[:, :, permutation], permuted)
    with pytest.raises(ValueError, match='conditioning time'):
        reader.unc_posterior_samples([969400], ids, present, cav, params, n_samples=5, t0_s=301)
    policy['idm_population_sha256'] = 'wrong'
    fit.write_json(tmp_path/'bad_policy.json', policy)
    with pytest.raises(ValueError, match='binding mismatch'):
        MotionFit(tmp_path, shrinkage_path=tmp_path/'bad_policy.json')


def test_generic_motion_posterior_needs_no_vid_and_rejects_noncausal_motion(tmp_path):
    import hashlib
    import torch
    from wms.physics.motion_fit import MotionFit

    support = np.stack([fit.INITIAL, fit.INITIAL*1.2])
    pop = dict(schema='wm_phys.idm_population.v1', parameter_order=list(fit.NAMES),
        distribution='finite_support_mixture', support=support.tolist(), weights=[.5, .5],
        **idm_mixture.moments(support, np.array([.5, .5])), delta=4, sigma_accel_mps2=.2)
    post = dict(schema='wm_phys.idm_posteriors.v1', parameter_order=list(fit.NAMES),
                conditioning_end_s=300, drivers={})
    fit.write_json(tmp_path/'idm_population.json', pop)
    fit.write_json(tmp_path/'idm_posteriors.json', post)
    policy = dict(schema='wm_phys.idm_shrinkage.v1', conditioning_end_s=300,
        idm_population_sha256=hashlib.sha256((tmp_path/'idm_population.json').read_bytes()).hexdigest(),
        population_only=False, prior_count=19.355678502354067)
    fit.write_json(tmp_path/'policy.json', policy)
    reader = MotionFit(tmp_path)
    before = np.array([7., 12., 17., 23.])
    prediction = fit.next_speed(before[:, None], np.array([30., 45., 60., 80.])[:, None],
                                before[:, None], support, .5)
    tensors = [torch.tensor(v) for v in (prediction, before, prediction[:, 1],
                                         np.full(4, .5), np.arange(298.5, 300.5, .5))]
    result = reader.condition_motion(*tensors, t0_s=300, policy_path=tmp_path/'policy.json')
    ll = -(((prediction-prediction[:, 1, None])/.5)**2).sum(0)/(2*.2**2)
    weights = np.exp(ll-ll.max()); weights /= weights.sum()
    r = 4/(4+policy['prior_count'])
    np.testing.assert_allclose(result['weights'], r*weights+(1-r)*.5, atol=1e-13)
    assert result['conditioning_end_s'] == 300 and result['n_observations'] == 4
    empty = [t[:0] for t in tensors]
    prior = reader.condition_motion(*empty, t0_s=301, policy_path=tmp_path/'policy.json')
    np.testing.assert_allclose(prior['weights'], [.5, .5])
    future = list(tensors); future[-1] = future[-1]+.5
    with pytest.raises(ValueError, match='causal last60s'):
        reader.condition_motion(*future, t0_s=300)
    old = list(tensors); old[-1] = old[-1]-60
    with pytest.raises(ValueError, match='causal last60s'):
        reader.condition_motion(*old, t0_s=300)
    overlap = list(tensors); overlap[-1] = torch.tensor([298.5, 298.75, 299.5, 300.])
    with pytest.raises(ValueError, match='nonoverlapping'):
        reader.condition_motion(*overlap, t0_s=300)


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
