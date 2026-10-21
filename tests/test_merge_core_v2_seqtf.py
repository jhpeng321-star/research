"""Round-8 recorded-state sequence likelihood for the secure-gap merge core."""
import math

import torch

from wms.model.merge_core import secure_gap_margin
from wms.model.merge_frailty import gauss_hermite
from wms.train.losses.base import UnrollTargets
from wms.train.losses.merge_core import _core_margin
from wms.train.losses.merge_core_seqtf import build_recorded_scene, seq_tf_chunk

INIT = {"s0": 2.160959664129727, "T": 0.8617146481156865,
        "b": 1.9684840575324012, "tau_a": 1.0793451040830009,
        "temperature": 1.2781399549008334}


def _state(s, lane, v, a=0.0, length=5.0):
    row = torch.zeros(7)
    row[0] = s
    row[1 + lane] = 1.0
    row[4] = v
    row[5] = a
    row[6] = length
    return row


def _targets(lanes, s, v, *, is_cav=None, a=None):
    t1, k = lanes.shape
    states = torch.zeros(1, t1, k, 7)
    a = torch.zeros_like(s) if a is None else a
    for t in range(t1):
        for j in range(k):
            states[0, t, j] = _state(float(s[t, j]), int(lanes[t, j]),
                                     float(v[t, j]), float(a[t, j]))
    n = t1 - 1
    valid = torch.ones(1, t1, k, dtype=torch.bool)
    cav = (torch.zeros(1, t1, k, dtype=torch.bool) if is_cav is None
           else is_cav.bool().expand(1, t1, k).clone())
    return UnrollTargets(
        states=states, n_steps=n, weights=torch.ones(1, n, 1),
        hdv_mask=torch.ones(1, n, k), aux_mask=torch.ones(1, n, k),
        aux_count=n * k, denom=torch.tensor(1.0),
        lane_target=lanes[:, 1:].clone(), loss_valid=valid, is_cav=cav, dt_s=0.5,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False)


def _scene(n=6, event=2, ramp_s=40.0, k=3, cav=None):
    lanes = torch.full((n + 1, k), 1, dtype=torch.long)
    s = torch.zeros(n + 1, k)
    v = torch.full((n + 1, k), 10.0)
    lanes[:, 0] = 2
    if event is not None:
        lanes[event + 1:, 0] = 1
    s[:, 0] = ramp_s
    if k > 1:
        s[:, 1] = ramp_s + 10.0
    if k > 2:
        s[:, 2] = ramp_s - 10.0
    targets = _targets(lanes, s, v, is_cav=cav)
    return targets, build_recorded_scene(
        targets, merge_lane=2, zone_s0_m=0.0, zone_s1_m=75.44, k=k)


def _clone(scene):
    return {key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in scene.items()}


def _params(**kw):
    out = {key: torch.tensor(float(value)) for key, value in INIT.items()}
    out.update(kw)
    return out


def _chunk(scene, c0=0, c1=None, *, log_post=None, sigma=0.3, n_quad=5, params=None,
           require_both=None):
    p = _params() if params is None else params
    c1 = scene["n_steps"] if c1 is None else c1
    return seq_tf_chunk(scene, c0, c1, s0=p["s0"], T=p["T"], b=p["b"],
                        tau_a=p["tau_a"], temperature=p["temperature"],
                        sigma=sigma, n_quad=n_quad, log_post=log_post,
                        require_both=require_both)


def test_scene_marks_event_survival_and_censoring_points():
    _, scene = _scene(n=6, event=2, k=1)
    assert scene["rec_event"][0, 0].item() == 2
    assert scene["last_obs"][0, 0].item() == 1
    assert scene["supervised"][0, :3, 0].all()
    assert scene["identity"][0, 0].item() == 0
    assert scene["entry"][0, 0].item() == 0

    _, censored = _scene(n=6, event=None, k=1)
    assert censored["rec_event"][0, 0].item() == -1
    assert censored["last_obs"][0, 0].item() == 5


def test_event_and_right_censored_cars_are_scored():
    _, scene = _scene(n=6, event=3, k=1)
    nll, n_cars, post, metrics = _chunk(scene)
    assert n_cars.item() == 1.0
    assert metrics["events"] == 1.0
    assert metrics["unsupported_events"] == 0.0
    assert nll.item() > 0.0 and math.isfinite(nll.item())
    assert torch.isfinite(post).all()
    assert torch.allclose(torch.logsumexp(post, dim=1), torch.zeros(1, 1), atol=1e-4)

    _, censored = _scene(n=6, event=None, k=1)
    cnll, _, _, cmetrics = _chunk(censored)
    assert cmetrics["events"] == 0.0
    assert cmetrics["censored"] == 1.0
    assert cnll.item() > 0.0


def test_infeasible_recorded_event_has_no_event_factor_but_is_counted():
    _, scene = _scene(n=6, event=2, k=1, ramp_s=40.0)
    bad = _clone(scene)
    bad["feasible"][0, 2, 0] = False
    nll_bad, _, _, metrics = _chunk(bad)
    assert metrics["events"] == 0.0
    assert metrics["unsupported_events"] == 1.0
    assert nll_bad.item() > 0.0

    ref = _clone(scene)
    ref["rec_event"][0, 0] = -1
    ref["last_obs"][0, 0] = 1
    nll_ref, _, _, _ = _chunk(ref)
    assert torch.allclose(nll_bad, nll_ref, atol=1e-5)


def test_passed_feasible_gaps_contribute_survival():
    _, scene = _scene(n=6, event=3, k=3)
    full, _, _, m_full = _chunk(scene)
    fewer = _clone(scene)
    fewer["feasible"][0, 1, 0] = False
    fewer_nll, _, _, m_fewer = _chunk(fewer)
    assert m_fewer["scored_rows"] < m_full["scored_rows"]
    assert fewer_nll.item() < full.item()


def test_chunk_split_equals_monolithic_at_fixed_parameters():
    _, scene = _scene(n=8, event=5, k=3)
    mono, _, _, _ = _chunk(scene, 0, 8, sigma=0.4, n_quad=5)
    first, _, post1, _ = _chunk(scene, 0, 3, sigma=0.4, n_quad=5)
    second, _, _, _ = _chunk(scene, 3, 8, sigma=0.4, n_quad=5, log_post=post1)
    assert torch.allclose(mono, first + second, atol=1e-5, rtol=1e-5)


def test_mixture_mass_sums_to_one_on_fully_supported_path():
    _, base = _scene(n=6, event=None, k=1)
    n = base["n_steps"]
    prior_mass = 0.0
    for event in range(n):
        scene = _clone(base)
        scene["rec_event"][0, 0] = event
        scene["last_obs"][0, 0] = event - 1
        for key in ("gap_l", "gap_f"):
            scene[key] = torch.full_like(scene[key], INIT["s0"])
        for key in ("v", "v_l", "v_f", "a", "a_l", "a_f"):
            scene[key] = torch.zeros_like(scene[key])
        scene["has_l"] = torch.ones_like(scene["has_l"])
        scene["has_f"] = torch.ones_like(scene["has_f"])
        nll, _, _, _ = _chunk(scene, n_quad=1, sigma=0.0)
        prior_mass += math.exp(-float(nll))
    censored = _clone(base)
    censored["rec_event"][0, 0] = -1
    censored["last_obs"][0, 0] = n - 1
    for key in ("gap_l", "gap_f"):
        censored[key] = torch.full_like(censored[key], INIT["s0"])
    for key in ("v", "v_l", "v_f", "a", "a_l", "a_f"):
        censored[key] = torch.zeros_like(censored[key])
    censored["has_l"] = torch.ones_like(censored["has_l"])
    censored["has_f"] = torch.ones_like(censored["has_f"])
    nll, _, _, _ = _chunk(censored, n_quad=1, sigma=0.0)
    prior_mass += math.exp(-float(nll))
    assert abs(prior_mass - 1.0) < 1e-6


def test_no_post_event_survival():
    _, scene = _scene(n=6, event=2, k=1)
    scene["has_l"] = torch.ones_like(scene["has_l"])
    scene["has_f"] = torch.ones_like(scene["has_f"])
    scene["gap_l"] = torch.full_like(scene["gap_l"], 10.8)
    scene["gap_f"] = torch.full_like(scene["gap_f"], 10.8)
    before, _, _, _ = _chunk(scene)
    after = _clone(scene)
    after["gap_l"][0, 3:, 0] = 1.0e4
    after["gap_f"][0, 3:, 0] = 1.0e4
    after["feasible"][0, 3:, 0] = True
    after_nll, _, _, _ = _chunk(after)
    assert torch.allclose(before, after_nll, atol=1e-6)


def test_gradient_direction_for_survival_and_event_factors():
    _, scene = _scene(n=6, event=None, k=1)
    low = _clone(scene)
    high = _clone(scene)
    for key in ("gap_l", "gap_f"):
        low[key] = torch.full_like(low[key], INIT["s0"] + 0.2)
        high[key] = torch.full_like(high[key], INIT["s0"] + 1.0)
    for key in ("v", "v_l", "v_f", "a", "a_l", "a_f"):
        low[key] = torch.zeros_like(low[key])
        high[key] = torch.zeros_like(high[key])
    low["has_l"] = low["has_f"] = torch.ones_like(low["has_l"])
    high["has_l"] = high["has_f"] = torch.ones_like(high["has_l"])
    low_nll, _, _, _ = _chunk(low, n_quad=1, sigma=0.0)
    high_nll, _, _, _ = _chunk(high, n_quad=1, sigma=0.0)
    assert high_nll.item() > low_nll.item()

    _, ev = _scene(n=6, event=3, k=1)
    ev["has_l"] = torch.ones_like(ev["has_l"])
    ev["has_f"] = torch.ones_like(ev["has_f"])
    low_h = _clone(ev)
    high_h = _clone(ev)
    low_h["gap_l"][0, 3, 0] = INIT["s0"] + 0.2
    high_h["gap_l"][0, 3, 0] = INIT["s0"] + 1.0
    low_h["gap_f"][0, 3, 0] = INIT["s0"] + 0.2
    high_h["gap_f"][0, 3, 0] = INIT["s0"] + 1.0
    low_ev, _, _, _ = _chunk(low_h, n_quad=1, sigma=0.0)
    high_ev, _, _, _ = _chunk(high_h, n_quad=1, sigma=0.0)
    assert high_ev.item() < low_ev.item()


def test_gradients_flow_to_physics_and_sigma():
    _, scene = _scene(n=6, event=3, k=3)
    scene["v_l"] = torch.full_like(scene["v"], 8.0)
    scene["v_f"] = torch.full_like(scene["v"], 12.0)
    scene["has_l"] = torch.ones_like(scene["has_l"])
    scene["has_f"] = torch.ones_like(scene["has_f"])
    params = {key: torch.tensor(float(value), requires_grad=(key != "temperature"))
              for key, value in INIT.items()}
    sigma = torch.tensor(0.35, requires_grad=True)
    nll, _, _, _ = _chunk(scene, sigma=sigma, n_quad=5, params=params)
    nll.backward()
    for name in ("s0", "T", "b", "tau_a"):
        grad = params[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().item() > 0.0, name
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all()
    assert sigma.grad.abs().item() > 0.0


def test_identity_change_resets_persistent_posterior():
    _, scene = _scene(n=6, event=3, k=1)
    _, _, post, _ = _chunk(scene, 0, 3, sigma=0.5)
    new = _clone(scene)
    new["identity"] = torch.full_like(new["identity"], 7)
    new.pop("_post_identity", None)
    new["_post_identity"] = scene["_post_identity"].clone()
    _, _, reset, _ = _chunk(new, 3, 6, sigma=0.5, log_post=post)
    _, prior = gauss_hermite(5)
    prior = torch.log(prior)
    assert torch.allclose(torch.logsumexp(post, dim=1), torch.zeros(1, 1), atol=1e-4)
    assert not torch.allclose(reset, post)
    assert torch.allclose(reset, prior.view(1, -1, 1), atol=1e-5)


def test_cav_neighbours_enter_gaps_but_cav_targets_are_not_supervised():
    cav = torch.tensor([[False, True, True]]).T.reshape(1, 1, 3)
    targets, scene = _scene(n=4, event=2, k=3, cav=cav)
    assert scene["has_l"][0, 0, 0]
    assert scene["gap_l"][0, 0, 0].item() == 5.0
    assert scene["gap_f"][0, 0, 0].item() == 5.0
    assert scene["supervised"][0, :3, 0].all()
    assert not scene["supervised"][0, :, 2].any()
    assert scene["identity"][0, 2].item() == -1


def test_margin_matches_runtime_core_on_the_same_scene():
    _, scene = _scene(n=4, event=2, k=3)
    alpha = torch.ones(4, 3)
    args = (scene["gap_l"], scene["gap_f"], scene["v"], scene["v_l"], scene["v_f"],
            scene["a"], scene["a_l"], scene["a_f"], scene["has_l"], scene["has_f"],
            torch.tensor(INIT["s0"]), torch.tensor(INIT["T"]), torch.tensor(INIT["b"]),
            torch.tensor(INIT["tau_a"]))
    internal = _core_margin(alpha, *args, None, None)
    runtime = secure_gap_margin(*args, alpha, None, None)
    assert torch.allclose(internal, runtime)


def test_zero_exposure_is_counted_once():
    lanes = torch.tensor([[2, 1], [2, 1], [2, 1], [2, 1]], dtype=torch.long)
    s = torch.tensor([[40.0, 50.0], [40.0, 50.0], [40.0, 50.0], [40.0, 50.0]])
    v = torch.full((4, 2), 10.0)
    targets = _targets(lanes, s, v)
    scene = build_recorded_scene(targets, merge_lane=2, zone_s0_m=100.0,
                                 zone_s1_m=200.0, k=2)
    _, n_cars, _, metrics = _chunk(scene, 0, 3)
    assert n_cars.item() == 1.0
    assert metrics["zero_exposure"] == 1.0
    _, _, _, later = _chunk(scene, 3, 3 + 0)
    assert later["zero_exposure"] == 0.0
