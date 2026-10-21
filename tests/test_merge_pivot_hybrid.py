"""Synthetic CPU contracts for the opt-in persistent-z / fresh-epsilon law.

The reference integrates the logistic density in z, independently of the
production quadrature. Time is the LAST axis of the public mass API. No traffic
datasets, evaluation runs, or external checkpoints are needed.
"""
from collections.abc import Mapping
from dataclasses import asdict
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from wms.model import ModelSpec, WmSModel
from wms.model import merge_pivot as pivot
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.physics.config import PhysicsConfig


DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _isolated_cpu_rng():
    # These are local synthetic unit-test seeds, not simulator/evaluation seeds.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71237)
        yield


def _tensor(value, *, grad=False):
    return torch.tensor(value, dtype=DTYPE, device="cpu", requires_grad=grad)


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_tree(item) for item in value)
    return value


def _assert_same_tree(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_same_tree(a, b)
    else:
        assert actual == expected


def _reference_log_masses(c, tau, w):
    """Composite Gauss-Legendre integral of the defining conditional law.

    Fixed z nodes make ordinary autograd an independent derivative reference.
    Breaks around each transition resolve small w/tau without needing a huge
    uniform grid. Detached node placement is intentional: only the integrand
    is differentiated. The omitted prior tail is less than 4e-28; reference
    fixtures keep their posterior support well inside these bounds.
    """
    values = c.detach().reshape(-1).tolist()
    ratio = float((w / tau).detach())
    breaks = {-64.0, -32.0, -16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0, 32.0, 64.0}
    for value in values:
        center = -value / float(tau.detach())
        for offset in (-16, -8, -4, -1, 0, 1, 4, 8, 16):
            point = center + offset * ratio
            if -64 < point < 64:
                breaks.add(point)
    edges = sorted(breaks)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    z, quadrature_weights = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        half = (right - left) / 2
        z.extend(((left + right) / 2 + half * nodes).tolist())
        quadrature_weights.extend((half * weights).tolist())
    z = c.new_tensor(z)
    log_weight = (c.new_tensor(quadrature_weights).log()
                  + F.logsigmoid(z) + F.logsigmoid(-z))
    logits = (c.unsqueeze(-1) + tau * z) / w
    logh, log1mh = F.logsigmoid(logits), F.logsigmoid(-logits)
    conditional_logS = log1mh.cumsum(-2)
    before = torch.cat((torch.zeros_like(conditional_logS[..., :1, :]),
                        conditional_logS[..., :-1, :]), dim=-2)
    return (torch.logsumexp(before + logh + log_weight, dim=-1),
            torch.logsumexp(conditional_logS + log_weight, dim=-1))


def _nll(log_event, log_survival):
    # A mixture of two observed event times and one right-censored trajectory.
    return -log_event[..., 1] - 0.7 * log_event[..., -1] - 0.4 * log_survival[..., -1]


def _tick(c, *, carry=None, tau=0.7, w=0.5, ramp=None, present=None,
          identity=None, uniform=None, epsilon_uniform=None, order=32):
    if not torch.is_tensor(c):
        c = _tensor([c])
    ones = torch.ones_like(c, dtype=torch.bool)
    return pivot.update_state(
        c, c.new_tensor(tau), ones if ramp is None else ramp,
        ones if present is None else present, carry=carry, identity=identity,
        uniform=uniform, w=c.new_tensor(w), epsilon_uniform=epsilon_uniform,
        quadrature_order=order,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conditional_log_hazards_broadcast_ordinary_scene_and_latent_axes(dtype):
    # Neither axis is a time axis here: broadcasting is ordinary PyTorch broadcasting.
    c = torch.tensor([[-1.2], [0.0], [0.7]], dtype=dtype)
    z = torch.tensor([[-2.0, -0.3, 0.4, 2.0]], dtype=dtype)
    tau, w = c.new_tensor(0.65), c.new_tensor(0.3)
    logh, log1mh = pivot.conditional_log_hazards(c, tau, w, z)
    logits = (c + tau * z) / w
    assert logh.shape == log1mh.shape == (3, 4)
    assert logh.dtype == log1mh.dtype == dtype
    assert logh.device.type == log1mh.device.type == "cpu"
    torch.testing.assert_close(logh, F.logsigmoid(logits))
    torch.testing.assert_close(log1mh, F.logsigmoid(-logits))
    torch.testing.assert_close(torch.logaddexp(logh, log1mh), torch.zeros_like(logh))
    eligible = torch.tensor([[True], [False], [True]])
    masked_h, masked_s = pivot.conditional_log_hazards(c, tau, w, z, eligible=eligible)
    assert torch.isneginf(masked_h[1]).all() and (masked_s[1] == 0).all()
    torch.testing.assert_close(masked_h[[0, 2]], logh[[0, 2]])
    torch.testing.assert_close(masked_s[[0, 2]], log1mh[[0, 2]])


def test_finite_boundaries_have_positive_log_support_even_in_extreme_tails():
    c = _tensor([-10000.0, -40.0, 0.0, 40.0, 10000.0], grad=True)
    z = _tensor([-2.0, 1.0, 0.0, -1.0, 2.0], grad=True)
    tau, w = _tensor(0.003, grad=True), _tensor(0.0002, grad=True)
    logh, log1mh = pivot.conditional_log_hazards(c, tau, w, z)
    assert torch.isfinite(logh).all() and torch.isfinite(log1mh).all()
    assert logh[0] < -1e7 and log1mh[-1] < -1e7  # no probability floor
    (-logh[:2].sum() - log1mh[-2:].sum() - logh[2]).backward()
    for value in (c, z, tau, w):
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_veto_and_infinite_atoms_are_exact_and_have_zero_masked_adjoints():
    c = _tensor([-math.inf, math.inf, 2.0, math.nan, math.inf, -0.4], grad=True)
    z = _tensor([0.0, 0.0, 1e6, math.nan, 1e6, 0.2], grad=True)
    eligible = torch.tensor([True, True, False, False, False, True])
    tau, w = _tensor(0.7, grad=True), _tensor(0.3, grad=True)
    logh, log1mh = pivot.conditional_log_hazards(c, tau, w, z, eligible=eligible)
    veto = torch.tensor([0, 2, 3, 4])
    assert torch.isneginf(logh[veto]).all()
    assert torch.equal(log1mh[veto], torch.zeros(4, dtype=DTYPE))
    assert logh[1] == 0 and torch.isneginf(log1mh[1])
    # Exclude impossible outcomes, but retain deterministic outcomes in the loss.
    loss = -log1mh[veto].sum() - logh[1] - logh[5]
    loss.backward()
    for value in (c, z, tau, w):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert torch.equal(c.grad[:5], torch.zeros(5, dtype=DTYPE))
    assert torch.equal(z.grad[:5], torch.zeros(5, dtype=DTYPE))
    assert c.grad[5].abs() > 0 and tau.grad.abs() > 0 and w.grad.abs() > 0


def test_masses_use_time_last_and_normalize_every_prefix_and_conditional_chunk():
    c = _tensor([[[-1.5, -1.5, -0.2, -0.8, 0.6],
                  [-3.0, -2.0, -2.5, 0.1, 1.2]],
                 [[0.2, -0.3, 0.4, -0.6, 0.0],
                  [-0.7, -0.7, -0.7, -0.7, -0.7]]])
    tau, w = _tensor(0.7), _tensor(0.45)
    event, survival = pivot.hybrid_log_masses(c, tau, w)
    assert event.shape == survival.shape == c.shape
    assert event.dtype == survival.dtype == c.dtype
    assert torch.isfinite(event).all() and torch.isfinite(survival).all()
    assert (event <= 0).all() and (survival <= 0).all()
    torch.testing.assert_close(event.exp().cumsum(-1) + survival.exp(),
                               torch.ones_like(c), atol=2e-12, rtol=2e-12)
    assert (survival[..., 1:] < survival[..., :-1]).all()
    # Repeated/decreasing c still have event support under fresh epsilon.
    assert (event[..., 1].exp() > 0).all()
    start = 2
    conditional = ((event[..., start:] - survival[..., start - 1, None]).exp().sum(-1)
                   + (survival[..., -1] - survival[..., start - 1]).exp())
    torch.testing.assert_close(conditional, torch.ones_like(conditional), atol=2e-12, rtol=2e-12)
    # Check no accidental reduction along a scene/slot axis.
    for i in range(2):
        for j in range(2):
            e, s = pivot.hybrid_log_masses(c[i, j], tau, w)
            torch.testing.assert_close(event[i, j], e, rtol=2e-12, atol=2e-12)
            torch.testing.assert_close(survival[i, j], s, rtol=2e-12, atol=2e-12)


def test_veto_ticks_and_certain_event_leave_exact_deterministic_atoms():
    c = _tensor([[-math.inf, -math.inf, math.inf, -2.0, math.inf],
                 [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf]])
    event, survival = pivot.hybrid_log_masses(c, _tensor(0.7), _tensor(0.3))
    expected_event = torch.full_like(c, -math.inf)
    expected_event[0, 2] = 0
    expected_survival = torch.zeros_like(c)
    expected_survival[0, 2:] = -math.inf
    torch.testing.assert_close(event, expected_event, rtol=0, atol=0)
    torch.testing.assert_close(survival, expected_survival, rtol=0, atol=0)
    for boundary, cross in ((-math.inf, False), (math.inf, True)):
        out = _tick(boundary, uniform=_tensor([0.01]), epsilon_uniform=_tensor([0.99]))
        assert out["cross"].item() is cross
        assert out["event_mass"].item() == float(cross)
        assert out["cdf"].item() == float(cross)


def test_equal_scales_at_zero_have_exact_uniform_mixture_path_masses():
    # If tau == w and c == 0, h(z) = sigmoid(z) is Uniform(0, 1).
    # Hence S_t = E[(1-h)^t] = 1/(t+1), p_t = 1/[t(t+1)].
    # This also rules out incorrectly integrating a fresh z at every decision.
    c, scale = _tensor([0.0] * 8), _tensor(0.6)
    t = torch.arange(1, 9, dtype=DTYPE)
    expected_event, expected_survival = (1 / (t * (t + 1))).log(), -(t + 1).log()
    reference_event, reference_survival = _reference_log_masses(c, scale, scale)
    torch.testing.assert_close(reference_event, expected_event, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(reference_survival, expected_survival, atol=2e-12, rtol=2e-12)
    event, survival = pivot.hybrid_log_masses(c, scale, scale, order=32)
    torch.testing.assert_close(event, expected_event, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(survival, expected_survival, atol=2e-12, rtol=2e-12)


def test_masked_zero_mass_and_post_atom_terms_do_not_poison_likelihood_gradients():
    c = _tensor([[-math.inf, -0.8, -math.inf, 0.3, math.inf, -0.1],
                 [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf]],
                grad=True)
    tau, w = _tensor(0.7, grad=True), _tensor(0.4, grad=True)
    event, survival = pivot.hybrid_log_masses(c, tau, w, order=64)
    loss = -(torch.where(torch.isfinite(event), event, torch.zeros_like(event)).sum()
             + torch.where(torch.isfinite(survival), survival, torch.zeros_like(survival)).sum())
    loss.backward()
    for value in (c, tau, w):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert c.grad[0, 1].abs() > 0 and c.grad[0, 3].abs() > 0
    assert torch.equal(c.grad[1], torch.zeros_like(c.grad[1]))
    assert torch.equal(c.grad[0, [0, 2, 4, 5]], torch.zeros(4, dtype=DTYPE))


@pytest.mark.parametrize("values,tau_value,w_value", [
    ([-1.4, -0.6, -0.9, 0.2, 0.8], 0.7, 0.4),
    ([-0.004, -0.001, -0.002, 0.001, 0.003], 0.001, 0.0002),
    ([-0.8, -0.2, -0.4, 0.3, 0.9], 0.7, 0.003),
    ([-40.0, -39.0, -41.0, -38.0, -37.0], 0.005, 0.02),
    ([37.0, 38.0, 36.0, 39.0, 40.0], 0.001, 0.02),
], ids=["ordinary", "small-scales", "narrow-epsilon", "negative-tail", "positive-tail"])
def test_quadrature_refinement_matches_independent_masses_nll_and_gradients(
        values, tau_value, w_value):
    c = _tensor(values, grad=True)
    tau, w = _tensor(tau_value, grad=True), _tensor(w_value, grad=True)
    ref_event, ref_survival = _reference_log_masses(c, tau, w)
    ref_nll = _nll(ref_event, ref_survival)
    ref_grad = torch.cat([g.reshape(-1) for g in torch.autograd.grad(ref_nll, (c, tau, w))])
    assert torch.isfinite(ref_nll) and torch.isfinite(ref_grad).all()
    errors = []
    for order in (32, 64, 128):
        event, survival = pivot.hybrid_log_masses(c, tau, w, order=order)
        loss = _nll(event, survival)
        grad = torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, (c, tau, w))])
        assert torch.isfinite(event).all() and torch.isfinite(survival).all()
        assert torch.isfinite(loss) and torch.isfinite(grad).all()
        mass_error = torch.cat(((event.exp() - ref_event.exp()).abs(),
                                (survival.exp() - ref_survival.exp()).abs())).max()
        nll_error = (loss - ref_nll).abs() / (1 + ref_nll.abs())
        gradient_error = ((grad - ref_grad).abs() / (1 + ref_grad.abs())).max()
        errors.append(torch.stack((mass_error, nll_error, gradient_error)).detach())
        if order == 128:
            torch.testing.assert_close(event, ref_event, atol=3e-4, rtol=3e-6)
            torch.testing.assert_close(survival, ref_survival, atol=3e-4, rtol=3e-6)
            torch.testing.assert_close(grad, ref_grad, atol=3e-3, rtol=3e-3)
    # Allow roundoff plateaus, but a higher order must not hide a regression.
    assert (errors[-1] <= errors[0] * 1.25 + 2e-7).all(), errors
    assert (errors[-1] < _tensor([2e-5, 2e-5, 3e-3])).all(), errors


def test_hybrid_nll_gradcheck_includes_boundary_population_scale_and_width():
    c = _tensor([-1.0, -0.4, -0.7, 0.5], grad=True)
    tau, w = _tensor(0.65, grad=True), _tensor(0.4, grad=True)

    def objective(c, tau, w):
        return _nll(*pivot.hybrid_log_masses(c, tau, w, order=64))

    assert torch.autograd.gradcheck(objective, (c, tau, w), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_width_to_zero_recovers_strict_running_max_masses_not_independent_hazards():
    c = _tensor([-1.5, -2.0, -0.5, -1.0, 0.4, 0.1, 1.1])
    tau = _tensor(0.7)
    maximum = c.cummax(-1).values
    previous = torch.cat((torch.full_like(c[:1], -math.inf), maximum[:-1]))
    strict_event = pivot.log_event_mass(previous, maximum, tau).exp()
    strict_survival = pivot.log_survival(maximum, tau).exp()
    errors = []
    for width in (0.1, 0.01, 0.001):
        event, survival = pivot.hybrid_log_masses(c, tau, _tensor(width), order=128)
        errors.append(max(float((event.exp() - strict_event).abs().max()),
                          float((survival.exp() - strict_survival).abs().max())))
    assert errors[2] < errors[1] < errors[0], errors
    assert errors[-1] < 3e-4, errors
    assert torch.equal(strict_event[[1, 3, 5]], torch.zeros(3, dtype=DTYPE))


def test_default_and_explicit_none_width_preserve_strict_latent_and_crossing():
    c, tau = _tensor([-1.0, 0.2, -0.5]), _tensor(0.7)
    u = _tensor([0.2, 0.5, 0.8])
    mask = torch.ones_like(c, dtype=torch.bool)
    default = pivot.update_state(c, tau, mask, mask, uniform=u)
    explicit = pivot.update_state(c, tau, mask, mask, uniform=u, w=None)
    _assert_same_tree(default, explicit)
    z = u.log() - torch.log1p(-u)
    torch.testing.assert_close(default["z"], z, rtol=0, atol=0)
    assert torch.equal(default["cross"], c + tau * z >= 0)
    next_c = c - 0.5
    a = pivot.update_state(next_c, tau, mask, mask, carry=default)
    b = pivot.update_state(next_c, tau, mask, mask, carry=explicit, w=None)
    _assert_same_tree(a, b)
    torch.testing.assert_close(a["maximum"], c, rtol=0, atol=0)
    torch.testing.assert_close(a["z"], z, rtol=0, atol=0)


def test_explicit_epsilon_draws_are_fresh_and_crossing_uses_both_latents():
    c, u = _tensor([-0.4, -0.4, -0.4]), _tensor([0.2, 0.5, 0.8])
    first = _tick(c, uniform=u, epsilon_uniform=_tensor([0.9, 0.1, 0.5]))
    second_u = _tensor([0.1, 0.9, 0.2])
    second = _tick(c, carry=first, uniform=1 - u, epsilon_uniform=second_u)
    torch.testing.assert_close(first["z"], second["z"], rtol=0, atol=0)
    epsilon = second_u.log() - torch.log1p(-second_u)
    torch.testing.assert_close(second["epsilon"], epsilon, rtol=0, atol=0)
    assert torch.equal(second["cross"], c + 0.7 * first["z"] + 0.5 * epsilon >= 0)
    assert torch.equal(second["decision_step"], first["decision_step"] + 1)
    assert not torch.equal(first["epsilon"], second["epsilon"])


def test_unrelated_identity_reset_cannot_change_an_existing_identity_epsilon_stream():
    c, identity = _tensor([-0.5, -0.5]), torch.tensor([73, 104])
    first = _tick(c, identity=identity, uniform=_tensor([0.5, 0.5]))
    normal = _tick(c, carry=_clone_tree(first), identity=identity)
    torch.manual_seed(297)
    replaced = _tick(c, carry=_clone_tree(first), identity=torch.tensor([73, 105]),
                     uniform=_tensor([0.5, 0.3]))
    torch.testing.assert_close(normal["epsilon"][0], replaced["epsilon"][0], rtol=0, atol=0)
    torch.testing.assert_close(normal["decision_step"][0], replaced["decision_step"][0],
                               rtol=0, atol=0)
    torch.testing.assert_close(normal["rng_seed"][0], replaced["rng_seed"][0], rtol=0, atol=0)
    assert first["epsilon"][0] != first["epsilon"][1]


def test_sampled_conditional_frequencies_match_analytic_hazard():
    n = 20000
    c = _tensor([-0.9, -0.2, 0.6])[:, None].expand(3, n)
    z = _tensor([-0.5, 0.0, 0.8])[:, None].expand_as(c)
    generator = torch.Generator(device="cpu").manual_seed(3721)
    # Interior uniforms avoid measure-zero RNG endpoints without clipping tails.
    epsilon_u = (torch.rand(c.shape, generator=generator, dtype=DTYPE) * (1 - 2e-12) + 1e-12)
    result = _tick(c, uniform=z.sigmoid(), epsilon_uniform=epsilon_u)
    expected = pivot.conditional_log_hazards(c[:, 0], _tensor(0.7), _tensor(0.5),
                                             z[:, 0])[0].exp()
    observed = result["cross"].to(DTYPE).mean(-1)
    tolerance = 6 * (expected * (1 - expected) / n).sqrt() + 1 / n
    assert ((observed - expected).abs() < tolerance).all(), (observed, expected)


def test_sampled_first_passage_timing_matches_integrated_path_masses():
    values = _tensor([-1.4, -1.4, -0.4, -1.1, 0.5])
    event, survival = pivot.hybrid_log_masses(values, _tensor(0.65), _tensor(0.5), order=64)
    n = 16000
    shape = (n,)
    alive = torch.ones(shape, dtype=torch.bool)
    identity = torch.arange(1000, 1000 + n, dtype=torch.long)
    carry, counts = None, []
    for value in values:
        carry = _tick(value.expand(shape), carry=carry, tau=0.65, w=0.5,
                      ramp=alive, identity=identity)
        crossed = carry["cross"] & alive
        counts.append(crossed.sum())
        alive = alive & ~crossed
    observed = torch.stack((*counts, alive.sum())).to(DTYPE) / n
    expected = torch.cat((event.exp(), survival[-1:].exp()))
    assert (event[1].exp() > 0.02).item()  # repeated c distinguishes this from strict passage
    tolerance = 6 * (expected * (1 - expected) / n).sqrt() + 1 / n
    assert ((observed - expected).abs() < tolerance).all(), (observed, expected)
    assert observed.sum().item() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("split", [1, 3])
def test_rng_chunk_state_roundtrip_ignores_global_rng_and_preserves_history(tmp_path, split):
    c = _tensor([[-1.2, -0.8, -1.0, 0.1, -0.5],
                 [-0.7, -0.7, -0.3, -0.9, 0.4]])
    identity = torch.tensor([41, 93])
    carry, uninterrupted = None, []
    for i in range(c.shape[-1]):
        carry = _tick(c[:, i], carry=carry, identity=identity,
                      uniform=_tensor([0.4, 0.6]), order=64)
        uninterrupted.append(_clone_tree(carry))
    saved = uninterrupted[split - 1]
    assert {"epsilon", "decision_step", "rng_seed", "logS", "event_mass", "cdf"} <= saved.keys()
    assert any("history" in key.lower() for key in saved), saved.keys()
    file = tmp_path / "hybrid_carry.pt"
    torch.save(saved, file)
    restored = torch.load(file, map_location="cpu", weights_only=True)
    _assert_same_tree(restored, saved)
    for i in range(split, c.shape[-1]):
        # A resumption need not restore the process-global random-number stream.
        torch.manual_seed(901 + i)
        torch.rand(113 + i)
        restored = _tick(c[:, i], carry=restored, identity=identity,
                         uniform=_tensor([0.99, 0.01]), order=64)
        _assert_same_tree(restored, uninterrupted[i])
    assert not torch.equal(uninterrupted[1]["epsilon"], uninterrupted[2]["epsilon"])
    for i, state in enumerate(uninterrupted):
        event, survival = pivot.hybrid_log_masses(c[:, :i + 1], _tensor(0.7),
                                                 _tensor(0.5), order=64)
        torch.testing.assert_close(state["logS"], survival[:, -1], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(state["event_mass"], event[:, -1].exp(),
                                   atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(state["cdf"], -torch.expm1(survival[:, -1]),
                                   atol=2e-12, rtol=2e-12)


def test_hybrid_history_gradients_survive_attached_chunks():
    c = _tensor([-1.0, -0.4, -0.8, 0.3], grad=True)
    tau, w = _tensor(0.7, grad=True), _tensor(0.4, grad=True)
    ones = torch.ones(1, dtype=torch.bool)
    carry = None
    for value in c:
        carry = pivot.update_state(value.reshape(1), tau, ones, ones, carry=carry, w=w,
                                   uniform=_tensor([0.5]), epsilon_uniform=_tensor([0.5]),
                                   quadrature_order=64)
    chunk_nll = -carry["event_mass"].log().sum() - 0.4 * carry["logS"].sum()
    chunk_grad = torch.autograd.grad(chunk_nll, (c, tau, w))
    event, survival = pivot.hybrid_log_masses(c, tau, w, order=64)
    full_nll = -event[-1] - 0.4 * survival[-1]
    full_grad = torch.autograd.grad(full_nll, (c, tau, w))
    torch.testing.assert_close(chunk_nll, full_nll, atol=2e-12, rtol=2e-12)
    for chunk, full in zip(chunk_grad, full_grad):
        assert torch.isfinite(chunk).all()
        torch.testing.assert_close(chunk, full, atol=2e-10, rtol=2e-10)
    assert (chunk_grad[0].abs() > 1e-8).all()  # even a non-record-setting earlier boundary matters


def test_hybrid_identity_reset_clears_old_history_and_risk_exit_is_absorbing():
    first = _tick(-1.0, identity=torch.tensor([11]), uniform=_tensor([0.4]),
                  epsilon_uniform=_tensor([0.5]))
    veto = _tick(-math.inf, carry=first, identity=torch.tensor([11]))
    assert not veto["terminal"].item() and not veto["cross"].item()
    torch.testing.assert_close(veto["logS"], first["logS"], rtol=0, atol=2e-12)
    exited = _tick(-0.5, carry=veto, identity=torch.tensor([11]), ramp=torch.tensor([False]))
    reentry = _tick(math.inf, carry=exited, identity=torch.tensor([11]))
    assert reentry["terminal"].item() and not reentry["cross"].item()
    torch.testing.assert_close(reentry["logS"], exited["logS"], rtol=0, atol=0)
    fresh = _tick(-0.3, carry=reentry, identity=torch.tensor([12]), uniform=_tensor([0.8]),
                  epsilon_uniform=_tensor([0.2]))
    expected_event, expected_survival = pivot.hybrid_log_masses(
        _tensor([-0.3]), _tensor(0.7), _tensor(0.5))
    assert not fresh["terminal"].item()
    torch.testing.assert_close(fresh["z"], _tensor([math.log(4)]))
    torch.testing.assert_close(fresh["logS"], expected_survival, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(fresh["event_mass"], expected_event.exp(), atol=2e-12, rtol=2e-12)
    absent = _tick(-0.3, carry=fresh, identity=torch.tensor([12]), present=torch.tensor([False]))
    back = _tick(-0.7, carry=absent, identity=torch.tensor([12]), uniform=_tensor([0.2]),
                 epsilon_uniform=_tensor([0.8]))
    _, survival = pivot.hybrid_log_masses(_tensor([-0.7]), _tensor(0.7), _tensor(0.5))
    torch.testing.assert_close(back["logS"], survival, atol=2e-12, rtol=2e-12)


def _native_state():
    # Tiny positive clearances keep the saved ramp identity at risk, so resumption
    # really exercises fresh draws and the nonempty likelihood posterior.
    s = torch.tensor([[30.0, 35.001, 24.999]])
    return {"s_m": s, "v_mps": torch.zeros_like(s), "a_mps2": torch.zeros_like(s),
            "length_m": torch.full_like(s, 5.0), "lane": torch.tensor([[2, 1, 1]]),
            "present": torch.ones_like(s, dtype=torch.bool),
            "is_cav": torch.zeros_like(s, dtype=torch.bool),
            "vehicle_id": torch.tensor([[51, 52, 53]])}


def _stationary_forward(state, **kwargs):
    lane = state["lane"]
    logits = torch.full((*lane.shape, 3), -20.0)
    logits.scatter_(-1, lane.unsqueeze(-1), 20.0)
    return {"accel_mps2": torch.zeros_like(state["s_m"]), "lane_logits": logits,
            "lane_feasible": torch.ones_like(logits, dtype=torch.bool),
            "hdv_mask": state["present"] & ~state["is_cav"],
            "driver_parameters": torch.ones((*lane.shape, 9)),
            "lane_age": torch.zeros_like(lane), "lane_v5_acc": kwargs.get("lane_v5_acc"),
            "lane_sample": torch.zeros_like(state["present"])}


def test_native_hybrid_checkpoint_and_saved_rollout_state_resume_exactly(tmp_path):
    spec = ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                     lane_decision="sample", merge_decision="first_passage",
                     merge_passage_law="hybrid", merge_w_init=0.45,
                     merge_quadrature_order=64)
    model = WmSModel(spec, physics=PhysicsConfig()).cpu().eval()
    assert "w" in model.merge_pivot.params()
    assert model.merge_pivot.params()["w"].item() == pytest.approx(0.45, rel=1e-6)
    with torch.no_grad():
        # A non-initial value proves the checkpoint restores the learned width.
        model.merge_pivot.w_raw.add_(0.23)
        with patch.object(model, "forward", side_effect=_stationary_forward):
            first = model.rollout(_native_state(), n_steps=2,
                                  lane_uniform=torch.full((1, 2, 3), 0.01))
    saved_state = first["state"]
    assert saved_state["lane"][0, 0] == 2
    assert not saved_state["merge_pivot_terminal"][0, 0]
    for name in ("epsilon", "decision_step", "rng_seed", "logS"):
        assert "merge_pivot_" + name in saved_state
    checkpoint = tmp_path / "native_hybrid.pt"
    save_checkpoint(checkpoint, model, {**asdict(spec), "physics": asdict(model.physics)},
                    extra={"rollout_state": _clone_tree(saved_state),
                           "rollout_history": _clone_tree(first["history"]),
                           "rollout_history_mask": _clone_tree(first["history_mask"])})
    loaded = load_checkpoint(checkpoint, device="cpu")
    restored = loaded.model
    assert restored.spec.merge_passage_law == "hybrid"
    assert restored.spec.merge_quadrature_order == 64
    _assert_same_tree(restored.state_dict(), model.state_dict())
    torch.manual_seed(291)
    with torch.no_grad(), patch.object(model, "forward", side_effect=_stationary_forward):
        expected = model.rollout(saved_state, n_steps=3, k0=2,
                                 history=first["history"], history_mask=first["history_mask"])
    torch.manual_seed(819)
    torch.rand(307)
    with torch.no_grad(), patch.object(restored, "forward", side_effect=_stationary_forward):
        actual = restored.rollout(loaded.payload_meta["rollout_state"], n_steps=3, k0=2,
                                  history=loaded.payload_meta["rollout_history"],
                                  history_mask=loaded.payload_meta["rollout_history_mask"])
    for key in ("positions", "speeds", "lanes", "merge_pivot_cdf", "merge_pivot_event_mass"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    _assert_same_tree(actual["state"], expected["state"])
    assert actual["state"]["merge_pivot_decision_step"][0, 0] > saved_state[
        "merge_pivot_decision_step"][0, 0]
