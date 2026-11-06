"""Exactness of the memory-bounded hybrid_log_masses against the original O(T^2) implementation.

``_reference`` is the pre-change implementation, kept verbatim as a private oracle (it builds one graph over the
whole [cars, knots x order] grid and loops over every time step). Synthetic CPU float64 only.
"""
import pytest
import torch

from wms.model import merge_pivot as pivot
from wms.model.merge_pivot import (_legendre, _logsumexp_supported, _scale, conditional_log_hazards,
                                   log_event_mass, log_survival)

NEG, POS = -torch.inf, torch.inf


def _reference(c, tau, w, order=32):
    if c.shape[-1] == 0:
        return c.clone(), c.clone()
    c = c.to(torch.float64)
    atom = (c == POS).long().cumsum(-1)
    after_atom = atom - (c == POS).long() > 0
    c = torch.where(after_atom, torch.full_like(c, NEG), c)
    tau, w = _scale(tau, c), _scale(w, c)
    finite = torch.isfinite(c)
    safe = torch.where(finite, c, torch.zeros_like(c))
    offsets = c.new_tensor([-8., -1., 0., 1., 8.])
    knots = ((-safe.unsqueeze(-1) + w * offsets) / tau).detach()
    knots = torch.where(finite.unsqueeze(-1), knots, torch.zeros_like(knots))
    knots = knots.flatten(-2).sort(-1).values
    knots = torch.unique_consecutive(knots, dim=-1)
    ends = torch.full_like(knots[..., :1], POS)
    knots = torch.cat((-ends, knots, ends), -1)
    a, b = knots[..., :-1], knots[..., 1:]
    logweights = log_event_mass(a, b, c.new_tensor(1.))
    nodes_np, weights_np = _legendre(order)
    u = torch.as_tensor(nodes_np, device=c.device, dtype=c.dtype)
    weights = torch.as_tensor(weights_np, device=c.device, dtype=c.dtype)
    la, lb = log_survival(-a, c.new_tensor(1.)), log_survival(-b, c.new_tensor(1.))
    sa, sb = log_survival(a, c.new_tensor(1.)), log_survival(b, c.new_tensor(1.))
    logu = torch.logaddexp(la.unsqueeze(-1) + torch.log1p(-u), lb.unsqueeze(-1) + u.log())
    logv = torch.logaddexp(sa.unsqueeze(-1) + torch.log1p(-u), sb.unsqueeze(-1) + u.log())
    z = (logu - logv).flatten(-2)
    lw = (logweights.unsqueeze(-1) + weights.log()).flatten(-2)
    running = torch.zeros_like(z)
    events, survival = [], []
    for step in range(c.shape[-1]):
        lh, ls = conditional_log_hazards(c[..., step, None], tau, w, z)
        le = _logsumexp_supported(lw + running + lh, -1)
        no_previous = (c[..., :step] == NEG).all(-1)
        le = torch.where(no_previous & (c[..., step] == POS), torch.zeros_like(le), le)
        events.append(le)
        running = running + ls
        surv = _logsumexp_supported(lw + running, -1)
        surv = torch.where((c[..., :step + 1] == NEG).all(-1), torch.zeros_like(surv), surv)
        survival.append(surv)
    return torch.stack(events, -1), torch.stack(survival, -1)


def _params():
    return (torch.tensor(0.55, dtype=torch.float64, requires_grad=True),
            torch.tensor(0.9, dtype=torch.float64, requires_grad=True))


def _paths(rows, steps, seed, live_fraction=0.15):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(rows, steps, generator=g, dtype=torch.float64) * 0.8 - 0.4
    c = torch.where(torch.rand(rows, steps, generator=g) < live_fraction, c, torch.full_like(c, NEG))
    return c


def _compare(c, order=16, node_budget=None, grad=True, rtol=1e-10):
    results = []
    for fn in (lambda cc, t, w_: _reference(cc, t, w_, order),
               lambda cc, t, w_: pivot.hybrid_log_masses(cc, t, w_, order, node_budget=node_budget)):
        cc = c.clone().requires_grad_(grad)
        tau, w = _params()
        ev, sv = fn(cc, tau, w)
        out = [ev.detach(), sv.detach()]
        if grad:
            # exercise every finite mass: event masses of positive events and all finite survivals
            loss = (torch.where(torch.isfinite(ev), ev, torch.zeros_like(ev)) * 0.7
                    + torch.where(torch.isfinite(sv), sv, torch.zeros_like(sv)) * 1.3).sum()
            out += list(torch.autograd.grad(loss, (cc, tau, w), allow_unused=True))
        results.append(out)
    ref, new = results
    for x, y in zip(ref[:2], new[:2]):
        assert torch.equal(torch.isneginf(x), torch.isneginf(y))
        assert torch.equal(torch.isposinf(x), torch.isposinf(y))
        m = torch.isfinite(x)
        err = ((x - y).abs()[m] / x.abs()[m].clamp_min(1.0)).max() if m.any() else torch.tensor(0.)
        assert err <= rtol, float(err)
    for x, y in zip(ref[2:], new[2:]):
        if x is None:
            assert y is None or float(y.abs().max()) == 0.0
            continue
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        assert float((x - y).abs().max() / x.abs().max().clamp_min(1.0)) <= 1e-8


@pytest.mark.parametrize("budget", [None, 1, 4000])
def test_equivalence_values_and_gradients_sparse_vetoes(budget):
    c = _paths(5, 80, 1)
    c[0] = NEG                       # an all-vetoed row
    _compare(c, node_budget=budget)


def test_equivalence_long_prefix_over_two_thousand_steps():
    # F2-like: a very long mostly vetoed prefix. The oracle's graph is O(T x knots), so keep the live set small.
    shared = _paths(3, 2100, 2, live_fraction=1.0)
    live = torch.zeros(2100, dtype=torch.bool)
    live[torch.tensor([1500, 1530, 1790, 2000, 2050, 2099])] = True   # same live columns, different values
    _compare(torch.where(live, shared, torch.full_like(shared, NEG)), order=8, node_budget=3000)
    # rows with DIFFERENT live columns: values only (no graph) at T > 2000
    ragged = _paths(3, 2100, 3, live_fraction=0.004)
    ragged[:, :1900] = NEG
    _compare(ragged, order=8, node_budget=None, grad=False)
    _compare(ragged, order=8, node_budget=1500, grad=False)


def test_equivalence_with_atoms_and_mixed_rows():
    c = _paths(6, 60, 3, live_fraction=0.5)
    c[1, 20] = POS                   # deterministic atom, later columns become impossible
    c[2, 5] = POS                    # atom as first live step (no previous)
    c[3, :] = NEG
    c[4, 59] = POS
    c[5, 10:30] = c[5, 10]           # repeated values -> coincident knots
    for budget in (None, 1, 777):
        _compare(c, node_budget=budget)


def test_equivalence_single_path_and_leading_dims():
    c1 = _paths(1, 50, 4, live_fraction=0.4)[0]
    _compare(c1)
    c3 = _paths(6, 40, 5, live_fraction=0.4).reshape(2, 3, 40)
    _compare(c3, node_budget=300)


def test_all_vetoed_and_empty():
    c = torch.full((2, 7), NEG, dtype=torch.float64)
    ev, sv = pivot.hybrid_log_masses(c, torch.tensor(0.5), torch.tensor(1.0))
    assert torch.isneginf(ev).all() and (sv == 0).all()
    ev, sv = pivot.hybrid_log_masses(c[0], torch.tensor(0.5), torch.tensor(1.0))
    assert torch.isneginf(ev).all() and (sv == 0).all()
    assert pivot.hybrid_log_masses(torch.zeros(2, 0, dtype=torch.float64), torch.tensor(0.5), torch.tensor(1.0))[0].shape == (2, 0)


def test_grouped_matches_ungrouped_at_high_order():
    c = _paths(4, 40, 6, live_fraction=0.6)
    _compare(c, order=64, node_budget=5000)
