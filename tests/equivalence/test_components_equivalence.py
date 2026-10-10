"""Small model components vs legacy: bounds, gate, history, stability, entries v2."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wms.model import entries, history, monotone_gate, soft_bound
from wms.model import stability as stab_new
from wms.model.residual_bound import bounded_value, bounded_value_asym

from .legacy import legacy, load_legacy


@pytest.fixture(scope="module", autouse=True)
def _legacy():
    load_legacy()


def test_bounds_equal_including_gradient():
    lh, bf, st = legacy("long_horizon"), legacy("brakefix"), legacy("stability")
    x = torch.linspace(-40, 40, 101, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = lh.soft_bound(x, 2.0, 4.0), soft_bound(y, 2.0, 4.0)
    assert torch.equal(a, b)
    a.sum().backward()
    b.sum().backward()
    assert torch.equal(x.grad, y.grad)
    z = torch.linspace(-9, 9, 37)
    assert torch.equal(bf.bounded_value_asym(z, 1.5, 3.0), bounded_value_asym(z, 1.5, 3.0))
    assert torch.equal(st.bounded_value(z, 2.0), bounded_value(z, 2.0))


def test_gate_zero_and_monotone():
    inc = torch.randn(240)
    g = monotone_gate(inc, torch.arange(0, 241), 240)
    assert g[0] == 0 and bool((g[1:] >= g[:-1]).all())
    assert torch.equal(g[:1], torch.zeros(1))


def test_history_spec_equal():
    lh = legacy("history")
    for args in ((None, 1.0, 0.5), (4.0, 1.0, 0.5), (10.0, 2.0, 0.5)):
        assert lh.history_spec(*args) == history.history_spec(*args)
    with pytest.raises(ValueError):
        history.history_spec(4.0, 0.7, 0.5)


def test_stability_utilities_equal():
    st = legacy("stability")
    masks = torch.rand(2, 6, 8) > 0.3
    old = st.ConservationTracker(masks[:, 0]).track_rollout(masks)
    new = stab_new.ConservationTracker(masks[:, 0]).track_rollout(masks)
    assert old.summary() == new.summary()
    errs = {1.0: 0.5, 5.0: 1.2, 10.0: 3.1, 30.0: 9.0}
    assert st.error_growth_diagnostic(errs) == stab_new.error_growth_diagnostic(errs)


def test_entry_v2_and_history_builders_equal():
    ent = legacy("entries")
    rng = np.random.default_rng(0)
    counter = rng.poisson(2.0, size=(2, 30)).astype(float)
    ho = ent.build_entrance_history(counter=counter)
    hn = entries.build_entrance_history(counter=counter)
    assert np.array_equal(ho.to_features(), hn.to_features())

    def run(mod):
        b, k = 2, 20
        state = {"s_m": torch.zeros(b, k), "v_mps": torch.zeros(b, k),
                 "a_mps2": torch.zeros(b, k), "lane": torch.zeros(b, k, dtype=torch.long),
                 "length_m": torch.full((b, k), 5.0),
                 "present": torch.zeros(b, k, dtype=torch.bool),
                 "is_cav": torch.zeros(b, k, dtype=torch.bool)}
        state["present"][:, :8] = True
        state["s_m"][:, :8] = torch.arange(8) * 30.0 - 2600.0
        state["v_mps"][:, :8] = 18.0
        state["lane"][:, :8] = torch.arange(8) % 2
        proc = mod.TorchEntryProcessV2(counter, seed=3, batch=2)
        masks = [proc.insert(state).clone() for _ in range(80)]
        return state, torch.stack(masks), proc.summary()

    (so, mo, summo), (sn, mn, summn) = run(ent), run(entries)
    assert torch.equal(mo, mn) and summo == summn
    for key in so:
        assert torch.equal(so[key], sn[key]), key
    assert int(mo.sum()) > 0
