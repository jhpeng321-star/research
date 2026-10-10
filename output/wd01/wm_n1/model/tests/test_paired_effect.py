"""Synthetic-only CPU contracts for the climatology-anchored paired-effect head.

No CAL/HOLDOUT/chunk-5/J4B/INC1 data is opened.  The one external file touched
(and only if present, else skipped) is the TRAIN-derived
``TRAIN_CLIMATOLOGY.json`` the task explicitly allows reading.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from model import constants as C
from model.paired_effect import (CLIMATOLOGY_SCHEMA, PLAN_RAW_F,
                                 PairedEffectHead, load_climatology,
                                 order_statistics, paired_effect_pinball,
                                 sha256_file,
                                 untrained_head_returns_climatology)

_TAUS = C.TAUS
_NQ = C.N_QUANTILES


def _cell(seed: int, tau: float) -> dict:
    rng = np.random.default_rng(seed)
    q = np.sort(rng.normal(loc=0.0, scale=10.0 + tau, size=_NQ))
    return {"n_pairs": 3, "n_worlds": 3, "quantiles": [float(x) for x in q]}


def _write_climatology(path: Path, contrast: str = "pi-native") -> Path:
    doc = {
        "schema": CLIMATOLOGY_SCHEMA,
        "contrasts": [contrast],
        "taus": [float(t) for t in _TAUS],
        "quantile_levels": list(C.QUANTILE_LEVELS),
        "reference_policy": "native",
        "cells": {contrast: {f"{float(t):g}": _cell(i + 1, t)
                             for i, t in enumerate(_TAUS)}},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# --------------------------------------------------------------- climatology --
def test_load_climatology_roundtrip_and_sha(tmp_path):
    path = _write_climatology(tmp_path / "clim.json")
    anchor, sha, contrast = load_climatology(path)
    assert anchor.shape == (len(_TAUS), _NQ)
    assert contrast == "pi-native"
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == sha256_file(path)
    # sorted quantiles on every tau
    assert np.all(np.diff(anchor, axis=-1) >= -1e-9)


def test_load_climatology_sha_pin_refuses_mismatch(tmp_path):
    path = _write_climatology(tmp_path / "clim.json")
    with pytest.raises(ValueError):
        load_climatology(path, expected_sha256="0" * 64)


def test_load_climatology_refuses_missing_cell(tmp_path):
    doc = json.loads(_write_climatology(tmp_path / "clim.json").read_text())
    doc["cells"]["pi-native"].pop("60")
    path = tmp_path / "clim.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load_climatology(path)


def test_load_climatology_refuses_wrong_schema(tmp_path):
    path = tmp_path / "clim.json"
    path.write_text(json.dumps({"schema": "other", "cells": {}}),
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_climatology(path)


# -------------------------------------------------------------------- head ----
def _anchor(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.sort(rng.normal(scale=5.0 + t, size=_NQ))
                     for t in _TAUS])


def test_untrained_head_returns_climatology_exactly():
    h = PairedEffectHead(d_model=16, hidden=8)
    a = _anchor()
    h.set_climatology(a)
    token = torch.randn(4, 16)
    q = h(token)
    ref = order_statistics(a)
    assert torch.equal(q, ref.expand_as(q))
    assert float(h.gate.detach()) == 0.0
    assert untrained_head_returns_climatology(h, a, token)
    # mutating the anchor changes the untrained output (the gate is the only
    # learned movement, and it is still exactly zero).
    h.set_climatology(a + 1.0)
    assert torch.equal(h(token), order_statistics(a + 1.0).expand_as(q))


def test_head_output_is_finite_and_monotone():
    h = PairedEffectHead(d_model=8, hidden=8)
    h.set_climatology(_anchor())
    with torch.no_grad():
        h.gate.fill_(0.5)
    q = h.dv_quantiles(torch.randn(3, 8))
    assert q.shape == (3, len(_TAUS), _NQ)
    assert torch.isfinite(q).all()
    assert torch.all(torch.diff(q, dim=-1) >= -1e-6)


def test_gate_and_pinball_loss_are_finite_and_get_gradient():
    h = PairedEffectHead(d_model=8, hidden=8)
    h.set_climatology(_anchor())
    token = torch.randn(5, 8, requires_grad=True)
    dy = torch.randn(5, len(_TAUS))
    mask = torch.ones(5, len(_TAUS), dtype=torch.bool)
    loss = paired_effect_pinball(h(token), dy, mask=mask, weight=0.25)
    assert torch.isfinite(loss)
    loss.backward()
    assert h.gate.grad is not None and torch.isfinite(h.gate.grad).all()
    assert token.grad is not None and torch.isfinite(token.grad).all()
    # a real gradient step moves the gate away from zero
    with torch.no_grad():
        assert float(h.gate.detach()) == 0.0
        h.gate -= 0.1 * h.gate.grad
    assert float(h.gate.detach()) != 0.0


def test_empty_mask_scores_exactly_zero():
    pred = torch.randn(3, len(_TAUS), _NQ)
    dy = torch.randn(3, len(_TAUS))
    empty = torch.zeros(3, len(_TAUS), dtype=torch.bool)
    loss = paired_effect_pinball(pred, dy, mask=empty)
    assert float(loss) == 0.0


# ------------------------------------------------- real TRAIN climatology -----
_REAL = Path(os.environ.get(
    "WM_N1_TRAIN_CLIMATOLOGY",
    "E:/research-dense/runs/active/WM-N1-SCORE-CALTUNE-20261004/"
    "TRAIN_CLIMATOLOGY.json"))


@pytest.mark.skipif(not _REAL.is_file(),
                    reason="TRAIN_CLIMATOLOGY.json absent")
def test_real_train_climatology_anchors_the_head():
    anchor, sha, contrast = load_climatology(_REAL)
    assert sha == ("970f5eff280e5afb3644a7854b631e073eed7dc1812e1636"
                   "ab0bcdc31b60b01d")
    h = PairedEffectHead(d_model=8, hidden=8)
    h.set_climatology(anchor)
    q = h(torch.zeros(1, 8))
    assert torch.equal(q, order_statistics(anchor)[None])


def test_loss_scale_is_q90_minus_q10_and_normalises_pinball():
    """A8 scale fix: s_tau = q90 - q10 of the anchor; scaled pinball is O(1)."""
    torch.manual_seed(0)
    n_tau = len(_TAUS)
    spread = torch.tensor([30.0, 100.0, 330.0, 1300.0, 3800.0])[:n_tau]
    base = torch.linspace(-1.3, 1.3, _NQ)
    anchor = base.unsqueeze(0) * (spread.unsqueeze(1) / 2.6)
    head = PairedEffectHead(8)
    head.set_climatology(anchor.numpy())
    lv = [round(float(x), 4) for x in C.QUANTILE_LEVELS]
    expect = anchor[:, lv.index(0.9)] - anchor[:, lv.index(0.1)]
    assert torch.allclose(head.loss_scale(), expect.clamp(min=1.0))
    dy = torch.randn(64, n_tau) * spread
    dv = head(torch.randn(64, 8))
    raw = paired_effect_pinball(dv, dy)
    scaled = paired_effect_pinball(dv, dy, scale=head.loss_scale())
    assert float(raw) > 50.0
    assert 0.01 < float(scaled) < 2.0
    assert untrained_head_returns_climatology(head, anchor.numpy(),
                                              torch.randn(4, 8))


# ----------------------------------------------------------- plan-aware A9 ----
def test_plan_raw_f_matches_the_correction_summary_width():
    from corr.summary import SUMMARY_RAW_F

    assert PLAN_RAW_F == SUMMARY_RAW_F


def test_plan_aware_untrained_head_returns_climatology_for_any_plan():
    a = _anchor()
    h = PairedEffectHead(d_model=8, hidden=8, plan_aware=True, plan_dim=8)
    h.set_climatology(a)
    token = torch.randn(4, 8)
    ref = order_statistics(a)
    for plan in (torch.zeros(4, PLAN_RAW_F), torch.randn(4, PLAN_RAW_F),
                 torch.full((4, PLAN_RAW_F), 3.0)):
        q = h(token, plan=plan)
        assert torch.equal(q, ref.expand_as(q))
        assert untrained_head_returns_climatology(h, a, token, plan=plan)
    assert float(h.gate.detach()) == 0.0
    assert h.plan_aware and h.plan_dim == 8 and h.plan_proj is not None
    # a non-plan-aware head carries no plan embedding and is unchanged
    plain = PairedEffectHead(d_model=8)
    assert plain.plan_aware is False and plain.plan_proj is None


def test_plan_aware_head_requires_a_plan_vector():
    h = PairedEffectHead(8, plan_aware=True)
    h.set_climatology(_anchor())
    with pytest.raises(ValueError):
        h(torch.randn(2, 8))


def test_plan_changes_output_after_a_gradient_step_and_loss_is_o1():
    torch.manual_seed(0)
    h = PairedEffectHead(8, hidden=8, plan_aware=True, plan_dim=8)
    h.set_climatology(_anchor())
    opt = torch.optim.Adam(h.parameters(), lr=0.05)
    token = torch.randn(16, 8)
    plan = torch.randn(16, PLAN_RAW_F)
    dy = torch.randn(16, len(_TAUS))
    mask = torch.ones(16, len(_TAUS), dtype=torch.bool)
    loss = None
    for _ in range(3):
        opt.zero_grad()
        loss = paired_effect_pinball(h(token, plan=plan), dy, mask=mask,
                                     scale=h.loss_scale())
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(h.gate.grad).all()
        assert torch.isfinite(h.plan_proj.weight.grad).all()
        opt.step()
    assert 0.0 <= float(loss.detach()) < 100.0        # scaled pinball is O(1)
    assert float(h.gate.detach()) != 0.0
    q1 = h(token, plan=torch.zeros_like(plan))
    q2 = h(token, plan=torch.ones_like(plan))
    assert not torch.equal(q1, q2)                    # the plan now matters
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
