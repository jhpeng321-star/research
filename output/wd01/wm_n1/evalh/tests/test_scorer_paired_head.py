"""Tests for the ``paired_head`` dV source (owner-approved 2026-10-04).

Generated fixtures only: no CAL/HOLDOUT data, no real checkpoint.  The paired
head is represented by a tiny fake exposing ``dv_quantiles(pair_infos)``.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evalh.protocol import N_QUANTILES, OutcomeRow
from evalh.scorer import cli
from evalh.scorer.baselines import ZeroEffectBaseline
from evalh.scorer.cli import ScorerCliRefusal
from evalh.scorer.metrics_loop import (DV_METHOD_COMONOTONE,
                                       DV_METHOD_PAIRED_HEAD, score)

_MEMBER_BYTES = b"paired-head-test-member-bytes"
_RELEASE_SHA = "a" * 64
_GIT_HEAD = "b" * 40


def _rows():
    rows = []
    for tau in (60.0, 120.0):
        rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                               tau_s=tau, policy="native", y_delay_s=2.0))
        rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                               tau_s=tau, policy="damping", y_delay_s=7.0))
    return rows


def _const_q(base):
    return np.stack([np.full(N_QUANTILES, float(base)) for _ in _rows()])


class FakePairedHead:
    """Predicts a constant dV quantile vector for every pair."""

    def __init__(self, base):
        self.base = float(base)
        self.calls = 0

    def dv_quantiles(self, pair_infos):
        self.calls += 1
        return np.stack([np.full(N_QUANTILES, self.base)
                         for _ in pair_infos])


def test_paired_head_uses_direct_quantiles_and_scores_perfectly():
    rows = _rows()
    q_a = _const_q(0.0)             # marginal A is irrelevant for this method
    head = FakePairedHead(5.0)      # dy = 7 - 2 = 5 exactly
    res = score(rows, {"A": q_a}, dv_models={"zero": ZeroEffectBaseline()},
                taus=(60, 120), n_boot=20, seed=1, dv_method=DV_METHOD_PAIRED_HEAD,
                paired_head=head)
    assert res["dv_method"] == DV_METHOD_PAIRED_HEAD
    cell = res["cells"]["damping-native"]["60"]
    assert cell["crps_A"] == 0.0
    assert cell["coverage_10_90_A"] == 1.0
    assert cell["sign_accuracy_A"] == 1.0
    assert head.calls == 1


def test_paired_head_differs_from_comonotone():
    rows = _rows()
    q_a = _const_q(0.0)
    com = score(rows, {"A": q_a}, dv_models={"zero": ZeroEffectBaseline()},
                taus=(60,), n_boot=20, seed=1, dv_method=DV_METHOD_COMONOTONE)
    head = score(rows, {"A": q_a}, dv_models={"zero": ZeroEffectBaseline()},
                 taus=(60,), n_boot=20, seed=1,
                 dv_method=DV_METHOD_PAIRED_HEAD, paired_head=FakePairedHead(5.0))
    assert com["cells"]["damping-native"]["60"]["crps_A"] != \
        head["cells"]["damping-native"]["60"]["crps_A"]


def test_paired_head_without_source_refused():
    rows = _rows()
    with pytest.raises(ValueError, match="paired_head"):
        score(rows, {"A": _const_q(0.0)},
              dv_models={"zero": ZeroEffectBaseline()}, taus=(60,),
              dv_method=DV_METHOD_PAIRED_HEAD)


def test_paired_head_refused_on_two_model_path():
    rows = _rows()
    with pytest.raises(ValueError, match="direct-DV comparator path"):
        score(rows, {"A": _const_q(0.0), "v2g": _const_q(1.0)}, taus=(60,),
              dv_method=DV_METHOD_PAIRED_HEAD, paired_head=FakePairedHead(5.0))


def test_cli_parser_accepts_paired_head():
    args = cli.build_parser().parse_args(
        ["score", "--release-spec", "r.json", "--release-sha", _RELEASE_SHA,
         "--checkpoint", "m.pt", "--out", "out",
         "--dv-method", "paired_head"])
    assert args.dv_method == "paired_head"


# ------------------------------------------------------------- CLI wiring -----
class _FakeSource:
    def __init__(self, spec, half="cal_tune"):
        self.spec = spec
        self.half = half
        self.selected = [object()]

    def outcome_rows(self):
        return _rows()


class _FakeBound:
    def __init__(self, sha):
        self.bundle = object()
        self.sha256 = sha
        self.member = 1


class _FakePredictor:
    def __init__(self, paired_head_dv=None):
        self.name = "A:teacher"
        self.offset = 0.0
        self.paired_head_dv = paired_head_dv

    def predict_quantiles(self, rows):
        return np.stack([np.full(N_QUANTILES, 5.0) for _ in rows])


def _member_file(tmp_path):
    p = tmp_path / "member_1.pt"
    p.write_bytes(_MEMBER_BYTES)
    return p, hashlib.sha256(_MEMBER_BYTES).hexdigest()


def _args(tmp_path, member, comparators="zero", dv_method="paired_head"):
    return SimpleNamespace(
        command="score", release_spec=str(tmp_path / "release.json"),
        release_sha=_RELEASE_SHA, checkpoint=str(member),
        out=str(tmp_path / "out"), params_sha256=None,
        comparators=comparators, cav_motion="teacher",
        dv_method=dv_method)


def _run(tmp_path, member, sha, predictor, out=None):
    spec = {"checkpoint_member_sha256": sha}
    return cli.run_score(
        _args(tmp_path, member, dv_method="paired_head"),
        source_factory=lambda spec_path,
        expected_release_sha256: _FakeSource(spec),
        checkpoint_loader=lambda path, **kw: _FakeBound(sha),
        rows_loader=lambda source: (_rows(), {}),
        a_factory=lambda bundle: predictor,
        git_head=_GIT_HEAD)


def test_cli_refuses_paired_head_without_head(tmp_path):
    member, sha = _member_file(tmp_path)
    with pytest.raises(ScorerCliRefusal, match="paired_head"):
        _run(tmp_path, member, sha, _FakePredictor(paired_head_dv=None))


def test_cli_reports_paired_head_source(tmp_path):
    member, sha = _member_file(tmp_path)
    done = _run(tmp_path, member, sha,
                _FakePredictor(paired_head_dv=FakePairedHead(5.0)))
    results = json.loads(
        (tmp_path / "out" / "RESULTS.json").read_text("utf-8"))
    assert results["inputs"]["dv_method"] == "paired_head"
    assert results["inputs"]["paired_head_source"] == "model_a"
    assert done["inputs"] == results["inputs"]
