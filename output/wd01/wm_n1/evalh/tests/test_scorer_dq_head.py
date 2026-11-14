"""Plan-aware paired-effect decision quality (WM-N1 A9 fix 2).

Synthetic fakes only: no CAL / HOLDOUT / chunk-5 / J4B / INC1 data, no real
checkpoint, no scoring run.  Pins:

* a plan-aware head that encodes the true dV gives zero WM regret;
* the plan-aware path reads each ARM's teacher ``summary_raw`` on the shared
  t0 state and the ``--dv-method paired_head`` path uses the treatment arm;
* a missing plan helper (A8 state-only head) falls back to the state-only
  head;
* the default ``--dq-source marginals`` is unchanged.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evalh.protocol import N_QUANTILES, OutcomeRow
from evalh.scorer import decision_quality as DQ
from evalh.scorer import predictors as P
from evalh.scorer.cli import ScorerCliRefusal, run_score
from evalh.scorer.predictors import ModelAPredictor

from model import constants as MC

TAU = 60.0
N_TAU = MC.N_TAU


def _q(value):
    return np.full(N_QUANTILES, float(value))


def _row(world, arm, y, *, tau=TAU, t0=0.0, dec=0.0, hb=0.0, position=0):
    return OutcomeRow(world=world, t0_s=t0, decision_t_s=dec, tau_s=tau,
                      policy=arm, y_delay_s=y, hard_brake_s=hb,
                      features={"pack_item_position": position})


# ------------------------------------------------------- synthetic modules --
class _FakeEnc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, kin, mask, counter):
        return torch.zeros(int(kin.shape[0]), 4, dtype=kin.dtype)


class _FakeHead(torch.nn.Module):
    """State-only paired head returning a fixed ``[N_TAU, NQ]`` array."""

    def __init__(self, arr):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.register_buffer("_arr", torch.as_tensor(arr, dtype=torch.float32))

    def dv_quantiles(self, token, *, monotone=True):
        b = int(token.shape[0])
        return self._arr.unsqueeze(0).expand(b, -1, -1).clone()


def _bundle(head, *, plan_aware=False, enc=True):
    cfg = SimpleNamespace(paired_effect_plan_aware=plan_aware)
    return SimpleNamespace(
        paired_effect=head,
        state_enc=_FakeEnc() if enc else None,
        config=cfg)


def _window(item="pi", world="w1", t0=0.0):
    return SimpleNamespace(
        item=item, run={}, world=world, t0_s=t0, is_cav_t0=np.zeros(3, bool),
        counter_t0=np.zeros((2, 30)))


def _predictor(bundle):
    return ModelAPredictor(bundle, source=SimpleNamespace(), cav_motion="teacher")


def _tau_slice(arr, tau_s=TAU):
    from model import constants as MC

    taus = [round(float(t), 6) for t in MC.TAUS]
    return arr[taus.index(round(float(tau_s), 6))]


def _fake_t0(recorder):
    def _t0(source, window):
        recorder["t0"].append(window.item)
        return (torch.zeros(1, 3, 7), torch.zeros(1, 3, dtype=torch.bool),
                torch.zeros(1, 2, 30))

    return _t0


# ------------------------------------------------------ plan-aware head ----
def test_plan_aware_head_reads_arm_summary_on_shared_t0(monkeypatch):
    calls = {"summary": [], "t0": []}
    head_arr = {"pi": 1.0, "wall": -2.0}

    def fake_helper(bundle, *, t0_kin, t0_mask, t0_counter, summary_raw):
        calls["summary"].append(summary_raw)
        base = head_arr[str(summary_raw)]
        return np.stack([np.full(N_QUANTILES, base + i)
                         for i in range(N_TAU)])

    monkeypatch.setattr(P, "_plan_aware_head_helper", lambda: fake_helper)
    monkeypatch.setattr(P, "training_summary_raw",
                        lambda source, item, run, *, n_steps, is_cav: item)
    monkeypatch.setattr(P, "_t0_state_channels", _fake_t0(calls))

    pred = _predictor(_bundle(_FakeHead(np.zeros((N_TAU, N_QUANTILES))),
                              plan_aware=True))
    assert pred.plan_aware is True
    pi = pred.paired_head_arm_dv(_window("pi"), TAU)
    wall = pred.paired_head_arm_dv(_window("wall"), TAU)
    assert calls["summary"] == ["pi", "wall"]
    assert calls["t0"] == ["pi", "wall"]           # shared t0 state per arm
    assert np.allclose(pi, 1.0 + _tau_index())
    assert np.allclose(wall, -2.0 + _tau_index())


def _tau_index(tau_s=TAU):
    from model import constants as MC

    return [round(float(t), 6) for t in MC.TAUS].index(round(float(tau_s), 6))


def test_state_only_fallback_when_helper_absent(monkeypatch):
    monkeypatch.setattr(P, "_plan_aware_head_helper", lambda: None)
    monkeypatch.setattr(P, "_t0_state_channels",
                        _fake_t0({"t0": []}))
    arr = np.stack([np.full(N_QUANTILES, 4.0 + i) for i in range(N_TAU)])
    pred = _predictor(_bundle(_FakeHead(arr), plan_aware=False))
    assert pred.plan_aware is False
    got = pred.paired_head_arm_dv(_window("pi"), TAU)
    assert np.allclose(got, 4.0 + _tau_index())


def test_state_only_head_is_not_plan_aware_with_helper_installed(monkeypatch):
    """Fix 2a: helper availability must not imply a plan-aware head."""
    def helper(bundle, *, t0_kin, t0_mask, t0_counter, summary_raw):
        return None

    monkeypatch.setattr(P, "_plan_aware_head_helper", lambda: helper)
    monkeypatch.setattr(P, "_t0_state_channels", _fake_t0({"t0": []}))
    arr = np.stack([np.full(N_QUANTILES, 4.0 + i) for i in range(N_TAU)])
    head = _FakeHead(arr)
    head.plan_aware = False
    head.plan_proj = None
    pred = _predictor(_bundle(head, plan_aware=False))
    assert pred.plan_aware is False
    got = pred.paired_head_arm_dv(_window("pi"), TAU)
    assert np.allclose(got, 4.0 + _tau_index())


def test_plan_aware_flag_read_from_head_not_config_or_helper(monkeypatch):
    monkeypatch.setattr(P, "_plan_aware_head_helper",
                        lambda: pytest.fail("helper must not be probed"))
    head = _FakeHead(np.zeros((N_TAU, N_QUANTILES)))
    head.plan_aware = True
    assert _predictor(_bundle(head, plan_aware=False)).plan_aware is True


def test_plan_aware_detected_from_plan_proj_presence():
    head = _FakeHead(np.zeros((N_TAU, N_QUANTILES)))
    head.plan_proj = torch.nn.Linear(1, 1)
    assert _predictor(_bundle(head, plan_aware=False)).plan_aware is True


def test_plan_aware_head_reads_plan_in_controller_mode(monkeypatch):
    calls = {"summary": []}

    def fake_helper(bundle, *, t0_kin, t0_mask, t0_counter, summary_raw):
        calls["summary"].append(summary_raw)
        return np.stack([np.full(N_QUANTILES, 7.0) for _ in range(N_TAU)])

    monkeypatch.setattr(P, "_plan_aware_head_helper", lambda: fake_helper)
    monkeypatch.setattr(P, "training_summary_raw",
                        lambda source, item, run, *, n_steps, is_cav: "plan")
    monkeypatch.setattr(P, "_t0_state_channels", _fake_t0({"t0": []}))
    arr = np.stack([np.full(N_QUANTILES, 0.0) for _ in range(N_TAU)])
    pred = ModelAPredictor(_bundle(_FakeHead(arr), plan_aware=True),
                           source=SimpleNamespace(), cav_motion="controller")
    assert np.allclose(pred.paired_head_arm_dv(_window(), TAU), 7.0)
    assert calls["summary"] == ["plan"]


def test_paired_head_dv_method_uses_treatment_arm_plan(monkeypatch):
    calls = {"summary": []}

    def fake_helper(bundle, *, t0_kin, t0_mask, t0_counter, summary_raw):
        calls["summary"].append(summary_raw)
        return np.stack([np.full(N_QUANTILES, float(summary_raw))
                         for _ in range(N_TAU)])

    monkeypatch.setattr(P, "_plan_aware_head_helper", lambda: fake_helper)
    monkeypatch.setattr(P, "training_summary_raw",
                        lambda source, item, run, *, n_steps, is_cav: item)
    monkeypatch.setattr(P, "_t0_state_channels", _fake_t0({"t0": []}))
    pred = _predictor(_bundle(_FakeHead(np.zeros((N_TAU, N_QUANTILES))),
                              plan_aware=True))
    pred._pos_by_world_arm[("w1", 0.0, "pi")] = 3
    pred._cache = SimpleNamespace(world=lambda position: _window(position))
    dv = P._PairedHeadDV(pred).dv_quantiles([
        dict(world="w1", t0_s=0.0, tau_s=TAU, contrast="pi-native")])
    assert dv.shape == (1, N_QUANTILES)
    assert np.allclose(dv[0], 3.0)
    assert calls["summary"] == [3]


# -------------------------------------------------- decision-quality path --
def test_paired_head_true_dv_gives_zero_regret():
    rows = [
        _row("w1", "native", 130.0, position=0),
        _row("w1", "pi", 100.0, position=1),
        _row("w2", "native", 90.0, position=2),
        _row("w2", "pi", 95.0, position=3),
    ]
    dV = {("w1", "native"): 0.0, ("w1", "pi"): -30.0,
          ("w2", "native"): 0.0, ("w2", "pi"): 5.0}
    q = np.stack([_q(dV[(r.world, r.policy)]) for r in rows])
    points, _ = DQ.decision_points_from_rows(rows, q, cost="delay_s")
    doc = DQ.evaluate(points, fixed_best=DQ.build_fixed_best(points),
                      n_boot=20, seed=1, source="paired_head")
    cell = doc["cells"]["delay_s"]["60"]
    assert doc["prediction_source"] == "paired_head"
    assert cell["wm_mean_regret"] == pytest.approx(0.0)
    assert cell["wm_risk_regret"] == pytest.approx(0.0)
    assert cell["wm_picks_oracle_fraction"] == pytest.approx(1.0)
    assert cell["native_regret"] == pytest.approx(15.0)   # (130-100 + 95-90)/2


# ------------------------------------------------------------- CLI wiring ---
class _FakeA:
    name = "A:teacher"
    summary_source = "training_packed_plan"

    def __init__(self, paired=False):
        self.paired_head_dv = object() if paired else None

    def predict_quantiles(self, rows):
        return np.stack([_q(r.y_delay_s) for r in rows])

    def predict_paired_head_row_quantiles(self, rows):
        native = {r.world: r.y_delay_s for r in rows if r.policy == "native"}
        return np.stack([_q(0.0 if r.policy == "native"
                            else r.y_delay_s - native[r.world]) for r in rows])


class _FakeSource:
    def __init__(self, spec):
        self.spec = spec
        self.half = "cal_tune"
        self.selected = [object()]


class _FakeBound:
    def __init__(self, sha):
        self.bundle = object()
        self.sha256 = sha


def _dq_rows():
    return [_row("w1", "native", 130.0, position=0),
            _row("w1", "pi", 100.0, position=1)]


def _member(tmp_path):
    p = tmp_path / "member_1.pt"
    p.write_bytes(b"dq-head-test-member")
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _fixed_best(tmp_path):
    rows = _dq_rows()
    pts = []
    for cost in DQ.COSTS:
        cost_points, _ = DQ.decision_points_from_rows(rows, None, cost=cost)
        pts.extend(cost_points)
    doc = DQ.build_fixed_best(pts, train_source={"path": "synthetic"})
    path = tmp_path / "fb.json"
    return path, DQ.write_fixed_best(doc, path)


def _args(tmp_path, member, out, **extra):
    base = dict(command="score", release_spec=str(tmp_path / "release.json"),
                release_sha="a" * 64, checkpoint=str(member), out=str(out),
                cav_motion="teacher", comparators="zero")
    base.update(extra)
    return SimpleNamespace(**base)


def _run(member, sha, args, fake_a):
    return run_score(
        args,
        source_factory=lambda spec_path, expected_release_sha256: _FakeSource(
            {"checkpoint_member_sha256": sha}),
        checkpoint_loader=lambda path, **kw: _FakeBound(sha),
        rows_loader=lambda source: (_dq_rows(), {"n_items": 1}),
        a_factory=lambda bundle: fake_a,
        git_head="b" * 40)


def test_parser_accepts_dq_source():
    from evalh.scorer import cli

    args = cli.build_parser().parse_args(
        ["score", "--release-spec", "r.json", "--release-sha", "a" * 64,
         "--checkpoint", "m.pt", "--out", "out", "--comparators", "zero",
         "--cav-motion", "teacher", "--dq-source", "paired_head"])
    assert args.dq_source == "paired_head"
    args2 = cli.build_parser().parse_args(
        ["score", "--release-spec", "r.json", "--release-sha", "a" * 64,
         "--checkpoint", "m.pt", "--out", "out", "--comparators", "zero",
         "--cav-motion", "teacher"])
    assert args2.dq_source == "marginals"


def test_cli_paired_head_source_end_to_end(tmp_path):
    member, sha = _member(tmp_path)
    fb_path, fb_sha = _fixed_best(tmp_path)
    out = tmp_path / "out"
    args = _args(tmp_path, member, out, decision_quality=True,
                 dq_fixed_best=str(fb_path), dq_fixed_best_sha256=fb_sha,
                 dq_n_boot=20, dq_source="paired_head")
    _run(member, sha, args, _FakeA(paired=True))
    results = __import__("json").loads(
        (out / "RESULTS.json").read_text("utf-8"))
    assert results["inputs"]["decision_quality"]["dq_source"] == "paired_head"
    assert results["decision_quality"]["prediction_source"] == "paired_head"
    assert results["decision_quality"]["cells"]["delay_s"]["60"][
        "wm_mean_regret"] == pytest.approx(0.0)


def test_cli_paired_head_without_head_refused(tmp_path):
    member, sha = _member(tmp_path)
    fb_path, fb_sha = _fixed_best(tmp_path)
    out = tmp_path / "out"
    args = _args(tmp_path, member, out, decision_quality=True,
                 dq_fixed_best=str(fb_path), dq_fixed_best_sha256=fb_sha,
                 dq_source="paired_head")
    with pytest.raises(ScorerCliRefusal, match="paired_head"):
        _run(member, sha, args, _FakeA(paired=False))


def test_cli_default_dq_source_marginals_unchanged(tmp_path):
    member, sha = _member(tmp_path)
    fb_path, fb_sha = _fixed_best(tmp_path)
    out = tmp_path / "out"
    args = _args(tmp_path, member, out, decision_quality=True,
                 dq_fixed_best=str(fb_path), dq_fixed_best_sha256=fb_sha,
                 dq_n_boot=20)
    _run(member, sha, args, _FakeA(paired=False))
    results = __import__("json").loads(
        (out / "RESULTS.json").read_text("utf-8"))
    assert results["inputs"]["decision_quality"]["dq_source"] == "marginals"
    assert results["decision_quality"]["prediction_source"] == "marginals"
