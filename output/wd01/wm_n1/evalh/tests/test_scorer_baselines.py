"""E9 scorer baseline tests: honest TRAIN climatology + zero effect.

Everything here runs on GENERATED synthetic sidecars under ``tmp_path``; no CAL
data, no real pack, no real checkpoint bytes are ever opened.  The tests cover
the registered exclusions, the SHA pin, missing-cell refusal, the zero-effect
CRPS identity and the full CLI comparator path.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evalh.metrics import crps_of_quantiles
from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS, OutcomeRow
from evalh.scorer import baselines as BL
from evalh.scorer import build_climatology as BC
from evalh.scorer import cli
from evalh.scorer.cli import ScorerCliRefusal, run_score

_ARM_KIND = {
    "native": "native_fleet", "native_base": "native_fleet",
    "base": "damping_fleet", "pi": "pi_fleet",
    "swnative": "switching", "fixed1": "fixed",
}
_DELAY = {"native": 100.0, "pi": 130.0, "base": 200.0,
          "native_base": 90.0, "swnative": 110.0, "fixed1": 150.0}
_COLUMNS = list(BC._COLUMNS)


def _rows(worlds=("w1", "w2"), taus=(60.0, 120.0),
          sources=("native", "pi", "base", "native_base", "swnative",
                   "fixed1"), split="TRAIN"):
    out = []
    for w in worlds:
        for src in sources:
            for tau in taus:
                out.append(dict(
                    world=w, source=src, arm_kind=_ARM_KIND[src],
                    policy=_ARM_KIND[src], split=split, t0_s=0.0, t=0.0,
                    tau=float(tau), tau_available=1.0,
                    total_delay_ref_s=_DELAY[src]))
    return out


def _sidecar(tmp_path, rows, name="sidecar"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_COLUMNS).to_parquet(
        d / BC.TRAIN_OUTCOMES_FILENAME, index=False)
    return d


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------- build tests --
def test_climatology_pairing_and_exclusions(tmp_path):
    d = _sidecar(tmp_path, _rows())
    doc = BC.compute_climatology(d, expected_sha256=None)
    assert doc["schema"] == BC.SCHEMA
    # base (damping treatment) and native_base are excluded, so damping-native
    # has no cell; swnative/fixed1 are unknown; only pi-native survives.
    assert set(doc["cells"]) == {"pi-native"}
    assert doc["contrasts"] == ["pi-native"]
    for tau in ("60", "120"):
        cell = doc["cells"]["pi-native"][tau]
        assert cell["n_pairs"] == 2 and cell["n_worlds"] == 2
        assert cell["quantiles"] == pytest.approx([30.0] * N_QUANTILES)
    used = doc["items_by_contrast"]["pi-native"]
    assert {it["source"] for it in used} == {"pi", "native"}
    roles = {it["source"]: it["role"] for it in used}
    assert roles == {"pi": "treatment", "native": "reference"}
    assert doc["counts"]["n_excluded_arm"] == 8      # base+native_base, 2x2x2
    assert doc["counts"]["n_excluded_unknown_policy"] == 8  # swnative+fixed1


def test_native_base_and_base_labels_absent(tmp_path):
    d = _sidecar(tmp_path, _rows())
    doc = BC.compute_climatology(d, expected_sha256=None)
    used = {it["source"] for it in doc["items_by_contrast"]["pi-native"]}
    assert "base" not in used and "native_base" not in used
    assert "damping-native" not in doc["cells"]
    assert "wall-native" not in doc["cells"]


def test_swnative_is_not_a_reference(tmp_path):
    assert BC._is_reference("native", "native") is True
    assert BC._is_reference("swnative", "switching") is False
    d = _sidecar(tmp_path, _rows(sources=("pi", "swnative")))
    doc = BC.compute_climatology(d, expected_sha256=None)
    assert doc["cells"] == {}                        # no native reference


def test_non_train_split_refused(tmp_path):
    rows = _rows(worlds=("w1",), taus=(60.0,), sources=("pi",))
    rows[0]["split"] = "CALIBRATION"
    d = _sidecar(tmp_path, rows)
    with pytest.raises(BC.ClimatologyBuildRefusal, match="non-TRAIN"):
        BC.compute_climatology(d, expected_sha256=None)


def test_sha_pin_mismatch_refused(tmp_path):
    d = _sidecar(tmp_path, _rows())
    good = _sha(d / BC.TRAIN_OUTCOMES_FILENAME)
    BC.compute_climatology(d, expected_sha256=good)          # matching pin ok
    with pytest.raises(BC.ClimatologyBuildRefusal, match="sha256"):
        BC.compute_climatology(d, expected_sha256="0" * 64)


def test_out_exists_refused(tmp_path):
    d = _sidecar(tmp_path, _rows())
    out = tmp_path / "clim.json"
    out.write_text("existing", encoding="utf-8")
    with pytest.raises(BC.ClimatologyBuildRefusal, match="already exists"):
        BC.run_build(d, out, expected_sha256=None)


# ------------------------------------------------------------ baseline tests --
def test_zero_baseline_crps_equals_abs_dy():
    zb = BL.ZeroEffectBaseline()
    assert zb.sign_defined is False
    dv = zb.dv_quantiles([{"contrast": "pi-native", "tau_s": 60.0}] * 3)
    assert dv.shape == (3, N_QUANTILES) and not dv.any()
    q = np.zeros(N_QUANTILES)
    for dy in (0.0, 5.0, -7.5, 30.0):
        assert crps_of_quantiles(q, dy, QUANTILE_LEVELS) == pytest.approx(
            abs(dy), abs=1e-12)


def test_zero_baseline_in_metrics_loop(tmp_path):
    from evalh.scorer.metrics_loop import score

    rows = _cli_rows()
    q_a = _fake_a_quantiles(rows)
    out = score(rows, {"A": q_a}, dv_models={"zero": BL.ZeroEffectBaseline()},
                n_boot=50, seed=20261001)
    assert out["models"] == ["A", "zero"]
    assert out["comparators"] == ["zero"]
    cell = out["cells"]["pi-native"]["60"]
    assert cell["crps_zero"] == pytest.approx(30.0)      # |dy|, dy=30
    assert cell["sign_accuracy_zero_defined"] is False
    assert np.isnan(cell["sign_accuracy_zero"])
    assert cell["diff_zero_mean"] == pytest.approx(cell["crps_A"] - 30.0)


def test_climatology_sha_mismatch_refused(tmp_path):
    d = _sidecar(tmp_path, _rows())
    out = tmp_path / "clim.json"
    BC.run_build(d, out, expected_sha256=None)
    BL.TrainClimatologyBaseline(out, expected_sha256=_sha(out))  # ok
    with pytest.raises(BL.BaselineRefusal, match="sha256"):
        BL.TrainClimatologyBaseline(out, expected_sha256="0" * 64)


def test_climatology_missing_cell_refused(tmp_path):
    d = _sidecar(tmp_path, _rows())
    out = tmp_path / "clim.json"
    BC.run_build(d, out, expected_sha256=None)
    base = BL.TrainClimatologyBaseline(out, expected_sha256=_sha(out))
    with pytest.raises(BL.BaselineRefusal, match="no cell"):
        base.dv_quantiles([{"contrast": "damping-native", "tau_s": 60.0}])


def test_make_comparator_refuses_v2g_and_unknown():
    with pytest.raises(BL.BaselineRefusal, match="unknown comparator"):
        BL.make_comparator("v2g")
    with pytest.raises(BL.BaselineRefusal, match="unknown comparator"):
        BL.make_comparator("oracle")


# ----------------------------------------------------------------- CLI tests --
def _cli_rows():
    rows = []
    for tau in (60.0, 120.0):
        rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                               tau_s=tau, policy="native", y_delay_s=100.0))
        rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                               tau_s=tau, policy="pi", y_delay_s=130.0))
    return rows


def _fake_a_quantiles(rows):
    return np.stack([np.full(N_QUANTILES, r.y_delay_s) for r in rows])


class _FakeSource:
    def __init__(self, spec):
        self.spec = spec
        self.half = "cal_tune"
        self.selected = [object()]


class _FakeBound:
    def __init__(self, sha):
        self.bundle = object()
        self.sha256 = sha


class _FakeA:
    name = "A:teacher"
    summary_source = "training_packed_plan"

    def predict_quantiles(self, rows):
        return _fake_a_quantiles(list(rows))


def _member(tmp_path):
    p = tmp_path / "member_1.pt"
    p.write_bytes(b"baseline-test-member")
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _args(tmp_path, member, out, **extra):
    base = dict(
        command="score", release_spec=str(tmp_path / "release.json"),
        release_sha="a" * 64, checkpoint=str(member), out=str(out),
        cav_motion="teacher")
    base.update(extra)
    return SimpleNamespace(**base)


def _run(tmp_path, member, sha, out, args):
    return run_score(
        args,
        source_factory=lambda spec_path, expected_release_sha256: _FakeSource(
            {"checkpoint_member_sha256": sha}),
        checkpoint_loader=lambda path, **kw: _FakeBound(sha),
        rows_loader=lambda source: (_cli_rows(), {"n_items": 1}),
        a_factory=lambda bundle: _FakeA(),
        git_head="b" * 40)


def test_cli_refuses_v2g_comparator(tmp_path):
    member, sha = _member(tmp_path)
    args = _args(tmp_path, member, tmp_path / "out", comparators="v2g",
                 climatology=None, climatology_sha256=None)
    with pytest.raises(ScorerCliRefusal, match="v2g"):
        _run(tmp_path, member, sha, tmp_path / "out", args)


def test_cli_end_to_end_with_comparators(tmp_path):
    side = _sidecar(tmp_path, _rows())
    clim = tmp_path / "clim.json"
    BC.run_build(side, clim, expected_sha256=None)
    clim_sha = _sha(clim)
    member, sha = _member(tmp_path)
    out = tmp_path / "out"
    args = _args(tmp_path, member, out, comparators="zero,train_climatology",
                 climatology=str(clim), climatology_sha256=clim_sha)
    done = _run(tmp_path, member, sha, out, args)
    results = json.loads((out / "RESULTS.json").read_text("utf-8"))
    inputs = results["inputs"]
    assert inputs["comparators"] == ["zero", "train_climatology"]
    assert inputs["climatology_sha256"] == clim_sha
    assert inputs["cav_motion"] == "teacher"
    assert inputs["summary_source"] == "training_packed_plan"
    assert results["comparators"] == ["zero", "train_climatology"]
    assert done["inputs"] == inputs
    assert (out / "DONE.json").is_file() and (out / "rows.csv").is_file()
    cell = results["metrics"]["cells"]["pi-native"]["60"]
    assert "diff_zero_ci" in cell and "diff_train_climatology_ci" in cell
    # the TRAIN climatology here is the exact constant dV=30, so A (also 30)
    # matches it and the zero baseline is 30 off.
    assert cell["crps_train_climatology"] == pytest.approx(0.0)
    assert cell["crps_zero"] == pytest.approx(30.0)
    assert cell["diff_train_climatology_mean"] == pytest.approx(0.0)
    assert cell["diff_zero_mean"] == pytest.approx(-30.0)


def test_cli_help_lists_comparators(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["score", "--help"])
    assert "--comparators" in capsys.readouterr().out
