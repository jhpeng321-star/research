"""E4 scorer CLI tests: injectable factories, generated fixtures only.

No CAL data, no real checkpoint, no deserialization: the member file is
arbitrary bytes and every reader (source, checkpoint loader, predictors,
rows loader, git head) is a test fake.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evalh.protocol import N_QUANTILES, OutcomeRow
from evalh.scorer import cli
from evalh.scorer.cli import ScorerCliRefusal, run_score

_RELEASE_SHA = "a" * 64
_GIT_HEAD = "b" * 40
_MEMBER_BYTES = b"scorer-cli-test-member-bytes"


def _rows():
    rows = []
    for tau in (60.0, 120.0):
        rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                               tau_s=tau, policy="native", y_delay_s=2.0))
        rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                               tau_s=tau, policy="damping", y_delay_s=7.0))
    return rows


def _const(base):
    return np.full(N_QUANTILES, float(base))


class FakeSource:
    def __init__(self, spec, half="cal_tune"):
        self.spec = spec
        self.half = half
        self.selected = [object() for _ in range(3)]
        self.outcome_reads = 0

    def outcome_rows(self):
        self.outcome_reads += 1
        return _rows()


class FakePredictor:
    def __init__(self, name, offset, params_sha=None):
        self.name = name
        self.offset = float(offset)
        if params_sha:
            self.params_lf_sha256 = params_sha

    def predict_quantiles(self, rows):
        q = [_const(2.0 + self.offset) if r.policy == "native"
             else _const(7.0 + self.offset) for r in rows]
        return np.stack(q)


class FakeBound:
    def __init__(self, sha256):
        self.bundle = object()
        self.sha256 = sha256
        self.member = 1


def _member_file(tmp_path):
    path = tmp_path / "member_1.pt"
    path.write_bytes(_MEMBER_BYTES)
    return path, hashlib.sha256(_MEMBER_BYTES).hexdigest()


def _args(tmp_path, member, out=None, params_sha=None):
    return SimpleNamespace(
        command="score", release_spec=str(tmp_path / "release.json"),
        release_sha=_RELEASE_SHA, checkpoint=str(member),
        out=str(tmp_path / "out") if out is None else str(out),
        params_sha256=params_sha)


def _run(tmp_path, member, sha, tmp_out, **overrides):
    spec = {"checkpoint_member_sha256": sha}
    kwargs = dict(
        source_factory=lambda spec_path, expected_release_sha256: FakeSource(
            spec),
        checkpoint_loader=lambda path, **kw: FakeBound(sha),
        rows_loader=lambda source: (_rows(), {"n_items": 2}),
        a_factory=lambda bundle: FakePredictor("A", 0.0),
        v2g_factory=lambda source: FakePredictor("v2g", 2.0,
                                                 params_sha="c" * 64),
        git_head=_GIT_HEAD)
    kwargs.update(overrides)
    return run_score(_args(tmp_path, member, out=tmp_out), **kwargs)


def test_end_to_end_writes_results_rows_and_done(tmp_path):
    member, sha = _member_file(tmp_path)
    out = tmp_path / "out"
    done = _run(tmp_path, member, sha, out)
    assert (out / "RESULTS.json").is_file()
    assert (out / "rows.csv").is_file()
    assert (out / "DONE.json").is_file()
    results = json.loads((out / "RESULTS.json").read_text("utf-8"))
    assert results["schema"] == "wm_n1_scorer_results/1"
    assert results["cal_half"] == "cal_tune"
    assert results["inputs"]["code_commit"] == _GIT_HEAD
    assert results["inputs"]["release_spec_sha256"] == _RELEASE_SHA
    assert results["inputs"]["checkpoint_sha256"] == sha
    assert results["inputs"]["params_sha256"] == "c" * 64
    assert results["dynamics_rollout_label"] == "extrapolation"
    assert results["census"] == {"n_items": 2}
    assert "damping-native" in results["metrics"]["cells"]
    cell = results["metrics"]["cells"]["damping-native"]["60"]
    assert cell["n_pairs"] == 1 and cell["n_worlds"] == 1
    lines = (out / "rows.csv").read_text("utf-8").strip().splitlines()
    assert len(lines) == len(_rows()) + 1
    assert lines[0].count(",") == lines[1].count(",")
    assert done["inputs"] == results["inputs"]
    assert done["n_rows"] == len(_rows())


def test_refuses_existing_out_dir(tmp_path):
    member, sha = _member_file(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ScorerCliRefusal, match="already exists"):
        _run(tmp_path, member, sha, out)


def test_refuses_non_tune_half(tmp_path):
    member, sha = _member_file(tmp_path)
    spec = {"checkpoint_member_sha256": sha}
    with pytest.raises(ScorerCliRefusal, match="cal_tune"):
        _run(tmp_path, member, sha, tmp_path / "out",
             source_factory=lambda spec_path,
             expected_release_sha256: FakeSource(spec, half="cal_cal"))


def test_refuses_checkpoint_pin_mismatch(tmp_path):
    member, sha = _member_file(tmp_path)
    spec = {"checkpoint_member_sha256": "d" * 64}
    with pytest.raises(ScorerCliRefusal, match="release-spec pin"):
        _run(tmp_path, member, sha, tmp_path / "out",
             source_factory=lambda spec_path,
             expected_release_sha256: FakeSource(spec))


def test_refuses_missing_params_sha(tmp_path):
    member, sha = _member_file(tmp_path)
    spec = {"checkpoint_member_sha256": sha}
    with pytest.raises(ScorerCliRefusal, match="params"):
        run_score(
            _args(tmp_path, member, tmp_path / "out", params_sha=None),
            source_factory=lambda spec_path,
            expected_release_sha256: FakeSource(spec),
            checkpoint_loader=lambda path, **kw: FakeBound(sha),
            rows_loader=lambda source: (_rows(), {}),
            a_factory=lambda bundle: FakePredictor("A", 0.0),
            v2g_factory=lambda source: FakePredictor("v2g", 2.0),
            git_head=_GIT_HEAD)


def test_dry_run_validates_metadata_without_payload(tmp_path, capsys):
    member, sha = _member_file(tmp_path)
    spec = {"checkpoint_member_sha256": sha}
    source = FakeSource(spec)

    def factory(spec_path, expected_release_sha256):
        return source

    rc = cli.main(["dry-run", "--release-spec",
                   str(tmp_path / "release.json"),
                   "--release-sha", _RELEASE_SHA,
                   "--checkpoint", str(member)],
                  source_factory=factory)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["schema"] == "wm_n1_scorer_dry_run/1"
    assert payload["planned_reads"]["checkpoint_sha256"] == sha
    assert payload["planned_reads"]["outcomes_payload_read"] is False
    assert payload["planned_reads"]["selected_run_count"] == 3
    assert source.outcome_reads == 0


def test_main_score_refusal_exit_code(tmp_path, capsys):
    member, sha = _member_file(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    spec = {"checkpoint_member_sha256": sha}
    rc = cli.main(
        ["score", "--release-spec", str(tmp_path / "release.json"),
         "--release-sha", _RELEASE_SHA, "--checkpoint", str(member),
         "--out", str(out)],
        source_factory=lambda spec_path,
        expected_release_sha256: FakeSource(spec))
    assert rc == 87
    assert "refused" in capsys.readouterr().err
