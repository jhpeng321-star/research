"""Generated metadata-only checks for the real-evaluation refusal boundary."""
from __future__ import annotations

import json

import pytest

from evalh import real_cli as R

_SHA = "a" * 64


def _argv(tmp_path, mode="--dry-run", **overrides):
    values = dict(release_spec=str(tmp_path / "release.json"),
                  release_sha256=_SHA, allocation_id="synthetic-allocation",
                  data_release_id="synthetic-release", cal_half="cal_tune")
    values.update(overrides)
    out = [mode]
    for name, value in values.items():
        out.extend(("--" + name.replace("_", "-"), str(value)))
    return out


class _MetadataSource:
    spec = {"allocation_id": "synthetic-allocation",
            "data_release_id": "synthetic-release", "half": "cal_tune"}
    half = "cal_tune"
    selected = ((0, {"world": "generated"}, {"key": "generated"}),)

    def outcome_rows(self, *_args, **_kwargs):
        raise AssertionError("dry-run opened outcome rows")

    def rollout_rows(self, *_args, **_kwargs):
        raise AssertionError("dry-run opened rollout arrays")


def test_real_cli_dry_run_only_checks_release_metadata(tmp_path, capsys):
    calls = []

    def source_factory(path, *, expected_release_sha256):
        calls.append((path, expected_release_sha256))
        return _MetadataSource()

    assert R.main(_argv(tmp_path), source_factory=source_factory) == 0
    result = json.loads(capsys.readouterr().out)
    assert calls == [(str(tmp_path / "release.json"), _SHA)]
    assert result["status"] == "metadata_valid_scoring_disabled"
    assert result["selected_run_count"] == 1
    assert result["outcome_rows_read"] is False
    assert result["prediction_or_scoring_performed"] is False
    assert result["use_marker_consumed"] is False
    assert list(tmp_path.iterdir()) == []


def test_real_cli_score_refuses_before_release_or_predictor(tmp_path, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("score path opened a release")

    assert R.main(_argv(tmp_path, "--score"), source_factory=forbidden) == 87
    error = capsys.readouterr().err
    assert "real scoring disabled" in error
    assert "summary parity" in error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("override", [
    {"allocation_id": "other"},
    {"data_release_id": "other"},
    {"cal_half": "cal_cal"},
])
def test_real_cli_refuses_identity_or_half_drift(tmp_path, override):
    args = R.build_parser().parse_args(_argv(tmp_path, **override))
    with pytest.raises(R.RealEvaluationRefusal,
                       match="identity differs from pinned release"):
        R.run(args, source_factory=lambda *_a, **_k: _MetadataSource())


def test_real_cli_bad_external_pin_refuses_before_source(tmp_path):
    args = R.build_parser().parse_args(_argv(tmp_path, release_sha256="bad"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unreviewed release was opened")

    with pytest.raises(R.RealEvaluationRefusal, match="64 lowercase hex"):
        R.run(args, source_factory=forbidden)


def test_real_cli_rejects_legacy_dynamic_source_options(tmp_path):
    with pytest.raises(SystemExit) as error:
        R.build_parser().parse_args(_argv(tmp_path) + ["--predictor", "real:ckpt"])
    assert error.value.code == 2
