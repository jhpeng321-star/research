"""Focused tests for the limited-domain P30 observation/value screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import tools.screen_idm_limited_domain_p30 as screen
from idm.paired_option_dataset import PairedOptionRecord


def _record(seed: int, penetration: float, demand: int, reward: float, utility: float, passes: bool, **overrides) -> PairedOptionRecord:
    payload = {
        "sample_id": f"s{seed}.p{penetration}.m{demand}",
        "seed": seed,
        "penetration": penetration,
        "demand": demand,
        "decision_index": 3,
        "decision_step": 100 + seed % 7,
        "vehicle_id": f"mainline.{seed}",
        "source": "runs/eval/fake",
        "observation": [0.0] * 134,
        "physical_utility": utility,
        "activate_reward": reward,
        "constraints_pass": passes,
        "dose_steps": 20,
        "classification": "neutral",
        "schema": "idm_paired_option_record/1",
    }
    return PairedOptionRecord.from_dict(payload | overrides)


def _dataset(tmp_path: Path, records: list[dict]) -> Path:
    records_bytes = "\n".join(json.dumps(row) for row in records).encode("utf-8")
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "records.jsonl").write_bytes(records_bytes)
    manifest = {
        "dataset_sha256": hashlib.sha256(b"x").hexdigest(),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "train_seeds": [4121, 4127],
        "validation_seeds": [4129],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _payload(seed: int, penetration: float, demand: int, reward: float, utility: float, passes: bool) -> dict:
    return {
        "sample_id": f"s{seed}.p{penetration}.m{demand}",
        "seed": seed,
        "penetration": penetration,
        "demand": demand,
        "decision_index": 3,
        "decision_step": 100,
        "vehicle_id": f"mainline.{seed}",
        "source": "runs/eval/fake",
        "observation": [0.0] * 134,
        "physical_utility": utility,
        "activate_reward": reward,
        "constraints_pass": passes,
        "dose_steps": 20,
        "classification": "neutral",
        "schema": "idm_paired_option_record/1",
        "split": "train",
    }


def test_analyze_restricts_to_five_and_ten_percent_and_keeps_penetration_out_of_features(tmp_path, monkeypatch):
    rows = []
    for seed, pen, demand in ((4121, 0.05, 1900), (4127, 0.05, 2000), (4129, 0.05, 2100), (4129, 0.10, 2000)):
        rows.append(_payload(seed, pen, demand, reward=0.1, utility=0.1, passes=True))
    rows.append(_payload(4121, 0.15, 1900, reward=0.0, utility=0.0, passes=True))  # outside domain
    dataset = _dataset(tmp_path, rows)
    monkeypatch.setattr(
        screen, "_context", lambda record: np.zeros(13, dtype=float), raising=True
    )
    result = screen.analyze(dataset)

    assert result["domain_record_counts"]["train_records"] == 2
    assert result["domain_record_counts"]["validation_records"] == 2
    assert result["domain_record_counts"]["penetration_counts"] == {"0.05": 3, "0.10": 1}
    assert result["rl_authorized"] is False
    value = result["value_screen"]["option_context_12"]
    assert value["validation"]["records"] == 2
    assert result["authorized_value_families"] == []
    assert "penetration (admission rule outside the actor)" in result["forbidden_actor_inputs"]


def test_option_context_without_penetration_drops_only_first_column(monkeypatch):
    monkeypatch.setattr(
        screen, "_context", lambda record: np.arange(13, dtype=float), raising=True
    )
    record = _record(4121, 0.05, 1900, 0.0, 0.0, True)
    context = screen.option_context_without_penetration(record)
    assert context.shape == (12,)
    assert context[0] == pytest.approx(1.0)  # original penetration column dropped


def test_classification_metrics_are_sane_on_small_samples():
    labels = np.asarray([1, 1, 1, 0], dtype=float)
    scores = np.asarray([3.0, 2.0, 1.0, 0.0], dtype=float)
    metrics = screen.classification_metrics([], scores, labels)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["positives"] == 3
    assert metrics["negatives"] == 1
