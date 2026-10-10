"""Real-member timing wrapper tests (tiny CPU; <= 1 min, no SUMO/training)."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from evalh import timing_gate as T

from bundle import timing as B
from bundle.e2e import write_synthetic_pack


def _real_pack(tmp_path):
    """A real-format TRAIN pack (k=192, sibling pair) for the gate path."""
    pack = tmp_path / "pack"
    write_synthetic_pack(pack, k=192, seed=0)
    return pack


def test_tiny_real_gate_writes_json(tmp_path):
    cfg = B.make_config(tiny=True)
    gate = B.run_timing_gate(cfg, device="cpu", tiny=True, warmup=1,
                             timed_steps=2, loss_steps=25)
    assert gate["schema"] == "wm_n1_timing_gate/1"
    assert gate["measured_model"] == B.MEASURED_MODEL
    assert "IdeaATrainer" in gate["measured_model"]
    assert gate["proxy"] is False
    assert gate["parameter_count"] > 0
    assert gate["throughput"]["steps_per_sec"] > 0
    assert gate["throughput"]["rollout_samples_per_sec"] > 0
    assert gate["loss_decrease"]["decreased"] is True
    # the gate measures the REAL Idea A corpus path, all registered terms live
    assert set(gate["active_loss_terms"]) == {
        "residual_nll", "rollout", "arrival", "correction_crps",
        "response_contrast"}
    # amendment 5: the registered Idea A gate runs on pack_v3 (K_max=192)
    assert gate["format"] == "v3"
    assert gate["k_max"] == 192
    assert gate["idea_a"]["throughput"]["k_max"] == 192
    # the explicit v2 fallback resolves the frozen K_max
    assert B.DEFAULT_GATE_FORMAT == "v3" and B.DEFAULT_GATE_K_MAX == 192
    assert B.n_vehicles_for_format("v2") == 128
    assert gate["v1_member"]["measured_model"] == B.V1_MEASURED_MODEL
    assert gate["checks"]["within_gate"] is True
    assert gate["thresholds"]["budget_hours_max"] == T.BUDGET_HOURS_MAX
    assert gate["thresholds"]["gate_limit_hours"] == T.GATE_LIMIT_HOURS
    assert gate["thresholds"]["rollout_fraction"] == T.ROLLOUT_FRACTION
    assert gate["thresholds"]["eval_hours"] == T.EVAL_HOURS
    assert "WM_N1_DESIGN.md:226" in gate["thresholds"]["citation"]

    out = tmp_path / "TIMING_GATE.json"
    sha = B.write_timing_gate(out, gate)
    raw = out.read_bytes()
    assert sha == hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    assert data["measured_model"] == B.MEASURED_MODEL
    assert data["architecture"]["d_model"] == cfg.d_model


def test_uses_real_model_not_evalh_toy():
    cfg = B.make_config(tiny=True)
    gate = B.run_timing_gate(cfg, device="cpu", tiny=True, warmup=1,
                             timed_steps=1, loss_steps=5)
    # the real net's config is recorded, and the default member is 2-5 M params
    assert gate["architecture"]["d_model"] == 16
    assert gate["architecture"]["counter_channels"] == 8
    big = B.make_config()
    assert big.wmn1_config().d_model == 256
    assert big.wmn1_config().n_layers == 4


def test_gate_pass_fail_shape():
    cfg = B.make_config(tiny=True)
    gate = B.run_timing_gate(cfg, device="cpu", tiny=True, warmup=1,
                             timed_steps=1, loss_steps=5,
                             budget_hours_max=0.0)
    assert gate["checks"]["within_budget"] is False
    assert gate["passed"] is False


def test_real_pack_path_reports_steps_terms_and_kmax(tmp_path):
    """The registered gate measures the REAL pack through IdeaACorpus + the
    pair-aware sampler, recording s/step, all five active terms and K_max."""
    pack = _real_pack(tmp_path)
    cfg = B.make_config(tiny=True)
    gate = B.run_timing_gate(cfg, device="cpu", tiny=True, pack_dir=str(pack),
                             warmup=1, timed_steps=2, loss_steps=5)
    # owner 2.3: pack_source is the EXACT supplied pack path; the kind is kept
    # separately (the old literal "real-pack" lost the actual location).
    assert gate["pack_source"] == str(pack)
    assert gate["source_kind"] == "real-pack"
    assert gate["pack_dir"] == str(pack)
    assert gate["k_max"] == 192
    assert gate["idea_a"]["throughput"]["k_max"] == 192
    assert gate["idea_a"]["throughput"]["s_per_step"] > 0
    assert set(gate["active_loss_terms"]) == {
        "residual_nll", "rollout", "arrival", "correction_crps",
        "response_contrast"}
    assert gate["loss_decrease"]["decreased"] is True
    assert "peak_memory_mb" in gate


def test_real_pack_path_accepts_a_cuda_request(tmp_path):
    """A CUDA device is a valid gate request: without a GPU the gate must fail
    with the standard CUDA-unavailable error, never the old CPU-only refusal."""
    pack = _real_pack(tmp_path)
    cfg = B.make_config(tiny=True)
    if torch.cuda.is_available():
        pytest.skip("CUDA present; the real GPU box covers the cuda run")
    with pytest.raises(RuntimeError, match="cuda requested but not available"):
        B.run_timing_gate(cfg, device="cuda", tiny=True, pack_dir=str(pack),
                          warmup=1, timed_steps=1, loss_steps=2)
