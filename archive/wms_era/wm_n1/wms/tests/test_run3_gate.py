"""Run-3 tests: multi-stage smoke gate, per-chunk loss logging, delay bin mean.

The gate must catch run 2's stage-60 traffic blow-up (fed here as the frozen
per-stage medians) and pass on a healthy per-chunk-mean log; the trainer must
log per-chunk means and per-term gradient-norm shares.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from output.wd01.wm_n1.wms.gate_stages import DEFAULT_TERMS, check, evaluate
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import (TrajectoryDataset, WmSTrainConfig,
                                         WmSTrainer, collate)
from output.wd01.wm_n1.wms.traffloss import (TrafficLossConfig, _aggregate_bins,
                                             section_metrics)
from hdv.config import HdvConfig

STAGES = (10.0, 30.0, 60.0, 120.0, 300.0)
#: Run-2 logged per-stage medians (train_log.jsonl) that the gate must reject.
LEGACY = {
    10.0: {"position": 11.97, "speed": 0.74, "lane": 0.14, "traffic": 0.0,
           "stability": 0.0, "conservation": 0.0},
    30.0: {"position": 64.15, "speed": 0.74, "lane": 0.14, "traffic": 0.0,
           "stability": 0.0, "conservation": 0.0},
    60.0: {"position": 197.7, "speed": 0.74, "lane": 0.14, "traffic": 7.77e6,
           "stability": 0.0, "conservation": 0.0},
    120.0: {"position": 394.9, "speed": 0.74, "lane": 0.14, "traffic": 6.58e7,
            "stability": 0.0, "conservation": 0.0},
    300.0: {"position": 448.4, "speed": 0.74, "lane": 0.14, "traffic": 3.33e8,
            "stability": 0.0, "conservation": 0.0},
}
HEALTHY = {
    10.0: {"position": 11.0, "speed": 0.7, "lane": 0.14, "traffic": 0.0,
           "stability": 0.01, "conservation": 0.02},
    30.0: {"position": 14.0, "speed": 0.8, "lane": 0.15, "traffic": 0.0,
           "stability": 0.01, "conservation": 0.02},
    60.0: {"position": 16.0, "speed": 0.9, "lane": 0.16, "traffic": 0.2,
           "stability": 0.01, "conservation": 0.02},
    120.0: {"position": 17.0, "speed": 1.0, "lane": 0.17, "traffic": 0.4,
            "stability": 0.01, "conservation": 0.02},
    300.0: {"position": 15.0, "speed": 1.1, "lane": 0.18, "traffic": 0.5,
            "stability": 0.01, "conservation": 0.02},
}


def _write(path, stages, overrides=None):
    overrides = overrides or {}
    rows = []
    step = 0
    for stage_s, terms in stages.items():
        merged = {**terms, **overrides.get(stage_s, {})}
        for _ in range(3):
            step += 1
            rows.append({"step": step, "stage_s": stage_s, "finite": True,
                         "gate_monotone": True, "gate_g0": 0.0,
                         "unroll_steps": int(round(stage_s / 0.5)), **merged})
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def test_gate_passes_on_healthy_multi_stage_log(tmp_path):
    log = _write(tmp_path / "train_log.jsonl", HEALTHY)
    result = evaluate(log)
    assert result["passed"] is True, result["failures"]
    assert set(result["per_stage"]) == set(STAGES)


def test_gate_catches_run2_legacy_traffic_blow_up(tmp_path):
    log = _write(tmp_path / "train_log.jsonl", LEGACY)
    result = evaluate(log)
    assert result["passed"] is False
    assert any("traffic" in f and "exceeds" in f for f in result["failures"]), \
        result["failures"]


def test_gate_check_raises_on_legacy_blow_up(tmp_path):
    log = _write(tmp_path / "train_log.jsonl", LEGACY)
    try:
        check(log)
    except SystemExit as exc:
        assert "GATE FAIL" in str(exc)
    else:
        raise AssertionError("check must fail closed on the legacy blow-up")


def test_gate_catches_nonfinite_term(tmp_path):
    log = _write(tmp_path / "train_log.jsonl", HEALTHY,
                 {60.0: {"traffic": float("nan")}})
    result = evaluate(log)
    assert result["passed"] is False
    assert any("nonfinite" in f for f in result["failures"])


def test_gate_catches_missing_stage(tmp_path):
    stages = {s: HEALTHY[s] for s in STAGES if s != 300.0}
    log = _write(tmp_path / "train_log.jsonl", stages)
    result = evaluate(log)
    assert result["passed"] is False
    assert any("absent" in f for f in result["failures"])


def test_gate_catches_missing_required_term(tmp_path):
    stages = {s: {k: v for k, v in HEALTHY[s].items() if k != "conservation"}
              for s in STAGES}
    log = _write(tmp_path / "train_log.jsonl", stages)
    result = evaluate(log)
    assert result["passed"] is False
    assert any("conservation" in f for f in result["failures"])


def test_gate_catches_post_ramp_growth(tmp_path):
    stages = {s: dict(HEALTHY[s]) for s in STAGES}
    stages[300.0]["speed"] = 30.0  # >10x the 120 s value (1.0), post-ramp
    log = _write(tmp_path / "train_log.jsonl", stages)
    result = evaluate(log)
    assert result["passed"] is False
    assert any("speed grew" in f for f in result["failures"]), result["failures"]


def test_gate_allows_expected_horizon_position_growth(tmp_path):
    # The bounded per-car Huber position loss grows with the unroll length; a
    # 14x stage-10 -> stage-30 jump (as seen in the CPU dry run) is horizon
    # growth, not a blow-up, because the allowance scales with that growth.
    stages = {s: dict(HEALTHY[s]) for s in STAGES}
    stages[10.0]["position"] = 8.4
    stages[30.0]["position"] = 118.3
    stages[60.0]["position"] = 101.5
    stages[120.0]["position"] = 205.3
    stages[300.0]["position"] = 258.5
    log = _write(tmp_path / "train_log.jsonl", stages)
    result = evaluate(log)
    assert result["passed"] is True, result["failures"]


def test_gate_exempts_intended_traffic_ramp_growth(tmp_path):
    stages = {s: dict(HEALTHY[s]) for s in STAGES}
    stages[120.0]["traffic"] = 20.0  # 100x stage 60 but still inside the ramp
    log = _write(tmp_path / "train_log.jsonl", stages)
    result = evaluate(log)
    assert result["passed"] is True, result["failures"]


def test_gate_reports_gradient_norm_shares_from_val_log(tmp_path):
    log = _write(tmp_path / "train_log.jsonl", HEALTHY)
    val = tmp_path / "val_log.jsonl"
    val.write_text(json.dumps({"stage_s": 300.0, "loss_share":
                               {"position": 0.6, "speed": 0.2, "lane": 0.05,
                                "traffic": 0.1, "stability": 0.03,
                                "conservation": 0.02}}) + "\n", encoding="utf-8")
    result = evaluate(log)
    assert result["passed"] is True
    assert abs(sum(result["loss_share"][300.0].values()) - 1.0) < 1e-6


def test_normalized_delay_is_averaged_per_bin_not_summed():
    cfg = TrafficLossConfig(section_temp_m=0.5)
    T, K = 60, 2
    s = torch.full((1, T, K), -100.0)      # upstream
    v = torch.zeros(1, T, K)               # fully stopped
    present = torch.ones(1, T, K, dtype=torch.bool)
    agg = _aggregate_bins(section_metrics(s, v, present, cfg=cfg),
                          torch.arange(T), cfg=cfg)
    # 2 stopped vehicles, free-speed-normalised, dt=0.5 -> 1.0 delay/step; the
    # bin value must be that per-step mean, not the 60-step sum (60.0).
    value = float(agg["upstream"]["normalized_delay"].mean())
    assert abs(value - 1.0) < 1e-3


def small_model(**kwargs):
    return WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                    hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
                    **kwargs)


def _states(n=1, t1=41, slots=4):
    lane = torch.zeros(n, slots, dtype=torch.long)
    lane[:, slots // 2:] = 1
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((torch.arange(slots).float().view(1, slots, 1).expand(n, -1, -1),
                     one_hot, torch.full((n, slots, 1), 12.0),
                     torch.zeros(n, slots, 2)), dim=-1)
    states = np.repeat(kin.numpy()[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    present = np.ones((n, t1, slots), dtype=bool)
    return TrajectoryDataset.from_arrays({"states": states, "present": present})


def test_logged_terms_are_horizon_invariant_to_chunk_count():
    dataset = _states()
    model = small_model()
    cfg20 = WmSTrainConfig(device="cpu", threads=4, batch_size=1,
                           unroll_curriculum_s=(20.0,), truncation_steps=20,
                           traffic_loss=True, traffic_ramp_start_s=0.0,
                           traffic_ramp_end_s=1.0)
    cfg5 = WmSTrainConfig(device="cpu", threads=4, batch_size=1,
                          unroll_curriculum_s=(20.0,), truncation_steps=5,
                          traffic_loss=True, traffic_ramp_start_s=0.0,
                          traffic_ramp_end_s=1.0)
    batch = collate([dataset[0]])
    r20 = WmSTrainer(model, cfg20)._forward_backward(batch, unroll_steps=40,
                                                     backward=False)
    r5 = WmSTrainer(model, cfg5)._forward_backward(batch, unroll_steps=40,
                                                   backward=False)
    assert r20["chunks"] == 2 and r5["chunks"] == 8
    # The logged traffic value must not scale with the 4x chunk count.
    assert 0.4 <= r5["traffic"] / max(r20["traffic"], 1e-9) <= 2.5
    # position is already a global-mean fraction and is likewise stable.
    assert 0.4 <= r5["position"] / r20["position"] <= 2.5


def test_loss_share_logging_reports_normalised_shares():
    dataset = _states()
    cfg = WmSTrainConfig(device="cpu", threads=4, batch_size=1,
                         unroll_curriculum_s=(20.0,), traffic_loss=True,
                         traffic_ramp_start_s=0.0, traffic_ramp_end_s=1.0,
                         stability_weight=0.01, conservation_weight=0.1,
                         loss_share_logging=True)
    batch = collate([dataset[0]])
    record = WmSTrainer(small_model(), cfg)._forward_backward(
        batch, unroll_steps=40, backward=False, share=True)
    share = record["loss_share"]
    assert set(DEFAULT_TERMS).issubset(share)
    assert abs(sum(share.values()) - 1.0) < 1e-6
    assert all(0.0 <= value <= 1.0 for value in share.values())


def test_loss_share_probe_uses_ramp_saturated_chunk_and_keeps_position():
    """The share probe chunk sits at ramp saturation, not the (empty) last one."""
    dataset = _states()
    cfg = WmSTrainConfig(device="cpu", threads=4, batch_size=1,
                         unroll_curriculum_s=(30.0,), traffic_loss=True,
                         traffic_ramp_start_s=0.0, traffic_ramp_end_s=12.5)
    batch = collate([dataset[0]])
    rec = WmSTrainer(small_model(), cfg)._forward_backward(
        batch, unroll_steps=40, backward=False, share=True)
    share = rec["loss_share"]
    assert share.get("traffic", 0.0) > 0.0
    assert share.get("position", 0.0) > 0.0


def test_log_off_path_omits_traffic_and_chunks():
    """When the traffic term is off the record must not gain new keys."""
    dataset = _states(n=1, t1=21, slots=4)
    cfg = WmSTrainConfig(device="cpu", threads=4, batch_size=1,
                         unroll_curriculum_s=(10.0,))
    batch = collate([dataset[0]])
    rec = WmSTrainer(small_model(), cfg)._forward_backward(
        batch, unroll_steps=20, backward=False)
    assert "traffic" not in rec and "chunks" not in rec


def test_fit_writes_per_stage_loss_share_without_validation(tmp_path):
    """The multi-stage smoke has no val set, yet must log per-stage shares."""
    dataset = _states()
    cfg = WmSTrainConfig(device="cpu", threads=4, batch_size=1,
                         out_dir=str(tmp_path / "run"),
                         unroll_curriculum_s=(10.0, 20.0), stage_steps=(1, 1),
                         traffic_loss=True, traffic_ramp_start_s=0.0,
                         traffic_ramp_end_s=1.0, stability_weight=0.01,
                         conservation_weight=0.1, loss_share_logging=True)
    WmSTrainer(small_model(), cfg).fit(dataset, val_dataset=None)
    share_log = tmp_path / "run" / "loss_share.jsonl"
    assert share_log.is_file()
    rows = [json.loads(x) for x in share_log.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    assert [round(r["stage_s"], 3) for r in rows] == [10.0, 20.0]
    for row in rows:
        assert abs(sum(row["loss_share"].values()) - 1.0) < 1e-6


def test_gate_reads_dedicated_loss_share_log(tmp_path):
    log = _write(tmp_path / "train_log.jsonl", HEALTHY)
    (tmp_path / "loss_share.jsonl").write_text(
        json.dumps({"stage_s": 300.0, "loss_share":
                    {"position": 0.5, "speed": 0.2, "lane": 0.1, "traffic": 0.1,
                     "stability": 0.05, "conservation": 0.05}}) + "\n",
        encoding="utf-8")
    result = evaluate(log)
    assert result["passed"] is True
    assert abs(sum(result["loss_share"][300.0].values()) - 1.0) < 1e-6

