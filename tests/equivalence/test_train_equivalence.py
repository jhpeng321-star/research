"""New loss composer / unroll / trainer vs the LEGACY WmSTrainer on fixed synthetic batches."""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit" / "train"))
import _compat  # noqa: E402

_compat.install()

from _compat import FlatConfig, legacy  # noqa: E402
from synth import make_arrays, make_batch, small_model  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402

train_mod = legacy("train")

R3 = dict(rollout_mode="hybrid", lane_head="history", residual_soft_bound_mps2=1.5,
          residual_mag_weight=0.05, lane_rate_weight=0.1, asym_speed_alpha=2.0,
          decel_weight=5.0, propagation_weight=1.0, history_s=None)
CASES = {
    "default": dict(),
    "braking_only": dict(asym_speed_alpha=2.0, decel_weight=5.0, propagation_weight=1.0),
    "r3": R3,
    "traffic": dict(traffic_loss=True, traffic_ramp_start_s=0.0, traffic_ramp_end_s=1.0),
    "stability_conservation": dict(stability_weight=0.5, conservation_weight=0.3,
                                   conservation_enabled=True),
    "projection_noise": dict(projection_enabled=True, state_noise_std_m=0.5,
                             state_noise_std_mps=0.1),
    "weights": dict(position_weight=0.7, speed_weight=1.3, lane_weight=0.4, step_weight_decay=0.05),
}


def _pair(extra, *, entry=False, cav=True):
    kwargs = dict(device="cpu", threads=1, truncation_steps=8, seed=17, **extra)
    cfg = train_mod.WmSTrainConfig(**kwargs)
    mode, head = cfg.rollout_mode, cfg.lane_head
    old = train_mod.WmSTrainer(small_model(mode, head), copy.deepcopy(cfg))
    new = WmSTrainer(FlatConfig.from_legacy(cfg), small_model(mode, head), dataset=None,
                     out_dir="unused")
    batch = make_batch(history=mode == "hybrid", entry=entry, cav=cav)
    return old, new, batch


def _close(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a) ^ set(b)}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=1e-5, abs=1e-5), path
    else:
        assert a == b, path


@pytest.mark.parametrize("name", sorted(CASES))
def test_forward_record_matches_legacy(name):
    old, new, batch = _pair(CASES[name])
    for p in (old, new):
        torch.manual_seed(5)
    ref = old._forward_backward(batch, unroll_steps=20, backward=False)
    torch.manual_seed(5)
    got = new.forward_backward(batch, unroll_steps=20, backward=False)
    for key in ("loss", "position", "speed", "lane"):
        assert key in got
    _close(ref, got)


@pytest.mark.parametrize("name", ["default", "r3", "traffic", "stability_conservation"])
def test_gradients_match_legacy(name):
    old, new, batch = _pair(CASES[name])
    torch.manual_seed(5)
    old.optimizer.zero_grad()
    ref = old._forward_backward(batch, unroll_steps=20)
    torch.manual_seed(5)
    new.optimizer.zero_grad()
    got = new.forward_backward(batch, unroll_steps=20)
    _close(ref, got)
    for (n1, p1), (n2, p2) in zip(old.model.named_parameters(), new.model.named_parameters()):
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None, n1
        else:
            torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-5, atol=1e-5)


def _checksum(model):
    h = hashlib.sha256()
    for _, p in sorted(model.state_dict().items()):
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("name", ["default", "r3"])
def test_three_training_steps_same_parameters(name):
    old, new, batch = _pair(CASES[name])
    for step in range(3):
        torch.manual_seed(100 + step)
        r_old = old.train_batch(batch, unroll_steps=20)
        torch.manual_seed(100 + step)
        r_new = new.train_batch(batch, unroll_steps=20)
        assert r_new["loss"] == pytest.approx(r_old["loss"], rel=1e-5, abs=1e-6)
    assert _checksum(old.model) == _checksum(new.model)


def test_loss_share_probe_matches_legacy():
    old, new, batch = _pair(dict(R3, loss_share_logging=True))
    torch.manual_seed(5)
    ref = old._forward_backward(batch, unroll_steps=20, backward=False, share=True)
    torch.manual_seed(5)
    got = new.forward_backward(batch, unroll_steps=20, backward=False, share=True)
    _close(ref["loss_share"], got["loss_share"])


def test_entries_v2_matches_legacy():
    old, new, batch = _pair(dict(entries_v2=True, entry_pad=4), entry=True)
    torch.manual_seed(5)
    ref = old._forward_backward(batch, unroll_steps=20, backward=False)
    torch.manual_seed(5)
    got = new.forward_backward(batch, unroll_steps=20, backward=False)
    assert got["entrants"] > 0
    _close(ref, got)


def test_fit_matches_legacy_fit(tmp_path):
    from output.wd01.wm_n1.wms.train import TrajectoryDataset

    kwargs = dict(device="cpu", threads=1, truncation_steps=8, seed=17, batch_size=2,
                  unroll_curriculum_s=(5.0, 10.0), stage_steps=(2, 2), loss_share_logging=True,
                  **{k: v for k, v in R3.items() if k not in ("rollout_mode", "lane_head")})
    ds = TrajectoryDataset.from_arrays(make_arrays())
    cfg_old = train_mod.WmSTrainConfig(out_dir=str(tmp_path / "old"), **kwargs)
    old = train_mod.WmSTrainer(small_model(), cfg_old)
    old.fit(ds)
    cfg_new = train_mod.WmSTrainConfig(out_dir=str(tmp_path / "new"), **kwargs)
    new = WmSTrainer(FlatConfig.from_legacy(cfg_new), small_model(), ds, out_dir=tmp_path / "new")
    result = new.fit()
    assert _checksum(old.model) == _checksum(new.model)
    import json
    for name in ("train_log.jsonl", "loss_share.jsonl"):
        a = [json.loads(x) for x in (tmp_path / "old" / name).read_text().splitlines()]
        b = [json.loads(x) for x in (tmp_path / "new" / name).read_text().splitlines()]
        assert len(a) == len(b)
        for ra, rb in zip(a, b):
            ra.pop("seconds", None), rb.pop("seconds", None)
            _close(ra, rb)
    assert result.receipt["steps"] == 4
    for name in ("wm_s.pt", "wm_s_stage5.pt", "wm_s_stage10.pt", "TRAIN_RECEIPT.json", "train_config.json"):
        assert (tmp_path / "new" / name).is_file(), name
