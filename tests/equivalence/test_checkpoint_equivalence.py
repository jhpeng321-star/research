"""Checkpoint loading: new loader vs the legacy plugin construction."""
from __future__ import annotations

import json

import pytest
import torch

from wms.model import ModelSpec, build_model, load_checkpoint, save_checkpoint

from . import inputs
from .legacy import C1_CHECKPOINT, legacy, load_legacy
from .test_model_equivalence import assert_same


@pytest.mark.skipif(not C1_CHECKPOINT.is_file(), reason="C1 checkpoint missing")
def test_c1_strict_load_and_forward():
    load_legacy()
    loaded = load_checkpoint(C1_CHECKPOINT, strict=True)
    payload = torch.load(C1_CHECKPOINT, map_location="cpu", weights_only=True)
    assert set(payload["model"]) <= set(loaded.model.state_dict())
    assert len(loaded.sha256) == 64
    leg = legacy("model")
    cfg = payload["config"]
    # The legacy plugin builds the model with the physics of the run's TRAIN_RECEIPT; the new loader
    # finds the same block next to the checkpoint (WP-I) -- the two must therefore agree.
    receipt = json.loads(C1_CHECKPOINT.with_name("TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
    allowed = set(leg.PhysicsConfig.__dataclass_fields__)
    physics = leg.PhysicsConfig(**{k: v for k, v in receipt["physics"].items() if k in allowed})
    assert loaded.model.physics.sumo_fidelity == physics.sumo_fidelity
    assert loaded.model.physics.merge_cal == physics.merge_cal
    old = leg.WmSModel(physics=physics,
                       history_features=int(cfg.get("history_features", 6)),
                       rollout_mode=cfg.get("rollout_mode", "residual"),
                       lane_head=cfg.get("lane_head", "off"),
                       residual_cap_mps2=cfg.get("residual_cap_mps2"),
                       residual_soft_bound_mps2=cfg.get("residual_soft_bound_mps2"),
                       residual_soft_bound_neg_mps2=cfg.get("residual_soft_bound_neg_mps2"),
                       history_s=cfg.get("history_s"))
    old.load_state_dict(payload["model"], strict=False)
    old.eval()
    state = inputs.make_state()
    hist, mask = inputs.make_history()
    kw = dict(n_steps=10, cav_teacher=inputs.make_teacher(state, 10), history=hist,
              history_mask=mask)
    with torch.no_grad():
        assert_same(old.rollout(dict(state), **kw), loaded.model.rollout(dict(state), **kw),
                    atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = build_model(ModelSpec(rollout_mode="hybrid", lane_head="history"))
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, model, {"rollout_mode": "hybrid", "lane_head": "history"},
                    history=[{"step": 1}], extra={"stage_s": 10.0})
    loaded = load_checkpoint(path, strict=True)
    assert loaded.payload_meta["stage_s"] == 10.0
    a, b = model.state_dict(), loaded.model.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
