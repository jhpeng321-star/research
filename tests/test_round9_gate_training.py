"""Round-9 logging is observational; saved smoke models really reload."""
from dataclasses import asdict
import json
import math
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from wms.config.train import TrainConfig
from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import save_checkpoint
from wms.physics.config import PhysicsConfig
from wms.train.round9_gate import check_checkpoints
from wms.train.trainer import WmSTrainer


def test_round9_resolved_smoke_config():
    from wms.config.load import load_train_config
    from wms.train.settings import TrainSettings

    path = Path(__file__).resolve().parents[1] / "configs/c4_wm_v10round9.toml"
    cfg = load_train_config(str(path), [
        "optim.batch_size=4", "optim.batch_size_per_stage=[4,4]",
        "curriculum.unroll_s=[120.0,300.0]", "curriculum.stage_steps=[2,2]",
        "curriculum.truncation_steps_per_stage=[120,60]",
    ])
    settings = TrainSettings.from_config(cfg)
    assert settings.merge_pivot_state_st is True
    assert cfg.physics.ramp_lane_end_stop is True
    assert settings.batch_size == 4 and settings.batch_size_per_stage == (4, 4)
    assert settings.unroll_curriculum_s == (120.0, 300.0)
    assert settings.stage_steps == (2, 2)


@pytest.mark.parametrize("clip", (0.0, 0.1))
@pytest.mark.parametrize("pivot_used", (False, True))
def test_gradient_logging_preserves_backward_clipping_and_optimizer(clip, pivot_used):
    model = torch.nn.Module()
    model.merge_pivot = torch.nn.Linear(2, 1, bias=False)
    model.other = torch.nn.Parameter(torch.tensor(2.0))
    model.merge_pivot.weight.data.fill_(1.0)
    trainer = WmSTrainer.__new__(WmSTrainer)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.settings = SimpleNamespace(grad_clip=clip)
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def backward(batch, **kwargs):
        loss = model.other.square()
        if pivot_used:
            loss = loss + model.merge_pivot(torch.tensor([[3.0, 4.0]])).sum()
        loss.backward()
        return {"loss": float(loss.detach())}

    trainer.forward_backward = backward
    # Independent reference: same loss/backward/clip/Adam, without train_batch logging.
    import copy
    reference = copy.deepcopy(model)
    optim = torch.optim.Adam(reference.parameters(), lr=0.01)
    loss = reference.other.square()
    if pivot_used:
        loss = loss + reference.merge_pivot(torch.tensor([[3.0, 4.0]])).sum()
    loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(reference.parameters(), clip)
    optim.step()
    row = trainer.train_batch({}, unroll_steps=1)
    assert row["merge_pivot_grad_norm"] == (5.0 if pivot_used else 0.0)
    assert math.isfinite(row["loss"])
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.equal(actual, expected)
        if expected.grad is None:
            assert actual.grad is None
        else:
            assert torch.equal(actual.grad, expected.grad)


@pytest.fixture
def checkpoints(tmp_path):
    cfg = TrainConfig.from_dict({
        "model": {"rollout_mode": "hybrid", "lane_model": "hazard_v5",
                  "merge_decision": "first_passage", "merge_passage_law": "hybrid",
                  "merge_pivot_state_st": True},
        "physics": {"ramp_lane_end_stop": True},
        "curriculum": {"unroll_s": [120.0, 300.0], "stage_steps": [2, 2]},
        "optim": {"batch_size": 4, "batch_size_per_stage": [4, 4]},
    })
    physics = PhysicsConfig(ramp_lane_end_stop=True)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics)
    rows = [{"step": i + 1, "stage_s": stage, "loss": 0.5}
            for i, stage in enumerate((120, 120, 300, 300))]
    (tmp_path / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")
    for stage, step in ((120, 2), (300, 4)):
        save_checkpoint(tmp_path / f"wm_s_stage{stage}.pt", model, cfg,
                        extra={"stage_s": stage, "step": step})
    save_checkpoint(tmp_path / "wm_s.pt", model, cfg, history=rows)
    return tmp_path, rows


def test_all_smoke_checkpoints_reload_exactly(checkpoints):
    reports = check_checkpoints(*checkpoints)
    assert len(reports) == 3
    assert all(report["reloaded"] and len(report["sha256"]) == 64 for report in reports)


@pytest.mark.parametrize("failure", (
    "missing", "corrupt", "step", "stage", "history", "state_st", "lane_stop",
    "batch", "stage_batch", "pivot_weights", "nonfinite",
))
def test_bad_checkpoint_refuses_launch(checkpoints, failure):
    directory, rows = checkpoints
    path = directory / ("wm_s.pt" if failure == "history" else "wm_s_stage300.pt")
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_bytes(b"not a checkpoint")
    elif failure == "lane_stop":
        physics_path = directory / "physics.json"
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        physics["ramp_lane_end_stop"] = False
        physics_path.write_text(json.dumps(physics), encoding="utf-8")
    else:
        payload = torch.load(path, weights_only=True)
        if failure in ("step", "stage"):
            payload["stage_s" if failure == "stage" else "step"] = -1
        elif failure == "history":
            payload["history"] = []
        elif failure == "state_st":
            payload["config"]["merge_pivot_state_st"] = False
        elif failure == "batch":
            payload["config"]["batch_size"] = 2
        elif failure == "stage_batch":
            payload["config"]["batch_size_per_stage"] = [4, 2]
        elif failure == "pivot_weights":
            payload["model"] = {k: v for k, v in payload["model"].items()
                                if not k.startswith("merge_pivot.")}
        elif failure == "nonfinite":
            next(v for k, v in payload["model"].items()
                 if k.startswith("merge_pivot.")).fill_(float("nan"))
        torch.save(payload, path)
    with pytest.raises(Exception):
        check_checkpoints(directory, rows)
