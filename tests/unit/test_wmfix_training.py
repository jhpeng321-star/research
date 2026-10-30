from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from tools.wmfixmerge.data import digest
from tools.wmfixmerge.feasibility import synthetic
from tools.wmfixmerge.train import check_gates, export
from wms.config.train import TrainConfig
from wms.model.merge_response import freeze_except_response
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


def test_opt_in_config_roundtrip_and_boolean_validation():
    cfg = TrainConfig()
    assert "merge_response" not in cfg.to_legacy_flat()
    cfg = replace(cfg, model=replace(cfg.model, merge_response=True))
    assert cfg.to_legacy_flat()["merge_response"] is True
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()).model.merge_response is True
    with pytest.raises(ValueError):
        ModelSpec(merge_response="false")


def test_synthetic_probe_is_real_free_running_and_reduces_error():
    torch.set_num_threads(1)
    result = synthetic()
    assert result["passed"] and result["finite_difference_ok"]
    assert result["after_step_loss"] < result["initial_loss"]
    assert result["frozen_bitwise"]


def test_no_synthetic_or_unbound_launch_gate(tmp_path):
    p = tmp_path / "feasibility.json"
    p.write_text(json.dumps(dict(mode="synthetic", launch_ready=True, passed=True)))
    args = SimpleNamespace(feasibility=p)
    with pytest.raises(ValueError, match="not PASS"):
        check_gates(args)
    p.write_text(json.dumps(dict(mode="real_train", launch_ready=True, passed=True, source_clock_match=True)))
    args.prepared = p
    with pytest.raises(ValueError, match="input changed"):
        check_gates(args)


def test_immutable_export_canonical_reload_and_frozen_guard(tmp_path):
    model = WmSModel(ModelSpec(merge_response=True))
    freeze_except_response(model)
    frozen = {n: v.detach().clone() for n, v in model.state_dict().items() if not n.startswith("merge_response.")}
    optimizer = torch.optim.Adam(model.merge_response.parameters())
    original = dict(format="wm_s/1", config=dict(kind="structured"), model={})
    p = export(model, original, frozen, optimizer, tmp_path, 0, dict(test=True))
    assert len(digest(p)) == 64
    with pytest.raises(FileExistsError):
        export(model, original, frozen, optimizer, tmp_path, 0, {})
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if not n.startswith("merge_response.")).add_(1)
    with pytest.raises(ValueError, match="frozen"):
        export(model, original, frozen, optimizer, tmp_path, 1, {})
