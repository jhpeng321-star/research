from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from tools.wmfixmerge.data import digest
from tools.wmfixmerge.feasibility import panel_gate, synthetic
from tools.wmfixmerge.train import check_gates, export
from wms.config.train import TrainConfig
from wms.train.settings import TrainSettings
from wms.model.merge_response import freeze_except_response
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


def test_opt_in_config_roundtrip_and_boolean_validation():
    cfg = TrainConfig()
    assert "merge_response" not in cfg.to_legacy_flat()
    cfg = replace(cfg, model=replace(cfg.model, merge_response=True))
    assert cfg.to_legacy_flat()["merge_response"] is True
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()).model.merge_response is True
    # Candidate configs must survive the r10/load_for_scoring settings path.
    assert TrainSettings.from_config(cfg).merge_response is True
    with pytest.raises(ValueError):
        ModelSpec(merge_response="false")


def test_synthetic_probe_is_real_free_running_and_reduces_error():
    torch.set_num_threads(1)
    result = synthetic()
    assert result["passed"] and result["finite_difference_ok"]
    assert result["after_step_loss"] < result["initial_loss"]
    assert result["frozen_bitwise"]


class _StubRt:
    """Two-row rollouts whose upstream term is the whole paired error."""

    class _Source:
        groups = list(range(12))

    source = _Source()

    def __init__(self, upstream_error):
        self.upstream_error = upstream_error
        self.calls = 0

    def rollout(self, group, horizon, arm_index=0):
        self.calls += 1
        pred = torch.tensor([[0., 0., 0., 0., 0.],
                             [self.upstream_error, self.upstream_error, 0., 0., 0.]])
        rec = torch.tensor([[0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0.]])
        return pred, rec, [0, 0], {}


def test_panel_gate_covers_all_groups_arms_and_reports_teacher_ceiling():
    rt = _StubRt(upstream_error=4.0)
    panel, rows, receipt = panel_gate(rt)
    assert receipt["frozen_rerun_noise"] == 0.0
    assert len(rows) == 48  # 2 horizons x 12 groups x 2 arms
    for h in ("120", "300"):
        assert panel[h]["rows"] == 24
        assert panel[h]["initial_paired_J_MAE"] == 4.0
        # substituting the recorded upstream removes the whole error: teacher 0
        assert panel[h]["teacher_paired_J_MAE"] == 0.0
    assert rt.calls == 50  # 48 panel rollouts + frozen rerun pair
    rt = _StubRt(upstream_error=0.0)
    panel, _, receipt = panel_gate(rt)
    # no headroom: teacher cannot be strictly below initial -> gate must fail downstream
    assert panel["120"]["teacher_paired_J_MAE"] == panel["120"]["initial_paired_J_MAE"]


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
