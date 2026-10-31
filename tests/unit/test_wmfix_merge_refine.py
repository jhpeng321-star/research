"""S-WMFIX-C fallback 2: geometry-concentrated merge-response spatial basis (CPU).

Default (0, 0) must be bitwise the registered uniform 4-basis head; the
refined layout must concentrate resolution over the observed 75 m merge
section and ramp acceleration lane from network geometry only, keep the
zero-init/frozen-WM2 guarantees and the merge_response.* parameter allowlist,
and survive the canonical checkpoint reload.
"""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from tools.wmfixmerge.data import digest
from tools.wmfixmerge.train import export
from wms.config.train import TrainConfig
from wms.model.merge_response import MergeResponseField, freeze_except_response, refined_spatial_layout
from wms.model.entries import ENTRY_S_M, SECTION_MERGE_END_M, SECTION_MERGE_START_M
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


@pytest.fixture(scope="module", autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def scene():
    kin = torch.zeros(1, 4, 7)
    kin[0, :, 0] = torch.tensor([-150., -100., -80., 100.])
    kin[0, :, 2] = 1
    kin[0, :, 4] = 10
    kin[0, :, 6] = 5
    return kin, torch.ones(1, 4, dtype=torch.bool), torch.tensor([[False, False, True, False]])


def committed_module():
    root = Path(__file__).resolve().parents[2]
    text = subprocess.check_output(
        ["git", "show", "5a4d64c2:src/wms/model/merge_response.py"], cwd=root, text=True)
    legacy = types.ModuleType("wms.model._wmfixc_registered_head")
    legacy.__package__ = "wms.model"
    sys.modules[legacy.__name__] = legacy
    exec(compile(text, "<registered wmfixmerge head>", "exec"), legacy.__dict__)
    return legacy


def test_default_construction_is_bitwise_the_registered_head():
    legacy = committed_module()
    torch.manual_seed(313)
    registered = legacy.MergeResponseField()
    torch.manual_seed(313)
    current = MergeResponseField()
    kin, present, cav = scene()
    assert current.merge_bases == current.ramp_bases == 0 and not current.refined
    assert registered.state_dict().keys() == current.state_dict().keys()
    for name, value in registered.state_dict().items():
        assert torch.equal(value, current.state_dict()[name]), name
    with torch.no_grad():
        assert torch.equal(registered.features(kin, present, cav),
                           current.features(kin, present, cav))
        assert torch.equal(registered(kin, present, cav), current(kin, present, cav))
    # explicit zeros are the same default construction, not a second path
    torch.manual_seed(313)
    explicit = MergeResponseField(merge_bases=0, ramp_bases=0)
    assert torch.equal(explicit(kin, present, cav), current(kin, present, cav))
    with pytest.raises(ValueError):
        MergeResponseField(merge_bases=1)
    with pytest.raises(ValueError):
        MergeResponseField(ramp_bases=True)


def test_refined_layout_concentrates_over_observed_geometry():
    centres, widths = refined_spatial_layout(4, 4, 4, dtype=torch.float64)
    start, end = min(ENTRY_S_M), SECTION_MERGE_END_M
    assert centres.ndim == 1 and len(centres) == 10           # 12 - shared endpoints
    assert bool(((centres >= start) & (centres <= end)).all())
    assert torch.isfinite(widths).all() and bool((widths > 0).all())
    merge = centres[(centres > SECTION_MERGE_START_M) & (centres <= SECTION_MERGE_END_M)]
    assert len(merge) >= 3
    assert bool((widths[centres <= SECTION_MERGE_END_M][centres <= SECTION_MERGE_END_M] > 0).all())
    merge_widths = widths[(centres > SECTION_MERGE_START_M) & (centres <= SECTION_MERGE_END_M)]
    assert float(merge_widths.max()) == pytest.approx(75.44 / 3, rel=1e-6)  # ~25 m resolution
    ramp = centres[(centres >= ENTRY_S_M[2]) & (centres < SECTION_MERGE_START_M)]
    assert len(ramp) == 3                                    # entrance to merge nose
    coarse = widths[centres < ENTRY_S_M[2]]
    assert float(coarse.max()) == pytest.approx((end - start) / 3, rel=1e-6)  # backbone intact


def test_refined_head_zero_init_padding_and_state_gradient():
    kin, present, cav = scene()
    field = MergeResponseField(merge_bases=4, ramp_bases=4)
    assert field.net[0].in_features == 8 + 9 * 10
    assert torch.equal(field(kin, present, cav), torch.zeros(1, 4))  # zero-init head
    features = field.features(kin, present, cav)
    assert torch.isfinite(features).all() and features.shape == (1, 4, 8 + 90)
    pad = torch.full((1, 2, 7), float("nan"))
    expanded = field(torch.cat((kin, pad), 1),
                     torch.cat((present, torch.zeros(1, 2, dtype=torch.bool)), 1),
                     torch.cat((cav, torch.zeros(1, 2, dtype=torch.bool)), 1))
    assert torch.isfinite(expanded).all() and torch.equal(expanded[:, :4], field(kin, present, cav))
    # merge-region resolution: two merge positions separate in the pooled field
    with torch.no_grad():
        near = features[0, 1, 8:]
        kin2 = kin.clone()
        kin2[0, 1, 0] = -100. + 60.                           # still inside the merge basin
        far = field.features(kin2, present, cav)[0, 1, 8:]
        default = MergeResponseField().features(kin, present, cav)[0, 1, 8:]
        default2 = MergeResponseField().features(kin2, present, cav)[0, 1, 8:]
    assert (near - far).abs().sum() > (default - default2).abs().sum()
    # finite-difference agreement of the refined state gradient
    field = field.double()
    with torch.no_grad():
        field.net[-1].weight.fill_(.1)
    kin = kin.double().requires_grad_(True)
    grad = torch.autograd.grad(field(kin, present, cav).sum(), kin)[0]
    eps = 1e-4
    plus, minus = kin.detach().clone(), kin.detach().clone()
    plus[0, 0, 4] += eps
    minus[0, 0, 4] -= eps
    fd = (field(plus, present, cav).sum() - field(minus, present, cav).sum()) / (2 * eps)
    torch.testing.assert_close(grad[0, 0, 4], fd, rtol=1e-5, atol=1e-8)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_refined_head_keeps_frozen_wm2_and_allowlist():
    kin, present, cav = scene()
    initial = dict(veh_kin=kin, present=present, is_cav=cav)
    torch.manual_seed(91)
    base = WmSModel(ModelSpec(merge_response=True))
    torch.manual_seed(91)
    refined = WmSModel(ModelSpec(merge_response=True, merge_response_merge_bases=4,
                                  merge_response_ramp_bases=4))
    for name, value in base.state_dict().items():
        if not name.startswith("merge_response."):
            assert torch.equal(value, refined.state_dict()[name]), name
    names = freeze_except_response(refined)
    assert names and all(n.startswith("merge_response.") for n in names)
    with torch.no_grad():
        off = base(initial)["accel_mps2"]
        zero_head = refined(initial)["accel_mps2"]
    assert torch.equal(off, zero_head)                        # zero-init: exact identity
    optimizer = torch.optim.Adam([p for p in refined.parameters() if p.requires_grad], lr=.001)
    refined(initial)["accel_mps2"].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in refined.parameters())
    optimizer.step()
    for name, value in base.state_dict().items():
        if not name.startswith("merge_response."):
            assert torch.equal(value, refined.state_dict()[name]), name


def test_refined_config_roundtrip_and_checkpoint_reload(tmp_path):
    counts = dict(merge_response_merge_bases=4, merge_response_ramp_bases=4)
    cfg = replace(TrainConfig().model, merge_response=True, **counts)
    flat = replace(TrainConfig(), model=cfg).to_legacy_flat()
    assert flat["merge_response_merge_bases"] == 4 and flat["merge_response_ramp_bases"] == 4
    assert TrainConfig.from_legacy_flat(flat).model.merge_response_ramp_bases == 4
    assert "merge_response_merge_bases" not in TrainConfig().to_legacy_flat()
    with pytest.raises(ValueError):
        ModelSpec(merge_response=True, merge_response_merge_bases=1)
    with pytest.raises(ValueError):
        ModelSpec(merge_response=True, merge_response_ramp_bases="4")
    assert ModelSpec.from_mapping({"merge_response": True, **counts}).merge_response_merge_bases == 4
    # canonical wm_s/1 reload rebuilds the refined basis with exact parity
    model = WmSModel(ModelSpec(merge_response=True, **counts))
    freeze_except_response(model)
    frozen = {n: v.detach().clone() for n, v in model.state_dict().items()
              if not n.startswith("merge_response.")}
    optimizer = torch.optim.Adam(model.merge_response.parameters())
    original = dict(format="wm_s/1", config=dict(kind="structured"), model={})
    path = export(model, original, frozen, optimizer, tmp_path, 0, dict(test=True))
    from wms.model.checkpoint import load_checkpoint
    loaded = load_checkpoint(path).model
    assert loaded.merge_response.net[0].in_features == 8 + 9 * 10
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    assert len(digest(path)) == 64
