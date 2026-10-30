from dataclasses import replace
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest
import torch

from wms.model.entries import EntryConfig, TorchEntryProcess, _EntryProcessBase
from wms.model.merge_response import MergeResponseField, freeze_except_response
from wms.model.source_ledger import SourceLedger
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.train.merge_response import paired_objective, smooth_crossings


@pytest.fixture(scope="module", autouse=True)
def small_cpu_probe_threads():
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


def test_no_drop_or_misattributed_entrance():
    proc = _EntryProcessBase(EntryConfig(max_pending=1, conserve_queue=True), .5,
                             (-20, -20, -10), 1, None)
    proc.pending[0] = [9, 3]
    proc._carry_over(0, 0, 12, 1)
    proc._carry_over(0, 1, 7, 0)  # pool exhausted: preserve EVERY waiting request
    assert proc.pending.tolist() == [[11, 7]]
    assert proc.stats[0].dropped_pending == 0
    assert proc._source_ledgers[(0, 0)].due == 12
    proc._carry_over(0, 0, 13, 2)
    assert proc.pending.tolist() == [[11, 7]]
    for (b, e), ledger in proc._source_ledgers.items():
        assert ledger.due - ledger.inserted == proc.pending[b, e]


def test_legacy_queue_clipping_exact_when_off():
    proc = _EntryProcessBase(EntryConfig(max_pending=1), .5, (-20, -20, -10), 1, None)
    proc._carry_over(0, 0, 12, 1)
    assert proc.pending[0, 0] == 1 and proc.stats[0].dropped_pending == 10
    assert not hasattr(proc, "_source_ledgers")


def test_ledger_invalid_admission_is_atomic():
    ledger = SourceLedger(2)
    with pytest.raises(ValueError):
        ledger.advance(1, 4)
    assert (ledger.due, ledger.inserted) == (2, 0)
    with pytest.raises(ValueError):
        SourceLedger(-1)


def test_real_insert_pool_exhaustion_preserves_requests():
    proc = TorchEntryProcess(np.zeros((2, 30)), cfg=EntryConfig(conserve_queue=True),
                             pool=np.array([False]))
    proc._draw = lambda: [(0, .1, .9), (1, .1, .9)]
    state = dict(s_m=torch.zeros(1, 1), v_mps=torch.zeros(1, 1),
                 a_mps2=torch.zeros(1, 1), lane=torch.zeros(1, 1, dtype=torch.long),
                 length_m=torch.ones(1, 1) * 5, present=torch.ones(1, 1, dtype=torch.bool),
                 is_cav=torch.zeros(1, 1, dtype=torch.bool))
    for i in range(3):
        assert not proc.insert(state).any()
        assert proc.pending.tolist() == [[i + 1, i + 1]]
    assert proc.stats[0].inserted == proc.stats[0].dropped_pending == 0


def test_off_matches_committed_legacy_forward_and_rollout():
    root = Path(__file__).resolve().parents[2]
    text = subprocess.check_output(
        ["git", "show", "fc69fcfd:src/wms/model/wms.py"], cwd=root, text=True)
    legacy = types.ModuleType("wms.model._wmfix_legacy")
    legacy.__package__ = "wms.model"
    exec(compile(text, "<committed WM2 runtime>", "exec"), legacy.__dict__)
    torch.manual_seed(902)
    old = legacy.WmSModel(ModelSpec())
    rng = torch.random.get_rng_state()
    torch.manual_seed(902)
    new = WmSModel(ModelSpec())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    kin, present, _ = scene()
    state = dict(s_m=kin[..., 0], lane=kin[..., 1:4].argmax(-1),
                 v_mps=kin[..., 4], a_mps2=kin[..., 5], length_m=kin[..., 6],
                 present=present, is_cav=torch.zeros_like(present))
    with torch.no_grad():
        for a, b in ((old(state), new(state)),
                     (old.rollout(state, n_steps=3), new.rollout(state, n_steps=3))):
            for key in a:
                if torch.is_tensor(a[key]):
                    assert torch.equal(a[key], b[key]), key


def test_zero_init_scope_and_padding():
    kin, present, cav = scene()
    field = MergeResponseField()
    assert torch.equal(field(kin, present, cav), torch.zeros(1, 4))
    with torch.no_grad():
        field.net[-1].bias.fill_(1.)
    assert torch.equal(field(kin, present, cav), torch.tensor([[1., 1., 0., 0.]]))
    pad = torch.full((1, 2, 7), float("nan"))
    expanded = field(torch.cat((kin, pad), 1),
                     torch.cat((present, torch.zeros(1, 2, dtype=torch.bool)), 1),
                     torch.cat((cav, torch.zeros(1, 2, dtype=torch.bool)), 1))
    assert torch.isfinite(expanded).all() and torch.equal(expanded[:, :4], field(kin, present, cav))


def test_spatial_state_gradient_matches_finite_difference():
    kin, present, cav = scene()
    field = MergeResponseField().double()
    with torch.no_grad():
        field.net[-1].weight.fill_(.1)
    kin = kin.double().requires_grad_(True)
    loss = field(kin, present, cav).sum()
    grad = torch.autograd.grad(loss, kin)[0]
    eps = 1e-4
    plus, minus = kin.detach().clone(), kin.detach().clone()
    plus[0, 0, 4] += eps
    minus[0, 0, 4] -= eps
    fd = (field(plus, present, cav).sum() - field(minus, present, cav).sum()) / (2 * eps)
    torch.testing.assert_close(grad[0, 0, 4], fd, rtol=1e-5, atol=1e-8)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_model_off_rng_keys_and_forward_bitwise():
    torch.manual_seed(91)
    base = WmSModel(ModelSpec())
    state = torch.random.get_rng_state()
    torch.manual_seed(91)
    explicit = WmSModel(replace(ModelSpec(), merge_response=False))
    assert torch.equal(torch.random.get_rng_state(), state)
    assert base.state_dict().keys() == explicit.state_dict().keys()
    for name, value in base.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    kin, present, cav = scene()
    initial = dict(veh_kin=kin, present=present, is_cav=cav)
    with torch.no_grad():
        a, b = base(initial), explicit(initial)
    for name in a:
        if torch.is_tensor(a[name]):
            assert torch.equal(a[name], b[name]), name
    old = {name: value.clone() for name, value in base.state_dict().items()}
    base.enable_merge_response()
    names = freeze_except_response(base)
    assert names and all(n.startswith("merge_response.") for n in names)
    with torch.no_grad():
        on = base(initial)
    assert torch.equal(a["accel_mps2"], on["accel_mps2"])
    optimizer = torch.optim.Adam([p for p in base.parameters() if p.requires_grad], lr=.001)
    base(initial)["accel_mps2"].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in base.parameters())
    optimizer.step()
    for name, value in old.items():
        assert torch.equal(value, base.state_dict()[name]), name


def test_paired_objective_moves_error_and_source_value_is_retained():
    theta = torch.nn.Parameter(torch.tensor(0.))
    target = torch.tensor([[0., 0.], [-4., 2.]])
    opt = torch.optim.SGD([theta], lr=.1)
    def value():
        pred = torch.stack((torch.stack((theta * 0, theta * 0)),
                            torch.stack((theta, theta * 0))))
        return paired_objective(pred, target, [0, 0], torch.ones(2))
    before = value().item()
    value().backward()
    assert torch.isfinite(theta.grad) and theta.grad > 0
    opt.step()
    assert value().item() < before and value().item() >= 2  # uncorrected source error remains


def test_disappearance_or_slot_reuse_is_not_discharge():
    pos = torch.tensor([[[74.], [76.], [100.]]], requires_grad=True)
    present = torch.tensor([[[True], [False], [True]]])
    newborn = torch.tensor([[[False], [False], [True]]])
    assert smooth_crossings(pos, present, newborn, 1.).item() == 0
    value = smooth_crossings(pos, torch.ones_like(present), torch.zeros_like(newborn), 1.)
    value.backward()
    assert torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0
