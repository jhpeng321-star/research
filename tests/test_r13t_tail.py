"""R13T: temporal read-out re-initialisation and the per-cell pinball tail-matching terms.

Synthetic CPU tensors only (no data, no SUMO).  Covers the quantile score (zero at the recorded
distribution, a proper score: minimised there; heavy tails penalised), gradients (finite, only through
predicted speeds, direction), masking (CAV / other-lane / absent rows never scored), the support rule, the
TRAIN-split guard, the shared cell convention (identical to the pre-refactor brake-persistence cells),
zero effect when off, the reinit switch and the round-13T / 13AT configs.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wms.config.load import load_raw, load_train_config
from wms.train.losses import brake_persist as bp
from wms.train.losses.accel_tail import (AccelTailConfig, AccelTailLoss, JerkTailLoss, TABLE_SCHEMA,
                                         _load_table, accel_tail_terms, cell_scores, support_ok)
from wms.train.losses.base import LossContext
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

from test_r10b_courtesy import sequence
from test_r11_courtesy import leaves, model_v

ROOT = Path(__file__).resolve().parents[1]
ROUND12 = ROOT / "configs/c4_wm_v10round12.toml"
ROUND13A = ROOT / "configs/c4_wm_v10round13a.toml"
ROUND13T = ROOT / "configs/c4_wm_v10round13t.toml"
ROUND13AT = ROOT / "configs/c4_wm_v10round13at.toml"
LEVELS = [m / 20 for m in range(1, 20)]                     # 0.05 .. 0.95 (test grid)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def normal_q(mu=0.0, sd=0.5):
    dist = torch.distributions.Normal(mu, sd)
    return [float(dist.icdf(torch.tensor(u))) for u in LEVELS]


def write_table(path, split=("train",), cell_n=10_000, jerk_q=None):
    q = normal_q()
    block = {"pooled": {"n": 100_000, "q": q}, "cells": {str(c): {"n": cell_n, "q": q} for c in range(64)}}
    jq = jerk_q or [0.1 * u for u in LEVELS]
    jblock = {"pooled": {"n": 100_000, "q": jq}, "cells": {}}
    path.write_text(json.dumps({"schema": TABLE_SCHEMA, "levels": LEVELS, "max_gap_m": 40.0, "dt_s": 0.5,
                                "lanes": [1], "split_roles": list(split), "accel": block,
                                "abs_jerk": jblock}), encoding="utf-8")
    _load_table.cache_clear()
    return str(path)


# ------------------------------------------------------------------ the score
def test_score_is_zero_at_the_recorded_distribution_and_positive_elsewhere():
    levels = torch.tensor(LEVELS, dtype=torch.float64)
    table = torch.tensor(normal_q(), dtype=torch.float64).expand(64, -1)
    torch.manual_seed(0)
    match = torch.distributions.Normal(0.0, 0.5).sample((20_000,)).double()
    cells = torch.zeros_like(match, dtype=torch.long)
    s_match, rows, n_cells = cell_scores(match, cells, table, levels)
    assert rows == 20_000 and n_cells == 1 and abs(float(s_match)) < 2e-3
    heavy = match.clone()
    heavy[:800] = -3.7                                       # 4 % spurious hard brakes (BRAKE_DIAG)
    shifted = match + 0.3
    s_heavy = float(cell_scores(heavy, cells, table, levels)[0])
    s_shift = float(cell_scores(shifted, cells, table, levels)[0])
    assert s_heavy > 10 * abs(float(s_match)) and s_shift > 10 * abs(float(s_match))


def test_score_is_minimised_at_the_recorded_quantile_per_level():
    """Proper score: for one level the expected pinball over the recorded grid is minimal at q_rec(u)."""
    levels = torch.tensor([0.25], dtype=torch.float64)
    y = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    from wms.train.losses.accel_tail import _pinball_grid
    grid = torch.linspace(-1.5, 1.5, 301, dtype=torch.float64)
    scores = torch.stack([_pinball_grid(q.view(1), y, levels) for q in grid])
    best = float(grid[scores.argmin()])
    assert abs(best - float(torch.quantile(y, 0.25))) <= 0.011


def test_gradient_pushes_a_heavy_lower_tail_up_and_is_finite():
    levels = torch.tensor(LEVELS, dtype=torch.float64)
    table = torch.tensor(normal_q(), dtype=torch.float64).expand(64, -1)
    torch.manual_seed(1)
    values = torch.distributions.Normal(0.0, 0.5).sample((4000,)).double()
    values[:400] = -3.7
    values.requires_grad_(True)
    score, _, _ = cell_scores(values, torch.zeros(4000, dtype=torch.long), table, levels)
    score.backward()
    g = values.grad
    assert torch.isfinite(g).all()
    assert float(g[:400].sum()) < 0.0                        # gradient descent raises the spurious brakes
    assert int((g != 0).sum()) <= 2 * len(LEVELS)            # only the order statistics at the levels


def test_support_rule_and_unsupported_cells():
    assert support_ok(20, 0.05) and not support_ok(19, 0.05) and support_ok(2, 0.5)
    levels = torch.tensor(LEVELS, dtype=torch.float64)
    table = torch.zeros(64, len(LEVELS), dtype=torch.float64)
    one = torch.tensor([5.0], dtype=torch.float64)
    score, rows, n_cells = cell_scores(one, torch.zeros(1, dtype=torch.long), table, levels)
    assert rows == 0 and n_cells == 0 and float(score) == 0.0


def test_table_guard_split_schema_and_fallback(tmp_path):
    path = write_table(tmp_path / "val.json", split=("val",))
    with pytest.raises(ValueError, match="TRAIN"):
        _load_table(path, True)
    assert _load_table(path, False)["accel"].shape == (64, len(LEVELS))
    # a thin cell (n below 1/min(u,1-u) = 20) falls back to the pooled quantiles
    path = write_table(tmp_path / "thin.json", cell_n=19)
    table = _load_table(path, True)
    assert not bool(table["accel_own_cell"].any())
    bad = json.loads(Path(path).read_text(encoding="utf-8"))
    bad["schema"] = "other"
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        _load_table(str(tmp_path / "bad.json"), True)


# ------------------------------------------------------------------ the chunk term (masking, gradient path)
def fake_ctx(n_steps=6, k=6, seed=0):
    torch.manual_seed(seed)
    v0 = torch.full((1, k), 10.0)
    accel = torch.randn(1, n_steps, k) * 0.5
    accel[0, 2, 1] = -4.0                                    # spurious hard brakes (slot 1, lane 1, HDV)
    accel[0, 3, 2] = -4.0                                    # on a CAV row: must not count
    accel[0, 2, 4] = -4.0                                    # on a lane-0 row: must not count
    speeds = (v0.unsqueeze(1) + accel.cumsum(1) * 0.5).requires_grad_(True)
    positions = torch.arange(k).float().view(1, 1, k) * 30.0 + torch.zeros(1, n_steps, k)
    lanes = torch.tensor([1, 1, 1, 1, 0, 1]).view(1, 1, k).expand(1, n_steps, k).clone()
    masks = torch.ones(1, n_steps, k, dtype=torch.bool)
    masks[0, :, 5] = False                                   # slot 5 absent
    current = {"v_mps": v0, "s_m": positions[:, 0], "lane": lanes[:, 0], "present": masks[:, 0].clone(),
               "a_mps2": torch.zeros(1, k)}
    current["present"][0, 5] = False
    is_cav = torch.zeros(1, n_steps + 1, k, dtype=torch.bool)
    is_cav[0, :, 2] = True
    pred = {"speeds": speeds, "positions": positions, "lanes": lanes, "masks": masks,
            "accel": accel}
    targets = SimpleNamespace(is_cav=is_cav, dt_s=0.5)
    return LossContext(pred=pred, current=current, targets=targets, c0=0, c1=n_steps), speeds


def test_term_masks_cav_other_lane_and_absent_rows_and_grads_only_speeds(tmp_path):
    path = write_table(tmp_path / "t.json")
    ctx, speeds = fake_ctx()
    term = AccelTailLoss(AccelTailConfig(weight=1.0, table=path))
    out = term(ctx)
    # eligible rows: slots 0, 1, 3 (HDV, lane 1, present) x 6 steps; cells with a single row have no
    # supported level (n * min(u, 1 - u) < 1 for every u) and are not scored
    from wms.train.losses.accel_tail import _chunk_rows
    rows = _chunk_rows(ctx, (1,), 40.0)["rows"]
    assert int(rows.sum()) == 18 and bool(rows[0, :, [0, 1, 3]].all())
    assert 0 < term.last_stats["rows"] <= 18
    out.value.backward()
    g = speeds.grad
    assert torch.isfinite(g).all()
    assert float(g[..., [2, 4, 5]].abs().sum()) == 0.0       # CAV, lane 0, absent: never scored
    assert float(g[..., [0, 1, 3]].abs().sum()) > 0.0
    ctx2, speeds2 = fake_ctx()
    jt = JerkTailLoss(AccelTailConfig(jerk_weight=1.0, table=path))
    jv = jt(ctx2)
    assert 0 < jt.last_stats["rows"] <= 3 * 5 and torch.isfinite(jv.value)
    jv.value.backward()
    assert float(speeds2.grad[..., [2, 4, 5]].abs().sum()) == 0.0


def test_weights_scale_and_off_builds_nothing(tmp_path):
    path = write_table(tmp_path / "t.json")
    assert accel_tail_terms(AccelTailConfig()) == []
    names = [t.name for t in accel_tail_terms(AccelTailConfig(weight=2.0, jerk_weight=1.0, table=path))]
    assert names == ["accel_tail", "jerk_tail"]
    ctx, _ = fake_ctx()
    v1 = AccelTailLoss(AccelTailConfig(weight=1.0, table=path))(ctx).value
    ctx, _ = fake_ctx()
    v2 = AccelTailLoss(AccelTailConfig(weight=2.5, table=path))(ctx).value
    assert torch.allclose(v2, 2.5 * v1)
    with pytest.raises(ValueError):
        AccelTailConfig(weight=1.0)                          # no table
    s = TrainSettings(truncation_steps=8)
    assert not s.accel_tail_on
    assert not {"accel_tail", "jerk_tail"} & {t.name for t in build_loss(s).chunk_terms}


def test_off_is_bit_identical_in_the_runner_and_on_adds_a_finite_term(tmp_path):
    path = write_table(tmp_path / "t.json")
    results = {}
    for name, extra in (("default", {}), ("zero", {"accel_tail_weight": 0.0, "accel_tail_jerk_weight": 0.0,
                                                   "accel_tail_table": path}),
                        ("on", {"accel_tail_weight": 1.0, "accel_tail_jerk_weight": 1.0,
                                "accel_tail_table": path})):
        model = model_v(randomize=True).train()
        cfg = TrainSettings(truncation_steps=8, **extra)
        res = UnrollRunner(model, cfg, build_loss(cfg), projection=None,
                           device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=True)
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        results[name] = (float(res["loss"]), grads, res)
    assert results["default"][0] == results["zero"][0]
    assert torch.equal(results["default"][1], results["zero"][1])
    assert results["on"][0] != results["default"][0] and torch.isfinite(results["on"][1]).all()


def test_leader_cells_refactor_matches_the_previous_implementation():
    """The brake-persistence cell ids are unchanged by the shared helper (pre-refactor code inlined)."""
    torch.manual_seed(4)
    b, n, k = 2, 5, 9
    states = torch.zeros(b, n + 1, k, 7)
    states[..., 0] = torch.rand(b, n + 1, k) * 120
    states[..., 4] = torch.rand(b, n + 1, k) * 15
    states[..., 5] = torch.randn(b, n + 1, k) * 1.5
    lane = torch.randint(0, 3, (b, n + 1, k))
    states[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    states[0, :, 3, 1:4] = 0.0                               # off road
    loss_valid = torch.rand(b, n + 1, k) > 0.1
    onset = torch.rand(b, n, k) > 0.3
    targets = SimpleNamespace(states=states, loss_valid=loss_valid)
    new = bp._cell_ids(targets, onset, max_gap_m=40.0)
    old = torch.full(onset.shape, -1, dtype=torch.long)
    for t in range(n):
        s_t = states[:, t, :, 0]
        ln = states[:, t, :, 1:4].argmax(dim=-1)
        on_road = states[:, t, :, 1:4].sum(dim=-1) > 0
        valid_t = loss_valid[:, t] & on_road
        gap = s_t.unsqueeze(-2) - s_t.unsqueeze(-1)
        same = ln.unsqueeze(-2) == ln.unsqueeze(-1)
        ahead = valid_t.unsqueeze(-2) & valid_t.unsqueeze(-1) & same & (gap > 0.0) & (gap <= 40.0)
        dist = torch.where(ahead, gap, torch.full_like(gap, float("inf")))
        nearest, index = dist.min(dim=-1)
        has = torch.isfinite(nearest)
        v, a = states[:, t, :, 4], states[:, t, :, 5]
        closing = v - torch.gather(v, -1, index)
        a_l = torch.gather(a, -1, index)
        gb = torch.where(nearest <= 10.0, 0, torch.where(nearest <= 20.0, 1, torch.where(has, 2, 3)))
        cb = torch.where(closing < 0.0, 0, torch.where(closing < 3.0, 1, torch.where(closing < 6.0, 2, 3)))
        db = torch.where(~has, 3, torch.where(a_l >= -1.0, 0, torch.where(a_l >= -2.5, 1, 2)))
        cell = (gb * 4 + cb) * 4 + db
        old[:, t] = torch.where(onset[:, t], cell, torch.full_like(cell, -1))
    assert torch.equal(new, old)


# ------------------------------------------------------------------ reinit switch
class _Stub:
    pass


def test_temporal_reinit_zeroes_only_the_accel_row_and_keeps_a_live_gradient():
    from wms.train.trainer import WmSTrainer
    model = model_v(randomize=True)
    torch.manual_seed(9)
    with torch.no_grad():
        model.temporal[-1].weight.normal_()
        model.temporal[-1].bias.normal_()
    before = model.temporal[-1].weight.detach().clone(), model.temporal[-1].bias.detach().clone()
    other = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("temporal.2")}
    stub = _Stub()
    stub.model, stub.settings = model, TrainSettings(temporal_accel_reinit=True)
    report = WmSTrainer._r12_reinit(stub)
    w, b = model.temporal[-1].weight, model.temporal[-1].bias
    assert report["temporal_accel_reinit"]["zeroed_rows"] == [0] and report["temporal_accel_reinit"]["abs_sum_before"] > 0
    assert float(w[0].abs().sum()) == 0.0 and float(b[0]) == 0.0
    assert torch.equal(w[1:], before[0][1:]) and torch.equal(b[1:], before[1][1:])
    assert all(torch.equal(p, other[n]) for n, p in model.named_parameters() if n in other)
    cfg = TrainSettings(truncation_steps=8)
    UnrollRunner(model.train(), cfg, build_loss(cfg), projection=None,
                 device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=True)
    assert float(w.grad[0].abs().sum()) > 0.0                # live gradient on the zeroed row
    stub.settings = TrainSettings()
    assert "temporal_accel_reinit" not in WmSTrainer._r12_reinit(stub)


def test_reinit_validation():
    with pytest.raises(ValueError):
        load_train_config(ROUND12, ["model.temporal_accel_reinit=true", "model.temporal_accel_off=true"])


# ------------------------------------------------------------------ configs
def test_round12_omits_the_r13t_keys():
    cfg = load_train_config(ROUND12)
    assert cfg.model.temporal_accel_reinit is False and cfg.loss.accel_tail.weight == 0.0
    flat = cfg.to_legacy_flat()
    assert not {"temporal_accel_reinit", "accel_tail_weight", "accel_tail_jerk_weight", "accel_tail_table",
                "accel_tail_lanes"} & set(flat)


R13T_KEYS = {"model.temporal_accel_reinit", "loss.accel_tail.weight", "loss.accel_tail.jerk_weight",
             "loss.accel_tail.table", "loss.accel_tail.lanes"}


INIT_KEYS = {"model.init_from", "model.driver_param_init", "model.residual_accel_reinit",
             "model.follower_courtesy_init_zero"}


@pytest.mark.parametrize("path,base,extra", ((ROUND13T, ROUND12, INIT_KEYS),
                                             (ROUND13AT, ROUND13A, set())))
def test_round13t_configs_add_exactly_the_temporal_fix(path, base, extra):
    a, b = dict(leaves(load_raw(base))), dict(leaves(load_raw(path)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    allowed = {"name", "runtime.out_dir"} | R13T_KEYS | extra
    assert changed <= allowed and R13T_KEYS <= changed, sorted(changed ^ allowed)
    raw = load_raw(path)
    assert raw["model"]["init_from"].endswith("c4_wm_v10_round12/wm_s_stage300.pt")
    cfg = load_train_config(path)
    assert cfg.model.temporal_accel_reinit is True and cfg.model.temporal_accel_off is False
    settings = TrainSettings.from_config(cfg)
    assert settings.accel_tail_on and tuple(settings.accel_tail_lanes) == (1,)
    rule = json.loads((ROOT / "docs/r13t_weight_rule.json").read_text(encoding="utf-8"))
    assert raw["loss"]["accel_tail"]["weight"] == rule["accel_weight"]
    assert raw["loss"]["accel_tail"]["jerk_weight"] == rule["jerk_weight"]
    assert raw["loss"]["accel_tail"]["lanes"] == [1]
    assert raw["model"]["temporal_accel_reinit"] is True
    model = load_train_config(path).model
    assert model.driver_param_init == "checkpoint" and model.residual_accel_reinit is False
