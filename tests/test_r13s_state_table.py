"""R13S: S-FACTORIAL state-conditional mean temporal emission (``model.temporal_accel_state_table``).

Synthetic CPU data only.  Bins/backoff identical to the S-FACTORIAL numpy definition (inlined copy of
fit_table.bins/predict), default off (no tensors, keys omitted), all-zero table = ``temporal_accel_off`` bit for
bit, emission = lookup of the current state + k, gradient only into the table cells, init switch, checkpoint
round trip and the round-13S config.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.config.load import load_raw, load_train_config
from wms.model import temporal_state_table as tst
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

from test_r10b_courtesy import sequence, state_at_step
from test_r11_courtesy import leaves, model_v

ROOT = Path(__file__).resolve().parents[1]
ROUND12 = ROOT / "configs/c4_wm_v10round12.toml"
ROUND13F = ROOT / "configs/c4_wm_v10round13f.toml"
ROUND13S = ROOT / "configs/c4_wm_v10round13s.toml"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def factorial_predict(tab, x):
    """Inlined S-FACTORIAL fit_table.bins + predict (numpy reference)."""
    lane, v, gap, clos, s, k = (x[:, i] for i in range(6))
    b = (np.clip(lane, 0, 2).astype(int), np.searchsorted(tst.V_B, v, side="left"),
         np.where(gap >= 999, 7, np.searchsorted(tst.GAP_B, gap, side="left")),
         np.searchsorted(tst.CL_B, clos, side="left"), np.searchsorted(tst.S_B, s, side="left"),
         np.searchsorted(tst.K_B, k, side="left"))
    pred = np.full(len(x), np.nan)
    for lv, keep in enumerate(tst.LEVELS):
        idx = tuple(b[i] for i in keep)
        m, n = tab[f"mean{lv}"][idx], tab[f"n{lv}"][idx]
        use = np.isnan(pred) & ((n >= 30) | (lv == 4))
        pred[use] = m[use]
    return pred


def random_rows(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.stack([rng.integers(0, 3, n), rng.uniform(0, 32, n),
                  np.where(rng.random(n) < 0.2, 1e3, rng.uniform(0, 100, n)), rng.normal(0, 3, n),
                  rng.uniform(-450, 250, n), rng.integers(0, 150, n)], -1).astype(np.float64)
    y = 0.5 * x[:, 0] - 0.05 * x[:, 1] + rng.normal(0, 1, n)
    return x, y


def test_bins_fit_and_backoff_match_the_factorial_definition():
    x, y = random_rows()
    tab = tst.fit(x, y)
    tst.check_table(tab)
    xt = torch.as_tensor(x, dtype=torch.float32)
    bins = tuple(t.long() for t in tst.bin_index(*(xt[:, i] for i in range(6))))
    got = tst.lookup([torch.as_tensor(tab[f"mean{lv}"]) for lv in range(5)],
                     [torch.as_tensor(tab[f"n{lv}"]) for lv in range(5)], bins).numpy()
    np.testing.assert_allclose(got, factorial_predict(tab, x), rtol=0, atol=1e-6)


def test_features_match_the_factorial_state_feats():
    kin = torch.zeros(1, 4, 7)
    kin[..., 0] = torch.tensor([0., 30., 12., 50.])
    kin[..., 4] = torch.tensor([10., 8., 9., 7.])
    kin[..., 6] = 5.
    kin[..., 1:4] = torch.nn.functional.one_hot(torch.tensor([[1, 1, 1, 0]]), 3).float()
    lane, v, gap, clos, s, k = tst.state_features(kin, torch.ones(1, 4, dtype=torch.bool), torch.tensor(7))
    assert lane.tolist() == [[1, 1, 1, 0]]
    assert gap.tolist() == [[7.0, 1e3, 13.0, 1e3]]                    # bumper gaps; no leader -> 1e3
    assert clos.tolist() == [[1.0, 0.0, 1.0, 0.0]] and k.tolist() == [[7.0] * 4]


def test_default_off_and_exclusive():
    assert WmSModel(ModelSpec()).temporal_state_mean is None
    assert not any(k.startswith("temporal_state") for k in WmSModel(ModelSpec()).state_dict())
    flat = load_train_config(ROUND12).to_legacy_flat()
    assert not {"temporal_accel_state_table", "temporal_state_table_init"} & set(flat)
    with pytest.raises(ValueError):
        ModelSpec(temporal_accel_state_table=True, temporal_accel_lane_bias=True)
    with pytest.raises(ValueError):
        load_train_config(ROUND12, ["model.temporal_state_table_init=x.npz"])


def table_model(fill=None):
    model = model_v(randomize=True, temporal_accel_state_table=True)
    if fill is not None:
        with torch.no_grad():
            for lv in range(5):
                model.temporal_state_mean[lv].copy_(fill(lv))
                getattr(model, f"temporal_state_n{lv}").fill_(100.0 if lv < 2 else 0.0)
    return model


def test_zero_table_equals_off_and_emission_is_the_lookup():
    off = model_v(randomize=True, temporal_accel_off=True)
    tab = table_model()
    tab.load_state_dict({**off.state_dict(), **{k: v for k, v in tab.state_dict().items()
                                                if k.startswith("temporal_state")}})
    batch = sequence(cav=False)
    with torch.no_grad():
        for t in range(3):
            state = state_at_step(batch, t)
            a, b = off(state, k_steps=1), tab(state, k_steps=1)
            for key in a:
                if torch.is_tensor(a[key]):
                    assert torch.equal(a[key], b[key]), key
        torch.manual_seed(3)
        for lv in range(5):
            tab.temporal_state_mean[lv].copy_(torch.randn(tst.level_shape(lv)))
            getattr(tab, f"temporal_state_n{lv}").fill_(100.0 if lv == 1 else 0.0)
        tab.diag_emit_temporal = True
        state = state_at_step(batch, 1)
        out = tab(state, k_steps=5)
        kin = state["veh_kin"]
        bins = tst.bin_index(*tst.state_features(kin, state["present"], torch.tensor(5)))
        want = tab.temporal_state_mean[1][tuple(bins[i] for i in tst.LEVELS[1])]   # level 1 is the finest with n>=30
        assert torch.equal(out["temporal_accel_pregate_mps2"], want)


def test_gradient_reaches_only_the_table():
    model = table_model(fill=lambda lv: torch.full(tst.level_shape(lv), 0.3)).train()
    cfg = TrainSettings(truncation_steps=8)
    UnrollRunner(model, cfg, build_loss(cfg), projection=None,
                 device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=True)
    g0 = model.temporal_state_mean[0].grad
    assert g0 is not None and float(g0.abs().sum()) > 0
    assert int((g0 != 0).sum()) < g0.numel()                                # only visited cells
    g = model.temporal[-1].weight.grad
    assert g is None or float(g[0].abs().sum()) == 0.0


def test_init_switch_and_checkpoint_round_trip(tmp_path):
    from wms.train.trainer import WmSTrainer
    x, y = random_rows()
    tab = tst.fit(x, y)
    np.savez(tmp_path / "table.npz", **tab)

    class Stub:
        pass
    stub = Stub()
    stub.model = WmSModel(ModelSpec(temporal_accel_state_table=True))
    stub.settings = TrainSettings(temporal_accel_state_table=True,
                                  temporal_state_table_init=str(tmp_path / "table.npz"))
    report = WmSTrainer._r12_reinit(stub)
    assert report["temporal_state_table_init"]["cells_used_level0"] == int((tab["n0"] >= 30).sum())
    for lv in range(5):
        assert torch.equal(stub.model.temporal_state_mean[lv].detach(), torch.as_tensor(tab[f"mean{lv}"]))
    (tmp_path / "physics.json").write_text(json.dumps(asdict(PhysicsConfig())), encoding="utf-8")
    flat = {k: v for k, v in asdict(stub.model.spec).items() if v is not None}
    save_checkpoint(tmp_path / "wm_s.pt", stub.model, flat)
    loaded = load_checkpoint(tmp_path / "wm_s.pt", strict=True).model
    for lv in range(5):
        assert torch.equal(loaded.temporal_state_mean[lv].detach(), stub.model.temporal_state_mean[lv].detach())
        assert torch.equal(getattr(loaded, f"temporal_state_n{lv}"), getattr(stub.model, f"temporal_state_n{lv}"))


def test_round13s_config_is_round13f_with_the_state_mean():
    a, b = dict(leaves(load_raw(ROUND13F))), dict(leaves(load_raw(ROUND13S)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    expected = {"name", "runtime.out_dir", "model.temporal_accel_off", "model.temporal_accel_state_table",
                "model.temporal_state_table_init"}
    assert changed == expected, sorted(changed ^ expected)
    cfg = load_train_config(ROUND13S)
    assert cfg.model.temporal_accel_state_table is True and cfg.model.temporal_accel_off is False
    path = ROOT / cfg.model.temporal_state_table_init
    tst.check_table(dict(np.load(path)))
    summary = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    import hashlib
    assert summary["npz_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert TrainSettings.from_config(cfg).temporal_accel_state_table is True
