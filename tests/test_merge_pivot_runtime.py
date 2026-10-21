"""Native round-9 rollout/checkpoint contracts, independent of local datasets."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch

from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model import merge_pivot as pivot
from wms.physics.config import PhysicsConfig
from wms.physics.state import state_kin


def model(mode="first_passage", width=1.0):
    torch.manual_seed(19)
    return WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                              lane_decision="sample", merge_decision=mode,
                              merge_st_width=width), physics=PhysicsConfig()).eval()


def state(positions=(30.0, 49.0, 10.0), speeds=(5.0, 5.0, 5.0), lanes=(2, 1, 1)):
    s = torch.tensor([positions])
    return {"s_m": s, "v_mps": torch.tensor([speeds]), "a_mps2": torch.zeros_like(s),
            "length_m": torch.full_like(s, 5.0), "lane": torch.tensor([lanes]),
            "present": torch.ones_like(s, dtype=torch.bool),
            "is_cav": torch.zeros_like(s, dtype=torch.bool)}


def fixed_forward(s, **kwargs):
    # Stay for all legacy heads; isolate the actual native pivot step.
    lane = s["lane"]
    logits = torch.full((*lane.shape, 3), -20.0)
    logits.scatter_(-1, lane.unsqueeze(-1), 20.0)
    return {"accel_mps2": torch.zeros_like(s["s_m"]), "lane_logits": logits,
            "lane_feasible": torch.ones_like(logits, dtype=torch.bool),
            "hdv_mask": s["present"] & ~s["is_cav"],
            "driver_parameters": torch.ones((*lane.shape, 9)),
            "lane_age": torch.zeros_like(lane), "lane_v5_acc": kwargs.get("lane_v5_acc"),
            "lane_sample": torch.zeros_like(s["present"])}


def test_round8_load_into_pivot_uses_loaded_parameters_and_preserves_every_tensor(tmp_path):
    old = model("secure_gap")
    with torch.no_grad():
        old.merge_core.s0_raw.fill_(0.7)
        old.merge_core.t_raw.fill_(0.3)
        old.merge_core.b_raw.fill_(1.1)
        old.merge_core.alpha_log_sigma_raw.fill_(-1.5)
    file = tmp_path / "round8.pt"
    save_checkpoint(file, old, {**old.spec.__dict__, "physics": old.physics.__dict__})
    new = load_checkpoint(file, model_config={"merge_decision": "first_passage",
                                              "merge_st_width": 0.4}).model
    for key, value in old.state_dict().items():
        assert torch.equal(new.state_dict()[key], value), key
    p, old_p = new.merge_pivot.params(), old.merge_core.params()
    for side in ("f", "l"):
        for name, old_name in (("s", "s0"), ("T", "T"), ("b", "b")):
            assert torch.allclose(p[f"{name}_{side}"], old_p[old_name], atol=1e-7, rtol=1e-6)
    assert p["tau"].item() == pytest.approx(old.merge_core.alpha_scale().item()
                                             * (3.0 ** 0.5) / torch.pi, rel=1e-6)
    assert not any(p.requires_grad for p in new.merge_core.parameters())
    file2 = tmp_path / "pivot.pt"
    save_checkpoint(file2, new, {**new.spec.__dict__, "physics": new.physics.__dict__})
    loaded = load_checkpoint(file2).model
    assert loaded.merge_st_width == 0.4
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in new.state_dict().items())


def test_default_off_rng_state_and_round8_outputs_identical():
    a = model("secure_gap")
    rng_a = torch.random.get_rng_state().clone()
    b = WmSModel(replace(a.spec, merge_st_width=0.01), physics=a.physics).eval()
    b.load_state_dict(a.state_dict())
    assert a.merge_pivot is b.merge_pivot is None
    assert not any(k.startswith("merge_pivot.") for k in a.state_dict())
    torch.random.set_rng_state(rng_a)
    x = a.step(state())
    torch.random.set_rng_state(rng_a)
    y = b.step(state())
    for key, value in x.items():
        if torch.is_tensor(value):
            assert torch.equal(value, y[key]), key


def test_step_order_single_motion_and_decision_and_shared_latent_parity():
    m, s = model(), state()
    with patch.object(m, "forward", side_effect=fixed_forward) as forward, patch.object(
            m.merge_pivot, "forward", wraps=m.merge_pivot.forward) as boundary:
        out = m.step(s, lane_uniform=torch.full_like(s["s_m"], 0.5))
    assert forward.call_count == boundary.call_count == 1
    moved = pivot.advance_longitudinal(state_kin(s), torch.zeros_like(s["s_m"]), 0.5)
    geo = pivot.geometry(moved, s["present"], m.physics)
    c = m.merge_pivot(geo)
    ref = pivot.update_state(c, m.merge_pivot.params()["tau"], geo["ramp"], s["present"],
                             uniform=torch.full_like(s["s_m"], 0.5))
    assert torch.equal(out["next_s_m"], moved[..., 0])
    assert torch.equal(out["merge_pivot_c"], c)
    assert torch.equal(out["merge_pivot_cross"], ref["cross"])
    assert torch.allclose(out["merge_pivot_cdf"], torch.sigmoid(
        ref["maximum"] / m.merge_pivot.params()["tau"]), atol=1e-6, rtol=0)
    assert torch.allclose(out["merge_pivot_event_mass"], out["merge_pivot_cdf"], atol=1e-6)
    assert "merge_core_alpha" not in out


def test_persistent_latent_and_hard_timing_across_nonzero_chunks():
    m, s = model(), state(positions=(30.0, 37.0, 23.0))
    uniforms = torch.full((1, 5, 3), 0.01)
    with patch.object(m, "forward", side_effect=fixed_forward):
        whole = m.rollout(s, n_steps=5, lane_uniform=uniforms)
        first = m.rollout(s, n_steps=2, lane_uniform=uniforms[:, :2])
        second = m.rollout(first["state"], n_steps=3, k0=2,
                           history=first["history"], history_mask=first["history_mask"],
                           lane_uniform=torch.full((1, 3, 3), 0.99))
    assert torch.equal(first["state"]["merge_pivot_z"], second["state"]["merge_pivot_z"])
    for key in ("positions", "speeds", "lanes", "merge_pivot_cdf", "merge_pivot_event_mass"):
        assert torch.equal(whole[key], torch.cat((first[key], second[key]), 1)), key


def test_native_cav_callback_once_and_no_recorded_future_motion_input():
    m, s = model(), state()
    s["is_cav"][0, 1] = True
    teacher = torch.tensor([[[999.0, 0.0, 99.0, 0.0]]]).expand(1, 3, 4)
    calls = []
    def callback(current):
        calls.append(1)
        return torch.stack((current["s_m"] + 1.0, current["lane"],
                            current["v_mps"], current["a_mps2"]), -1)
    with patch.object(m, "forward", side_effect=fixed_forward):
        out = m.step(s, cav_teacher=teacher, cav_controller=callback,
                     lane_uniform=torch.full_like(s["s_m"], 0.5))
    assert len(calls) == 1
    assert out["next_s_m"][0, 1] == s["s_m"][0, 1] + 1.0
    moved = state_kin(s).clone()
    moved[..., 0] = out["next_s_m"]
    assert torch.equal(out["merge_pivot_c"], m.merge_pivot(pivot.geometry(moved, s["present"], m.physics)))


@pytest.mark.parametrize("positions", [(30., 30., 10.), (30., 32., 10.)])
def test_native_exact_overlap_veto(positions):
    m, s = model(), state(positions=positions)
    with patch.object(m, "forward", side_effect=fixed_forward):
        out = m.step(s, lane_uniform=torch.full_like(s["s_m"], 0.9999))
    assert not out["merge_pivot_eligible"][0, 0]
    assert out["next_lane"][0, 0] == 2


def test_native_simultaneous_gap_conflict_and_st_backward_width():
    m, s = model(), state(positions=(30., 31., 100.), lanes=(2, 2, 1))
    with patch.object(m, "forward", side_effect=fixed_forward):
        out = m.step(s, lane_uniform=torch.full_like(s["s_m"], 0.5))
    assert out["merge_pivot_cross"][0, :2].tolist() == [True, False]
    assert out["next_lane"].tolist() == [[1, 2, 1]]
    assert out["merge_pivot_committed"].tolist() == [[True, False, False]]
    assert out["merge_pivot_st"][0, 1] == 0
    out["merge_pivot_st"].sum().backward()
    assert m.merge_pivot.s_l_raw.grad is not None
    assert torch.isfinite(m.merge_pivot.s_l_raw.grad)


def test_merge_reservation_is_independent_of_competing_willingness_and_mainline_draws():
    m, s = model(), state(positions=(30., 31., 48.), lanes=(2, 0, 1))
    with torch.no_grad():
        m.merge_pivot.tau_raw.fill_(1. + torch.log(-torch.expm1(torch.tensor(-1.))))
    def incoming(current, **kwargs):
        out = fixed_forward(current, **kwargs)
        out["lane_logits"][0, 1] = torch.tensor([-20., 20., -20.])
        return out
    with patch.object(m, "forward", side_effect=incoming):
        wait = m.step(s, lane_uniform=torch.full_like(s["s_m"], .01))
        merge = m.step(s, lane_uniform=torch.full_like(s["s_m"], .99))
    assert not wait["merge_pivot_cross"][0, 0] and merge["merge_pivot_cross"][0, 0]
    assert wait["next_lane"].tolist() == [[2, 0, 1]]
    assert merge["next_lane"].tolist() == [[1, 0, 1]]
    for key in ("merge_pivot_c", "merge_pivot_eligible", "merge_pivot_cdf", "merge_pivot_event_mass"):
        assert torch.equal(wait[key], merge[key]), key
    assert torch.equal(merge["merge_pivot_cross"], merge["merge_pivot_committed"])


def test_current_action_cav_insertion_veto_uses_prechange_neighbor_features():
    m, s = model(), state(positions=(30., 31., 48.), lanes=(2, 0, 1))
    s["is_cav"][0, 1] = True
    def callback(current):
        lanes = current["lane"].clone()
        lanes[0, 1] = 1
        return torch.stack((current["s_m"], lanes, current["v_mps"], current["a_mps2"]), -1)
    with patch.object(m, "forward", side_effect=fixed_forward):
        out = m.step(s, cav_controller=callback,
                     lane_uniform=torch.full_like(s["s_m"], .99))
    assert not out["merge_pivot_eligible"][0, 0]
    assert out["next_lane"].tolist() == [[2, 1, 1]]
    geo = m.pivot_decision_geometry(s, out["accel_mps2"], next_s=out["next_s_m"],
                                    next_v=out["next_v_mps"], fixed_next_lane=out["next_lane"])
    assert geo["gap_l"][0, 0] == 13.0  # pre-change leader at 48+2.5, not incoming CAV at31
    assert not geo["has_f"][0, 0]


def test_no_command_cav_fallback_holds_lane_and_ignores_future_teacher():
    m, s = model(), state(positions=(30., 31., 48.), lanes=(2, 0, 1))
    s["is_cav"][0, 1] = True
    def future_change(current, **kwargs):
        out = fixed_forward(current, **kwargs)
        out["lane_logits"][0, 1] = torch.tensor([-20., 20., -20.])
        return out
    with patch.object(m, "forward", side_effect=future_change):
        a = m.step(s, cav_teacher=torch.full((1, 3, 4), 999.),
                   lane_uniform=torch.full((1, 3), .5))
        b = m.step(s, cav_teacher=torch.full((1, 3, 4), -999.),
                   lane_uniform=torch.full((1, 3), .5))
    assert a["next_lane"][0, 1] == 0
    assert torch.equal(a["next_s_m"], s["s_m"] + .5 * s["v_mps"])
    for key in ("next_lane", "next_s_m", "next_v_mps", "merge_pivot_c"):
        assert torch.equal(a[key], b[key])


def test_hard_lane_transition_has_a_downstream_residual_adjoint():
    m = WmSModel(replace(model().spec, rollout_mode="hybrid")).eval()
    s = state(positions=(30., 38., 20.))
    # A lane-dependent residual probe isolates the path without relying on random weights.
    original = m._residual_prediction
    def probe(kin, present, conditioning, courtesy=None):
        result = original(kin, present, conditioning, courtesy)
        result["accel_delta_mps2"] = kin[..., 2]
        return result
    # Native hard weights must carry a nonzero ST adjoint into the next call.
    out = m.step(s, lane_uniform=torch.full_like(s["s_m"], 0.5))
    weights = out["merge_pivot_lane_weights"]
    assert torch.equal(weights.detach(), torch.nn.functional.one_hot(out["next_lane"], 3).float())
    current = {**s, "s_m": out["next_s_m"], "v_mps": out["next_v_mps"],
               "a_mps2": out["accel_mps2"], "lane": out["next_lane"],
               "merge_pivot_lane_weights": weights}
    with patch.object(m, "_residual_prediction", side_effect=probe):
        following = m.forward(current, k_steps=2)
    following["raw_residual_mps2"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.merge_pivot.parameters())


def test_draft_config_four_synthetic_updates_freeze_every_legacy_tensor():
    from wms.config.load import load_train_config
    from wms.train.settings import TrainSettings
    from wms.train.trainer import build_optimizer

    cfg = load_train_config("configs/c4_wm_v10round9.toml")
    settings = TrainSettings.from_config(cfg)
    m = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat())).eval()
    optimizer, report = build_optimizer(m, lr=5e-5, weight_decay=0.,
                                        freeze_prefixes=settings.freeze_prefixes,
                                        lr_scale=settings.lr_scale)
    assert report["trainable_tensors"] == report["trainable_parameters"] == 8
    assert not report["unmatched_freeze_prefixes"]
    assert cfg.model.merge_negotiation is False and cfg.loss.merge_core.yield_weight == 0
    frozen = {key: val.clone() for key, val in m.state_dict().items()
              if not key.startswith("merge_pivot.")}
    scenes = [state(speeds=(8., 4., 12.)),
              state(positions=(30., 55., 0.), speeds=(8., 4., 6.))]
    geometries = [pivot.geometry(state_kin(s), s["present"], m.physics) for s in scenes]
    before = {name: p.clone() for name, p in m.merge_pivot.named_parameters()}
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(-pivot.hybrid_log_masses(m.merge_pivot(g)[:, :1],
                                            m.merge_pivot.params()["tau"],
                                            m.merge_pivot.params()["w"])[1].sum() for g in geometries)
        assert torch.isfinite(loss)
        loss.backward()
        for name, p in m.merge_pivot.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad) and p.grad.abs() > 0, name
        optimizer.step()
    for key, value in frozen.items():
        assert torch.equal(value, m.state_dict()[key]), key
    for name, p in m.merge_pivot.named_parameters():
        assert torch.isfinite(p) and not torch.equal(p, before[name]), name


@pytest.fixture
def launcher_checkpoint_dir(tmp_path):
    # The launcher intentionally refuses checkpoints outside its own worktree.
    # Prefer pytest's temp tree when local, with a local fallback for default pytest runs.
    root = Path(__file__).resolve().parents[1]
    parent = tmp_path if tmp_path.resolve().is_relative_to(root) else root
    with TemporaryDirectory(prefix="round9-launcher-", dir=parent) as directory:
        yield Path(directory)


def test_draft_launcher_uses_round8_stage300_and_preserves_frozen_recipe(
        launcher_checkpoint_dir, monkeypatch):
    from tools.launch_round9 import main
    import wms.cli.train as cli

    old = model("secure_gap")
    file = launcher_checkpoint_dir / "round8_stage300.pt"
    save_checkpoint(file, old, old.spec.__dict__, extra={"stage_s": 300.})
    seen = []
    monkeypatch.setattr(cli, "execute", lambda cfg, mode: seen.append((cfg, mode)) or 0)
    assert main(["--init-from", str(file)]) == 0
    cfg, mode = seen[0]
    assert cfg.model.init_from == str(file.resolve())
    assert cfg.model.merge_decision == "first_passage" and mode == "preflight"
    assert cfg.loss.merge_core.z_tf == 395.2 and cfg.loss.merge_core.z_on == 2.15


@pytest.mark.parametrize("stage", [120., 240.])
def test_draft_launcher_rejects_non_stage300_init(launcher_checkpoint_dir, stage, capsys):
    from tools.launch_round9 import main

    old = model("secure_gap")
    file = launcher_checkpoint_dir / "wrong_stage.pt"
    save_checkpoint(file, old, old.spec.__dict__, extra={"stage_s": stage})
    with pytest.raises(SystemExit) as exc:
        main(["--init-from", str(file)])
    assert exc.value.code == 2
    assert "--init-from must record stage_s=300" in capsys.readouterr().err
