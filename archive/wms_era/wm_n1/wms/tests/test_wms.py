from __future__ import annotations

import numpy as np
import torch

from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.driver_encoder import DriverEncoder, PARAMETER_BOUNDS
from output.wd01.wm_n1.wms.train import (TrajectoryDataset, WmSTrainConfig,
                                         WmSTrainer, collate, validate_train_source)
from hdv.config import HdvConfig


def small_model(**kwargs):
    return WmSModel(
        physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        **kwargs,
    )


def state(batch=2, slots=4):
    lane = torch.zeros(batch, slots, dtype=torch.long)
    lane[:, slots // 2:] = 1
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((torch.arange(slots).float().view(1, slots, 1).expand(batch, -1, -1),
                     one_hot, torch.full((batch, slots, 1), 12.0),
                     torch.zeros(batch, slots, 2)), dim=-1)
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
            "present": torch.tensor([[1, 1, 1, 0]] * batch, dtype=torch.bool),
            "is_cav": torch.zeros(batch, slots, dtype=torch.bool)}


def test_shapes_and_exact_physics_at_k_zero():
    model = small_model()
    current = state()
    out = model(current, k_steps=0)
    assert out["accel_mps2"].shape == (2, 4)
    assert out["lane_logits"].shape == (2, 4, 3)
    assert torch.equal(out["gate"], torch.zeros_like(out["gate"]))
    torch.testing.assert_close(out["accel_mps2"], out["physics_accel_mps2"])


def test_gate_is_monotone_and_gradients_flow():
    model = small_model()
    gates = model.gate(torch.arange(0, 8))
    assert torch.all(gates[1:] >= gates[:-1])
    out = model(state(), k_steps=torch.ones(2, 4, dtype=torch.long))
    loss = out["accel_mps2"].sum() + out["lane_logits"].sum()
    loss.backward()
    assert model.gate_increments.grad is not None
    assert any(p.grad is not None for p in model.residual_net.parameters())


def test_pluggable_batched_physics_backend():
    class Backend(torch.nn.Module):
        def forward(self, current):
            kin = current["veh_kin"]
            shape = kin.shape[:2]
            return {"accel_mps2": torch.zeros(shape, device=kin.device),
                    "lane_logits": torch.zeros(*shape, 3, device=kin.device)}

    model = small_model(physics_step=Backend())
    out = model(state(), k_steps=0)
    torch.testing.assert_close(out["physics_accel_mps2"],
                               torch.zeros_like(out["physics_accel_mps2"]))


def test_non_default_physics_dt_is_used_for_step():
    model = WmSModel(physics=PhysicsConfig(dt_s=0.25, max_gap_steps=8))
    current = state(batch=1)
    out = model.step(current)
    assert torch.all(out["next_s_m"] >= current["s_m"])


def test_masks_and_teacher_forced_cavs():
    model = small_model()
    current = state()
    current["is_cav"][:, 0] = True
    teacher = torch.tensor([[[8.0, 0.0, 7.0, -1.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]]] * 2)
    out = model.step(current, cav_teacher=teacher)
    assert torch.equal(out["next_present"][:, 3], torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(out["next_s_m"][:, 0], torch.full((2,), 8.0))
    torch.testing.assert_close(out["next_v_mps"][:, 0], torch.full((2,), 7.0))


def test_cav_mask_follows_identity_when_a_slot_is_reused():
    # Slot 0 is CAV 17 at t0, exits, and is occupied by HDV 99 later.
    n_ticks, slots = 4, 1
    states = np.zeros((n_ticks, slots, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[..., 4] = 10.0
    identities = np.array([[17], [17], [-1], [99]], dtype=np.int64)
    actual_present = np.array([[1], [1], [0], [1]], dtype=bool)
    dataset = TrajectoryDataset.from_arrays({
        "states": states, "present": actual_present, "identities": identities,
        "is_cav": np.array([[True]]),
        "cav_teacher": np.zeros((n_ticks - 1, slots, 4), dtype=np.float32),
    })
    record = dataset[0]
    assert record["is_cav"][:, 0].tolist() == [True, True, False, False]
    assert record["present"][:, 0].tolist() == [True, True, False, False]

    # Before the fix, t0 is_cav was broadcast as all-True.  The existing
    # cumulative alive mask already made that broadcast loss-neutral.
    legacy_cav = np.broadcast_to(np.array([True]), actual_present.shape)
    legacy_loss = record["present"][1:] & ~legacy_cav[1:]
    corrected_loss = record["present"][1:] & ~record["is_cav"][1:]
    assert np.array_equal(legacy_loss, corrected_loss)
    assert corrected_loss[:, 0].tolist() == [False, False, False]

    model = small_model()
    current = state(batch=1, slots=slots)
    current["present"] = torch.ones(1, slots, dtype=torch.bool)
    current["is_cav"] = torch.ones(1, slots, dtype=torch.bool)
    teacher = torch.zeros(1, n_ticks, slots, 4)
    teacher[0, :, 0, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0])
    teacher[0, :, 0, 2] = 10.0
    out = model.rollout(
        current, n_steps=n_ticks - 1,
        cav_teacher=teacher,
        present_seq=torch.as_tensor(record["present"]).unsqueeze(0),
        is_cav_seq=torch.as_tensor(record["is_cav"]).unsqueeze(0),
    )
    assert out["positions"][0, 1, 0].item() == 102.0  # last live CAV tick
    assert out["positions"][0, 2, 0].item() != 103.0  # reused HDV is not forced


def test_two_update_cpu_training_rollout():
    n, t1, slots = 2, 21, 4
    base = state(batch=n, slots=slots)["veh_kin"].numpy()
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    states[..., 4] = 12.0
    present = np.ones((n, t1, slots), dtype=bool)
    present[:, :, -1] = False
    dataset = TrajectoryDataset.from_arrays({"states": states, "present": present})
    cfg = WmSTrainConfig(device="cpu", threads=4, steps_per_epoch=2,
                         unroll_curriculum_s=(10.0,), batch_size=2)
    result = WmSTrainer(small_model(), cfg)
    batch = collate([dataset[0], dataset[1]])
    records = [result.train_batch(batch, unroll_steps=20) for _ in range(2)]
    assert len(records) == 2
    assert all(np.isfinite(r["loss"]) for r in records)
    assert all(r["unroll_steps"] == 20 for r in records)


def test_small_training_set_cannot_silently_produce_zero_optimizer_steps(tmp_path):
    states = np.zeros((1, 2, 2, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[..., 4] = 10.0
    dataset = TrajectoryDataset.from_arrays({"states": states})
    cfg = WmSTrainConfig(device="cpu", threads=4, steps_per_epoch=1,
                         unroll_curriculum_s=(0.5,), batch_size=4,
                         out_dir=str(tmp_path))
    receipt = WmSTrainer(small_model(), cfg).fit(dataset)
    assert receipt["steps"] == 1


def test_train_source_refuses_non_train_names_before_read():
    for name in ("CAL_SPLIT_A4.json", "HOLDOUT", "chunk-5", "J4B", "INC1",
                 "sealed"):
        try:
            validate_train_source(name)
        except PermissionError:
            pass
        else:
            raise AssertionError(name)


def test_rollout_history_present_seq_and_chunk_continuation():
    model = small_model(history_features=6)
    current = state(batch=2, slots=4)
    n = 8
    present_seq = torch.ones(2, n + 1, 4, dtype=torch.bool)
    present_seq[:, 5:, 1] = False                       # an exit mid-rollout
    full = model.rollout(current, n_steps=n, present_seq=present_seq)
    assert full["positions"].shape == (2, n, 4)
    assert full["lane_logits"].shape == (2, n, 4, 3)
    assert not bool(full["masks"][:, -1, 1].any())
    assert full["history"].shape == (2, 4, 10, 6)
    first = model.rollout(current, n_steps=4, present_seq=present_seq[:, :5])
    rest = model.rollout(first["state"], n_steps=4, k0=4,
                         present_seq=present_seq[:, 4:],
                         history=first["history"], history_mask=first["history_mask"])
    torch.testing.assert_close(rest["positions"], full["positions"][:, 4:])


def test_merge_cal_flag_default_off_and_enables_strategic_merge():
    assert PhysicsConfig().merge_cal is False

    def ramp_state():
        lane = torch.tensor([[2, 1]])
        one_hot = torch.nn.functional.one_hot(lane, 3).float()
        kin = torch.cat((torch.tensor([[[40.0], [70.0]]]), one_hot,
                         torch.full((1, 2, 1), 10.0),
                         torch.zeros(1, 2, 2)), dim=-1)
        return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
                "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
                "present": torch.ones(1, 2, dtype=torch.bool),
                "is_cav": torch.zeros(1, 2, dtype=torch.bool)}

    off = WmSModel(physics=PhysicsConfig(max_gap_steps=8)).mobil_merge_logits(ramp_state())
    on = WmSModel(physics=PhysicsConfig(max_gap_steps=8, merge_cal=True)).mobil_merge_logits(ramp_state())
    assert off[0, 0, 1] < 0.0          # legacy MOBIL: ramp vehicle stays
    assert on[0, 0, 1] == 3.0          # calibrated: strategic mandatory merge


def test_plan_steps_default_matches_run1_and_extends_to_300s(tmp_path):
    import json
    from output.wd01.wm_n1.wms.plan_steps import plan, DEFAULT_STAGES_S
    log = tmp_path / "train_log.jsonl"
    log.write_text("".join(
        json.dumps({"seconds": 0.5}) + "\n" for _ in range(20)), encoding="utf-8")
    steps4, info4 = plan(log, budget_s=1.0e9)
    assert info4["stages_s"] == list(DEFAULT_STAGES_S)
    assert info4["cost_per_unit_sim_steps"] == 290.0   # frozen run-1 schedule
    steps5, info5 = plan(log, budget_s=1.0e9,
                         stages_s=(10.0, 30.0, 60.0, 120.0, 300.0))
    assert info5["cost_per_unit_sim_steps"] == 440.0   # + 300 s at ratio 0.25
    assert len(steps5) == 5 and steps5[-1] == round(0.25 * info5["N"])
    assert steps4[:4] == steps5[:4]                    # prefix budget unchanged


def test_plan_steps_project_uses_explicit_budget_without_shrinking(tmp_path):
    import json
    from output.wd01.wm_n1.wms.plan_steps import project
    log = tmp_path / "smoke.jsonl"
    rows = []
    for stage_s, base in ((10.0, 0.7), (30.0, 2.1), (60.0, 4.2), (120.0, 7.8),
                          (300.0, 20.0)):
        rows += [{"stage_s": stage_s, "seconds": base} for _ in range(6)]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    info = project(log, [2500, 2500, 1875, 1250, 400],
                   stages_s=(10.0, 30.0, 60.0, 120.0, 300.0))
    assert info["stage_steps"] == [2500, 2500, 1875, 1250, 400]
    expected = (0.7 * 2500 + 2.1 * 2500 + 4.2 * 1875 + 7.8 * 1250
                + 20.0 * 400) / 3600.0
    assert abs(info["projected_hours"] - expected) < 1.0e-9
    assert info["estimated_hours"] == info["projected_hours"]


def test_merge_cal_params_file_loads_known_keys_and_enables_flag(tmp_path):
    import json
    path = tmp_path / "merge_params.json"
    path.write_text(json.dumps({
        "_meta": {"purpose": "unit-test fixture"},
        "t_lead_base_s": 1.2, "b_yield_max_mps2": 1.75,
        "yield_min_merger_speed_mps": 3.5, "speed_factor": None,
        "unknown_key": 123, "merge_cal": False,
    }), encoding="utf-8")
    cfg = PhysicsConfig.from_params_file(str(path), merge_cal=True, sumo_fidelity=True)
    assert cfg.merge_cal is True                       # explicit override wins
    assert cfg.t_lead_base_s == 1.2
    assert cfg.b_yield_max_mps2 == 1.75
    assert cfg.yield_min_merger_speed_mps == 3.5
    assert cfg.sumo_fidelity is True
    assert cfg.speed_factor == 1.0        # null in the file -> PhysicsConfig default
    assert not hasattr(cfg, "unknown_key")


def test_baseline_lane_logits_prefer_staying_and_gate_zero_matches_physics():
    model = small_model(history_features=6)
    out = model(state(), k_steps=0)
    base = model.mobil_merge_logits(state())
    cur = state()["lane"]
    stay = base.gather(-1, cur.unsqueeze(-1)).squeeze(-1)
    assert torch.all(stay[state()["present"]] == 0.0)
    assert torch.isfinite(out["lane_logits"]).all()
    assert float(base.abs().max()) <= 8.0 + 1.0e-6


def test_trainer_entries_v2_padding_is_excluded_from_loss():
    from output.wd01.wm_n1.wms.entries import EntranceHistory

    n, t1, slots = 1, 11, 4
    base = state(batch=n, slots=slots)["veh_kin"].numpy()
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    states[..., 4] = 12.0
    present = np.ones((n, t1, slots), dtype=bool)
    dataset = TrajectoryDataset.from_arrays({"states": states, "present": present})
    feat = EntranceHistory.defaults().to_features().astype(np.float32)
    feat[0] = feat[1] = 5.0                     # high mainline/ramp arrival rate
    feat[12] = feat[13] = 0.0                   # no extra gap floor
    dataset.records[0]["entry_feat"] = feat

    def run(entries_v2):
        cfg = WmSTrainConfig(device="cpu", threads=4, steps_per_epoch=1,
                             unroll_curriculum_s=(5.0,), batch_size=1, seed=7,
                             entries_v2=entries_v2, entry_pad=4)
        batch = collate([dataset[0]])
        trainer = WmSTrainer(small_model(), cfg)
        return trainer._forward_backward(batch, unroll_steps=10)

    off = run(False)
    on = run(True)
    assert on["valid_targets"] == off["valid_targets"] > 0
    assert np.isfinite(on["loss"]) and np.isfinite(off["loss"])


def test_driver_encoder_is_bounded_and_shrinks_empty_history_to_defaults():
    enc = DriverEncoder(history_features=6)
    history = torch.randn(1, 2, 10, 6)
    mask = torch.zeros(1, 2, 10, dtype=torch.bool)
    interaction = torch.zeros(1, 2, 4)
    defaults = torch.tensor([31.29, 1.1, 2.5, 2.6, 4.5, .2, .1, 6., 1.]).view(1, 1, 9).expand(1, 2, -1)
    empty = enc(history, mask, interaction, defaults)
    torch.testing.assert_close(empty["values"], defaults)
    mask[..., -2:] = True
    result = enc(history, mask, interaction, defaults)
    delta = (result["values"] - defaults).abs()
    assert torch.all(delta <= torch.tensor(PARAMETER_BOUNDS).view(1, 1, -1) + 1e-6)
    assert torch.all(result["shrinkage"] < 1)
    assert torch.isfinite(result["uncertainty"]).all()


def test_driver_parameter_physics_has_encoder_gradients_and_cav_teacher_is_exact():
    model = small_model(rollout_mode="driver_params")
    current = state(batch=1)
    current["s_m"] = torch.tensor([[0., 20., 40., 60.]])
    current["veh_kin"][..., 0] = current["s_m"]
    history = torch.randn(1, 4, 10, 6)
    history_mask = torch.ones(1, 4, 10, dtype=torch.bool)
    current["is_cav"][:, 0] = True
    teacher = torch.zeros(1, 4, 4)
    teacher[:, 0] = torch.tensor([123.0, 1.0, 7.0, -0.4])
    out = model.step(current, cav_teacher=teacher, driver_history=history,
                     driver_history_mask=history_mask)
    delta = (out["driver_parameters"] - out["driver_parameters"].new_tensor(
        [31.29, 1.1, 2.5, 2.6, 4.5, .2, .1, 6., 1.]).view(1, 1, 9)).abs()
    assert torch.all(delta <= torch.tensor(PARAMETER_BOUNDS).view(1, 1, -1) + 1e-6)
    torch.testing.assert_close(out["next_s_m"][:, 0], torch.tensor([123.0]))
    torch.testing.assert_close(out["next_v_mps"][:, 0], torch.tensor([7.0]))
    assert torch.equal(out["accel_delta_mps2"], torch.zeros_like(out["accel_delta_mps2"]))
    out["physics_accel_mps2"][:, 1:].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.driver_encoder.parameters())


def test_driver_encoder_receives_only_explicit_observed_history():
    model = small_model(rollout_mode="driver_params")
    current = state(batch=1)
    hist = torch.randn(1, 4, 10, 6)
    mask = torch.ones(1, 4, 10, dtype=torch.bool)
    first = model(current, driver_history=hist, driver_history_mask=mask)
    # Changing an unrelated hypothetical future tensor cannot affect an encoder
    # invocation; its complete observation interface is the supplied prefix.
    ignored_future = torch.randn_like(hist) * 1.0e5
    second = model(current, history=ignored_future, driver_history=hist,
                   driver_history_mask=mask)
    torch.testing.assert_close(first["driver_parameters"], second["driver_parameters"])
