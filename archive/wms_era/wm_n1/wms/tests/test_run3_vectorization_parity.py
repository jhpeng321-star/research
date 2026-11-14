"""Run-3 CPU fast-path parity: vectorised projection / entries vs reference.

These tests pin the run-3 performance work to identical numerics:

* ``FeasibilityProjection._project_gaps`` (vectorised segmented prefix-min)
  matches the original per-vehicle loop to float64 exactness (<= 1e-6), and on
  the production float32 dtype to a few ulps with identical gap/clamp counts.
* ``TorchEntryProcessV2.insert`` materialisation fast-path (skip the per-step
  host copies when no arrival is due and none is pending) is bit-identical to
  the forced-materialisation path over a long rollout.
* A full ``WmSTrainer`` step with every run-2/run-3 flag on gives the same loss
  whether the vectorised projection or the reference loop is installed.

Synthetic CPU-only, no real packs, no SUMO, no GPU.
"""
from __future__ import annotations

import numpy as np
import torch

from hdv.config import HdvConfig
from output.wd01.wm_n1.wms.entries import (EntranceHistory, EntryConfig,
                                           TorchEntryProcessV2,
                                           build_entrance_history)
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.stability import FeasibilityProjection
from output.wd01.wm_n1.wms.train import (TrajectoryDataset, WmSTrainConfig,
                                         WmSTrainer, collate)


def _random_case(b, k, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(b, k, generator=g, dtype=dtype) * 200.0
    length = torch.full((b, k), 5.0, dtype=dtype)
    lane = torch.randint(0, 3, (b, k), generator=torch.Generator().manual_seed(seed + 1))
    present = torch.rand(b, k, generator=torch.Generator().manual_seed(seed + 2)) > 0.3
    projectable = present & (
        torch.rand(b, k, generator=torch.Generator().manual_seed(seed + 3)) > 0.15)
    return s, length, lane, present, projectable


def test_projection_vectorised_is_algorithmically_exact_in_float64():
    worst = 0.0
    for seed in range(300):
        b = int(torch.randint(1, 6, (1,), generator=torch.Generator().manual_seed(seed)).item())
        k = int(torch.randint(2, 50, (1,), generator=torch.Generator().manual_seed(seed + 99)).item())
        s, length, lane, present, projectable = _random_case(b, k, torch.float64, seed)
        ref, g_ref, c_ref = FeasibilityProjection._project_gaps_reference(
            s, length, lane, present, projectable, 2.0)
        got, g_got, c_got = FeasibilityProjection._project_gaps(
            s, length, lane, present, projectable, 2.0, n_lanes=3)
        worst = max(worst, float((ref - got).abs().max()))
        assert g_ref == g_got
        assert c_ref == c_got
    assert worst <= 1.0e-6, worst


def test_projection_float32_agrees_to_rounding_and_counts():
    worst = 0.0
    for seed in range(300):
        b = int(torch.randint(1, 6, (1,), generator=torch.Generator().manual_seed(seed)).item())
        k = int(torch.randint(2, 50, (1,), generator=torch.Generator().manual_seed(seed + 7)).item())
        s, length, lane, present, projectable = _random_case(b, k, torch.float32, seed)
        ref, g_ref, c_ref = FeasibilityProjection._project_gaps_reference(
            s, length, lane, present, projectable, 2.0)
        got, g_got, c_got = FeasibilityProjection._project_gaps(
            s, length, lane, present, projectable, 2.0, n_lanes=3)
        worst = max(worst, float((ref - got).abs().max()))
        assert g_ref == g_got
        assert c_ref == c_got
    # float32 ulp at |s| ~ 200 m is ~1.5e-5; allow a small multiple.
    assert worst <= 2.0e-4, worst


def test_projection_gradients_flow_through_vectorised_path():
    s, _, lane, present, projectable = _random_case(1, 12, torch.float32, 3)
    s = s.clone().requires_grad_(True)
    length = torch.full_like(s, 5.0)
    out, _, _ = FeasibilityProjection._project_gaps(
        s, length, lane, present, projectable, 2.0, n_lanes=3)
    out.sum().backward()
    assert s.grad is not None and bool(torch.isfinite(s.grad).all())


def _entry_process(seed):
    return TorchEntryProcessV2(
        history=build_entrance_history(counter=None,
                                       defaults=EntranceHistory.defaults()),
        seed=seed, cfg=EntryConfig(rate_scale=50.0), batch=1)


def _entry_state(slots=32, present=4):
    lane = torch.zeros(1, slots, dtype=torch.long)
    lane[0, 0::3] = 1
    lane[0, 2::3] = 2
    kin = torch.zeros(1, slots, 7)
    kin[..., 0] = -2000.0 + torch.arange(slots).float() * 30.0
    kin[..., 1:4] = torch.nn.functional.one_hot(lane.clamp_min(0), 3).float()
    kin[..., 4] = 20.0
    kin[..., 6] = 5.0
    pres = torch.zeros(1, slots, dtype=torch.bool)
    pres[0, :present] = True
    return {"veh_kin": kin, "s_m": kin[..., 0].clone(), "v_mps": kin[..., 4].clone(),
            "a_mps2": kin[..., 5].clone(), "length_m": kin[..., 6].clone(),
            "lane": lane, "present": pres,
            "is_cav": torch.zeros(1, slots, dtype=torch.bool)}


def test_entries_fastpath_matches_forced_materialisation():
    def rollout(force):
        TorchEntryProcessV2._FORCE_MATERIALISE = force
        try:
            ep = _entry_process(seed=13)
            w = _entry_state()
            masks, states, summary = [], [], None
            for _ in range(60):
                masks.append(ep.insert(w).clone())
                states.append((w["s_m"].clone(), w["present"].clone(),
                               w["is_cav"].clone()))
            summary = ep.summary()
            return masks, states, summary
        finally:
            TorchEntryProcessV2._FORCE_MATERIALISE = False

    masks_fast, states_fast, summary_fast = rollout(False)
    masks_forced, states_forced, summary_forced = rollout(True)
    for a, b in zip(masks_fast, masks_forced):
        assert torch.equal(a, b)
    for (s1, p1, c1), (s2, p2, c2) in zip(states_fast, states_forced):
        assert torch.equal(s1, s2)
        assert torch.equal(p1, p2)
        assert torch.equal(c1, c2)
    assert summary_fast == summary_forced
    assert summary_fast["inserted"] > 0


def _tiny_arrays(n=2, t=41, k=8):
    rng = np.random.RandomState(0)
    lane = rng.randint(0, 3, size=(n, t, k))
    one_hot = np.eye(3, dtype=np.float32)[lane]
    s = np.cumsum(rng.uniform(5.0, 15.0, size=(n, t, k)), axis=1).astype(np.float32)
    v = rng.uniform(5.0, 20.0, size=(n, t, k)).astype(np.float32)
    states = np.concatenate((s[..., None], one_hot, v[..., None],
                             np.zeros((n, t, k, 2), np.float32)), axis=-1)
    states[..., 6] = 5.0
    present = np.ones((n, t, k), bool)
    return {"states": states, "present": present}


def _tiny_model():
    return WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                    hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16))


def test_trainer_step_new_vs_reference_projection_parity():
    torch.manual_seed(4)
    ds = TrajectoryDataset.from_arrays(_tiny_arrays())
    batch = collate([ds[i] for i in range(len(ds))])
    cfg = WmSTrainConfig(device="cpu", threads=2, batch_size=2, lr=1e-3,
                         truncation_steps=20, projection_enabled=True,
                         projection_min_gap_m=2.0, residual_cap_mps2=1.5,
                         lane_logit_cap=3.0, stability_weight=0.01,
                         stability_margin=1.0, stability_probes=1,
                         conservation_enabled=True, conservation_weight=0.01,
                         traffic_loss=True)
    model = _tiny_model()
    trainer = WmSTrainer(model, cfg)
    init = {name: value.clone() for name, value in model.state_dict().items()}
    new_fn = FeasibilityProjection._project_gaps
    ref_fn = FeasibilityProjection._project_gaps_reference

    def reference_wrapper(s, length, lane, present, projectable, min_gap, n_lanes=3):
        return ref_fn(s, length, lane, present, projectable, min_gap)

    def run(use_reference):
        FeasibilityProjection._project_gaps = staticmethod(
            reference_wrapper if use_reference else new_fn)
        model.load_state_dict(init)
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        trainer.optimizer.zero_grad(set_to_none=True)
        return trainer.train_batch(batch, unroll_steps=40)

    try:
        new = run(False)
        old = run(True)
    finally:
        FeasibilityProjection._project_gaps = staticmethod(new_fn)
    # float32 ulp-level agreement (the 60-step loss is O(300), so allow a
    # relative float32 rounding band around the reference).
    assert abs(float(new["loss"]) - float(old["loss"])) <= 1.0e-6 * abs(
        float(old["loss"])) + 1.0e-5
    assert new["projection_active_total"] == old["projection_active_total"]
    assert new["projection_s_clamped"] == old["projection_s_clamped"]
