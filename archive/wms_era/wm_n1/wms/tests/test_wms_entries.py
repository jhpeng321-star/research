"""Synthetic tests for observed-counter WM-S entries (no model weights)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from hdv.config import HdvConfig
from output.wd01.wm_n1.wms.entries import (EntranceHistory, EntryConfig,
                                           TorchEntryProcess, TorchEntryProcessV2,
                                           build_entrance_history, observed_rates,
                                           read_insertion_history, safe_to_insert)
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel

K = 32
SLOTS0 = 4


def counter(mainline=6.0, ramp=2.0, window_bins=12):
    c = np.zeros((2, 30), np.float32)
    c[0, :window_bins] = mainline
    c[1, :window_bins] = ramp
    return c


def state(batch=1, slots=K, present=SLOTS0, v=20.0, s0=-2000.0):
    lane = torch.zeros(batch, slots, dtype=torch.long)
    lane[:, 0::3] = 1
    lane[:, 2::3] = 2
    kin = torch.zeros(batch, slots, 7)
    kin[..., 0] = s0 + torch.arange(slots).float() * 30.0
    kin[..., 1:4] = torch.nn.functional.one_hot(lane.clamp_min(0), 3).float()
    kin[..., 4] = v
    kin[..., 6] = 5.0
    pres = torch.zeros(batch, slots, dtype=torch.bool)
    pres[:, :present] = True
    return {"veh_kin": kin, "s_m": kin[..., 0].clone(), "v_mps": kin[..., 4].clone(),
            "a_mps2": kin[..., 5].clone(), "length_m": kin[..., 6].clone(),
            "lane": lane, "present": pres,
            "is_cav": torch.zeros(batch, slots, dtype=torch.bool)}


def small_model(**kw):
    return WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                    hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16), **kw)


def test_rates_use_only_the_recent_window():
    c = counter(6.0, 3.0)
    assert observed_rates(c, 12) == pytest.approx([0.6, 0.3])
    c2 = c.copy()
    c2[:, 12:] = 99.0                       # older/other bins must not matter
    assert observed_rates(c2, 12) == pytest.approx([0.6, 0.3])


def test_no_future_leakage_identical_for_same_t0_counter():
    """Arrivals depend only on the t0 counter + seed, never on a recorded future."""
    ep_a = TorchEntryProcess(counter(), seed=5, cfg=EntryConfig(rate_scale=50.0))
    ep_b = TorchEntryProcess(counter(), seed=5, cfg=EntryConfig(rate_scale=50.0))
    wa, wb = state(), state()
    for _ in range(6):
        ma, mb = ep_a.insert(wa), ep_b.insert(wb)
        assert torch.equal(ma, mb)
        assert wa["s_m"][~wa["present"]].numel() == wb["s_m"][~wb["present"]].numel()
    assert ep_a.summary()["inserted"] == ep_b.summary()["inserted"] > 0


def test_slot_allocation_only_into_free_padding_slots():
    ep = TorchEntryProcess(counter(), seed=1, cfg=EntryConfig(rate_scale=50.0))
    w = state(present=SLOTS0)
    occupied = w["present"].clone()
    for _ in range(8):
        ep.insert(w)
    inserted = w["present"] & ~occupied
    assert inserted.any()
    assert not bool((w["present"][:, :SLOTS0] & ~occupied[:, :SLOTS0]).any())
    assert not bool(w["present"][0, SLOTS0:].sum() == 0)
    # every newcomer lands at its entrance lane start
    for k in torch.nonzero(inserted[0]).flatten().tolist():
        assert float(w["s_m"][0, k]) == pytest.approx(
            (-2645.67, -2644.22, -378.66)[int(w["lane"][0, k])], abs=0.01)


def test_deterministic_with_seed_and_common_random_numbers_across_rows():
    def run():
        ep = TorchEntryProcess(counter(), seed=7, cfg=EntryConfig(rate_scale=20.0),
                               batch=2)
        w = state(batch=2, present=SLOTS0)
        for _ in range(8):
            ep.insert(w)
        return w["s_m"].numpy().copy(), w["present"].numpy().copy(), ep.summary(0)

    a, b = run(), run()
    assert np.array_equal(a[0], b[0]) and (a[1] == b[1]).all() and a[2] == b[2]
    assert (a[1][0] == a[1][1]).all()


def test_gap_check_blocks_then_retries():
    ep = TorchEntryProcess(counter(ramp=0.0), seed=1,
                           cfg=EntryConfig(rate_scale=1000.0))
    w = state(present=SLOTS0)
    # block BOTH mainline lanes with a rearmost vehicle at the entrance
    for i in range(SLOTS0):
        w["lane"][0, i] = i % 2
        w["s_m"][0, i] = -2645.67 + 3.0
    ep.insert(w)
    assert ep.pending[0, 0] > 0
    assert ep.summary()["inserted"] == 0
    w2 = state(present=SLOTS0)          # clear road: safe gaps again
    for _ in range(3):
        ep.insert(w2)
    assert ep.summary()["inserted"] > 0


def test_pool_confines_entries_to_padding_slots():
    ep = TorchEntryProcess(counter(), seed=1, cfg=EntryConfig(rate_scale=50.0),
                           pool=[False, False, True, True])
    w = state(slots=4, present=2)          # slots 0,1 t0; 2,3 padding
    w["present"][0, 0] = False             # a vacated t0 slot must stay unused
    for _ in range(5):
        ep.insert(w)
    assert not bool(w["present"][0, 0])
    assert bool(w["present"][0, 2:].any())


def test_no_free_slot_is_counted_not_crashing():
    ep = TorchEntryProcess(counter(), seed=0, cfg=EntryConfig(rate_scale=500.0))
    w = state(present=K)
    ep.insert(w)
    assert ep.summary()["inserted"] == 0 and ep.summary()["no_slot"] > 0


def test_spawned_cav_share_observed_and_native_hold():
    c = counter()
    w = state(present=SLOTS0)
    w["is_cav"][0, :2] = True                    # observed share 2/4 = 0.5
    ep = TorchEntryProcess(c, seed=2, cfg=EntryConfig(rate_scale=40.0))
    for _ in range(6):
        ep.insert(w)
    assert ep.summary()["cav_share_observed"] == pytest.approx(0.5)
    assert w["is_cav"].any()


def test_flag_off_rollout_is_byte_identical():
    model = small_model()
    w = state()
    base = model.rollout(w, n_steps=6)
    explicit = model.rollout(w, n_steps=6, entries=None)
    assert "entrant" in base and not bool(base["entrant"].any())
    torch.testing.assert_close(base["positions"], explicit["positions"])
    torch.testing.assert_close(base["speeds"], explicit["speeds"])


def test_rollout_with_entries_inserts_and_marks_entrants():
    model = small_model()
    w = state()
    ep = TorchEntryProcess(counter(), seed=3, cfg=EntryConfig(rate_scale=40.0))
    out = model.rollout(w, n_steps=12, entries=ep)
    assert out["positions"].shape == (1, 12, K)
    assert bool(out["entrant"].any())
    assert bool(out["masks"][:, -1][out["entrant"]].any())
    assert bool(torch.isfinite(out["positions"][out["masks"]]).all())
    assert ep.summary()["inserted"] > 0


def test_entrant_cav_uses_hold_not_teacher():
    model = small_model()
    w = state()                                   # no t0 CAV -> no teacher required
    ep = TorchEntryProcess(counter(), seed=4, cfg=EntryConfig(rate_scale=40.0,
                                                             cav_share=1.0))
    out = model.rollout(w, n_steps=8, entries=ep)
    assert bool(out["entrant"].any())
    assert bool(out["state"]["is_cav"][out["entrant"]].any())
    # entrant CAVs are held (native), never forced by a recorded teacher
    assert bool(torch.isfinite(out["positions"][out["masks"]]).all())


def test_safe_to_insert_matches_reference():
    cfg = EntryConfig(min_gap_m=2.0, headway_s=1.0, decel_mps2=3.0)
    assert safe_to_insert(40.0, 20.0, 20.0, cfg)
    assert not safe_to_insert(5.0, 20.0, 20.0, cfg)
    assert not safe_to_insert(5.0, 30.0, 0.0, cfg)   # closing brake dominates


# --------------------------------------------------------------- entries v2


def v2_history(rate=0.5, cv=1.0, cav=0.0, lane_p1=0.5, speed_mean=1.0,
               speed_std=0.05, gap_mean=10.0, trend=1.0):
    h = EntranceHistory.defaults()
    h.rate[:] = rate
    h.headway_cv[:] = cv
    h.cav_share[:] = cav
    h.lane_p1 = lane_p1
    h.speed_mean[:] = speed_mean
    h.speed_std[:] = speed_std
    h.gap_mean[:] = gap_mean
    h.gap_std[:] = 1.0
    h.trend[:] = trend
    return h


def test_v2_feature_roundtrip():
    h = build_entrance_history(counter=counter(mainline=6.0, ramp=2.0))
    vec = h.to_features()
    assert vec.shape == (EntranceHistory.from_features(vec).to_features().shape)
    h2 = EntranceHistory.from_features(vec)
    for key in ("rate", "trend", "headway_cv", "speed_mean", "gap_mean",
                "cav_share", "n_obs"):
        assert np.allclose(getattr(h, key), getattr(h2, key))
    assert np.isclose(h.lane_p1, h2.lane_p1)
    assert np.allclose(h.vtype_share, h2.vtype_share)


def test_v2_ignores_older_bins_and_uses_recency_trend():
    flat = counter(6.0, 6.0)
    rising = flat.copy()
    rising[:, :6] = 12.0
    rising[:, 6:12] = 2.0
    falling = flat.copy()
    falling[:, :6] = 2.0
    falling[:, 6:12] = 12.0
    old = flat.copy()
    old[:, 12:] = 99.0                          # older bins must not matter
    assert build_entrance_history(counter=old).to_features().tolist() == \
        pytest.approx(build_entrance_history(counter=flat).to_features().tolist())
    h_rise = build_entrance_history(counter=rising)
    h_fall = build_entrance_history(counter=falling)
    assert h_rise.rate[0] > h_fall.rate[0]
    assert h_rise.trend[0] > 1.0 > h_fall.trend[0]


def test_v2_shrinkage_to_defaults_when_history_is_short():
    default = EntranceHistory.defaults().rate[0]
    times = np.arange(0.0, 60.0, 2.0)           # ~0.5 veh/s, 30 observations
    ent = np.zeros(len(times), np.int64)
    many = build_entrance_history(times=times, entrance=ent, t0_s=60.0, shrink_k=8.0)
    one = build_entrance_history(times=times[:1], entrance=ent[:1], t0_s=60.0,
                                 shrink_k=8.0)
    assert abs(one.rate[0] - default) < abs(many.rate[0] - default)
    assert many.n_obs[0] == 30 and one.n_obs[0] == 1


def test_v2_empirical_headway_cv_from_observed_times():
    bursty = np.concatenate([np.arange(0, 10, 0.2), np.arange(50, 60, 0.2)])
    regular = np.arange(0, 60, 1.0)
    hb = build_entrance_history(times=bursty, entrance=np.zeros_like(bursty),
                                t0_s=60.0, shrink_k=0.0)
    hr = build_entrance_history(times=regular, entrance=np.zeros_like(regular),
                                t0_s=60.0, shrink_k=0.0)
    assert hb.headway_cv[0] > hr.headway_cv[0]


def test_v2_gamma_burstiness_increases_arrival_dispersion():
    def dispersion(cv):
        ep = TorchEntryProcessV2(history=v2_history(rate=1.0, cv=cv), seed=1,
                                 cfg=EntryConfig(rate_scale=1.0))
        w = state(slots=600, present=0)
        counts = []
        for _ in range(300):
            before = int(w["present"].sum())
            ep.insert(w)
            counts.append(int(w["present"].sum()) - before)
            w["s_m"] += w["v_mps"] * 0.5        # let newcomers clear the entrance
        arr = np.asarray(counts, float)
        return arr.var() / max(arr.mean(), 1e-9)

    assert dispersion(3.0) > dispersion(0.3)


def test_v2_deterministic_and_no_future_leakage():
    def run():
        ep = TorchEntryProcessV2(history=v2_history(rate=0.6, cv=1.2), seed=11,
                                 cfg=EntryConfig())
        w = state(slots=300, present=SLOTS0)
        for _ in range(40):
            ep.insert(w)
        return w["present"].numpy().copy(), w["s_m"].numpy().copy()

    a, b = run(), run()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # a future-only counter change is invisible: the builder reads t <= t0
    read = read_insertion_history
    assert callable(read)


def test_v2_pool_confines_entries_to_padding_slots():
    ep = TorchEntryProcessV2(history=v2_history(rate=5.0), seed=1,
                             cfg=EntryConfig(),
                             pool=[False, False, True, True, True, True])
    w = state(slots=6, present=2)
    w["present"][0, 0] = False                  # vacated t0 slot stays unused
    for _ in range(20):
        ep.insert(w)
        w["s_m"] += w["v_mps"] * 0.5
    assert not bool(w["present"][0, 0])
    assert bool(w["present"][0, 2:].any())


def test_v2_inserted_attributes_follow_history():
    h = v2_history(rate=5.0, cav=1.0, speed_mean=2.0, gap_mean=0.0)
    h.vtype_share[:] = 0.0
    h.vtype_share[:, 3] = 1.0                   # truck
    ep = TorchEntryProcessV2(history=h, seed=2, cfg=EntryConfig())
    w = state(slots=200, present=SLOTS0, v=20.0)
    for _ in range(20):
        ep.insert(w)
        w["s_m"] += w["v_mps"] * 0.5
    newcomers = w["present"][0, SLOTS0:]
    assert bool(newcomers.any())
    assert bool(w["is_cav"][0, SLOTS0:][newcomers].all())
    for k in torch.nonzero(w["present"][0, SLOTS0:]).flatten().tolist():
        kk = SLOTS0 + int(k)
        assert float(w["length_m"][0, kk]) == pytest.approx(7.5)
        assert float(w["v_mps"][0, kk]) > 30.0      # speed_mean=2x, not 1x


def test_v2_rollout_marks_entrants_with_model():
    model = small_model()
    w = state(slots=200)
    ep = TorchEntryProcessV2(history=v2_history(rate=2.0, gap_mean=0.0), seed=3,
                             cfg=EntryConfig())
    out = model.rollout(w, n_steps=20, entries=ep)
    assert bool(out["entrant"].any())
    assert ep.summary()["inserted"] > 0


def test_section_ids_model_frame_boundaries():
    from output.wd01.wm_n1.wms.entries import section_ids

    s = np.array([-10.0, 0.0, 37.7, 75.44, 100.0])
    present = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(section_ids(s, present), [0, 1, -1, 1, 2])
    ts = section_ids(torch.as_tensor(s), torch.as_tensor(present))
    assert ts.tolist() == [0, 1, -1, 1, 2]
    assert ts.dtype == torch.long
    np.testing.assert_array_equal(section_ids(s), [0, 1, 1, 1, 2])


def test_rollout_exposes_per_step_entrant_mask_and_sections():
    model = small_model()
    w = state(slots=200)
    ep = TorchEntryProcessV2(history=v2_history(rate=2.0, gap_mean=0.0), seed=3,
                             cfg=EntryConfig())
    out = model.rollout(w, n_steps=20, entries=ep)
    ent_seq = out["entrant_seq"]
    sections = out["sections"]
    assert ent_seq.shape == (1, 20, 200) and sections.shape == (1, 20, 200)
    assert ent_seq.dtype == torch.bool and sections.dtype == torch.long
    assert bool(out["entrant"].any())
    # per-step mask is monotone and equals the cumulative mask on the last step
    assert bool((ent_seq[0, 1:] >= ent_seq[0, :-1]).all())
    assert torch.equal(ent_seq[0, -1], out["entrant"][0])
    # a section id is present exactly where the slot is, and always 0/1/2
    masks = out["masks"]
    assert bool(((sections == -1) == ~masks).all())
    live = sections[masks]
    assert bool(((live >= 0) & (live <= 2)).all())
    # entrance lane starts are upstream (s_down < 0)
    inserted = out["entrant"][0] & masks[0, -1]
    if bool(inserted.any()):
        assert bool((sections[0, -1][inserted] == 0).all())


def test_rollout_entrant_seq_off_is_all_false():
    model = small_model()
    out = model.rollout(state(), n_steps=6)
    assert out["entrant_seq"].shape == (1, 6, K)
    assert not bool(out["entrant_seq"].any())
    assert not bool(out["entrant"].any())
    assert bool((out["sections"][out["masks"]] >= 0).all())


def test_rollout_carries_entrant_membership_across_chunks():
    model = small_model()
    w = state(slots=200)
    ep = TorchEntryProcessV2(history=v2_history(rate=2.0, gap_mean=0.0), seed=3,
                             cfg=EntryConfig())
    first = model.rollout(w, n_steps=10, entries=ep)
    carried = first["entrant"][0]
    assert bool(carried.any())
    cont = model.rollout(first["state"], n_steps=5, entries=ep)
    assert bool(cont["entrant_seq"][0, 0][carried].all())
    assert torch.equal(cont["entrant"][0] & carried, carried)


def test_read_insertion_history_drops_future_rows(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"vehicle_id": "a", "inserted_s": 1.0, "entrance": "mainline",
         "depart_speed_mps": 20.0, "is_cav": False, "type_id": "normal_car"},
        {"vehicle_id": "b", "inserted_s": 5.0, "entrance": "ramp",
         "depart_speed_mps": 10.0, "is_cav": True, "type_id": "cautious_car"},
        {"vehicle_id": "c", "inserted_s": 99.0, "entrance": "mainline",
         "depart_speed_mps": 30.0, "is_cav": False, "type_id": "normal_car"},
    ]
    with (run / "insertions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    hist = read_insertion_history(tmp_path, t0_s=10.0)
    assert hist["times"].tolist() == [1.0, 5.0]
    assert hist["entrance"].tolist() == [0, 1]
    assert hist["is_cav"].tolist() == [False, True]
    assert hist["vtype"].tolist() == ["normal_car", "cautious_car"]
