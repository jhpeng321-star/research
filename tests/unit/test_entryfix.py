"""ENTRYFIX (ENTRYFIX_NOTES.md): switch-guarded SUMO-like insertion of the entries-v2 process.

``EntryConfig.insert_rule = "v2"`` (default) keeps the legacy rule bit-identical; ``"sumo_like"``
reproduces SUMO's departSpeed="max" / departLane="best" / departPos="base" insertion.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.model import entries as E
from wms.model.entries import EntranceHistory, EntryConfig, TorchEntryProcessV2

DT = 0.5


def _state(cars, n_pad=8):
    """WM-S state [1, K] from ``cars = [(s_front, v, lane, length), ...]`` + empty pad slots."""
    k = len(cars) + n_pad
    s = torch.zeros(1, k, dtype=torch.float64)
    v = torch.zeros(1, k, dtype=torch.float64)
    lane = torch.zeros(1, k, dtype=torch.long)
    length = torch.full((1, k), 5.0, dtype=torch.float64)
    present = torch.zeros(1, k, dtype=torch.bool)
    for i, (si, vi, li, Li) in enumerate(cars):
        s[0, i], v[0, i], lane[0, i], length[0, i], present[0, i] = si, vi, li, Li, True
    pool = np.zeros(k, bool)
    pool[len(cars):] = True
    return {"s_m": s, "v_mps": v, "a_mps2": torch.zeros(1, k, dtype=torch.float64), "lane": lane,
            "length_m": length, "present": present,
            "is_cav": torch.zeros(1, k, dtype=torch.bool)}, pool


def _hist(rate=(0.0, 0.0), gap_mean=(40.0, 40.0), gap_std=(10.0, 10.0), speed=(1.0, 1.0)):
    return dataclasses.replace(EntranceHistory.defaults(), rate=np.asarray(rate, float),
                               gap_mean=np.asarray(gap_mean, float),
                               gap_std=np.asarray(gap_std, float),
                               speed_mean=np.asarray(speed, float),
                               speed_std=np.asarray([1e-3, 1e-3]))


def _proc(rule, hist, pool, pending=(0, 0), seed=0, **kw):
    cfg = EntryConfig(max_pending=10 ** 9, insert_rule=rule, **kw)
    proc = TorchEntryProcessV2(histories=[hist], dt_s=DT, seed=seed, batch=1, cfg=cfg, pool=pool)
    proc.pending[0] = list(pending)
    return proc


def _no_arrivals(proc):
    proc._emit_due = lambda b: []
    return proc


# ------------------------------------------------------------------------------ defaults
def test_default_rule_is_v2_and_invalid_rules_refused():
    assert EntryConfig().insert_rule == "v2"
    with pytest.raises(ValueError):
        EntryConfig(insert_rule="sumo")
    with pytest.raises(ValueError):
        EntryConfig(insert_rule="sumo_like", sumo_lane_mode="free")


def _busy_road(seed):
    rng = np.random.default_rng(seed)
    cars = []
    for ln, s0 in ((0, E.ENTRY_S_M[0]), (1, E.ENTRY_S_M[1]), (2, E.ENTRY_S_M[2])):
        s = s0 + rng.uniform(5.0, 40.0)
        for _ in range(4):
            cars.append((s, rng.uniform(0.0, 25.0), ln, 5.0))
            s += rng.uniform(8.0, 60.0)
    return cars


def test_default_and_explicit_v2_are_bit_identical():
    """Default config == explicit ``insert_rule="v2"`` on busy roads with real arrivals."""
    hist = _hist(rate=(0.8, 0.3), gap_mean=(15.0, 12.0), gap_std=(6.0, 4.0))
    for seed in range(4):
        outs = []
        for cfg in (EntryConfig(max_pending=10 ** 9),
                    EntryConfig(max_pending=10 ** 9, insert_rule="v2")):
            state, pool = _state(_busy_road(seed), n_pad=40)
            proc = TorchEntryProcessV2(histories=[hist], dt_s=DT, seed=seed, batch=1, cfg=cfg,
                                       pool=pool)
            masks, backlog = [], []
            for _ in range(40):
                with torch.no_grad():
                    state["s_m"] += state["v_mps"] * DT * state["present"]
                masks.append(proc.insert(state).clone())
                backlog.append(proc.backlog().copy())
            outs.append((state, masks, backlog, proc.summary()))
        (s1, m1, b1, x1), (s2, m2, b2, x2) = outs
        assert x1 == x2
        assert all(torch.equal(a, b) for a, b in zip(m1, m2))
        assert all(np.array_equal(a, b) for a, b in zip(b1, b2))
        for key in s1:
            assert torch.equal(s1[key], s2[key]), key


def test_rules_share_the_arrival_stream():
    """Common random numbers: on a free road both rules admit the same arrivals at the same steps
    (sumo_like "any" lane mode; "best_free" may pick the occupied lane, as SUMO does)."""
    hist = _hist(rate=(0.7, 0.2))
    per = {}
    for rule in ("v2", "sumo_like"):
        state, pool = _state([], n_pad=120)
        proc = TorchEntryProcessV2(histories=[hist], dt_s=DT, seed=3, batch=1,
                                   cfg=EntryConfig(max_pending=10 ** 9, insert_rule=rule,
                                                   sumo_lane_mode="any"),
                                   pool=pool)
        counts = []
        for _ in range(60):
            with torch.no_grad():                      # move entrants far away: no blocking
                state["s_m"] += 200.0 * state["present"]
            counts.append(int(proc.insert(state).sum()))
        per[rule] = counts
    assert per["v2"] == per["sumo_like"] and sum(per["v2"]) > 10


# ------------------------------------------------------------------------- sumo_like rule
def test_sumo_like_inserts_at_reduced_speed_where_v2_refuses():
    """Slow ramp leader 14 m beyond the SUMO entry front: v2 (sampled required gap ~40 m, full
    local speed) refuses; sumo_like enters at the reduced safe speed, front at start + length."""
    ln = 2
    lead_rear = E.ENTRY_S_M[ln] + 5.0 + E.SUMO_POSITION_EPS_M + 14.0
    cars = [(lead_rear + 5.0, 3.0, ln, 5.0)]
    hist = _hist(speed=(1.0, 5.0))           # v2 speed ratio 5 x local 3 m/s = 15 m/s
    state, pool = _state(cars)
    v2 = _no_arrivals(_proc("v2", hist, pool, pending=(0, 1)))
    assert not bool(v2.insert(state).any()) and v2.backlog()[0] == 1.0
    state, pool = _state(cars)
    sl = _no_arrivals(_proc("sumo_like", hist, pool, pending=(0, 1)))
    mask = sl.insert(state)
    assert int(mask.sum()) == 1 and sl.backlog()[0] == 0.0
    k = int(torch.nonzero(mask[0])[0])
    v_new = float(state["v_mps"][0, k])
    v_cap = E.SUMO_LANE_SPEED_MPS[ln] * E.VTYPE_SPEED_FACTOR[0]
    assert 0.0 < v_new < v_cap
    assert float(state["s_m"][0, k]) == pytest.approx(E.ENTRY_S_M[ln] + 5.0 + E.SUMO_POSITION_EPS_M)
    gap = lead_rear - float(state["s_m"][0, k])
    assert gap == pytest.approx(14.0)
    assert E.sumo_insert_ok(gap, v_new, 3.0, E.VTYPE_TAU_S[0], E.VTYPE_DECEL_MPS2[0],
                            E.VTYPE_MIN_GAP_M[0])
    # the safe speed is maximal: a slightly faster entrant would violate the SUMO safety gap
    assert not E.sumo_insert_ok(gap, v_new + 0.05, 3.0, E.VTYPE_TAU_S[0],
                                E.VTYPE_DECEL_MPS2[0], E.VTYPE_MIN_GAP_M[0])


def test_sumo_like_refuses_inside_min_gap_and_free_lane_enters_at_cap():
    ln = 2
    front = E.ENTRY_S_M[ln] + 5.0 + E.SUMO_POSITION_EPS_M
    state, pool = _state([(front + 2.0 + 5.0, 0.0, ln, 5.0)])     # bumper gap 2.0 < minGap 2.5
    proc = _no_arrivals(_proc("sumo_like", _hist(), pool, pending=(0, 1)))
    assert not bool(proc.insert(state).any()) and proc.backlog()[0] == 1.0
    state, pool = _state([])
    proc = _no_arrivals(_proc("sumo_like", _hist(), pool, pending=(1, 1)))
    mask = proc.insert(state)
    v = sorted(float(x) for x in state["v_mps"][mask])
    assert v == pytest.approx(sorted([E.SUMO_LANE_SPEED_MPS[0], E.SUMO_LANE_SPEED_MPS[2]]))


@pytest.mark.parametrize("lane_mode", ["best_free", "any"])
def test_sumo_like_never_inserts_unsafe_or_overlapping(lane_mode):
    rng = np.random.default_rng(7)
    hist = _hist(rate=(1.5, 0.6))
    n_checked = 0
    for trial in range(30):
        cars = []
        for ln in (0, 1, 2):
            s = E.ENTRY_S_M[ln] + rng.uniform(0.0, 30.0)
            for _ in range(int(rng.integers(0, 4))):
                cars.append((s, rng.uniform(0.0, 30.0), ln, float(rng.choice([5.0, 7.5]))))
                s += rng.uniform(5.5, 40.0)
        state, pool = _state(cars, n_pad=20)
        proc = _proc("sumo_like", hist, pool, pending=(int(rng.integers(0, 6)),
                                                       int(rng.integers(0, 3))),
                     seed=trial, sumo_lane_mode=lane_mode)
        before = state["present"].clone()
        mask = proc.insert(state)[0]
        for k in torch.nonzero(mask).flatten().tolist():
            ln = int(state["lane"][0, k])
            s_k, v_k = float(state["s_m"][0, k]), float(state["v_mps"][0, k])
            # the leader is any vehicle ahead on the lane (pre-existing or inserted earlier)
            ahead = [j for j in range(state["s_m"].shape[1])
                     if bool(state["present"][0, j]) and j != k
                     and int(state["lane"][0, j]) == ln and float(state["s_m"][0, j]) > s_k]
            assert v_k >= 0.0
            if not ahead:
                continue
            j = min(ahead, key=lambda i: float(state["s_m"][0, i]))
            gap = float(state["s_m"][0, j] - state["length_m"][0, j]) - s_k
            assert gap >= E.VTYPE_MIN_GAP_M[0] - 1e-9          # no overlap, minGap kept
            if bool(before[0, j]):                              # leader known at insertion
                assert E.sumo_insert_ok(gap, v_k, float(state["v_mps"][0, j]),
                                        E.VTYPE_TAU_S[0], E.VTYPE_DECEL_MPS2[0],
                                        E.VTYPE_MIN_GAP_M[0])
            n_checked += 1
    assert n_checked > 10


def test_sumo_like_backlog_drains_when_gaps_exist():
    """12 queued mainline arrivals behind free lanes drain within a few steps (one per lane per
    step, as SUMO's departPos="base"); v2 with a sampled 40 m required gap drains slower."""
    hist = _hist(gap_mean=(40.0, 40.0), gap_std=(10.0, 10.0))
    left = {}
    for rule in ("v2", "sumo_like"):
        state, pool = _state([], n_pad=40)
        proc = _no_arrivals(_proc(rule, hist, pool, pending=(12, 0)))
        trace = []
        for _ in range(12):
            proc.insert(state)
            with torch.no_grad():                      # entrants move on at their speed
                state["s_m"] += state["v_mps"] * DT * state["present"]
            trace.append(float(proc.backlog()[0]))
        left[rule] = trace
    assert left["sumo_like"][-1] == 0.0
    assert left["sumo_like"][5] <= 4.0
    assert sum(left["sumo_like"]) < sum(left["v2"])


# ------------------------------------------------------------------------- config plumbing
def test_trainer_and_training_switches_reach_the_process():
    from wms.control.dreamer import DreamerConfig, _entry_process
    from wms.train.settings import TrainSettings

    assert DreamerConfig().entry_insert_rule == "v2"
    with pytest.raises(ValueError):
        DreamerConfig(entry_insert_rule="bogus")
    spec = {"histories": [EntranceHistory.defaults()], "seeds": [1], "k0": 4}
    assert _entry_process(spec, [0], 8).cfg.insert_rule == "v2"
    assert _entry_process(spec, [0], 8, "sumo_like").cfg.insert_rule == "sumo_like"
    assert TrainSettings().entry_insert_rule == "v2"


def test_training_process_uses_data_switch():
    from wms.train.inputs import make_entry_process
    from wms.train.settings import TrainSettings

    feat = torch.as_tensor(EntranceHistory.defaults().to_features(), dtype=torch.float32)[None]
    present = torch.ones(1, 5, 3, dtype=torch.bool)

    class _Phys:
        dt_s = 0.5

    class _Model:
        physics = _Phys()

    for auto in (False, True):
        legacy = make_entry_process(TrainSettings(entries_v2=True, entry_pad_auto=auto),
                                    {"entry_feat": feat}, present, _Model(),
                                    torch.device("cpu"), 4)
        on = make_entry_process(TrainSettings(entries_v2=True, entry_pad_auto=auto,
                                              entry_insert_rule="sumo_like"),
                                {"entry_feat": feat}, present, _Model(), torch.device("cpu"), 4)
        assert legacy.cfg == (EntryConfig(max_pending=10 ** 9) if auto else EntryConfig())
        assert on.cfg.insert_rule == "sumo_like"
        assert on.cfg.max_pending == legacy.cfg.max_pending


def test_data_config_key_maps_to_settings():
    from wms.config.legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT

    from wms.config.train import TrainConfig
    from wms.train.settings import TrainSettings

    assert FLAT_TO_PATH["entry_insert_rule"] == "data.entry_insert_rule"
    assert "entry_insert_rule" in OMIT_WHEN_DEFAULT
    assert "entry_insert_rule" not in TrainConfig().to_legacy_flat()      # legacy flat unchanged
    cfg = TrainConfig.from_dict({"data": {"entries_v2": True, "entry_insert_rule": "sumo_like"}})
    assert TrainSettings.from_config(cfg).entry_insert_rule == "sumo_like"
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()).data.entry_insert_rule == "sumo_like"
    assert TrainSettings.from_config(TrainConfig()).entry_insert_rule == "v2"


# ------------------------------------------------------- recorded SUMO episode (skip if absent)
_EPISODE = Path("runs/active/c4_dev/b1_native")


@pytest.mark.skipif(not (_EPISODE / "episode.npz").is_file(),
                    reason="recorded SUMO episode not available on this host")
def test_recorded_sumo_insertions_admitted_by_sumo_like():
    """On SUMO's recorded pre-insertion states, sumo_like admits >= 90% of SUMO's actual
    insertions (v2 admits far fewer: its sampled required gap over-blocks)."""
    with np.load(_EPISODE / "episode.npz", allow_pickle=False) as z:
        ep = {k: z[k] for k in ("s_m", "v_mps", "lane", "length_true_m", "present",
                                "vid_index", "vehicle_ids")}
    names = {str(n): i for i, n in enumerate(ep["vehicle_ids"])}
    ins = {}
    with gzip.open(_EPISODE / "run" / "physics_steps.jsonl.gz", "rt") as fh:
        for line in fh:
            row = json.loads(line)
            for vid in row["departed_ids"]:
                ins.setdefault(vid, float(row["time_s"]))
    hist = _hist(gap_mean=(40.0, 40.0), gap_std=(30.0, 30.0))   # typical v2 feature values
    rates = {}
    for rule in ("v2", "sumo_like"):
        ok = n = 0
        for name, t in sorted(ins.items(), key=lambda x: x[1])[:60]:
            tick = int(round(t / DT))
            same_tick = [names[m] for m, tm in ins.items() if tm == t and m in names]
            pres = ep["present"][tick] & ~np.isin(ep["vid_index"][tick], same_tick)
            k = np.flatnonzero(ep["present"][tick] & (ep["vid_index"][tick] == names.get(name, -9)))
            if k.size == 0:
                continue
            ln = int(ep["lane"][tick, k[0]])
            e = 1 if ln == 2 else 0
            cars = [(float(ep["s_m"][tick, j]), float(ep["v_mps"][tick, j]),
                     int(ep["lane"][tick, j]), float(ep["length_true_m"][tick, j]))
                    for j in np.flatnonzero(pres)]
            state, pool = _state(cars, n_pad=2)
            proc = _no_arrivals(_proc(rule, hist, pool, pending=(1 - e, e), seed=n))
            proc.insert(state)
            ok += int(proc.stats[0].inserted > 0)
            n += 1
        rates[rule] = ok / max(n, 1)
    assert rates["sumo_like"] >= 0.9
    assert rates["v2"] < rates["sumo_like"]

