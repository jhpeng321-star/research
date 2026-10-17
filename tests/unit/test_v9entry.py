"""V9ENTRY: the imagined entrance (source) queue carries the recorded SUMO backlog and demand.

Before V9ENTRY every imagined rollout started with an EMPTY entrance queue and drew arrivals at
the observed INSERTION-counter rate, which on a saturated entrance is the service capacity, not
the demand; the WM could therefore not represent the standing queue that dominates SUMO's source
waiting, nor the extra waiting that slower entrance service causes.  ``EntranceQueue`` (t0
backlog + demand rate per entrance) fixes both; ``EntryConfig.fresh_no_credit`` stops a free lane
from admitting two entrants at one position in a single step (SUMO inserts at most one).
Defaults are bit-identical (switch off).
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from wms.control import srcgrad
from wms.control.dreamer import DreamerConfig, _entry_process, with_entry_slots
from wms.control.start_states import StartBatch, StartStateSampler
from wms.model import entries as E
from wms.model.entries import (EntranceHistory, EntranceQueue, EntryConfig,
                               TorchEntryProcessV2, with_queue_demand)
from tests.unit.test_c4_loop import write_episode

DT = 0.5


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _state(cars, n_pad=12):
    k = len(cars) + n_pad
    s = torch.zeros(1, k, dtype=torch.float64)
    v = torch.zeros(1, k, dtype=torch.float64)
    lane = torch.zeros(1, k, dtype=torch.long)
    length = torch.full((1, k), 5.0, dtype=torch.float64)
    present = torch.zeros(1, k, dtype=torch.bool)
    for i, (si, vi, li) in enumerate(cars):
        s[0, i], v[0, i], lane[0, i], present[0, i] = si, vi, li, True
    pool = np.zeros(k, bool)
    pool[len(cars):] = True
    return {"s_m": s, "v_mps": v, "a_mps2": torch.zeros(1, k, dtype=torch.float64), "lane": lane,
            "length_m": length, "present": present,
            "is_cav": torch.zeros(1, k, dtype=torch.bool)}, pool


def _hist(rate=(0.3, 0.1)):
    return dataclasses.replace(EntranceHistory.defaults(), rate=np.asarray(rate, float))


def _queue(backlog=(10, 2), demand=(0.7, 0.1)):
    return EntranceQueue(backlog=np.asarray(backlog, float), demand_rate=np.asarray(demand, float))


def _blocked_cars():
    """Both mainline lanes and the ramp held by a stopped vehicle right at the insertion front."""
    return [(E.ENTRY_S_M[ln] + 5.0 + E.SUMO_POSITION_EPS_M + 5.0 + 0.5, 0.0, ln) for ln in (0, 1, 2)]


# ----------------------------------------------------------------------------- the vector
def test_queue_vector_round_trip_and_validation():
    q = EntranceQueue.from_vector([12.4, 0.0, 0.66, 0.11])
    np.testing.assert_allclose(q.backlog, [12.0, 0.0])
    np.testing.assert_allclose(q.demand_rate, [0.66, 0.11])
    assert q.demand_cv == 1.0
    np.testing.assert_allclose(q.to_vector(), [12.0, 0.0, 0.66, 0.11])
    with pytest.raises(ValueError):
        EntranceQueue.from_vector([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        EntranceQueue.from_vector([1.0, 2.0, 3.0, 4.0, 0.5])
    with pytest.raises(ValueError):
        EntranceQueue.from_vector([1.0, np.nan, 0.5, 0.1])
    h = with_queue_demand(_hist(), q)
    np.testing.assert_allclose(h.rate, [0.66, 0.11])
    np.testing.assert_allclose(h.headway_cv, [1.0, 1.0])
    np.testing.assert_allclose(h.vtype_share, _hist().vtype_share)     # class mix kept
    np.testing.assert_allclose(h.cav_share, _hist().cav_share)         # no CAV share: kept
    q6 = EntranceQueue.from_vector([5, 1, 0.7, 0.1, 0.4, 1.3])
    np.testing.assert_allclose(q6.cav_share, [0.4, 1.0])              # clipped to [0, 1]
    np.testing.assert_allclose(q6.to_vector(), [5, 1, 0.7, 0.1, 0.4, 1.0])
    np.testing.assert_allclose(with_queue_demand(_hist(), q6).cav_share, [0.4, 1.0])
    assert EntryConfig().fresh_no_credit is False


# ------------------------------------------------------------------------ the process
def test_no_queue_is_bit_identical_to_the_legacy_construction():
    """queues=None and fresh_no_credit=False (the defaults) == the constructor without them."""
    outs = []
    for kw in ({}, {"queues": None}):
        rng = np.random.default_rng(5)
        cars = [(E.ENTRY_S_M[ln] + rng.uniform(6, 40), rng.uniform(0, 20), ln)
                for ln in (0, 1, 2) for _ in range(2)]
        state, pool = _state(cars, n_pad=40)
        proc = TorchEntryProcessV2(histories=[_hist((0.8, 0.3))], dt_s=DT, seed=3, batch=1,
                                   cfg=EntryConfig(max_pending=10 ** 9, insert_rule="sumo_like_v2"),
                                   pool=pool, **kw)
        masks = []
        for _ in range(40):
            with torch.no_grad():
                state["s_m"] += state["v_mps"] * DT * state["present"]
            masks.append(proc.insert(state).clone())
        outs.append((state["s_m"].clone(), masks, proc.summary()))
    assert outs[0][2] == outs[1][2] and torch.equal(outs[0][0], outs[1][0])
    assert all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))
    assert "queue_backlog0" not in outs[0][2]


def test_backlog_starts_pending_and_grows_at_the_demand_rate_while_blocked():
    state, pool = _state(_blocked_cars())
    q = _queue(backlog=(30, 3), demand=(0.8, 0.2))
    proc = TorchEntryProcessV2(histories=[_hist((0.3, 0.05))], dt_s=DT, seed=1, batch=1,
                               cfg=EntryConfig(max_pending=10 ** 9, insert_rule="sumo_like_v2",
                                               fresh_no_credit=True), pool=pool, queues=[q])
    np.testing.assert_array_equal(proc.pending[0], [30, 3])
    assert proc.backlog()[0] == 33.0                       # charged from the first step
    np.testing.assert_allclose(proc._hist(0).rate, [0.8, 0.2])  # arrivals at the demand
    n_steps = 2000
    for _ in range(n_steps):
        assert not proc.insert(state).any()                 # stopped heads: nothing enters
    grown = proc.pending[0] - np.array([30, 3])
    expected = np.array([0.8, 0.2]) * n_steps * DT
    assert np.all(np.abs(grown - expected) < 4.0 * np.sqrt(expected))   # Poisson, cv = 1
    assert proc.summary()["queue_backlog0"] == [30.0, 3.0]


def test_free_entrance_with_backlog_admits_at_most_one_per_lane_and_step():
    """A free entrance with a backlog: fresh_no_credit admits one entrant per lane in the step (SUMO);
    without it a free lane admits a second entrant at the SAME position (overlap)."""
    for fresh in (True, False):
        state, pool = _state([], n_pad=20)
        proc = TorchEntryProcessV2(histories=[_hist((0.0, 0.0))], dt_s=DT, seed=0, batch=1,
                                   cfg=EntryConfig(max_pending=10 ** 9, insert_rule="sumo_like_v2",
                                                   fresh_no_credit=fresh),
                                   pool=pool, queues=[_queue(backlog=(8, 0), demand=(0.0, 0.0))])
        proc._emit_due = lambda b: []
        new = proc.insert(state)
        lanes = state["lane"][new].tolist()
        n_stacked = len(lanes) - len(set(zip(lanes, state["s_m"][new].tolist())))
        if fresh:
            assert lanes.count(0) == 1 and lanes.count(1) == 1 and n_stacked == 0
            assert proc.pending[0, 0] == 6
        else:       # legacy sumo_like_v2: same-position entrants (credit from a fresh leader)
            assert lanes.count(0) >= 2 and n_stacked >= 2


def test_slow_vehicle_at_the_entrance_raises_the_queue():
    """The mechanism the WM lacked: with a standing backlog, a slower vehicle leaving the
    entrance blocks insertions longer, so the queue (source waiting) grows more."""
    def run(v_head):
        cars = [(E.ENTRY_S_M[ln] + 5.0 + E.SUMO_POSITION_EPS_M + 7.0, v_head, ln) for ln in (0, 1)]
        state, pool = _state(cars, n_pad=60)
        proc = TorchEntryProcessV2(histories=[_hist((0.0, 0.0))], dt_s=DT, seed=2, batch=1,
                                   cfg=EntryConfig(max_pending=10 ** 9, insert_rule="sumo_like_v2",
                                                   fresh_no_credit=True),
                                   pool=pool, queues=[_queue(backlog=(20, 0), demand=(0.6, 0.0))])
        waited = 0.0
        for _ in range(60):
            proc.insert(state)
            waited += DT * float(proc.backlog()[0])
            with torch.no_grad():            # entrants accelerate to v_head, heads keep v_head
                v = state["v_mps"]
                v.copy_(torch.where(state["present"], torch.clamp(v + 1.0 * DT, max=v_head), v))
                state["s_m"] += v * DT * state["present"]
        return waited
    assert run(2.0) > run(8.0) > 0.0


def test_srcgrad_history_reads_the_demand_rate():
    state, pool = _state(_blocked_cars())
    proc = TorchEntryProcessV2(histories=[_hist((0.3, 0.05))], dt_s=DT, seed=1, batch=1,
                               cfg=EntryConfig(max_pending=10 ** 9, insert_rule="sumo_like_v2"),
                               pool=pool, queues=[_queue(demand=(0.75, 0.12))])
    hist = srcgrad.history_tensors(proc, 1, "cpu", torch.float64)
    np.testing.assert_allclose(hist["rate"].numpy()[0], [0.75, 0.12])


# ------------------------------------------------------------------- dreamer plumbing
def _start_batch(with_queue=True, b=2, k=6):
    state = {"s_m": torch.zeros(b, k), "v_mps": torch.zeros(b, k), "a_mps2": torch.zeros(b, k),
             "length_m": torch.full((b, k), 5.0), "lane": torch.full((b, k), -1),
             "present": torch.zeros(b, k, dtype=torch.bool),
             "is_cav": torch.zeros(b, k, dtype=torch.bool)}
    feat = torch.as_tensor(np.stack([_hist((0.5, 0.1)).to_features()] * b)).float()
    queue = torch.tensor([[40.0, 0.0, 0.7, 0.1], [5.0, 1.0, 0.6, 0.1]])[:b] if with_queue else None
    return StartBatch(state, {}, [{} for _ in range(b)], feat, entry_queue=queue)


def test_dreamer_config_switch_and_slots():
    assert DreamerConfig().entry_queue_state is False
    assert DreamerConfig(entry_queue_state=True).entry_queue_state is True
    with pytest.raises(ValueError):
        DreamerConfig(entry_queue_state=1)
    g = torch.Generator().manual_seed(0)
    off = with_entry_slots(_start_batch(), 30.0, 256, torch.Generator().manual_seed(0))
    assert "queues" not in off.entry_spec
    rate = float(np.sum(EntranceHistory.from_features(_start_batch().entry_feat[0].numpy()).rate))
    assert off.entry_spec["pad"] == int(np.ceil(rate * 30.0 * 1.5)) + 16
    on = with_entry_slots(_start_batch(), 30.0, 256, g, queue_state=True)
    assert on.entry_spec["seeds"] == off.entry_spec["seeds"]         # same CRN seeds
    np.testing.assert_allclose(on.entry_spec["queues"][0].backlog, [40.0, 0.0])
    demand = float(np.sum(_start_batch().entry_queue[0, 2:].numpy().astype(np.float64)))
    assert on.entry_spec["pad"] == int(np.ceil((demand * 30.0 + 15.0) * 1.5)) + 16
    assert torch.equal(on.entry_queue, _start_batch().entry_queue)
    with pytest.raises(ValueError, match="entry_queue"):
        with_entry_slots(_start_batch(with_queue=False), 30.0, 256, None, queue_state=True)
    proc = _entry_process(on.entry_spec, [0, 0, 1], on.state["s_m"].shape[1], "sumo_like_v2")
    assert proc.cfg.fresh_no_credit is True and proc.cfg.insert_rule == "sumo_like_v2"
    np.testing.assert_array_equal(proc.pending, [[40, 0], [40, 0], [5, 1]])
    six = _start_batch()
    six.entry_queue = torch.cat((six.entry_queue, torch.tensor([[0.4, 0.0], [0.3, 0.1]])), 1)
    on6 = with_entry_slots(six, 30.0, 256, torch.Generator().manual_seed(0), queue_state=True)
    p6 = _entry_process(on6.entry_spec, [0, 1], on6.state["s_m"].shape[1], "sumo_like_v2")
    np.testing.assert_allclose(p6._hist(0).cav_share, [0.4, 0.0], rtol=1e-6)
    np.testing.assert_allclose(p6._hist(1).cav_share, [0.3, 0.1], rtol=1e-6)
    np.testing.assert_allclose(proc._hist(0).cav_share, _hist().cav_share)   # 4-vector: kept
    legacy = _entry_process(off.entry_spec, [0, 1], off.state["s_m"].shape[1], "sumo_like_v2")
    assert legacy.queues is None and not legacy.pending.any() and not legacy.cfg.fresh_no_credit


# ------------------------------------------------------------ data: converter + loader
def _requests_xml(path, rows):
    lines = ['<routes>', '  <route id="mainline_route" edges="a b" />',
             '  <route id="ramp_route" edges="r b" />']
    for name, dep, route in rows:
        lines.append(f'  <vehicle id="{name}" type="t" depart="{dep:.2f}" route="{route}" '
                     'departLane="best" departSpeed="max" />')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["</routes>"]) + "\n")


def test_queue_arrays_follow_the_sumo_ledger(tmp_path):
    from wms.data import c4_episodes as C4

    time_s = np.arange(0, 21) * 0.5
    vid = np.full((21, 2), -1, np.int32)
    present = np.zeros((21, 2), bool)
    vid[4:, 0], present[4:, 0] = 0, True          # v0 inserted at 2.0 s
    vid[10:, 1], present[10:, 1] = 1, True        # v1 inserted at 5.0 s
    names = ["m.0", "m.1", "m.2", "r.0"]
    depart = np.array([1.0, 1.5, 3.0, 9.0])
    entrance = np.array([0, 0, 0, 1])
    arrays, stats = C4.episode_queue_arrays(time_s, vid, present, ["m.0", "m.1"], names, depart,
                                            entrance)
    bl = arrays["entry_backlog"]
    assert bl.shape == (21, 2) and arrays["request_counter"].shape == (21, 2, 30)
    np.testing.assert_array_equal(bl[np.searchsorted(time_s, [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 9.0])],
                                  [[0, 0], [1, 0], [2, 0], [1, 0], [2, 0], [1, 0], [1, 1]])
    assert stats == {"n_requests": 4, "queue_unknown": 0, "not_inserted_by_end": 2}
    np.testing.assert_array_equal(arrays["request_counter"][20, :, 0], [3, 1])  # all within 10 s

    xml = tmp_path / "scenario" / "requests.rou.xml"
    _requests_xml(xml, [(n, d, "ramp_route" if e else "mainline_route")
                        for n, d, e in zip(names, depart, entrance)])
    got = C4.read_requests(xml)
    assert got[0] == names
    np.testing.assert_allclose(got[1], depart)
    np.testing.assert_array_equal(got[2], entrance)


def test_cav_entry_counter_is_the_cav_part_of_the_counter():
    from wms.data import c4_episodes as C4

    t_n = 41
    time_s = np.arange(t_n) * 0.5
    vid = np.full((t_n, 3), -1, np.int32)
    present = np.zeros((t_n, 3), bool)
    is_cav = np.zeros((t_n, 3), bool)
    lane = np.zeros((t_n, 3), np.int64)
    vid[4:, 0], present[4:, 0], is_cav[4:, 0] = 0, True, True     # CAV enters at 2 s (mainline)
    vid[10:, 1], present[10:, 1] = 1, True                        # HDV enters at 5 s
    vid[20:, 2], present[20:, 2], is_cav[20:, 2], lane[20:, 2] = 2, True, True, 2   # CAV, ramp, 10 s
    full = C4.episode_counter(time_s, vid, present, lane)
    cav = C4.cav_entry_counter(time_s, vid, present, is_cav, lane)
    assert (cav <= full).all()
    np.testing.assert_array_equal(full[-1].sum(1), [2, 1])
    np.testing.assert_array_equal(cav[-1].sum(1), [1, 1])


def test_converter_and_loader_carry_entry_queue(tmp_path):
    from wms.data import c4_episodes as C4

    collect = tmp_path / "collect" / "episodes" / "random"
    ep = write_episode(collect, seed=4107)
    # requests: v0..v5 (the episode's vehicles) + 3 never inserted mainline requests
    first = {"v0": 0.0, "v1": 0.0, "v2": 0.0, "v3": 0.0, "v4": 17.0, "v5": 12.5}
    rows = [(n, max(0.0, t - 1.0), "ramp_route" if n in ("v4", "v5") else "mainline_route")
            for n, t in first.items()]
    rows += [("m.late0", 5.0, "mainline_route"), ("m.late1", 8.0, "mainline_route"),
             ("m.late2", 14.0, "mainline_route")]
    _requests_xml(ep / "scenario" / "requests.rou.xml", rows)
    write_episode(collect, seed=4108)                  # no request file: no queue arrays
    out = tmp_path / "src"
    doc = C4.convert([tmp_path / "collect"], out, val_frac=0.0)
    by_world = {e["world"]: e for e in doc["episodes"]}
    assert by_world["w4107m2700p20t10"]["queue_arrays"]["n_requests"] == 9
    assert "queue_arrays" not in by_world["w4108m2700p20t10"]
    built = C4.build_c4_datasets(out, horizon_steps=20, entries_v2=True, stride_s=10.0)
    recs = [built["train"][i] for i in range(len(built["train"]))]
    with_q = [r for r in recs if "entry_queue" in r]
    assert with_q and all(r["meta"]["world"].startswith("w4107") for r in with_q)
    r20 = next(r for r in with_q if r["meta"]["t0_s"] == 20.0)
    # t0 = 20 s: every episode vehicle was inserted 1 s after its request (v4, ramp: 16 -> 17 s), so
    # only the 3 never-inserted mainline requests are waiting.
    assert r20["entry_queue"].shape == (6,)
    np.testing.assert_allclose(r20["entry_queue"][:2], [3.0, 0.0])
    np.testing.assert_allclose(r20["entry_queue"][4:], [0.0, 0.0])        # only HDVs entered
    np.testing.assert_allclose(r20["entry_queue"][2:4], [3 / 20.0, 2 / 20.0])  # ages < 20 s (2 bins)
    plain = C4.build_c4_datasets(out, horizon_steps=20, entries_v2=False, stride_s=10.0)["train"][0]
    assert "entry_queue" not in plain

    sampler = StartStateSampler(_ListDataset(with_q), seed=0)
    batch = sampler._batch([sampler._load(i) for i in range(len(with_q))], "cpu")
    assert batch.entry_queue is not None and batch.entry_queue.shape == (len(with_q), 6)
    assert all("_entry_queue" not in m for m in batch.metadata)
    json.dumps(batch.metadata)


class _ListDataset(torch.utils.data.Dataset):
    receipt = {"split": "TRAIN"}

    def __init__(self, records):
        self.records = [{**r, "meta": {**r["meta"], "world": r["meta"]["world"]}} for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


# ---------------------------------------------------------------- WM training plumbing
def test_wm_training_entry_process_switch():
    from wms.train.inputs import make_entry_process
    from wms.train.settings import TrainSettings

    feat = torch.as_tensor(np.stack([_hist((0.5, 0.1)).to_features()] * 2)).float()
    queue = torch.tensor([[40.0, 0.0, 0.7, 0.1], [5.0, 1.0, 0.6, 0.1]])
    present = torch.zeros(2, 61, 6, dtype=torch.bool)

    class _M:
        class physics:
            dt_s = 0.5

    base = dict(entries_v2=True, entry_pad=8, entry_pad_auto=True, entry_insert_rule="sumo_like_v2")
    off = make_entry_process(TrainSettings(**base), {"entry_feat": feat, "entry_queue": queue},
                             present, _M, torch.device("cpu"))
    assert off.queues is None and not off.pending.any()
    on = make_entry_process(TrainSettings(**base, entry_queue_state=True),
                            {"entry_feat": feat, "entry_queue": queue}, present, _M,
                            torch.device("cpu"))
    np.testing.assert_array_equal(on.pending, [[40, 0], [5, 1]])
    six = torch.cat((queue, torch.tensor([[0.9, 0.9], [0.9, 0.9]])), 1)
    on6 = make_entry_process(TrainSettings(**base, entry_queue_state=True),
                             {"entry_feat": feat, "entry_queue": six}, present, _M,
                             torch.device("cpu"))
    assert all(q.cav_share is None for q in on6.queues)       # no CAV entrants in WM training
    np.testing.assert_allclose(on6._hist(0).cav_share, _hist().cav_share)
    assert on.cfg.fresh_no_credit and on.train_pad > off.train_pad
    with pytest.raises(ValueError, match="entry_queue"):
        make_entry_process(TrainSettings(**base, entry_queue_state=True), {"entry_feat": feat},
                           present, _M, torch.device("cpu"))
    with pytest.raises(ValueError, match="entry_pad_auto"):
        make_entry_process(TrainSettings(**{**base, "entry_pad_auto": False},
                                         entry_queue_state=True),
                           {"entry_feat": feat, "entry_queue": queue}, present, _M,
                           torch.device("cpu"))


def test_config_key_round_trip():
    from wms.config.legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT
    from wms.config.train import TrainConfig
    from wms.train.settings import TrainSettings

    assert FLAT_TO_PATH["entry_queue_state"] == "data.entry_queue_state"
    assert "entry_queue_state" in OMIT_WHEN_DEFAULT
    assert "entry_queue_state" not in TrainConfig().to_legacy_flat()      # legacy flat unchanged
    cfg = TrainConfig.from_dict({"data": {"entries_v2": True, "entry_queue_state": True}})
    assert TrainSettings.from_config(cfg).entry_queue_state is True
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()).data.entry_queue_state is True
    assert TrainSettings.from_config(TrainConfig()).entry_queue_state is False
