"""V6ENTRY (ENTRY_DIAG.md): the "sumo_like_v2" saturated-entrance insertion rule.

SUMO 1.27 with ``departSpeed="max"`` on a QUEUED entrance (the v5 textbook world's mainline
source, demand 2400 > ~1930 veh/h capacity) admits a departure the moment the END-OF-STEP
bumper gap to the lane's rearmost vehicle clears the vType minGap (the rearmost vehicle
keeps moving during the insertion step) and inserts at the STOP speed behind that rear
position (the leader gets no speed credit; recorded median entry speed 0.15 m/s at a 3.4
m/s leader).  ``sumo_like`` tests the pre-step gap and credits the leader speed, so on the
recorded saturated states it refuses ~100 % of SUMO's actual insertions.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.control import srcgrad
from wms.control.contract import PHYSICS_DT_S
from wms.model import entries as E
from wms.model.entries import EntranceHistory, EntryConfig, TorchEntryProcessV2

REPO = Path(__file__).resolve().parents[2]
DT = 0.5


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


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


def _hist(rate=(0.0, 0.0)):
    return dataclasses.replace(EntranceHistory.defaults(), rate=np.asarray(rate, float))


def _proc(rule, pool, pending=(0, 0), hist=None, seed=0, **kw):
    proc = TorchEntryProcessV2(histories=[hist or _hist()], dt_s=DT, seed=seed, batch=1,
                               cfg=EntryConfig(max_pending=10 ** 9, insert_rule=rule, **kw),
                               pool=pool)
    proc._emit_due = lambda b: []
    proc.pending[0] = list(pending)
    return proc


def _rear_car(ln, bumper_gap, v_lead, length=5.0):
    """A lane-``ln`` rearmost vehicle whose REAR sits ``bumper_gap`` ahead of the SUMO
    departPos="base" insertion front (entry start + length + eps)."""
    front = E.ENTRY_S_M[ln] + 5.0 + E.SUMO_POSITION_EPS_M
    return [(front + bumper_gap + length, v_lead, ln, length)]


# --------------------------------------------------------------------------- defaults
def test_rules_and_defaults():
    assert EntryConfig().insert_rule == "v2"
    assert "sumo_like_v2" in E.INSERT_RULES == ("v2", "sumo_like", "sumo_like_v2")
    with pytest.raises(ValueError):
        EntryConfig(insert_rule="sumo_v3")


def test_default_and_sumo_like_bit_identical_to_their_namesakes():
    """Default == explicit "v2"; "sumo_like" keeps the follow-safe branch (checked analytically
    below), so only "sumo_like_v2" takes the new branch."""
    cars = _busy(3)
    hist = _hist(rate=(0.8, 0.3))
    outs = []
    for rule in (None, "v2", "sumo_like"):
        state, pool = _state(cars, n_pad=40)
        cfg = EntryConfig(max_pending=10 ** 9) if rule is None \
            else EntryConfig(max_pending=10 ** 9, insert_rule=rule)
        proc = TorchEntryProcessV2(histories=[hist], dt_s=DT, seed=3, batch=1, cfg=cfg, pool=pool)
        masks = []
        for _ in range(30):
            with torch.no_grad():
                state["s_m"] += state["v_mps"] * DT * state["present"]
            masks.append(proc.insert(state).clone())
        outs.append((state["s_m"].clone(), masks, proc.summary()))
    (s1, m1, x1), (s2, m2, x2) = outs[0], outs[1]
    assert x1 == x2 and torch.equal(s1, s2)
    assert all(torch.equal(a, b) for a, b in zip(m1, m2))


def _busy(seed):
    rng = np.random.default_rng(seed)
    cars = []
    for ln in (0, 1, 2):
        s = E.ENTRY_S_M[ln] + rng.uniform(5.0, 40.0)
        for _ in range(4):
            cars.append((s, rng.uniform(0.0, 25.0), ln, 5.0))
            s += rng.uniform(8.0, 60.0)
    return cars


# ------------------------------------------------------------------- the rule's semantics
def test_saturated_queue_sumo_like_v2_admits_where_sumo_like_refuses():
    """The recorded saturated-entrance scene: both mainline lanes carry a near-stopped queue
    head (bumper gap 1.0/0.8 m ahead of the insertion front, moving 3.4/3.0 m/s).  SUMO
    inserts (end-of-step gap 1.0 + 3.4*0.5 = 2.7 m >= minGap 2.0); sumo_like refuses both
    lanes (pre-step gaps < 2.5); sumo_like_v2 admits on the larger-gap lane at the stop
    speed."""
    cars = _rear_car(0, bumper_gap=1.0, v_lead=3.4) + _rear_car(1, bumper_gap=0.8, v_lead=3.0)
    state, pool = _state(cars)
    like = _proc("sumo_like", pool, pending=(1, 0))
    assert not bool(like.insert(state).any()) and like.backlog()[0] == 1.0
    state, pool = _state(cars)
    v2 = _proc("sumo_like_v2", pool, pending=(1, 0))
    mask = v2.insert(state)
    assert int(mask.sum()) == 1 and v2.backlog()[0] == 0.0
    k = int(torch.nonzero(mask[0])[0])
    assert int(state["lane"][0, k]) == 0                       # larger end-of-step gap
    v_new = float(state["v_mps"][0, k])
    expected = E.sumo_safe_insert_speed(1.0 + 3.4 * DT, 0.0, E.SUMO_V2_TAU_S[0],
                                        E.SUMO_V2_DECEL_MPS2[0], E.SUMO_V2_MIN_GAP_M[0])
    assert v_new == pytest.approx(expected, abs=1e-9)
    assert 0.0 < v_new < 1.5                    # stop speed, not the 2+ m/s follow speed
    assert float(state["s_m"][0, k]) == pytest.approx(E.ENTRY_S_M[0] + 5.0 + E.SUMO_POSITION_EPS_M)


def test_sumo_like_v2_stop_speed_is_maximal_safe_and_refuses_inside_min_gap():
    ln = 2
    bumper = 3.0                                  # end-of-step gap with a stationary leader
    cars = _rear_car(ln, bumper_gap=bumper, v_lead=0.0)
    state, pool = _state(cars)
    v2 = _proc("sumo_like_v2", pool, pending=(0, 1))
    mask = v2.insert(state)
    assert int(mask.sum()) == 1
    k = int(torch.nonzero(mask[0])[0])
    v_new = float(state["v_mps"][0, k])
    assert E.sumo_insert_ok(bumper, v_new, 0.0, E.SUMO_V2_TAU_S[0], E.SUMO_V2_DECEL_MPS2[0],
                            E.SUMO_V2_MIN_GAP_M[0])
    assert not E.sumo_insert_ok(bumper, v_new + 0.05, 0.0, E.SUMO_V2_TAU_S[0],
                                E.SUMO_V2_DECEL_MPS2[0], E.SUMO_V2_MIN_GAP_M[0])
    # a moving leader buys gap (its within-step movement) but no speed credit: two scenes
    # with the same END-OF-STEP gap enter at the same stop speed
    cars = _rear_car(ln, bumper_gap=3.0, v_lead=20.0)          # end-of-step gap 3 + 10 = 13
    state, pool = _state(cars)
    v2 = _proc("sumo_like_v2", pool, pending=(0, 1))
    mask = v2.insert(state)
    k = int(torch.nonzero(mask[0])[0])
    v_moving = float(state["v_mps"][0, k])
    cars = _rear_car(ln, bumper_gap=13.0, v_lead=0.0)          # same end-of-step gap, still leader
    state, pool = _state(cars)
    v2 = _proc("sumo_like_v2", pool, pending=(0, 1))
    mask = v2.insert(state)
    k = int(torch.nonzero(mask[0])[0])
    assert v_moving == pytest.approx(float(state["v_mps"][0, k]), abs=1e-9)
    # inside minGap even after the leader's step: refused
    cars = _rear_car(ln, bumper_gap=1.5, v_lead=0.5)   # end-of-step gap 1.75 < 2.0
    state, pool = _state(cars)
    v2 = _proc("sumo_like_v2", pool, pending=(0, 1))
    assert not bool(v2.insert(state).any()) and v2.backlog()[0] == 1.0
    # free lane: both caps
    state, pool = _state([])
    v2 = _proc("sumo_like_v2", pool, pending=(1, 1))
    mask = v2.insert(state)
    v = sorted(float(x) for x in state["v_mps"][mask])
    assert v == pytest.approx(sorted([E.SUMO_LANE_SPEED_MPS[0], E.SUMO_LANE_SPEED_MPS[2]]))


@pytest.mark.parametrize("lane_mode", ["best_free", "any"])
def test_sumo_like_v2_never_inserts_unsafe_or_overlapping(lane_mode):
    rng = np.random.default_rng(11)
    hist = _hist(rate=(1.5, 0.6))
    checked = 0
    for trial in range(30):
        cars = []
        for ln in (0, 1, 2):
            s = E.ENTRY_S_M[ln] + rng.uniform(5.0, 35.0)
            for _ in range(int(rng.integers(0, 4))):
                cars.append((s, rng.uniform(0.0, 30.0), ln, float(rng.choice([5.0, 7.5]))))
                s += rng.uniform(5.5, 40.0)
        state, pool = _state(cars, n_pad=20)
        proc = _proc("sumo_like_v2", pool,
                     pending=(int(rng.integers(0, 6)), int(rng.integers(0, 3))),
                     hist=hist, seed=trial, sumo_lane_mode=lane_mode)
        before = state["present"].clone()
        mask = proc.insert(state)[0]
        for k in torch.nonzero(mask).flatten().tolist():
            ln = int(state["lane"][0, k])
            s_k, v_k = float(state["s_m"][0, k]), float(state["v_mps"][0, k])
            ahead = [j for j in range(state["s_m"].shape[1])
                     if bool(state["present"][0, j]) and j != k
                     and int(state["lane"][0, j]) == ln and float(state["s_m"][0, j]) > s_k]
            assert v_k >= 0.0
            if not ahead:
                continue
            j = min(ahead, key=lambda i: float(state["s_m"][0, i]))
            gap = float(state["s_m"][0, j] - state["length_m"][0, j]) - s_k
            vt = 0
            assert gap + float(state["v_mps"][0, j]) * DT \
                >= E.SUMO_V2_MIN_GAP_M[vt] - 1e-9     # end-of-step minGap kept, no overlap
            if bool(before[0, j]):
                g_eff = gap + float(state["v_mps"][0, j]) * DT
                assert E.sumo_insert_ok(g_eff, v_k, 0.0, E.SUMO_V2_TAU_S[vt],
                                        E.SUMO_V2_DECEL_MPS2[vt], E.SUMO_V2_MIN_GAP_M[vt])
            checked += 1
    assert checked > 10


def test_sumo_like_v2_admits_a_saturated_queue_one_step_earlier():
    """Service of a crawling queue: with the leader 1.0 m ahead at 3.4 m/s, sumo_like_v2
    serves the first arrival immediately; sumo_like must wait for the pre-step bumper gap
    to clear its 2.5 m minGap (a later step)."""
    for rule in ("sumo_like", "sumo_like_v2"):
        cars = _rear_car(0, bumper_gap=1.0, v_lead=3.4) + _rear_car(1, bumper_gap=0.8, v_lead=3.0)
        state, pool = _state(cars, n_pad=10)
        proc = _proc(rule, pool, pending=(6, 0))
        trace = []
        for _ in range(8):
            proc.insert(state)
            with torch.no_grad():                 # the queue leader creeps forward
                state["s_m"] += state["v_mps"] * DT * state["present"]
            trace.append(float(proc.backlog()[0]))
        if rule == "sumo_like_v2":
            assert trace[0] == 4.0                # both lanes serve one arrival in step 0
        else:
            assert trace[0] == 6.0                # refused: pre-step gaps 1.0/0.8 < 2.5


# ------------------------------------------------------------------------------ plumbing
def test_trainer_and_data_plumbing():
    from wms.config.train import TrainConfig
    from wms.control.dreamer import DreamerConfig, _entry_process
    from wms.train.settings import TrainSettings

    assert DreamerConfig().entry_insert_rule == "v2"
    cfg = DreamerConfig(entry_insert_rule="sumo_like_v2")
    assert cfg.entry_insert_rule == "sumo_like_v2"
    spec = {"histories": [EntranceHistory.defaults()], "seeds": [1], "k0": 4}
    assert _entry_process(spec, [0], 8, "sumo_like_v2").cfg.insert_rule == "sumo_like_v2"
    with pytest.raises(ValueError):
        DreamerConfig(entry_insert_rule="bogus")
    full = TrainConfig.from_dict({"data": {"entries_v2": True,
                                           "entry_insert_rule": "sumo_like_v2"}})
    assert TrainSettings.from_config(full).entry_insert_rule == "sumo_like_v2"
    assert TrainConfig.from_legacy_flat(full.to_legacy_flat()).data.entry_insert_rule \
        == "sumo_like_v2"
    assert TrainSettings.from_config(TrainConfig()).entry_insert_rule == "v2"


def test_v6_entry_config_file():
    from wms.control.dreamer import DreamerConfig
    from wms.control.train_cli import _read_config

    raw = _read_config(REPO / "configs" / "c4_policy_v6_entry.toml")
    v5 = _read_config(REPO / "configs" / "c4_policy_v5.toml")
    cfg = DreamerConfig(**raw["trainer"])
    assert cfg.entry_insert_rule == "sumo_like_v2"
    assert cfg.source_wait_grad and cfg.cav_exec_cap == "idm_normal:textbook"
    changed = {k for k in set(raw["trainer"]) | set(v5["trainer"])
               if raw["trainer"].get(k) != v5["trainer"].get(k)}
    assert changed == {"entry_insert_rule"}


# ------------------------------------------------------------------ the soft queue surrogate
def _hist_tensors(vtype=None):
    h = EntranceHistory.defaults()
    if vtype is not None:
        h.vtype_share = np.asarray(vtype, np.float64)
    names = ("rate", "speed_mean", "gap_mean", "gap_std", "vtype_share")
    return h, {n: torch.as_tensor(getattr(h, n), dtype=torch.float64)[None] for n in names}


def _ramp_state(rear_s, rear_v, k=4):
    s = torch.full((1, k), -2000.0, dtype=torch.float64)
    s[0, 0] = rear_s
    v = torch.zeros(1, k, dtype=torch.float64)
    v[0, 0] = rear_v
    present = torch.zeros(1, k, dtype=torch.bool)
    present[0, 0] = True
    return {"s_m": s, "v_mps": v, "lane": torch.full((1, k), 2), "present": present,
            "a_mps2": torch.zeros(1, k, dtype=torch.float64),
            "is_cav": torch.zeros(1, k, dtype=torch.bool),
            "length_m": torch.full((1, k), 5.0, dtype=torch.float64)}


def test_surrogate_sumo_like_v2_follows_the_end_of_step_gap():
    _, hist = _hist_tensors()
    # rearmost ramp vehicle with bumper gap 1.0 m to the insertion front, moving 3.4 m/s:
    # off = 5.1 (front offset) + 1.0 (bumper) measured from the entry start
    state = _ramp_state(E.ENTRY_S_M[2] + 5.0 + 5.1 + 1.0, 3.4)
    like = srcgrad.entrance_block_probability(state, state["present"], hist, E.ENTRY_S_M,
                                              EntryConfig(insert_rule="sumo_like"))
    v2 = srcgrad.entrance_block_probability(state, state["present"], hist, E.ENTRY_S_M,
                                            EntryConfig(insert_rule="sumo_like_v2"))
    assert float(like[0, 1]) > 0.75           # pre-step gap 1.0 < 2.5: blocked-leaning
    assert float(v2[0, 1]) < 0.5              # end-of-step 2.7 >= 2.0: free-leaning
    assert float(v2[0, 1]) < float(like[0, 1]) - 0.3
    still = _ramp_state(E.ENTRY_S_M[2] + 5.0 + 5.1 + 1.5, 0.5)  # end-of-step 1.75 < 2.0
    v2b = srcgrad.entrance_block_probability(still, still["present"], hist, E.ENTRY_S_M,
                                             EntryConfig(insert_rule="sumo_like_v2"))
    assert float(v2b[0, 1]) > 0.5
    # the leader's within-step movement is what lowers the probability
    parked = _ramp_state(E.ENTRY_S_M[2] + 5.0 + 5.1 + 1.5, 0.0)
    v2c = srcgrad.entrance_block_probability(parked, parked["present"], hist, E.ENTRY_S_M,
                                             EntryConfig(insert_rule="sumo_like_v2"))
    assert float(v2b[0, 1]) < float(v2c[0, 1])


def test_surrogate_v2_agrees_with_the_process_decision():
    share = np.zeros((2, 4))
    share[:, 0] = 1.0
    _, hist = _hist_tensors(share)
    cfg = EntryConfig(insert_rule="sumo_like_v2")
    rng = np.random.default_rng(5)
    agree = 0
    for _ in range(120):
        off = float(rng.uniform(0.0, 25.0))
        rear_v = float(rng.uniform(0.0, 25.0))
        state = _ramp_state(E.ENTRY_S_M[2] + 5.0 + off, rear_v)
        p = float(srcgrad.entrance_block_probability(state, state["present"], hist,
                                                     E.ENTRY_S_M, cfg)[0, 1])
        proc = TorchEntryProcessV2(history=EntranceHistory.defaults(), dt_s=PHYSICS_DT_S,
                                   seed=0, batch=1,
                                   cfg=EntryConfig(max_pending=10 ** 9,
                                                   insert_rule="sumo_like_v2"))
        proc._observe(state)
        proc._started[:] = True
        proc.t_next[:] = float("inf")
        proc.pending[0, 1] = 1
        admitted = bool(proc.insert(state).any())
        assert admitted == (off - 5.1 + max(0.0, rear_v) * DT >= E.SUMO_V2_MIN_GAP_M[0])
        if abs(off - 5.1 + max(0.0, rear_v) * DT - E.SUMO_V2_MIN_GAP_M[0]) > 3.0:
            agree += int((p < 0.5) == admitted)
    assert agree > 100


def test_surrogate_v2_gradient_sign():
    _, hist = _hist_tensors()
    cfg = EntryConfig(insert_rule="sumo_like_v2")
    s = torch.tensor(E.ENTRY_S_M[2] + 5.0 + 5.1 + 0.5, dtype=torch.float64, requires_grad=True)
    state = _ramp_state(0.0, 3.4)
    state["s_m"] = torch.cat((s[None, None], state["s_m"][:, 1:]), -1)
    p = srcgrad.entrance_block_probability(state, state["present"], hist, E.ENTRY_S_M, cfg)
    (grad,) = torch.autograd.grad(p[0, 1], s)
    assert 0.2 < float(p[0, 1].detach()) < 0.8 and float(grad) < 0.0


# ------------------------------------------------ recorded v5 episode (skip when absent)
_POLICY_EP = REPO / "_ref" / "eval" / "c01" / "episodes" / "policy_best" / "s4107_m2400_p0400"


@pytest.mark.skipif(not (_POLICY_EP / "episode.npz").is_file(),
                    reason="recorded v5 policy episode not available on this host")
def test_recorded_v5_saturated_insertions():
    """On SUMO's recorded pre-insertion states of the v5 policy episode: sumo_like admits
    < 5 % of the saturated mainline insertions SUMO actually made; sumo_like_v2 >= 70 %."""
    with np.load(_POLICY_EP / "episode.npz", allow_pickle=False) as z:
        t, present = z["time_s"], z["present"]
        lane, s, v = z["lane"], z["s_m"], z["v_mps"]
        vid_index, ln = z["vid_index"], z["length_true_m"]
    ins_tick, ins_lane = {}, {}
    for i in range(len(t)):
        for k in np.flatnonzero(present[i] & (vid_index[i] >= 0)):
            vi = int(vid_index[i, k])
            if vi not in ins_tick:
                ins_tick[vi] = i
                ins_lane[vi] = int(lane[i, k])
    by_tick = {}
    for vi, i in ins_tick.items():
        by_tick.setdefault(i, set()).add(ins_lane[vi])
    rates = {}
    for rule in ("sumo_like", "sumo_like_v2"):
        ok = n = 0
        for i in range(600, 1798):
            for l_idx in (0, 1):
                if l_idx not in by_tick.get(i + 1, set()):
                    continue
                es = E.ENTRY_S_M[l_idx]
                sel = np.flatnonzero(present[i] & (lane[i] == l_idx))
                if not sel.size:
                    continue
                k = int(sel[np.argmin(s[i, sel])])
                gap = float(s[i, k]) - es - float(ln[i, k]) - (5.0 + E.SUMO_POSITION_EPS_M)
                v_lead = float(v[i, k])
                if rule == "sumo_like":
                    admitted = E.sumo_safe_insert_speed(gap, v_lead, E.VTYPE_TAU_S[0],
                                                        E.VTYPE_DECEL_MPS2[0],
                                                        E.VTYPE_MIN_GAP_M[0]) >= 0.0
                else:
                    admitted = E.sumo_safe_insert_speed(
                        gap + max(0.0, v_lead) * DT, 0.0, E.SUMO_V2_TAU_S[0],
                        E.SUMO_V2_DECEL_MPS2[0], E.SUMO_V2_MIN_GAP_M[0]) >= 0.0
                ok += int(admitted)
                n += 1
        rates[rule] = ok / max(1, n)
    assert n > 200
    assert rates["sumo_like"] < 0.05
    assert rates["sumo_like_v2"] > 0.70
