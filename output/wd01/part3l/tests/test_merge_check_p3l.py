"""Part 3l TASK A tests: merge-rejection cause attribution + follower
slot semantics.  Synthetic scenes only (Rollout.__new__), no data, no
weights, no SUMO -- same pattern as part3/tests/test_p3_merge.py.

The key-name test pins p3l_lib's MS_* constants to the engine's actual
merge_stats dict at runtime, so a transcription typo in either file
fails loudly here instead of silently in a long run.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part3"))

import engine as E  # noqa: E402
import p3l_lib as L  # noqa: E402


def mini_rollout(lane, s, v, merge_params=None):
    """Hand-built idm_merge Rollout stub with only what
    _gap_accept_merge touches (no p3d nets: this method is pure
    numpy bookkeeping + the pure acceptance function)."""
    n = len(lane)
    r = E.Rollout.__new__(E.Rollout)
    r.dev = torch.device("cpu")
    r.kind = "idm_merge"
    r.vid = np.arange(n)
    r.lane = torch.tensor(lane, dtype=torch.long)
    r.s = torch.tensor(s, dtype=torch.float64)
    r.v = torch.tensor(v, dtype=torch.float64)
    r.typ = np.ones(n, dtype=np.int64)
    r.vtype = np.ones(n, dtype=np.int64)      # normal: s0 = 2.5
    r.mp = dict(E.MERGE_PARAMS_DEFAULT)
    if merge_params:
        r.mp.update(merge_params)
    return r


def acc_parts(gap_lead, v_ego, gap_foll, v_foll, s0=2.5,
              tau=1.1, b=3.0):
    ok, ol, of = E.merge_acceptance(
        np.array([gap_lead]), np.array([v_ego]), np.array([s0]),
        np.array([gap_foll]), np.array([v_foll]), np.array([s0]),
        tau, b, return_parts=True)
    return bool(ok[0]), bool(ol[0]), bool(of[0])


# --------------------------------------------------- attribution units --

def test_acceptance_parts_follower_only_rejection():
    # leader far (lead ok), fast follower 5 m behind closing at +15 m/s:
    # b_req = (25^2-10^2) / (2*(5-2.5)) = 210 > 3
    ok, ol, of = acc_parts(250.0, 10.0, 5.0, 25.0)
    assert (ok, ol, of) == (False, True, False)


def test_acceptance_parts_lead_only_rejection():
    # 10 m gap to leader at v=10: minGap + 10*1.1 = 13.5 > 10; follower
    # 200 m behind and slower: b_req < 0
    ok, ol, of = acc_parts(10.0, 10.0, 200.0, 0.0)
    assert (ok, ol, of) == (False, False, True)


def test_acceptance_parts_both_rejected():
    ok, ol, of = acc_parts(5.0, 10.0, 5.0, 25.0)
    assert (ok, ol, of) == (False, False, False)


def test_acceptance_default_path_unchanged():
    gl = np.array([250.0, 5.0, 13.5])
    ve = np.array([10.0, 10.0, 10.0])
    gf = np.array([250.0, 250.0, 250.0])
    vf = np.array([0.0, 0.0, 0.0])
    s0 = np.full(3, 2.5)
    a = E.merge_acceptance(gl, ve, s0, gf, vf, s0, 1.1, 3.0)
    b, ol, of = E.merge_acceptance(gl, ve, s0, gf, vf, s0, 1.1, 3.0,
                                   return_parts=True)
    assert np.array_equal(a, b)


# ------------------------------------------- engine stats key spelling --

def test_merge_stats_keys_match_lib_constants():
    """Engine merge_stats must use EXACTLY the p3l_lib MS_* value
    strings (this is the guard against key transcription typos)."""
    r = mini_rollout([2], [400.0], [20.0])
    out = r._gap_accept_merge()
    assert bool(out[0]) is True
    assert set(r.merge_stats.keys()) == {
        L.MS_CHECKS, L.MS_ACCEPTED, L.MS_REJ_LEAD, L.MS_REJ_FOLL,
        L.MS_REJ_BOTH, L.MS_WITH_FOLL, L.MS_BREQ_HIST, L.MS_BREQ_FOLL_REJ}
    assert r.merge_stats[L.MS_CHECKS] == 1
    assert r.merge_stats[L.MS_ACCEPTED] == 1


def test_stats_count_follower_only_rejection():
    # ego d = 378.66-430 = -51.34; follower d = 2644.22-2690.56 = -46.34
    # (5 m behind), closing at +5 m/s from 25 vs 20: b_req = 45 > 3
    r = mini_rollout([2, 1], [430.0, 2690.56], [20.0, 25.0],
                     merge_params=dict(tau_accept=1.9, b_accept=3.0))
    out = r._gap_accept_merge()
    assert not bool(out[0])
    assert r.merge_stats[L.MS_REJ_FOLL] == 1
    assert r.merge_stats[L.MS_REJ_LEAD] == 0
    assert r.merge_stats[L.MS_REJ_BOTH] == 0
    assert r.merge_stats[L.MS_WITH_FOLL] == 1
    breq = r.merge_stats[L.MS_BREQ_FOLL_REJ]
    assert len(breq) == 1 and breq[0] == pytest.approx(45.0)
    # histogram: b_req = 45 lands in the > 3 bin
    assert r.merge_stats[L.MS_BREQ_HIST] == [0, 0, 0, 0, 0, 1]


def test_stats_count_lead_only_rejection():
    # d decreases downstream: the leader sits 10 m AHEAD (d_ego - 10),
    # the parked vehicle 200 m BEHIND (d_ego + 200)
    d_ego = 378.66 - 430.0                    # -51.34
    s_lead = 2644.22 - (d_ego - 10.0)         # ahead
    s_foll = 2644.22 - (d_ego + 200.0)        # behind
    r = mini_rollout([2, 1, 1], [430.0, s_lead, s_foll],
                     [20.0, 20.0, 0.0])
    out = r._gap_accept_merge()
    assert not bool(out[0])
    assert r.merge_stats[L.MS_REJ_LEAD] == 1
    assert r.merge_stats[L.MS_REJ_FOLL] == 0
    assert r.merge_stats[L.MS_WITH_FOLL] == 1


def test_stats_count_both_rejected():
    # tau_accept = 5.0 -> lead needs 2.5 + 20*5 = 102.5 m (gap is 50);
    # b_accept = 0.5 -> follower b_req = 45 must also reject
    d_ego = 378.66 - 430.0
    s_lead = 2644.22 - (d_ego - 50.0)
    s_foll = 2644.22 - (d_ego + 5.0)
    r = mini_rollout([2, 1, 1], [430.0, s_lead, s_foll],
                     [20.0, 0.0, 25.0],
                     merge_params=dict(tau_accept=5.0, b_accept=0.5))
    out = r._gap_accept_merge()
    assert not bool(out[0])
    assert r.merge_stats[L.MS_REJ_BOTH] == 1


# ----------------------------------------------------- follower slot ----

def test_follower_slot_is_nearest_behind():
    """The M2 follower term must use the NEAREST main_right vehicle
    behind the candidate (larger d).  Scene: leader 60 m ahead (clears
    tau=1.9 acceptance 2.5+20*1.9=40.5), nearest follower 5 m behind
    parked (b_req <= 0 -> passes), second follower 55 m behind closing
    fast (b_req ~ 4.8 > 3 -> would reject if the WRONG, farther slot
    were used).  Merge must fire."""
    d_ego = 378.66 - 430.0                    # -51.34
    lane = [2, 1, 1, 1]
    d = [d_ego, d_ego - 60.0, d_ego + 5.0, d_ego + 55.0]
    v = [20.0, 0.0, 0.0, 30.0]
    s = [430.0] + [2644.22 - x for x in d[1:]]
    r = mini_rollout(lane, s, v,
                     merge_params=dict(tau_accept=1.9, b_accept=3.0))
    out = r._gap_accept_merge()
    assert bool(out[0])                       # nearest (slow) slot used
    assert r.merge_stats[L.MS_ACCEPTED] == 1


def test_follower_slot_rejects_on_fast_nearest():
    """Mirror scene: the NEAREST follower is fast and close (b_req = 45),
    the farther one parked.  The term must bind (reject) -- proving the
    near slot, not the far one, feeds the follower check."""
    d_ego = 378.66 - 430.0
    lane = [2, 1, 1, 1]
    d = [d_ego, d_ego - 60.0, d_ego + 5.0, d_ego + 55.0]
    v = [20.0, 0.0, 25.0, 0.0]
    s = [430.0] + [2644.22 - x for x in d[1:]]
    r = mini_rollout(lane, s, v,
                     merge_params=dict(tau_accept=1.9, b_accept=3.0))
    out = r._gap_accept_merge()
    assert not bool(out[0])
    assert r.merge_stats[L.MS_REJ_FOLL] == 1
    assert r.merge_stats[L.MS_BREQ_FOLL_REJ] == [pytest.approx(45.0)]
