"""REGISTRATION_ADDENDUM.md section 2 unit proof.

The part-1 RAW overlap counter (t2_run.raw_passthrough, imported unmodified)
sorts by CURRENT d and then tests d[i] <= d[leader] along that same order --
false by construction, so it returns 0 even on a pair that physically passes
through.  The corrected counter (wd2lib.corrected_passthrough) tests the
order at t+1 among pairs adjacent in the d-order at t and flags the follower.
"""
import numpy as np

import wd2lib as W


def part1_raw_passthrough():
    return W.part1_t2().raw_passthrough


def test_old_counter_returns_zero_on_inverted_pair():
    raw_passthrough = part1_raw_passthrough()
    # t: idx1 ahead (d=90), idx0 behind (d=100), same lane group.
    # t+1: follower idx0 ends at 89 < leader idx1 at 95 -> order inverted.
    d_prev = np.array([100.0, 90.0])
    g = np.array([1, 1])
    assert raw_passthrough(d_prev, g).sum() == 0, (
        "part-1 counter flags a genuine inverted pair?!")
    # and on the inverted pair it stays 0 no matter the t+1 state, because it
    # never looks at it: feed the inverted t+1 order back in as its input and
    # it STILL reports nothing.
    assert raw_passthrough(np.array([89.0, 95.0]), g).sum() == 0


def test_corrected_counter_flags_the_inverted_pair():
    d_prev = np.array([100.0, 90.0])
    d_next = np.array([89.0, 95.0])          # follower passes the leader
    g = np.array([1, 1])
    flags = W.corrected_passthrough(d_prev, d_next, g)
    assert flags.tolist() == [True, False]   # the follower is flagged


def test_corrected_counter_no_false_positives():
    d_prev = np.array([100.0, 90.0])
    g = np.array([1, 1])
    # order preserved (leader still ahead)
    assert W.corrected_passthrough(
        d_prev, np.array([95.0, 94.0]), g).sum() == 0
    # equality counts as inversion, per the frozen definition
    assert W.corrected_passthrough(
        d_prev, np.array([95.0, 95.0]), g).tolist() == [True, False]
    # crossing in another lane group is not a violation here
    assert W.corrected_passthrough(
        d_prev, np.array([89.0, 95.0]), np.array([0, 1])).sum() == 0


def test_corrected_counter_adjacency_only():
    # order at t, front to back: idx2 (90) < idx1 (95) < idx0 (100)
    d_prev = np.array([100.0, 95.0, 90.0])
    # idx1 jumps past its adjacent leader idx2 (89 <= 91): flagged.
    # idx0 stays behind idx1: not flagged.  The pair (idx0, idx2) is not
    # adjacent in the t-order, so their relative motion is not counted.
    d_next = np.array([100.0, 89.0, 91.0])
    g = np.array([0, 0, 0])
    flags = W.corrected_passthrough(d_prev, d_next, g)
    assert flags.tolist() == [False, True, False]


def test_corrected_counter_per_lane_group_independence():
    rng = np.random.default_rng(7)
    d_prev = np.sort(rng.uniform(0.0, 300.0, size=40))[::-1].copy()
    d_next = d_prev + rng.normal(0.0, 8.0, size=40)
    g = rng.integers(0, 4, size=40)
    flags = W.corrected_passthrough(d_prev, d_next, g)
    # per-group cross-check against a plain reference implementation
    ref = np.zeros(40, dtype=bool)
    for grp in np.unique(g):
        mem = np.where(g == grp)[0]
        order = mem[np.argsort(d_prev[mem], kind="stable")]
        for j in range(1, len(order)):
            i, lead = int(order[j]), int(order[j - 1])
            if d_next[i] <= d_next[lead]:
                ref[i] = True
    assert (flags == ref).all()
