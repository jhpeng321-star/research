"""Grouping: NaN-aware deviations, edges, components, cap splitting."""
import numpy as np
import pytest

from traffic_shepherd.planner import (
    InteractionEdge,
    MockWorldModel,
    PlannerConfig,
    build_edges,
    cap_split,
    components,
    influences,
    trajectory_deviation,
)
from tests.planner._helpers import make_samples, plan

CFG = PlannerConfig(trajectory_s=10.0)


def traj_array(positions, speeds):
    """[1, 1, T, 4] array from position/speed sequences (lane 0)."""
    t = np.arange(len(positions), dtype=float)
    arr = np.zeros((1, 1, len(positions), 4))
    arr[0, 0, :, 0] = t
    arr[0, 0, :, 1] = positions
    arr[0, 0, :, 3] = speeds
    return arr


def edge(a, b, ratio):
    """InteractionEdge whose threshold ratio (= weight) is ``ratio``."""
    return InteractionEdge(a=a, b=b, pos_dev_m=ratio * CFG.interaction_pos_m,
                           speed_dev_mps=0.0, weight=float(ratio))


def chain(n, top_weight):
    """n-node path; edge i (node i -> i+1) has weight top_weight - i."""
    ids = [f"n{i}" for i in range(n)]
    edges = [edge(ids[i], ids[i + 1], top_weight - i) for i in range(n - 1)]
    return ids, edges


def step1_predictions(couplings, cav_ids, k=2, speed_couplings=None):
    """Step-1 style predictions: current plan (all native) + each CAV damping.

    ``couplings``: {(i, j): offset} -- j's position shifts by offset when i
    takes the damping action. ``speed_couplings`` analogous for speed.
    """
    world = MockWorldModel(cav_ids=cav_ids)
    for (i, j), offset in (couplings or {}).items():
        world.traj_coupling_pos[(i, j)] = {"damping": offset}
    for (i, j), offset in (speed_couplings or {}).items():
        world.traj_coupling_speed[(i, j)] = {"damping": offset}

    def predict(overrides):
        return world.predict_many(make_samples(k),
                                  [plan({**{c: "native" for c in cav_ids},
                                         **overrides})],
                                  trajectory_s=10)[0]

    alternatives = {
        cav: {"native": predict({}),
              "damping": predict({cav: "damping"})}
        for cav in cav_ids
    }
    return alternatives, alternatives[cav_ids[0]]["native"]


class TestTrajectoryDeviation:
    def test_identical_trajectories_zero(self):
        arr = traj_array([0, 20, 40], [20, 20, 20])
        pos, speed = trajectory_deviation(arr, arr)
        assert pos == 0.0 and speed == 0.0

    def test_position_and_speed_deviate_independently(self):
        a = traj_array([0, 20, 40], [20, 20, 20])
        b = traj_array([0, 22, 40], [20, 18, 20])
        pos, speed = trajectory_deviation(a, b)
        assert pos == pytest.approx(2.0)
        assert speed == pytest.approx(2.0)

    def test_both_nan_contributes_zero(self):
        nan_arr = traj_array([np.nan, np.nan], [np.nan, np.nan])
        pos, speed = trajectory_deviation(nan_arr, nan_arr.copy())
        assert pos == 0.0 and speed == 0.0

    def test_one_sided_exit_is_infinite_deviation(self):
        a = traj_array([0, 20, np.nan], [20, 20, np.nan])
        b = traj_array([0, 20, 40], [20, 20, 20])
        pos, _ = trajectory_deviation(a, b)
        assert np.isinf(pos)

    def test_max_over_time(self):
        a = traj_array([0, 20, 40], [20, 20, 20])
        b = traj_array([1, 25, 42], [20, 20, 20])
        pos, _ = trajectory_deviation(a, b)
        assert pos == pytest.approx(5.0)


class TestEdges:
    def test_no_interaction_no_edges(self):
        alts, ref = step1_predictions({}, ("c0", "c1"))
        pairwise = influences(alts, ref, ["c0", "c1"])
        assert build_edges(["c0", "c1"], pairwise, CFG) == ()
        assert components(["c0", "c1"], ()) == (("c0",), ("c1",))

    def test_position_threshold_both_sides(self):
        alts, ref = step1_predictions({("c0", "c1"): 2.1}, ("c0", "c1"))
        edges = build_edges(["c0", "c1"], influences(alts, ref, ["c0", "c1"]), CFG)
        assert len(edges) == 1
        assert edges[0].a == "c0" and edges[0].b == "c1"
        assert edges[0].pos_dev_m == pytest.approx(2.1)

        alts, ref = step1_predictions({("c0", "c1"): 2.0}, ("c0", "c1"))
        # exactly at the threshold does NOT exceed it
        assert build_edges(["c0", "c1"], influences(alts, ref, ["c0", "c1"]),
                           CFG) == ()

    def test_speed_threshold_both_sides(self):
        alts, ref = step1_predictions({}, ("c0", "c1"),
                                      speed_couplings={("c0", "c1"): 0.55})
        edges = build_edges(["c0", "c1"], influences(alts, ref, ["c0", "c1"]), CFG)
        assert len(edges) == 1 and edges[0].speed_dev_mps == pytest.approx(0.55)

        alts, ref = step1_predictions({}, ("c0", "c1"),
                                      speed_couplings={("c0", "c1"): 0.5})
        # exactly at the threshold does NOT exceed it
        assert build_edges(["c0", "c1"], influences(alts, ref, ["c0", "c1"]),
                           CFG) == ()

    def test_symmetric_max_of_directions(self):
        alts, ref = step1_predictions({("c0", "c1"): 3.0}, ("c0", "c1"))
        pairwise = influences(alts, ref, ["c0", "c1"])
        assert pairwise == (("c0", "c1", pytest.approx(3.0), pytest.approx(0.0)),)

    def test_median_over_samples_of_max_over_time(self):
        # offset only in one of two samples: max-over-time 4 in that sample,
        # 0 in the other; median over the 2 cells = 2 (below threshold 2? no:
        # strictly greater required) -> no edge.
        world = MockWorldModel(cav_ids=("c0", "c1"))
        world.traj_coupling_pos[("c0", "c1")] = {"damping": np.array([4.0, 0.0])}

        def predict(overrides):
            return world.predict_many(
                make_samples(2), [plan({"c0": overrides.get("c0", "native"),
                                        "c1": overrides.get("c1", "native")})],
                trajectory_s=10)[0]

        alts = {"c0": {"native": predict({}), "damping": predict({"c0": "damping"})},
                "c1": {"native": predict({}), "damping": predict({"c1": "damping"})}}
        edges = build_edges(["c0", "c1"], influences(alts, alts["c0"]["native"],
                                                     ["c0", "c1"]), CFG)
        assert edges == ()

        world.traj_coupling_pos[("c0", "c1")] = {"damping": np.array([5.0, 0.0])}
        alts["c0"]["damping"] = predict({"c0": "damping"})
        edges = build_edges(["c0", "c1"], influences(alts, alts["c0"]["native"],
                                                     ["c0", "c1"]), CFG)
        assert len(edges) == 1 and edges[0].pos_dev_m == pytest.approx(2.5)


class TestComponentsAndCaps:
    def test_chain_stays_whole_below_cap(self):
        ids, edges = chain(5, 10.0)
        assert cap_split(ids, edges, 5) == (tuple(ids),)
        ids, edges = chain(9, 10.0)
        assert cap_split(ids, edges, 9) == (tuple(ids),)

    def test_cap8_splits_off_the_weakest_edge(self):
        ids, edges = chain(9, 10.0)  # weights 10,9,...,3; weakest is n7-n8
        comps = cap_split(ids, edges, 8)
        assert comps == (("n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"),
                         ("n8",))

    def test_cap5_splits_repeatedly_at_weakest(self):
        ids, edges = chain(9, 10.0)
        comps = cap_split(ids, edges, 5)
        assert comps == (("n0", "n1", "n2", "n3", "n4"), ("n5",), ("n6",),
                         ("n7",), ("n8",))

    def test_cap3_splits_repeatedly_at_weakest(self):
        ids, edges = chain(9, 10.0)
        comps = cap_split(ids, edges, 3)
        assert comps == (("n0", "n1", "n2"), ("n3",), ("n4",), ("n5",),
                         ("n6",), ("n7",), ("n8",))

    def test_tie_breaks_by_sorted_ids(self):
        edges = [edge("a", "b", 5.0), edge("b", "c", 5.0)]
        comps = cap_split(["a", "b", "c"], edges, 2)
        # (a,b) removed first: smallest id pair among equal weights
        assert comps == (("a",), ("b", "c"))

    def test_components_sorted_deterministic(self):
        edges = [edge("d", "a", 5.0), edge("b", "c", 4.0)]
        comps = components(["d", "a", "b", "c"], edges)
        assert comps == (("a", "d"), ("b", "c"))

    def test_two_clusters_split_independently(self):
        edges = [edge("a", "b", 10.0), edge("c", "d", 1.0)]
        comps = cap_split(["a", "b", "c", "d"], edges, 1)
        assert comps == (("a",), ("b",), ("c",), ("d",))

    def test_cap_below_one_rejected(self):
        with pytest.raises(ValueError):
            cap_split(["a"], [], 0)
