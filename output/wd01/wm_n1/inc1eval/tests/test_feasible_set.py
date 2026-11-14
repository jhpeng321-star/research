"""A1.3/A1.4 feasible-set handling in the INC1 metrics (zero simulation)."""
from __future__ import annotations

import numpy as np

from inc1eval import metrics as M

ARMS = ["R0", "R1", "R2", "R3", "R4"]


def _inputs():
    true = np.array([[10.0, 5.0, 20.0, 30.0, 1.0],
                     [0.0, 50.0, 10.0, 20.0, 5.0]])
    pred = np.array([[10.0, 4.0, 5.0, 6.0, 0.0],
                     [5.0, 50.0, 3.0, 20.0, 5.0]])
    feasible = np.array([[True, True, True, True, False],
                         [True, False, True, True, False]])
    return M.MetricsInputs(
        worlds=["w0", "w1"], arms=list(ARMS),
        delay=M.Channel(true=true, idea_a=pred, v2g=pred.copy()),
        ttc_lt3_s=M.Channel(true=true * 0.1, idea_a=pred * 0.1, v2g=pred * 0.1),
        hard_brake_s=M.Channel(true=true * 0.2, idea_a=pred * 0.2,
                               v2g=pred * 0.2),
        feasible=feasible,
        feasibility={"reasons": {"R4": ["proposer_refusal:no_pair"],
                                 "R1": ["r1_no_qualifier"]},
                     "per_world": {"w1": {"R1": "r1_no_qualifier"}}})


def test_feasible_regret_is_primary_and_computed_on_feasible_set():
    report = M.compute_report(_inputs(), n_boot=16, seed=1)
    d = report["delay"]["idea_a"]
    # world0: WM picks R1 (feasible); feasible best is R1 -> regret 0
    # world1: WM picks R2; feasible best is R0 -> regret 10
    assert d["feasible_set_regret_s_per_veh"]["point"] == 5.0
    # the R4 (infeasible) prediction never contaminates the pick
    assert report["feasible_set"]["primary_metric"] == \
        "feasible_set_regret_s_per_veh"


def test_feasible_top1_and_spearman_use_feasible_arms_only():
    report = M.compute_report(_inputs(), n_boot=16, seed=1)
    d = report["delay"]["idea_a"]
    assert d["top1_hit"]["point"] == 0.5     # w0 hit, w1 miss
    # R4 is infeasible in both worlds, so it is excluded from the counts
    assert report["feasible_set"]["arm_feasible_counts"]["R4"] == 0
    assert report["feasible_set"]["arm_feasible_counts"]["R1"] == 1


def test_feasible_set_sizes_and_reasons_reported():
    report = M.compute_report(_inputs(), n_boot=16, seed=1)
    sizes = report["feasible_set"]["per_world_size"]
    assert sizes == {"w0": 4, "w1": 3}
    assert report["feasible_set"]["mean_size"] == 3.5
    assert report["feasible_set"]["reasons"]["R4"] == [
        "proposer_refusal:no_pair"]


def test_v2g_feasible_regret_matches_registered_definition():
    report = M.compute_report(_inputs(), n_boot=16, seed=1)
    v2g = report["delay"]["v2g"]
    assert v2g["feasible_set_regret_s_per_veh"]["point"] == 5.0


def test_a2_4_descriptive_breakdowns():
    """ADDENDUM_2 A2.4: multi- vs single-CAV regret, multi-CAV-best top-1, OOD."""
    arms = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
    true = np.array([[10.0, 8.0, 9.0, 5.0, 6.0, 3.0, 4.0],
                     [10.0, 2.0, 9.0, 5.0, 6.0, 7.0, 8.0]])
    pred = np.array([[10.0, 8.0, 9.0, 5.0, 6.0, 9.0, 4.0],
                     [10.0, 2.0, 9.0, 5.0, 6.0, 7.0, 8.0]])
    feasible = np.ones_like(true, dtype=bool)
    ood = np.tile(np.arange(7, dtype=float), (2, 1))
    inputs = M.MetricsInputs(
        worlds=["w0", "w1"], arms=arms,
        delay=M.Channel(true=true, idea_a=pred, v2g=true.copy()),
        ttc_lt3_s=M.Channel(true=true, idea_a=pred, v2g=pred.copy()),
        hard_brake_s=M.Channel(true=true, idea_a=pred, v2g=pred.copy()),
        feasible=feasible, ood=ood)
    report = M.compute_report(inputs, n_boot=8, seed=1)
    br = report["a2_4_breakdown"]
    assert br["multi_cav_arms"] == ["R3", "R4", "R5", "R6"]
    assert br["single_cav_arms"] == ["R1", "R2"]
    # world0 SUMO-best is R5 (multi); world1 SUMO-best is R1 (single)
    assert br["worlds_with_multi_cav_sumo_best"] == 1
    # WM mispicks R5 in world0 (regret 4-3=1); world1 is correct -> mean 0.5
    multi = br["delay_feasible_set_regret_s_per_veh"]["multi_cav"]["idea_a"]
    assert multi["point"] == 0.5 and multi["n"] == 2
    single = br["delay_feasible_set_regret_s_per_veh"]["single_cav"]["idea_a"]
    assert single["point"] == 0.0
    assert br["top1_on_multi_cav_best_worlds"]["idea_a"]["point"] == 0.0
    assert br["top1_on_multi_cav_best_worlds"]["v2g"]["point"] == 1.0
    assert br["ood_per_arm"]["R5"] == 5.0
    assert br["ood_per_arm"]["R0"] == 0.0


def test_world_with_no_feasible_arm_is_skipped():
    true = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    feasible = np.array([[True, True], [False, False]])
    inputs = M.MetricsInputs(
        worlds=["w0", "w1"], arms=["R0", "R1"],
        delay=M.Channel(true=true, idea_a=pred, v2g=pred),
        ttc_lt3_s=M.Channel(true=true, idea_a=pred, v2g=pred),
        hard_brake_s=M.Channel(true=true, idea_a=pred, v2g=pred),
        feasible=feasible)
    report = M.compute_report(inputs, n_boot=8, seed=1)
    point = report["delay"]["idea_a"]["feasible_set_regret_s_per_veh"]["point"]
    assert point == 0.0                    # the empty world is excluded
    assert report["feasible_set"]["per_world_size"]["w1"] == 0
    assert report["feasible_set"]["per_world_size"]["w0"] == 2
