"""Owner action-descriptor requirement (2026-10-01): descriptor + strata."""
from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import pytest

from evalh import idea_a as ia
from evalh.protocol import QUANTILE_LEVELS, OutcomeRow


def _d(events, t0=0.0, wall=None):
    return ia.descriptor_from_events(events, t0, wall=wall)


def test_damp60_then_native_differs_from_damp570():
    d60 = _d([{"vehicle_id": "c0", "time_s": 0.0, "action": "damping"},
              {"vehicle_id": "c0", "time_s": 60.0, "action": "native"}])
    d570 = _d([{"vehicle_id": "c0", "time_s": 0.0, "action": "damping"}])
    native = ia.ActionDescriptor()
    assert d60 != d570
    assert d60.describe() != d570.describe()
    diff60 = ia.schedule_difference(d60, native)
    diff570 = ia.schedule_difference(d570, native)
    assert diff60.differs and diff60.duration_stratum == "<=60s"
    assert diff60.onset_bin == "0-30s" and diff60.first_60_only
    assert diff570.duration_stratum == "whole-window"
    assert not diff570.first_60_only
    assert ia.schedule_difference(d60, d570).differs


def test_stratifier_assigns_duration_and_onset_bins():
    native = ia.ActionDescriptor()
    d120 = _d([{"vehicle_id": "c", "time_s": 0.0, "action": "damping"},
               {"vehicle_id": "c", "time_s": 120.0, "action": "native"}])
    diff = ia.schedule_difference(d120, native)
    assert diff.duration_stratum == "60-240s" and diff.onset_bin == "0-30s"

    late60 = _d([{"vehicle_id": "c", "time_s": 60.0, "action": "damping"},
                 {"vehicle_id": "c", "time_s": 90.0, "action": "native"}])
    diff = ia.schedule_difference(late60, native)
    assert diff.duration_stratum == "<=60s" and diff.onset_bin == "60-240s"
    assert not diff.first_60_only

    mid_onset = _d([{"vehicle_id": "c", "time_s": 30.0, "action": "damping"},
                    {"vehicle_id": "c", "time_s": 60.0, "action": "native"}])
    assert ia.schedule_difference(mid_onset, native).onset_bin == "30-60s"

    very_late = _d([{"vehicle_id": "c", "time_s": 300.0, "action": "damping"},
                    {"vehicle_id": "c", "time_s": 330.0, "action": "native"}])
    diff = ia.schedule_difference(very_late, native)
    assert diff.onset_bin == "after-240s" and diff.duration_stratum == "<=60s"


def test_wall_scalar_change_stratifies_wall_segments():
    native = ia.ActionDescriptor()
    early = _d([], wall={"target_time_s": 0.0, "hold_s": 240.0,
                         "speed_factor": 0.6, "speed_floor_mps": 2.5})
    fast = _d([], wall={"target_time_s": 0.0, "hold_s": 240.0,
                        "speed_factor": 0.85, "speed_floor_mps": 2.5})
    assert early != fast
    diff = ia.schedule_difference(early, fast)
    assert diff.differs and diff.wall_differs
    assert diff.duration_stratum == "60-240s" and diff.onset_bin == "0-30s"
    assert ia.schedule_difference(early, native).duration_stratum == "60-240s"


def _rows_two_worlds():
    out = []
    for w, y in (("w0", 0.0), ("w1", 0.0)):
        for pol in ("native", "damping"):
            out.append(OutcomeRow(world=w, t0_s=0.0, decision_t_s=0.0,
                                  tau_s=60.0, policy=pol,
                                  y_delay_s=y + (1.0 if pol == "damping"
                                                 else 0.0)))
    return out


def test_build_action_pairs_carries_only_descriptors():
    rows = _rows_two_worlds()
    d60 = _d([{"vehicle_id": "c0", "time_s": 0.0, "action": "damping"},
              {"vehicle_id": "c0", "time_s": 60.0, "action": "native"}])
    d120 = _d([{"vehicle_id": "c0", "time_s": 0.0, "action": "damping"},
               {"vehicle_id": "c0", "time_s": 120.0, "action": "native"}])

    def descriptors(world, policy, row):
        return ia.ActionDescriptor() if policy == "native" \
            else (d60 if world == "w0" else d120)

    n = len(rows)
    vq = np.zeros((n, QUANTILE_LEVELS.size))
    hq = np.zeros((n, QUANTILE_LEVELS.size))
    vq[[i for i, r in enumerate(rows) if r.policy == "damping"]] = 1.0
    pairs = ia.build_action_pairs(rows, vq, hq, descriptors, taus=(60.0,))
    assert len(pairs) == 2
    fields = {f.name for f in dataclasses.fields(ia.ActionPair)}
    assert not ({"contrast", "arm", "arm_name", "policy"} & fields)
    cells = {ia._cell_key(p.difference) for p in pairs}
    assert cells == {"<=60s|0-30s", "60-240s|0-30s"}


def test_stratified_metrics_report_ci_and_n_per_stratum():
    rows = _rows_two_worlds()
    d60 = _d([{"vehicle_id": "c0", "time_s": 0.0, "action": "damping"},
              {"vehicle_id": "c0", "time_s": 60.0, "action": "native"}])
    d120 = _d([{"vehicle_id": "c0", "time_s": 0.0, "action": "damping"},
               {"vehicle_id": "c0", "time_s": 120.0, "action": "native"}])

    def descriptors(world, policy, row):
        return ia.ActionDescriptor() if policy == "native" \
            else (d60 if world == "w0" else d120)

    n = len(rows)
    vq = np.zeros((n, QUANTILE_LEVELS.size))
    hq = np.zeros((n, QUANTILE_LEVELS.size))
    vq[[i for i, r in enumerate(rows) if r.policy == "damping"]] = 1.0
    pairs = ia.build_action_pairs(rows, vq, hq, descriptors, taus=(60.0,))
    out = ia.stratified_action_metrics(pairs, n_boot=64, alpha=0.10, seed=3)
    assert out["schema"] == "wm_n1_stratified_action_metrics/1"
    assert set(out["strata"]) == {"<=60s|0-30s", "60-240s|0-30s"}
    for key, by_tau in out["strata"].items():
        cell = by_tau["60"]
        assert cell["n_pairs"] == 1 and cell["n_worlds"] == 1
        assert len(cell["variant_vs_zero_ci"]) == 2
        assert len(cell["variant_vs_cellmean_ci"]) == 2
        assert np.isfinite(cell["variant_vs_zero_diff"])
        assert np.isfinite(cell["variant_vs_cellmean_diff"])
        assert cell["sign_n_used"] >= 1
        assert 0.0 <= cell["variant_sign_accuracy"] <= 1.0
    means = ia.action_cell_means(pairs)
    assert means[("<=60s|0-30s", 60.0)] == 1.0


def test_no_arm_name_input_path():
    sig = inspect.signature(ia.build_action_pairs)
    assert not ({"contrast", "contrasts", "arm", "arm_name"}
                & set(sig.parameters))
    ad = {f.name for f in dataclasses.fields(ia.ActionDescriptor)}
    assert "arm" not in ad and "contrast" not in ad
    rows = _rows_two_worlds()
    n = len(rows)
    z = np.zeros((n, QUANTILE_LEVELS.size))
    with pytest.raises(TypeError):
        ia.build_action_pairs(rows, z, z, {}, taus=(60.0,),
                              contrasts=("damping-native",))
    with pytest.raises(ia.ActionDescriptorError):
        ia.build_action_pairs(rows, z, z, {}, taus=(60.0,))


def _write_sched(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schedule": {"arm": "switching",
                                             "events": events}}),
                    encoding="utf-8")


def _synthetic_root(tmp_path):
    root = tmp_path / "root"
    plan = {
        "worlds": [
            {"world": "w0", "split": "TRAIN", "t0_s": 0.0,
             "branches": [
                 {"arm": "native", "run": "rn"},
                 {"arm": "switching", "run": "r60"},
                 {"arm": "switching", "run": "r570"},
             ]},
            {"world": "w_cal", "split": "CALIBRATION", "t0_s": 0.0,
             "branches": [{"arm": "native", "run": "rcal"}]},
        ]
    }
    plan_path = root / "output" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    _write_sched(root / "runs" / "r60" / "run" / "switch_schedule.json",
                 [{"vehicle_id": "c0", "time_s": 0.0, "action": "damping"},
                  {"vehicle_id": "c0", "time_s": 60.0, "action": "native"}])
    _write_sched(root / "runs" / "r570" / "run" / "switch_schedule.json",
                 [{"vehicle_id": "c0", "time_s": 0.0, "action": "damping"}])
    return plan_path, root


def test_representation_counts_and_report(tmp_path):
    plan_path, root = _synthetic_root(tmp_path)
    counts = ia.representation_counts(
        specs=[("synthetic", str(plan_path), str(root / "runs"))], root=root)
    assert counts["schema"] == "wm_n1_action_representation/1"
    summary = counts["summary"]
    assert summary["n_pairs"] == 3
    assert summary["first_60_only"] == 1
    assert summary["by_duration"]["<=60s"] == 1
    assert summary["by_duration"]["whole-window"] == 2
    assert summary["n_train_worlds"] == 1
    line = ia.disclosure_line(counts)
    assert "whole-window" in line and "2/3" in line and "first 60 s" in line
    json_path, md_path = ia.write_representation_report(
        tmp_path / "reports", counts)
    assert json_path.is_file() and md_path.is_file()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["disclosure"] == line
    md = md_path.read_text(encoding="utf-8")
    assert "ACTION_REPRESENTATION" in md and "first-60-only" in md


def test_descriptor_from_plan_tensor_wall_and_actions():
    plan = np.zeros((ia.PLAN_N_SEGMENTS, 1, ia.PLAN_F))
    plan[:, 0, ia.PLAN_ACTIONS.index("native")] = 1.0
    plan[0:2, 0, ia.PLAN_ACTIONS.index("native")] = 0.0
    plan[0:2, 0, ia.PLAN_ACTIONS.index("damping")] = 1.0
    desc = ia.descriptor_from_plan_tensor(plan, cav_ids=["c0"])
    assert desc.cav_ids() == ("c0",)
    diff = ia.schedule_difference(desc, ia.ActionDescriptor())
    assert diff.duration_stratum == "<=60s" and diff.first_60_only
