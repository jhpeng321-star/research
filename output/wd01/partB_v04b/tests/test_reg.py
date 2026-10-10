"""Registration binding + J4 plan split integrity (registration s.2)."""
from __future__ import annotations

import json

import pytest

from partB_v04b import j4plan, reg


def test_registration_binding():
    """The served registration file IS the registered one (e00cd98e...)."""
    assert reg.reg_sha().startswith(reg.REG_SHA_EXPECTED_PREFIX)
    assert reg.lf_sha256(reg.REG_PATH) == reg.reg_sha()


def test_registered_fixed_grid_and_taus():
    """The registered fixed grid of 6 candidates + tau/quantile sets."""
    assert len(reg.GRID) == 6
    assert (300, 0.05, 4, 40) in reg.GRID and (150, 0.1, 8, 20) in reg.GRID
    assert reg.TAUS == (30.0, 60.0, 120.0, 300.0, 570.0)
    assert reg.QUANTILES == (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    assert reg.TAUS_A3 == (60.0, 120.0, 300.0)
    assert reg.COVERAGE_BAND == (0.75, 0.85)
    assert reg.N_BOOT == 2000 and reg.BOOT_SEED == 20261001


def test_split_counts_by_name():
    train = j4plan.split_by_name()
    cal = j4plan.cal_worlds()
    assert len(train) == 39 and len(cal) == 5
    assert set(train) & set(cal) == set()
    # the plan is the split authority; the single recorded name drift
    # (registration text w8662m2700p10t300 vs plan w8662m2600p10t300)
    # is disclosed, never silent
    assert set(j4plan.TRAIN_WORLDS) - j4plan.SPLIT_NAME_DRIFT.keys() | \
        set(j4plan.SPLIT_NAME_DRIFT.values()) == set(train)
    assert "w8662m2600p10t300" in train
    assert "w8662m2700p10t300" not in train
    assert set(j4plan.CAL_WORLDS) == set(cal)
    for w, rec in train.items():
        assert rec["split"] == "TRAIN"
    for w, rec in cal.items():
        assert rec["split"] == "CALIBRATION"
    assert len(j4plan.wall_branches(next(iter(train.values())))) == 4


def test_split_refuses_an_unrecorded_name_drift(tmp_path):
    def rename(plan):
        for w in plan["worlds"]:
            if w["world"] == "w8659m2700p10t420":
                w["world"] = "w9999m2700p10t420"
    root = _write_tmp_plan(tmp_path, rename)
    with pytest.raises(j4plan.PlanMismatch, match="beyond"):
        j4plan.split_by_name(root)


def _write_tmp_plan(tmp_path, mutate):
    real = j4plan.load_plan()
    plan = json.loads(json.dumps(real))
    mutate(plan)
    d = tmp_path / "output" / "campaign_walls_v1" / "results_plan"
    d.mkdir(parents=True)
    (d / "WORLD_PLAN.json").write_text(json.dumps(plan),
                                       encoding="utf-8")
    return tmp_path


def test_split_refuses_a_drifted_plan(tmp_path):
    def flip(plan):
        for w in plan["worlds"]:
            if w["world"] == "w8659m2700p10t420":
                w["split"] = "CALIBRATION"
    root = _write_tmp_plan(tmp_path, flip)
    with pytest.raises(j4plan.PlanMismatch):
        j4plan.split_by_name(root)


def test_split_refuses_a_shifted_count(tmp_path):
    def promote(plan):
        for w in plan["worlds"]:
            if w["world"] == "w8674m2600p20t300":
                w["split"] = "TRAIN"      # 40 TRAIN / 4 CAL: never
    root = _write_tmp_plan(tmp_path, promote)
    with pytest.raises(j4plan.PlanMismatch, match="counts"):
        j4plan.split_by_name(root)


def test_split_refuses_a_renamed_world(tmp_path):
    def rename(plan):
        plan["worlds"][1]["world"] = "w9999m2700p10t300"
    root = _write_tmp_plan(tmp_path, rename)
    with pytest.raises(j4plan.PlanMismatch):
        j4plan.split_by_name(root)


def test_split_refuses_a_foreign_schema(tmp_path):
    def schema(plan):
        plan["schema"] = "someone_else/9"
    root = _write_tmp_plan(tmp_path, schema)
    with pytest.raises(j4plan.PlanMismatch):
        j4plan.load_plan(root)


def test_wall_branch_arms_are_asserted():
    rec = {"world": "wX", "branches": [{"arm": "W-lo-early"},
                                       {"arm": "W-mid-late"}]}
    with pytest.raises(j4plan.PlanMismatch):
        j4plan.wall_branches(rec)
