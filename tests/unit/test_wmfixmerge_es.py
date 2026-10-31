"""Unit tests for the S-WMFIX-ES derivative-free calibration tool."""
import json
import numpy as np
import pytest

from tools.wmfixmerge.es import (ANCHOR_COLUMNS, DEFAULTS, GRADIENT_COLUMNS, PARAMS,
                                 es_update, fitness_from_cells, from_unit,
                                 incumbents_from_resolved, params_doc, to_unit)


def test_unit_roundtrip_and_clip():
    values = [p["lo"] + 0.3 * (p["hi"] - p["lo"]) for p in PARAMS]
    assert np.allclose(from_unit(to_unit(values)), values)
    assert (to_unit([p["lo"] for p in PARAMS]) == 0).all()
    assert (to_unit([p["hi"] for p in PARAMS]) == 1).all()
    assert (to_unit([p["lo"] - 100 for p in PARAMS]) == 0).all()


def test_es_update_moves_toward_better_candidate():
    x = np.full(len(PARAMS), 0.5)
    eps = np.zeros((1, len(PARAMS)))
    eps[0, 0] = 1.0
    sigma, lr = 0.1, 0.5
    losses = [0.0, 1.0]  # plus candidate (x + sigma*e0) strictly better
    x2 = es_update(x, eps, losses, lr, sigma)
    assert x2[0] > x[0]  # centre moves toward the better candidate
    losses = [1.0, 0.0]  # minus candidate better
    x3 = es_update(x, eps, losses, lr, sigma)
    assert x3[0] < x[0]
    # orthogonal winners move orthogonal coordinates only
    x4 = es_update(x, np.eye(len(PARAMS))[:2], [0.0, 1.0, 0.0, 1.0], lr, sigma)
    assert x4[0] > x[0] and x4[1] > x[1] and x4[2] == x[2]
    # centre stays inside bounds
    x5 = es_update(np.zeros(len(PARAMS)), np.ones((1, len(PARAMS))), [0.0, 1.0], 10., 1.)
    assert (x5 >= 0).all() and (x5 <= 1).all()


def test_fitness_from_cells_matches_hand_computation():
    scales = [10., 20., 30., 40., 50.]
    anchors = [11., 12., 13., 14., 15.]
    predicted = [[100., 1., 2., 3., 4.], [110., 2., 3., 4., 5.]]
    recorded = [[90., 0., 2., 2., 3.], [95., 1., 3., 3., 4.]]
    cells = [dict(group=0, horizon=300, predicted=predicted, recorded=recorded)]
    fitness, reports = fitness_from_cells(cells, scales, anchors)
    error = [np.asarray(predicted[1]) - np.asarray(predicted[0])
             - (np.asarray(recorded[1]) - np.asarray(recorded[0]))]
    pair = np.mean([(error[0][c] / scales[c]) ** 2 for c in GRADIENT_COLUMNS])
    absolute = np.mean(np.square(
        (np.asarray(predicted) - np.asarray(recorded))[:, list(ANCHOR_COLUMNS)]
        / np.asarray(anchors)[list(ANCHOR_COLUMNS)]))
    assert fitness == pytest.approx(pair + absolute)
    assert reports["paired_J_MAE"] == pytest.approx(abs(error[0][0]))
    assert reports["source_contrast_MAE"] == pytest.approx(abs(error[0][2]))
    assert reports["cells"] == 1


def test_incumbents_from_resolved(tmp_path):
    resolved = dict(physics=dict(mainline_courtesy_zone_m=123.0))
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(resolved))
    values = incumbents_from_resolved(path)
    doc = params_doc(values)
    assert doc["mainline_courtesy_zone_m"] == 123.0
    for p in PARAMS:
        if p["name"] != "mainline_courtesy_zone_m":
            assert doc[p["name"]] == DEFAULTS[p["name"]]


def test_defaults_match_live_dataclasses():
    from src.wms.model.entries import EntryConfig
    from src.wms.physics.config import LANE_PHYSICAL_MARGIN_M, MAINLINE_COURTESY_ZONE_M, PhysicsConfig
    live = PhysicsConfig()
    assert DEFAULTS["accel_min_mps2"] == live.accel_min_mps2
    assert DEFAULTS["accel_max_mps2"] == live.accel_max_mps2
    assert DEFAULTS["lane_physical_margin_m"] == LANE_PHYSICAL_MARGIN_M == live.lane_physical_margin_m
    assert DEFAULTS["mainline_courtesy_zone_m"] == MAINLINE_COURTESY_ZONE_M
    assert DEFAULTS["free_road_gap_m"] == live.free_road_gap_m
    assert DEFAULTS["speed_limit_mps"] == live.speed_limit_mps
    assert DEFAULTS["sumo_min_speed_mps"] == EntryConfig().sumo_min_speed_mps


def test_cell_name_carries_corpus_tag():
    from tools.wmfixmerge.es import cell_name
    assert cell_name(3, 7, "wm", 35, 1, 300) == "CELL_g003_c007_wm_grp35_arm1_h300.json"
    assert cell_name(0, 0, "fit", 4, 0, 300) == "CELL_g000_c000_fit_grp4_arm0_h300.json"


def test_fitness_subset_every_fifth_wmdata_group():
    from tools.wmfixmerge.prepare_combined import fitness_subset
    fit = [f"fitpair{i}" for i in range(12)]
    wm = [f"wmpair{i}" for i in range(40)]
    seeds_of = {(p, k): 969400 + i for p, ks in (("fit", fit), ("wm", wm))
                for i, k in enumerate(ks)}
    subset = fitness_subset(fit, wm, seeds_of)
    assert [g["corpus"] for g in subset[:12]] == ["fit"] * 12
    assert [g["group"] for g in subset[12:]] == [0, 5, 10, 15, 20, 25, 30, 35]
    assert all(g["arm"] == 0 and g["arm_name"] == "ctrl0" for g in subset[12:])
    assert all(g["arm"] == 0 for g in subset[:12])


def test_check_registry_binds_objective_and_digests(tmp_path):
    import hashlib
    from tools.wmfixmerge.prepare_combined import check_registry
    fit = dict(scales=[1., 2., 3., 4., 5.], anchor_scales=[1., 2., 3., 4., 5.],
               crossing_width=4.25)
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(fit))
    registry = dict(schema="wmfix_train_comb/1",
                    parts=dict(fit=dict(sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest()),
                               wm=dict(sha256="0" * 64,
                                       groups=[f"w{i}" for i in range(40)])),
                    fitness_groups=[], scales=fit["scales"],
                    anchor_scales=fit["anchor_scales"], crossing_width=fit["crossing_width"])
    good = tmp_path / "reg.json"
    good.write_text(json.dumps(registry))
    doc = check_registry(good, fit_path)
    assert doc["schema"] == "wmfix_train_comb/1"
    registry["scales"] = [9., 9., 9., 9., 9.]
    bad = tmp_path / "reg2.json"
    bad.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="scales"):
        check_registry(bad, fit_path)
    registry["scales"] = fit["scales"]
    registry["parts"]["wm"]["groups"] = ["w0"]
    worse = tmp_path / "reg3.json"
    worse.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="40"):
        check_registry(worse, fit_path)
