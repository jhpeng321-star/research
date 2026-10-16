"""HDV driver population switch: legacy3 (default) vs textbook."""
from __future__ import annotations

import csv
import math
import statistics

import pytest

from traffic_shepherd.c4 import population as pop
from traffic_shepherd.scenarios import build_scenario, request_schedule

# Golden values produced by the PRE-population code (commit 1b249be traffic_shepherd/scenarios.py).
LEGACY_GOLDEN = {
    (4107, 2700, 0.2, None): ("1bfc736c1b4baa493fe819ded8df287c4123d43742cbbb1a162147dff03aff68",
                              "85ea04b41b5c458d66666d54099e893be814d765c8b06895936069c4373c45d6"),
    (4110, 2400, 0.4, "normal_car"): (
        "500d8e319b75805d27c80b82ba6bb97a56b61ab2a36eda9f5c909ab52d0f6c22",
        "c9046078657f27c330c925baeaef46f9c464b05606bb603dc7397060e0dec928"),
}
LEGACY_MANIFEST_KEYS = {
    "schema", "seed", "mainline_vph", "penetration", "episode_seconds", "demand_end_seconds",
    "population", "cav_population", "routes_sha256", "network_sha256", "requests_sha256",
    "assignment", "cav_vtype"}


@pytest.mark.parametrize("case", sorted(LEGACY_GOLDEN, key=str))
def test_legacy3_is_bit_identical_to_pre_population_code(case, tmp_path):
    seed, demand, pen, cav_vtype = case
    _, requests, manifest = build_scenario(tmp_path / "a", seed, demand, penetration=pen,
                                           cav_vtype=cav_vtype)
    _, requests2, manifest2 = build_scenario(tmp_path / "b", seed, demand, penetration=pen,
                                             cav_vtype=cav_vtype, population="legacy3")
    assert manifest == manifest2 and requests == requests2
    assert (manifest["routes_sha256"], manifest["requests_sha256"]) == LEGACY_GOLDEN[case]
    assert set(manifest) == LEGACY_MANIFEST_KEYS
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["requests.rou.xml",
                                                                  "scenario.sumocfg"]


def test_textbook_keeps_departures_and_cav_identity():
    for seed, demand, pen in ((4107, 2700, 0.2), (4109, 2400, 0.4)):
        legacy = request_schedule(seed, demand, penetration=pen, cav_vtype="normal_car")
        book = request_schedule(seed, demand, penetration=pen, population="textbook")
        assert len(legacy) == len(book)
        for old, new in zip(legacy, book):
            assert (old.vehicle_id, old.depart_s, old.route, old.is_cav) == (
                new.vehicle_id, new.depart_s, new.route, new.is_cav)
            assert new.driver_type == (pop.TEXTBOOK_CAV_VTYPE if new.is_cav
                                       else "tb." + new.vehicle_id)


def test_textbook_rejects_foreign_cav_vtype_and_unknown_population():
    with pytest.raises(ValueError):
        request_schedule(4107, 2700, penetration=0.2, population="textbook",
                         cav_vtype="normal_car")
    with pytest.raises(ValueError):
        request_schedule(4107, 2700, penetration=0.2, population="mixed")


def test_textbook_parameter_distributions():
    rows = [pop.sample_driver(4107, f"mainline.{i}") for i in range(5000)]

    def q(name, p):
        values = sorted(r[name] for r in rows)
        return values[int(p * (len(values) - 1))]

    for name, (lo, hi) in pop.BOUNDS.items():
        assert all(lo <= r[name] <= hi for r in rows), name
    for name, (lo, hi) in pop.UNIFORM.items():
        assert all(lo <= r[name] <= hi for r in rows), name
        assert abs(q(name, 0.5) - (lo + hi) / 2) < 0.05 * (hi - lo)
    medians = {"speed_factor": 1.0, "tau_s": 1.2, "accel_mps2": 1.2, "decel_mps2": 2.0,
               "min_gap_m": 2.0}
    for name, median in medians.items():
        assert abs(q(name, 0.5) - median) < 0.03 * median, name
    # 10-90 % ranges of the lognormal parameters
    for name, (q10, q90) in {"tau_s": (0.9, 1.6), "accel_mps2": (0.8, 1.8),
                             "decel_mps2": (1.5, 2.8)}.items():
        assert abs(q(name, 0.1) - q10) < 0.1 and abs(q(name, 0.9) - q90) < 0.15, name
    assert abs(q("speed_factor", 0.9) - q("speed_factor", 0.1) - 2 * 1.2816 * 0.1) < 0.03
    assert all(r["delta"] == 4.0 and r["lc_strategic"] == 1.0 for r in rows)
    corr = statistics.correlation([math.log(r["accel_mps2"]) for r in rows],
                                  [-math.log(r["tau_s"]) for r in rows])
    assert 0.25 < corr < 0.45          # rho 0.4 before truncation; positive (aggressive = close)


def test_textbook_draws_are_deterministic_and_per_identity():
    assert pop.sample_driver(4107, "ramp.3") == pop.sample_driver(4107, "ramp.3")
    assert pop.sample_driver(4107, "ramp.3") != pop.sample_driver(4108, "ramp.3")
    assert pop.sample_driver(4107, "ramp.3") != pop.sample_driver(4107, "ramp.4")


def test_exports_for_wm_prior_and_cav_cap():
    m = pop.TEXTBOOK_POPULATION_MEDIANS
    assert (m["accel"], m["time_headway"], m["decel"], m["min_gap"], m["speed_factor"]) == (
        1.2, 1.2, 2.0, 2.0, 1.0)
    c = pop.TEXTBOOK_CAV_PARAMS
    assert set(c) == set(pop.PARAM_ORDER)
    assert (c["accel_mps2"], c["tau_s"], c["decel_mps2"], c["min_gap_m"], c["speed_factor"]) == (
        1.2, 1.2, 2.0, 2.0, 1.0)
    assert (c["lc_cooperative"], c["lc_assertive"], c["lc_speed_gain"], c["impatience"]) == (
        0.65, 1.0, 1.0, 0.3)


def test_textbook_scenario_routes_manifest_and_truth(tmp_path):
    _, requests, manifest = build_scenario(tmp_path / "s", 4107, 2700, penetration=0.2,
                                           population="textbook")
    _, requests2, manifest2 = build_scenario(tmp_path / "t", 4107, 2700, penetration=0.2,
                                             population="textbook")
    assert manifest == manifest2 and requests == requests2           # deterministic per seed
    root = tmp_path / "s"
    import xml.etree.ElementTree as ET
    xml = ET.parse(root / "requests.rou.xml").getroot()
    vtypes = {v.attrib["id"]: v.attrib for v in xml.findall("vType")}
    hdvs = [r for r in requests if not r.is_cav]
    cavs = [r for r in requests if r.is_cav]
    assert cavs and set(vtypes) == {pop.TEXTBOOK_CAV_VTYPE} | {r.driver_type for r in hdvs}
    assert {v.attrib["type"] for v in xml.findall("vehicle")
            if v.attrib["id"] in {r.vehicle_id for r in cavs}} == {pop.TEXTBOOK_CAV_VTYPE}
    cav = vtypes[pop.TEXTBOOK_CAV_VTYPE]
    assert (cav["accel"], cav["decel"], cav["tau"], cav["minGap"], cav["speedFactor"]) == (
        "1.2", "2.0", "1.2", "2.0", "1.0")
    for v in vtypes.values():
        assert v["carFollowModel"] == "IDM" and v["speedDev"] == "0" and v["sigma"] == "0"
        assert v["emergencyDecel"] == "9.0"
    assert manifest["driver_population"] == "textbook"
    assert manifest["cav_vtype"] == pop.TEXTBOOK_CAV_VTYPE
    with (root / "hetdata_truth.csv").open(encoding="utf-8", newline="") as fh:
        truth = {row["vehicle_id"]: row for row in csv.DictReader(fh)}
    assert set(truth) == {r.vehicle_id for r in requests}
    sample = hdvs[5]
    row = truth[sample.vehicle_id]
    params = pop.sample_driver(4107, sample.vehicle_id)
    assert {k: float(row[k]) for k in pop.PARAM_ORDER} == params
    assert float(vtypes[sample.driver_type]["accel"]) == params["accel_mps2"]
    assert float(vtypes[sample.driver_type]["tau"]) == params["tau_s"]
    assert row["driver_class"] == "textbook" and row["is_cav"] == "0"
    assert truth[cavs[0].vehicle_id]["driver_class"] == pop.TEXTBOOK_CAV_VTYPE
    # readable by the WM evaluation-only truth loader
    from wms.data.driver_truth import load_driver_sidecar
    side = load_driver_sidecar(root / "hetdata_truth.csv")
    assert side.n == len(requests)


def test_env_population_kwarg():
    import shutil
    if shutil.which("sumo") is None:
        pytest.skip("SUMO not on PATH")
    from traffic_shepherd.c4.env import CavSumoEnv
    assert CavSumoEnv().population == "legacy3"
    env = CavSumoEnv(population="textbook")
    assert env.cav_vtype == pop.TEXTBOOK_CAV_VTYPE
    with pytest.raises(ValueError):
        CavSumoEnv(population="nope")


def test_collect_cli_has_population_flag():
    from traffic_shepherd.c4 import collect
    src = __import__("inspect").getsource(collect.main)
    assert '"--population"' in src and "textbook" in src
