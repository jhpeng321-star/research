"""Ported exogenous-demand, original hashes, sealed seeds and startup lock tests."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from threading import Event
import xml.etree.ElementTree as ET

import pytest

from traffic_shepherd.config import load_config
from traffic_shepherd.scenarios import build_scenario, request_schedule, scenario_identity
from traffic_shepherd.seeds import PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS, exclusion_provenance
from traffic_shepherd.simulation import launch_lock


@pytest.mark.parametrize("seed,loading,expected", [
    (4107,2000,"55b428e3dad8c610d03386b2e262f81942ea10f5c4d6f5b23a30e34de241228f"),
    (4109,2000,"0aed210f3e0be3f820f0a078ef3adc2926251573bd510082bf9cdada07eff4bd"),
    (4107,2200,"2994be3264577bee51d7a6dbd9966308380145f883f295415e5c016d5364798d"),
    (4109,2200,"61c24c9303b39ce8f4e04147acfc0a9cc0b23234069f511808c0fd4e9dbc25a0"),
])
def test_original_request_population_hashes_unchanged(seed, loading, expected):
    requests = request_schedule(seed, loading, penetration=.2)
    assert hashlib.sha256(json.dumps([asdict(r) for r in requests], sort_keys=True).encode()).hexdigest() == expected


def test_penetration_only_changes_nested_cav_designation():
    panels = [request_schedule(911001, 2000, role="development", penetration=p) for p in (.05,.1,.2,.3)]
    for first, second in zip(panels, panels[1:]):
        assert [replace(r, is_cav=False) for r in first] == [replace(r, is_cav=False) for r in second]
        assert {r.vehicle_id for r in first if r.is_cav} <= {r.vehicle_id for r in second if r.is_cav}


def test_scenario_preserves_idm_speeddev_and_refuses_overwrite(tmp_path):
    # Small local template avoids a dependency on shared SUMO installation in unit tests.
    template = tmp_path / "template.xml"
    template.write_text('<routes><vTypeDistribution><vType id="normal_car" carFollowModel="IDM" probability="1"/></vTypeDistribution><route id="mainline_route" edges="a b"/></routes>')
    network = tmp_path / "network.xml"
    network.write_text("test network")
    cfg = replace(load_config(), scenario_template_path=template, network_path=network)
    scene, requests, manifest = build_scenario(tmp_path / "scene", 4107, 2000, penetration=.2, config=cfg)
    assert requests == request_schedule(4107, 2000, penetration=.2)
    assert build_scenario(scene.parent, 4107, 2000, penetration=.2, config=cfg)[2] == manifest
    vtype = ET.parse(scene.parent / "requests.rou.xml").getroot().find("vType")
    assert vtype.attrib == dict(id="normal_car", carFollowModel="IDM", speedDev="0")
    with pytest.raises(ValueError, match="existing scenario"):
        build_scenario(scene.parent, 4107, 2200, penetration=.2, config=cfg)


@pytest.mark.parametrize("seed", [4201,4601,5001,912001,9001])
def test_reserved_or_unregistered_seed_generation_rejected(seed):
    with pytest.raises(ValueError):
        request_schedule(seed, 2000, penetration=.2)


def test_seed_exclusions_preserve_training_development_and_reservations():
    assert {4107,911001,910001,912001,8001} <= set(PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS)
    assert exclusion_provenance()["absence_proves_freshness"] is False


def test_real_byte_zero_lock_serializes_and_does_not_grow(tmp_path):
    path = tmp_path / "launch.lock"
    entered = Event()
    def contender():
        with launch_lock(path):
            entered.set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with launch_lock(path):
            pending = pool.submit(contender)
            assert not entered.wait(.1)
        pending.result(timeout=4)
    assert entered.is_set() and path.read_bytes() == b"0"


@pytest.mark.parametrize("seed,loading", [(911001.0,2000),(911001,2000.0),(True,2000),(911001,True)])
def test_float_or_bool_aliases_cannot_change_rng_identity(seed, loading):
    with pytest.raises(ValueError, match="exact integers"):
        request_schedule(seed, loading, penetration=.2, role="development")


def test_network_content_change_alters_active_scenario_identity(tmp_path):
    scene = tmp_path / "scenario.sumocfg"
    scene.write_text('<configuration><input><net-file value="network.xml"/></input></configuration>')
    (tmp_path / "requests.rou.xml").write_text("fixed requests")
    network = tmp_path / "network.xml"
    network.write_text("network before")
    requests = request_schedule(4107,2000,penetration=.2)
    before = scenario_identity(scene,requests)
    assert before == scenario_identity(scene,requests,network_path=network)
    network.write_text("network after")
    assert scenario_identity(scene,requests) != before
