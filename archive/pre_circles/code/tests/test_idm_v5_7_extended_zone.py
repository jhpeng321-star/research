from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from experiments import REPO_ROOT
from experiments.idm.wall_v5_7_extended_zone_contract import (
    ASSEMBLY_MIN_STEPS, HOLD_S, MAX_DECELERATION_MPS2, TAPER_S,
    TOTAL_TREATMENT_S, UPSTREAM_EXTENSION_EDGE, profile_manifest,
)
from tools.build_idm_v5_7_extended_scenario import build


def test_v5_7_contract_is_low_authority_and_phase_explicit():
    manifest = profile_manifest()
    overrides = manifest["overlay_overrides"]
    assert manifest["profile_id"] == "idm_v5_7_extended_zone_low_authority_wall"
    assert overrides["compact_assembly_a_max_dec_mps2"] == MAX_DECELERATION_MPS2
    assert overrides["compact_servo_a_max_dec_mps2"] == MAX_DECELERATION_MPS2
    assert overrides["compact_high_authority_assembly_steps"] == 0
    assert overrides["compact_min_assembly_steps"] == ASSEMBLY_MIN_STEPS
    assert overrides["compact_requested_hold_duration_s"] == HOLD_S
    assert overrides["taper_duration_s"] == TAPER_S
    assert overrides["planned_treatment_duration_s"] == TOTAL_TREATMENT_S


def test_extended_scenario_preserves_downstream_and_adds_only_extension():
    manifest = build()
    network = Path(manifest["derived_network"])
    root = ET.parse(network).getroot()
    extension = root.find(f"edge[@id='{UPSTREAM_EXTENSION_EDGE}']")
    assert extension is not None
    assert len(extension.findall("lane")) == 2
    assert root.find("edge[@id='1487606179']") is not None
    route_text = Path(manifest["derived_route"]).read_text(encoding="utf-8")
    assert f'edges="{UPSTREAM_EXTENSION_EDGE} 1191554636 1487606179 1487606178"' in route_text
    stored = json.loads((REPO_ROOT / "scenarios" / "idm_v5_7_extended_zone" / "manifest.json").read_text())
    assert stored["source_network_sha256"] == manifest["source_network_sha256"]
