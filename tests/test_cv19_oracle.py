"""CV19 oracle derivation and scope tests, no simulation.

The preserved CV-18 evidence test runs the full streaming oracle over the
real completed 4107-retain run (~3 s) and is skipped if that evidence is
absent, so the suite stays hermetic on fresh checkouts.
"""

import hashlib
import json
from pathlib import Path

import pytest

from output.controller_verification_v19 import independent_review as oracle

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "output/controller_verification_v15/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v19/independent_review.py"
FROZEN_SHA = ("3e41822a3681c9906b7641619b0ae5f22e690e9301e14497fb3b86b2296332e5")
PRESERVED_RUN = ROOT / "runs/active/controller_verification_v18/jobs/4107-retain"
CV18_REGISTRATION = ROOT / "output/controller_verification_v18/REGISTRATION.json"


def test_frozen_oracle_is_unchanged():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_SHA


def test_derived_oracle_differs_only_in_scope():
    """Reverting the three reviewed edits must reproduce the frozen bytes."""
    frozen = FROZEN.read_text(encoding="utf-8")
    derived = DERIVED.read_text(encoding="utf-8")
    # revert docstring
    start = derived.index('"""')
    end = derived.index('"""', start + 3) + 3
    reverted = '"""Streaming CV-15 saved-evidence review; no production law or simulator imports."""' \
        + derived[end:]
    # revert constant block
    reverted = reverted.replace(
        "\n#: Registered CV-19 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(4107, 2000), (4108, 2000)}\n", "")
    # revert scope assertion
    reverted = reverted.replace(
        '        and e["penetration"] == 0.2\n'
        '        and (e["seed"], e["loading"]) in REGISTERED_CELLS,\n'
        '        "registered CV19 scope",',
        '        and e["seed"] == 4108\n'
        '        and e["loading"] == 2700\n'
        '        and e["penetration"] == 0.2,\n'
        '        "registered fixed CV15 scope",')
    assert reverted == frozen


def test_registered_cells_exact():
    assert oracle.REGISTERED_CELLS == {(4107, 2000), (4108, 2000)}


def test_out_of_scope_scene_fails_closed(tmp_path):
    run = PRESERVED_RUN / "run"
    if not CV18_REGISTRATION.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV18 evidence absent")
    reg = json.loads(CV18_REGISTRATION.read_text(encoding="utf-8"))
    expected = dict(reg["jobs"]["4107-retain"]["expected"])
    expected["seed"] = 4109
    result = oracle.verify_run(str(run), expected)
    assert result["integrity_pass"] is False
    assert result["failures"] == ["manifest seed"]


def test_preserved_completed_run_adjudicates_genuine_pass():
    run = PRESERVED_RUN / "run"
    if not CV18_REGISTRATION.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV18 evidence absent")
    reg = json.loads(CV18_REGISTRATION.read_text(encoding="utf-8"))
    result = oracle.verify_run(str(run), reg["jobs"]["4107-retain"]["expected"])
    assert result["integrity_pass"] is True
    assert result["safety_pass"] is True
    assert result["failures"] == []
    assert result["physical_outcomes"]["inserted"] == 1150
    assert result["physical_outcomes"]["arrived"] == 1150
    assert result["physical_outcomes"]["totals"]["collision_involvements"] == 0
