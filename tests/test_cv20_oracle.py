"""CV20 oracle derivation and scope tests, no simulation.

The preserved CV-19 evidence test runs the full streaming oracle over the
real completed cv19_4107-retain run and is skipped if that evidence is
absent, keeping the suite hermetic on fresh checkouts.
"""

import hashlib
import json
from pathlib import Path

import pytest

from output.controller_verification_v20 import independent_review as oracle

ROOT = Path(__file__).resolve().parents[1]
V19 = ROOT / "output/controller_verification_v19/independent_review.py"
DERIVED = ROOT / "output/controller_verification_v20/independent_review.py"
V19_SHA = "64098e0b9d09d78da6f4864de9231103f38b0d734c8d1697c96ea6321c5bdc6f"
PRESERVED_RUN = ROOT / "runs/active/controller_verification_v19/jobs/4107-retain"
CV19_REGISTRATION = ROOT / "output/controller_verification_v19/REGISTRATION.json"


def test_v19_oracle_is_unchanged():
    assert hashlib.sha256(V19.read_bytes()).hexdigest() == V19_SHA


def test_derived_oracle_differs_only_in_scope():
    """Everything after the docstring must equal CV-19's body except the
    reviewed cell-set constant and the scope message."""
    def split(text):
        end = text.index('"""', text.index('"""') + 3) + 3
        return text[:end], text[end:]

    _, v19_body = split(V19.read_text(encoding="utf-8"))
    _, derived_body = split(DERIVED.read_text(encoding="utf-8"))
    reverted = derived_body.replace(
        "\n#: Registered CV-20 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(s, 2000) for s in range(7001, 7007)}"
        " | {(s, 2700) for s in range(7007, 7013)}\n",
        "\n#: Registered CV-19 (seed, mainline_vph) cells, mirroring the registration.\n"
        "REGISTERED_CELLS = {(4107, 2000), (4108, 2000)}\n")
    reverted = reverted.replace(
        '        "registered CV20 scope",', '        "registered CV19 scope",')
    assert reverted == v19_body


def test_registered_cells_exact():
    assert oracle.REGISTERED_CELLS == (
        {(s, 2000) for s in range(7001, 7007)} | {(s, 2700) for s in range(7007, 7013)})
    assert (4107, 2000) not in oracle.REGISTERED_CELLS


def test_out_of_scope_scene_fails_closed(tmp_path):
    run = PRESERVED_RUN / "run"
    if not CV19_REGISTRATION.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV19 evidence absent")
    reg = json.loads(CV19_REGISTRATION.read_text(encoding="utf-8"))
    expected = dict(reg["jobs"]["4107-retain"]["expected"])
    expected["seed"] = 4109
    result = oracle.verify_run(str(run), expected)
    assert result["integrity_pass"] is False
    assert result["failures"] == ["manifest seed"]


def test_preserved_out_of_scope_run_fails_closed_on_scope():
    """The CV-19 scene is deliberately outside CV-20's registered cells: the
    CV-20 oracle must refuse it at the scope gate (its genuine-pass validation
    happens on the first real CV-20 arm, as there is no preserved in-scope
    run for a fresh-seed corpus)."""
    run = PRESERVED_RUN / "run"
    if not CV19_REGISTRATION.is_file() or not (run / "manifest.json").is_file():
        pytest.skip("preserved CV19 evidence absent")
    reg = json.loads(CV19_REGISTRATION.read_text(encoding="utf-8"))
    result = oracle.verify_run(str(run), reg["jobs"]["4107-retain"]["expected"])
    assert result["integrity_pass"] is False
    assert result["safety_pass"] is False
    assert result["failures"] == ["registered CV20 scope"]
