"""Import-time safety (WALL-2 audit): no SUMO/TraCI at import.

Every package module - and the frozen campaign infrastructure it imports
read-only - must import WITHOUT pulling in traci/sumolib (the pilot's
``run_pilot`` module MAY be imported - ``run_chunk`` does so at module
level - but its ``import traci`` sits inside ``branch_sumo_session`` and
must stay there).  A fresh subprocess interpreter makes the check
airtight (sys.modules cannot be polluted by earlier tests): if any
module-level code imported traci or launched SUMO, ``traci`` would show
up in sys.modules.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TESTS_DIR.parent                 # output/campaign_walls_v1
OUTPUT_ROOT = PACKAGE_ROOT.parent               # output/
REPO_ROOT = OUTPUT_ROOT.parent                  # <repo>

MODULES = [
    "campaign_walls_v1.common",
    "campaign_walls_v1.plan_walls_pilot",
    "campaign_walls_v1.walls_pilot_registration",
    "campaign_walls_v1.wall_request",
    "campaign_walls_v1.wall_driver",
    "campaign_walls_v1.run_wall_branch",
    "campaign_walls_v1.smoke_walls",
    "campaign_walls_v1.launch_walls",
]


def test_importing_every_package_module_never_touches_traci_or_sumo():
    bootstrap = (
        "import sys, json\n"
        f"sys.path[:0] = [{str(REPO_ROOT)!r}, {str(OUTPUT_ROOT)!r}]\n"
        + "".join(f"import {module}\n" for module in MODULES)
        + "print(json.dumps({\n"
        "  'traci': 'traci' in sys.modules,\n"
        "  'sumolib': 'sumolib' in sys.modules,\n"
        "}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", bootstrap], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload == {"traci": False, "sumolib": False}, payload
