"""EXAM v1 labelled-dataset format (S-COOPBANDIT export) and restricted-path guard.

Format produced by tools/coopbandit/export.py in the executor worktree:
  states.jsonl     per decision state: state_id, world_id (seed), t0,
                   features (numeric observable dict), observed_vehicles,
                   prefix_sha256
  labels.csv       long: state_id, world_id, cell, t0, setting, option O0..O4,
                   horizon_s 120/300/600, available, D, D_road, D_source, J,
                   collisions, teleports, ttc_below_3_vs, hard_brake_vs,
                   override_share; D = J(O0) - J(option) >= 0 means saved delay
  group_credit.csv long: state_id, world_id, option, horizon_s, available,
                   D_G, D_0, D_1, sum_D_i, C_G
  worlds.json      selected worlds (seed/cell/repeat/penetration/demand)
  provenance.json  design hash, per-source sha256, budget, partial flag
  README.md        schema text; MANIFEST.json sha256 of the export files
Metadata (world_id/cell/t0/setting) are NEVER model features. Dev worlds
981080..981091 are live-test only and are rejected at load time.
"""
import hashlib
import json
from pathlib import Path
import re

OPTIONS = ("O1", "O2", "O3", "O4")
ALL_OPTIONS = ("O0", "O1", "O2", "O3", "O4")
HORIZONS = (120, 300, 600)
POLICY_HORIZON = 600
FOLDS = 5
INNER_FOLDS = 4
CONF_LEVEL = 0.95
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261031
STATES = "states.jsonl"
LABELS = "labels.csv"
CREDIT = "group_credit.csv"
WORLDS = "worlds.json"
PROVENANCE = "provenance.json"
MANIFEST = "MANIFEST.json"
README = "README.md"
LABEL_COLUMNS = ("state_id", "world_id", "cell", "t0", "setting", "option", "horizon_s", "available",
                 "D", "D_road", "D_source", "J", "collisions", "teleports", "ttc_below_3_vs",
                 "hard_brake_vs", "override_share")
CREDIT_REQUIRED = ("state_id", "world_id", "option", "horizon_s", "available", "D_G", "D_0", "D_1",
                   "sum_D_i", "C_G")
STATE_REQUIRED = ("state_id", "world_id", "t0", "features", "prefix_sha256")
DEV_WORLDS = range(981080, 981092)
FULL_PANEL_STATES = 300
FORBIDDEN_FEATURES = frozenset(("world_id", "world", "seed", "cell", "penetration", "demand",
                                "t0", "t_decision", "state_id"))
FORBIDDEN = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|vtype|assignment", re.I)


def checked(path):
    """Reject restricted paths before any filesystem access."""
    p = Path(path)
    if FORBIDDEN.search(str(p)) or FORBIDDEN.search(str(p.resolve())):
        raise ValueError("restricted path: " + str(p))
    return p


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes")
