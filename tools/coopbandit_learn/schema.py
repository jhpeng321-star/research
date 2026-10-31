"""EXAM v1 labelled-dataset format: constants, restricted-path guard, hashing.

Format (documented in fixture.SCHEMA and HANDOFF.md, mirrors COOPBANDIT_DESIGN.md):
  states.jsonl     one JSON object per decision state: id, world, cell, t_decision,
                   setting, available flags, group members, observable features, roster
  labels.csv       long format: state_id, option, horizon, D (+ components, safety)
  group_credit.csv long format: state_id, option, C_G (+ solo components)
  splits.json      world-group CV folds: {"fold_of": {<world>: 0..4}}
  provenance.json  sha256 manifest of every dataset file
Metadata (world/cell/penetration/demand/setting) are NEVER model features.
"""
import hashlib
import json
from pathlib import Path
import re

OPTIONS = ("O0", "O1", "O2", "O3", "O4")
GROUP_OPTIONS = ("O1", "O2", "O3", "O4")
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
SPLITS = "splits.json"
PROVENANCE = "provenance.json"
SCHEMA_DOC = "SCHEMA.md"
LABEL_REQUIRED = ("state_id", "option", "horizon", "D")
CREDIT_REQUIRED = ("state_id", "option", "C_G")
STATE_REQUIRED = ("state_id", "world", "cell", "t_decision", "setting", "available", "features")
FORBIDDEN_FEATURES = frozenset(("world", "seed", "cell", "penetration", "demand", "t_decision", "state_id"))
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


def write_json(path, value):
    checked(path).write_text(canonical(value), encoding="utf-8")
