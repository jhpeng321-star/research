"""Load and validate an EXAM v1 labelled dataset (see schema.py and SCHEMA.md).

Every filesystem access goes through schema.checked(); provenance SHAs are
verified; the feature matrix is built strictly from the recorded observable
features (metadata identifiers are rejected as features).
"""
from dataclasses import dataclass, field
import json
import math

import numpy as np
import pandas as pd

from .schema import (CREDIT, CREDIT_REQUIRED, FORBIDDEN_FEATURES, GROUP_OPTIONS, HORIZONS, LABELS,
                     LABEL_REQUIRED, OPTIONS, PROVENANCE, SPLITS, STATES, STATE_REQUIRED, checked, sha)


@dataclass
class Dataset:
    states: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    credit: pd.DataFrame
    fold_of: dict
    feature_names: list
    provenance: dict
    _cache: dict = field(default_factory=dict)

    def feature_matrix(self, state_ids):
        key = ("X", tuple(state_ids))
        if key not in self._cache:
            self._cache[key] = self.features.loc[list(state_ids), self.feature_names].to_numpy(dtype=float)
        return self._cache[key]

    def world_of(self, state_ids):
        return self.states.set_index("state_id").loc[list(state_ids), "world"].to_numpy()

    def available(self, state_id):
        return tuple(o for o in GROUP_OPTIONS if bool(self.states.set_index("state_id").loc[state_id, o]))

    def labels_for(self, state_id, horizon):
        key = ("lab", state_id, horizon)
        if key not in self._cache:
            sub = self.labels[(self.labels.state_id == state_id) & (self.labels.horizon == horizon)]
            self._cache[key] = dict(zip(sub.option, sub.D))
        return self._cache[key]


def _verify_provenance(data_dir):
    manifest = json.loads(checked(data_dir / PROVENANCE).read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        path = checked(data_dir / name)
        if not path.exists() or sha(path) != digest:
            raise ValueError("provenance mismatch: " + name)
    return manifest


def _states(data_dir):
    rows = []
    with open(checked(data_dir / STATES), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    seen = set()
    for r in rows:
        missing = [k for k in STATE_REQUIRED if k not in r]
        if missing:
            raise ValueError("state missing keys " + str(missing) + ": " + str(r.get("state_id")))
        if r["state_id"] in seen:
            raise ValueError("duplicate state_id " + r["state_id"])
        seen.add(r["state_id"])
        feats = r["features"]
        if not isinstance(feats, dict) or not feats:
            raise ValueError("empty features for " + r["state_id"])
        bad = [k for k in feats if k in FORBIDDEN_FEATURES]
        if bad:
            raise ValueError("metadata leaked into features " + str(bad) + " for " + r["state_id"])
        for k, v in feats.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError("non-finite feature " + k + " for " + r["state_id"])
        for o in ("O1", "O2", "O3", "O4"):
            if o not in r["available"]:
                raise ValueError("missing availability for " + o + " in " + r["state_id"])
    return pd.DataFrame(rows)


def _labels(data_dir, states):
    labels = pd.read_csv(checked(data_dir / LABELS))
    missing = [c for c in LABEL_REQUIRED if c not in labels.columns]
    if missing:
        raise ValueError("labels.csv missing columns " + str(missing))
    if set(labels.horizon) != set(HORIZONS):
        raise ValueError("labels horizons must be exactly " + repr(HORIZONS))
    bad_option = set(labels.option) - set(OPTIONS)
    if bad_option:
        raise ValueError("unknown options in labels: " + repr(sorted(bad_option)))
    if not np.isfinite(labels.D).all():
        raise ValueError("non-finite D labels")
    avail = {r.state_id: {o for o in GROUP_OPTIONS if r.available.get(o)} for r in states.itertuples()}
    labelled = labels.groupby("state_id").option.apply(set).to_dict()
    for sid, avail_opts in avail.items():
        have = {o for o in labelled.get(sid, set()) if o != "O0"}
        if have - avail_opts:
            raise ValueError("labels for unavailable options in " + sid + ": " + repr(sorted(have - avail_opts)))
        for o in avail_opts:
            got = labels[(labels.state_id == sid) & (labels.option == o)].horizon
            if set(got) != set(HORIZONS):
                raise ValueError("incomplete horizons for " + sid + "/" + o)
    key = ["state_id", "option", "horizon"]
    if labels.duplicated(key).any():
        raise ValueError("duplicate state/option/horizon rows")
    return labels


def _credit(data_dir):
    credit = pd.read_csv(checked(data_dir / CREDIT))
    missing = [c for c in CREDIT_REQUIRED if c not in credit.columns]
    if missing:
        raise ValueError("group_credit.csv missing columns " + str(missing))
    if not np.isfinite(credit.C_G).all():
        raise ValueError("non-finite C_G")
    return credit


def _splits(data_dir, states):
    raw = json.loads(checked(data_dir / SPLITS).read_text(encoding="utf-8"))
    fold_of = {str(w): int(f) for w, f in raw["fold_of"].items()}
    worlds = set(states.world.astype(str))
    if set(fold_of) != worlds:
        raise ValueError("splits worlds do not match states worlds")
    k = int(raw.get("folds", max(fold_of.values()) + 1))
    if k < 2:
        raise ValueError("need >= 2 folds")
    return fold_of


def load(data_dir, verify_provenance=True):
    """Read and validate a complete EXAM v1 dataset directory."""
    data_dir = checked(data_dir)
    if not data_dir.is_dir():
        raise ValueError("not a directory: " + str(data_dir))
    provenance = _verify_provenance(data_dir) if verify_provenance else {}
    states = _states(data_dir)
    labels = _labels(data_dir, states)
    credit = _credit(data_dir)
    fold_of = _splits(data_dir, states)
    feature_names = sorted(states.iloc[0]["features"])
    if not all(sorted(r["features"]) == feature_names for r in states.to_dict("records")):
        raise ValueError("inconsistent feature keys across states")
    features = pd.DataFrame([r["features"] for r in states.to_dict("records")])
    features.index = states.state_id.to_numpy()
    features = features[feature_names]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("non-finite feature values")
    for o in GROUP_OPTIONS:
        if o not in states.columns:
            states[o] = states["available"].apply(lambda a: bool(a.get(o)))
    return Dataset(states=states, features=features, labels=labels, credit=credit, fold_of=fold_of,
                   feature_names=feature_names, provenance=provenance)
