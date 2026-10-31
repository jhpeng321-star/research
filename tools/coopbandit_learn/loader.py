"""Load and validate an EXAM v1 labelled dataset (executor export format).

Every filesystem access goes through schema.checked(); the export MANIFEST
sha256s are verified; dev worlds 981080..981091 are rejected (live-test only).
Availability comes from labels.csv (per option); O0 is always available.
Partial (completion-order) exports load fine and are flagged exploratory.
"""
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

from .schema import (CREDIT, CREDIT_REQUIRED, DEV_WORLDS, FORBIDDEN_FEATURES, FULL_PANEL_STATES,
                     HORIZONS, LABEL_COLUMNS, LABELS, MANIFEST, OPTIONS, POLICY_HORIZON, PROVENANCE,
                     README, STATES, STATE_REQUIRED, WORLDS, ALL_OPTIONS, checked, sha, truthy)


@dataclass
class Dataset:
    states: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    credit: pd.DataFrame
    feature_names: list
    provenance: dict
    worlds_info: dict
    _cache: dict = field(default_factory=dict)

    def feature_matrix(self, state_ids):
        key = ("X", tuple(state_ids))
        if key not in self._cache:
            self._cache[key] = self.features.loc[list(state_ids), self.feature_names].to_numpy(dtype=float)
        return self._cache[key]

    def world_of(self, state_ids):
        return self.states.set_index("state_id").loc[list(state_ids), "world_id"].to_numpy()

    def available(self, state_id):
        key = ("avail", state_id)
        if key not in self._cache:
            row = self.states.set_index("state_id").loc[state_id]
            self._cache[key] = tuple(o for o in OPTIONS if bool(row[o]))
        return self._cache[key]

    def availability_map(self, state_ids):
        rows = self.states.set_index("state_id").loc[list(state_ids)]
        return {o: rows[o].to_numpy(dtype=bool) for o in OPTIONS}

    def labels_for(self, state_id, horizon_s):
        key = ("lab", state_id, horizon_s)
        if key not in self._cache:
            sub = self.labels[(self.labels.state_id == state_id) & (self.labels.horizon_s == horizon_s)]
            self._cache[key] = dict(zip(sub.option, sub.D))
        return self._cache[key]

    @property
    def partial(self):
        if self.provenance.get("partial"):
            return True
        return len(self.states) < FULL_PANEL_STATES


def _verify_manifest(data_dir):
    manifest = json.loads(checked(data_dir / MANIFEST).read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        path = checked(data_dir / name)
        if not path.exists() or sha(path) != digest:
            raise ValueError("MANIFEST mismatch: " + name)
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
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                raise ValueError("non-finite feature " + k + " for " + r["state_id"])
        if int(r["world_id"]) in DEV_WORLDS:
            raise ValueError("live dev world in fit data: " + str(r["world_id"]))
    return pd.DataFrame(rows)


def _labels(data_dir, states):
    labels = pd.read_csv(checked(data_dir / LABELS))
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        raise ValueError("labels.csv missing columns " + str(missing))
    if set(labels.horizon_s) != set(HORIZONS):
        raise ValueError("labels horizons must be exactly " + repr(HORIZONS))
    bad_option = set(labels.option) - set(ALL_OPTIONS)
    if bad_option:
        raise ValueError("unknown options in labels: " + repr(sorted(bad_option)))
    if not np.isfinite(labels.D).all():
        raise ValueError("non-finite D labels")
    zero = labels[labels.option == "O0"]
    if not (zero.D.abs() < 1e-9).all():
        raise ValueError("O0 rows must carry D=0")
    key = ["state_id", "option", "horizon_s"]
    if labels.duplicated(key).any():
        raise ValueError("duplicate state/option/horizon rows")
    raw_world, raw_t0 = states.world_id.copy(), states.t0.copy()
    wide = labels[labels.horizon_s == POLICY_HORIZON].pivot(index="state_id", columns="option",
                                                            values="available").apply(lambda c: c.map(truthy))
    wide = wide.reindex(states.state_id.to_numpy())
    if wide.isna().any().any():
        raise ValueError("missing option availability rows for some states")
    for o in OPTIONS:
        states[o] = wide[o].to_numpy(dtype=bool)
    meta = labels[(labels.horizon_s == POLICY_HORIZON) & (labels.option == "O0")].set_index("state_id")
    for col in ("world_id", "cell", "t0", "setting"):
        states[col] = states.state_id.map(meta[col])
    if not (states.world_id == states.apply(lambda r: int(str(r.state_id).split("_")[0][1:]), axis=1)).all():
        raise ValueError("state_id/world_id inconsistent")
    if not ((states.world_id == raw_world) & np.isclose(states.t0, raw_t0)).all():
        raise ValueError("labels metadata contradict states.jsonl")
    if not np.isclose(states.t0, states.state_id.str.split("_t").str[1].astype(float)).all():
        raise ValueError("state_id/t0 inconsistent")
    for r in states.itertuples():
        for o in OPTIONS:
            if not r.__getattribute__(o) and not np.isclose(
                    labels[(labels.state_id == r.state_id) & (labels.option == o)].D.abs().max(), 0.0):
                raise ValueError("unavailable option with non-zero D: " + r.state_id + "/" + o)
    return labels


def _credit(data_dir):
    credit = pd.read_csv(checked(data_dir / CREDIT))
    missing = [c for c in CREDIT_REQUIRED if c not in credit.columns]
    if missing:
        raise ValueError("group_credit.csv missing columns " + str(missing))
    if not np.isfinite(credit.C_G).all():
        raise ValueError("non-finite C_G")
    return credit


def load(data_dir, verify_manifest=True):
    """Read and validate a complete EXAM v1 export directory."""
    data_dir = checked(data_dir)
    if not data_dir.is_dir():
        raise ValueError("not a directory: " + str(data_dir))
    manifest = _verify_manifest(data_dir) if verify_manifest else {}
    checked(data_dir / README)
    provenance = json.loads(checked(data_dir / PROVENANCE).read_text(encoding="utf-8"))
    worlds_info = json.loads(checked(data_dir / WORLDS).read_text(encoding="utf-8"))
    states = _states(data_dir)
    seeds = {int(w["world"]["seed"]) for w in worlds_info["worlds"]}
    if set(states.world_id.astype(int)) != seeds:
        raise ValueError("states.jsonl worlds do not match worlds.json selection")
    labels = _labels(data_dir, states)
    credit = _credit(data_dir)
    feature_names = sorted(states.iloc[0]["features"])
    records = states.to_dict("records")
    if not all(sorted(r["features"]) == feature_names for r in records):
        raise ValueError("inconsistent feature keys across states")
    features = pd.DataFrame([r["features"] for r in records])
    features.index = states.state_id.to_numpy()
    features = features[feature_names]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("non-finite feature values")
    return Dataset(states=states, features=features, labels=labels, credit=credit,
                   feature_names=feature_names, provenance=provenance, worlds_info=worlds_info,
                   _cache={"manifest": manifest})
