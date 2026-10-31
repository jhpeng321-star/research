"""Dependency-free runtime for an exported learned policy (numpy + stdlib only).

This module is stageable to a remote box as plain text (it must never import
pandas/sklearn). It loads a policy JSON artifact written by artifact.py and
reproduces the documented choice rule: argmax predicted D600 among available
group options, executed only when predicted D minus the calibrated radius is
positive, otherwise O0 (MAPPO).
"""
import json
from pathlib import Path

import numpy as np


class PolicyRuntime:
    VERSION = 1

    def __init__(self, artifact):
        if int(artifact.get("version", 0)) != self.VERSION:
            raise ValueError("unsupported policy artifact version")
        self.feature_names = list(artifact["feature_names"])
        self.options = list(artifact["options"])
        self.radius = float(artifact["radius"])
        self.horizon_s = int(artifact["horizon_s"])
        self.models = {name: self._Model(spec) for name, spec in artifact["models"].items()}

    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_text(cls, text):
        return cls(json.loads(text))

    def feature_vector(self, features):
        try:
            return np.array([[float(features[k]) for k in self.feature_names]])
        except KeyError as e:
            raise ValueError("missing observable feature: " + str(e)) from None

    def predict(self, features):
        x = self.feature_vector(features)
        return {name: float(model.predict(x)[0]) for name, model in self.models.items()}

    def choose(self, features, available_options):
        pred = self.predict(features)
        best, best_pred = "O0", -np.inf
        for option in available_options:
            if option not in pred:
                continue
            if pred[option] > best_pred:
                best, best_pred = option, pred[option]
        if best != "O0" and best_pred - self.radius > 0.0:
            return best
        return "O0"

    class _Model:
        def __init__(self, spec):
            self.init = float(spec["init"])
            self.learning_rate = float(spec["learning_rate"])
            self.trees = [self._Tree(t) for t in spec["trees"]]

        def predict(self, x):
            out = np.full(len(x), self.init)
            for tree in self.trees:
                out += self.learning_rate * tree.predict(x)
            return out

        class _Tree:
            def __init__(self, spec):
                self.feature = np.asarray(spec["feature"], dtype=int)
                self.threshold = np.asarray(spec["threshold"], dtype=float)
                self.left = np.asarray(spec["left"], dtype=int)
                self.right = np.asarray(spec["right"], dtype=int)
                self.value = np.asarray(spec["value"], dtype=float)
                self.leaf = self.left < 0

            def predict(self, x):
                node = np.zeros(len(x), dtype=int)
                while True:
                    idx = np.flatnonzero(~self.leaf[node])
                    if idx.size == 0:
                        break
                    go_right = x[idx, self.feature[node[idx]]] > self.threshold[node[idx]]
                    node[idx] = np.where(go_right, self.right[node[idx]], self.left[node[idx]])
                return self.value[node]
