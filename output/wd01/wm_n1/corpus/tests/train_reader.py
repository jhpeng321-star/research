"""A minimal MemmapCorpus-equivalent reader (copy of the train-side contract).

This reproduces the parts of ``train/dataset.py`` the CORPUS parity test needs:
manifest schema, item validation/selection, array access, outcome mapping and
the Dataset ``__getitem__`` shapes.  It exists so the corpus tests can prove the
packed layout without importing (or merging) the TRAIN branch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model import constants as C

PACK_SCHEMA = "wm_n1_pack/1"
FORBIDDEN_SPLITS = ("HOLDOUT", "TEST", "SEALED")
FORBIDDEN_CHUNKS = ("chunk5",)
_ITEM_REQUIRED = ("world", "arm", "chunk", "split", "t0_s", "pair_key",
                  "tick", "plan_row", "outcome_rows")
_TENSORS = ("veh_step", "veh_mask", "plan", "wall", "wall_mask", "counter",
            "accel_next", "target_mask", "lane_next", "arrivals")


class CorpusRefusal(PermissionError):
    pass


class TrainReader:
    def __init__(self, pack_dir, split="TRAIN"):
        self.pack_dir = Path(pack_dir)
        self.split = split
        self.manifest = self._load_manifest()
        self.items = self._select_items()
        self._mm = {}
        self._rows = None

    def _load_manifest(self):
        path = self.pack_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != PACK_SCHEMA:
            raise CorpusRefusal(f"schema {manifest.get('schema')!r}")
        for it in manifest.get("items", []):
            self._refuse(it)
        return manifest

    @staticmethod
    def _refuse(it):
        if str(it.get("split", "")).upper() in FORBIDDEN_SPLITS:
            raise CorpusRefusal("sealed split")
        if any(str(it.get("chunk", "")).startswith(c)
               for c in FORBIDDEN_CHUNKS):
            raise CorpusRefusal("sealed chunk")

    def _select_items(self):
        out = []
        for raw in self.manifest.get("items", []):
            missing = [k for k in _ITEM_REQUIRED if k not in raw]
            if missing:
                raise CorpusRefusal(f"item misses {missing}")
            if str(raw["split"]).upper() == self.split:
                out.append(dict(raw))
        return out

    def _array(self, name):
        if name not in self._mm:
            spec = self.manifest["arrays"][name]
            self._mm[name] = np.load(self.pack_dir / spec["file"],
                                     mmap_mode="r")
        return self._mm[name]

    def _outcomes(self):
        if self._rows is None:
            import pandas as pd

            spec = self.manifest.get("outcomes", {"file": "outcomes.parquet"})
            self._rows = pd.read_parquet(self.pack_dir / spec["file"])
        return self._rows

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        tick, plan_row = int(it["tick"]), int(it["plan_row"])
        veh = self._array("veh_step")[tick]
        rows = self._outcomes()
        pos = [int(p) for p in it["outcome_rows"]]
        sub = rows.iloc[[p for p in pos if p >= 0]] if pos else rows.iloc[0:0]
        by_tau = {round(float(r["tau"]), 6): r for _, r in sub.iterrows()}
        targets = np.zeros((len(C.TAUS), len(C.OUTCOME_TARGETS)), np.float32)
        omask = np.zeros(len(C.TAUS), bool)
        for i2, tau in enumerate(C.TAUS):
            r = by_tau.get(round(float(tau), 6))
            if r is None:
                continue
            omask[i2] = bool(r.get("tau_available", True))
            targets[i2, 0] = float(r.get("total_delay_ref_s", 0.0))
            targets[i2, 1] = float(r.get("ttc_lt3_veh_s", 0.0))
            targets[i2, 2] = float(r.get("hard_brake_s", 0.0))
        return {
            "veh_step": np.asarray(veh, np.float32),
            "veh_mask": self._array("veh_mask")[tick].astype(bool),
            "plan": np.asarray(self._array("plan")[plan_row], np.float32),
            "wall": np.asarray(self._array("wall")[tick], np.float32),
            "wall_mask": self._array("wall_mask")[tick].astype(bool),
            "counter": np.asarray(self._array("counter")[tick], np.float32),
            "accel": np.asarray(self._array("accel_next")[tick], np.float32),
            "target_mask": self._array("target_mask")[tick].astype(bool),
            "lane": self._array("lane_next")[tick].astype(np.int64),
            "arrivals": np.asarray(self._array("arrivals")[tick], np.float32),
            "outcome_quantiles": targets,
            "outcome_mask": omask,
            "world": it["world"], "arm": it["arm"], "chunk": it["chunk"],
            "split": it["split"], "t0_s": float(it["t0_s"]),
            "pair_key": it["pair_key"], "policy": it.get("policy"),
        }

    def worlds(self):
        seen = []
        for it in self.items:
            if it["world"] not in seen:
                seen.append(it["world"])
        return sorted(seen)
