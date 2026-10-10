"""Real :class:`evalh.sources.EvalSource` over packed WM-N1 shards.

This closes INTERFACES **Mismatch 5**: ``evalh/sources.py`` ships only
``SyntheticSource``; the evaluation harness needs a source that reads the
realised rows of a packed corpus.  ``PackedEvalSource`` consumes the train
read-side pack (``manifest.json`` schema ``wm_n1_pack/1`` + ``outcomes.parquet``
+ the concatenated memmap arrays) and emits, in the frozen
``evalh.protocol`` shapes:

* :meth:`outcome_rows` -> ``list[OutcomeRow]``, one per
  ``(world, t0, decision_t, tau, policy)`` label row of the requested split;
* :meth:`rollout_rows` -> ``list[RolloutRow]``, the true short-horizon
  trajectories reconstructed from the packed per-tick vehicle states.

HOLDOUT, TEST and chunk-5 items are refused when the pack is opened, so a
sealed pack can never be read through this source.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _util as U
from .pack import MANIFEST_NAME, OUTCOMES_NAME, TRAIN_SCHEMA

SPLIT_TO_PACK = {"train_oof": "TRAIN", "cal": "CALIBRATION"}
FORBIDDEN_SPLITS = ("HOLDOUT", "TEST", "SEALED")
FORBIDDEN_CHUNKS = ("chunk5", "chunk_5", "chunk-5")

DAMPING = "damping"
NATIVE = "native"


class PackedSourceRefusal(PermissionError):
    """A sealed split/item was requested from a packed source."""


def _num(value, default=0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _eval_protocol():
    U.ensure_wm_n1_on_path()
    from evalh import protocol

    return protocol


class PackedEvalSource:
    """Map a packed corpus directory to ``evalh`` outcome/rollout rows."""

    def __init__(self, pack_dir, *, horizons_s=None):
        U.ensure_wm_n1_on_path()
        protocol = _eval_protocol()
        self.protocol = protocol
        self.horizons_s = tuple(horizons_s if horizons_s is not None
                                else protocol.HORIZONS_S)
        self.pack_dir = Path(pack_dir)
        path = self.pack_dir / MANIFEST_NAME
        if not path.is_file():
            raise PackedSourceRefusal(f"no packed manifest at {path}")
        manifest = U.read_json(path)
        if manifest.get("schema") != TRAIN_SCHEMA:
            raise PackedSourceRefusal(
                f"pack schema {manifest.get('schema')!r} != {TRAIN_SCHEMA!r}")
        for item in manifest.get("items", []):
            self._refuse_item(item)
        self.manifest = manifest
        self.items = list(manifest.get("items", []))
        self._arrays: dict = {}
        self._outcomes = None
        self.sign_threshold = 0.0

    # -- guards -------------------------------------------------------------
    @staticmethod
    def _refuse_item(item: dict) -> None:
        chunk = str(item.get("chunk", ""))
        split = str(item.get("split", "")).upper()
        world = item.get("world")
        if split in FORBIDDEN_SPLITS:
            raise PackedSourceRefusal(
                f"packed item {world}/{item.get('arm')} has forbidden split "
                f"{split!r}; HOLDOUT/TEST are sealed")
        if str(world).upper().startswith("HOLDOUT"):
            raise PackedSourceRefusal(f"HOLDOUT world {world!r} is sealed")
        if any(chunk.startswith(c) for c in FORBIDDEN_CHUNKS):
            raise PackedSourceRefusal(
                f"packed item {world} is under {chunk!r} (chunk-5 TEST is "
                "sealed)")

    def _pack_split(self, split: str) -> str:
        if split == "final_test":
            raise PackedSourceRefusal(
                "chunk-5 final TEST is sealed and is never present in a pack")
        if split not in SPLIT_TO_PACK:
            raise PackedSourceRefusal(
                f"unknown split {split!r}; expected train_oof|cal|final_test")
        return SPLIT_TO_PACK[split]

    def _array(self, name):
        if name not in self._arrays:
            spec = self.manifest["arrays"][name]
            self._arrays[name] = np.load(self.pack_dir / spec["file"],
                                         mmap_mode="r")
        return self._arrays[name]

    def _outcomes_df(self):
        if self._outcomes is None:
            import pandas as pd

            spec = self.manifest.get("outcomes", {"file": OUTCOMES_NAME})
            self._outcomes = pd.read_parquet(self.pack_dir / spec["file"])
        return self._outcomes

    def _items_for(self, split: str) -> list:
        pack_split = self._pack_split(split)
        return [it for it in self.items
                if str(it.get("split", "")).upper() == pack_split]

    # -- protocol -----------------------------------------------------------
    def outcome_rows(self, split: str):
        OutcomeRow = self.protocol.OutcomeRow
        df = self._outcomes_df()
        rows = []
        n = len(df)
        for it in self._items_for(split):
            world = str(it["world"])
            t0 = _num(it["t0_s"])
            policy = str(it.get("policy", it.get("arm", "unknown")))
            for pos in it.get("outcome_rows", []):
                pos = int(pos)
                if not 0 <= pos < n:
                    continue
                r = df.iloc[pos]
                available = r.get("tau_available", True)
                if available is False or (available is not None
                                          and not bool(available)):
                    continue
                dec = _num(r.get("t"), t0)
                rows.append(OutcomeRow(
                    world=world, t0_s=t0, decision_t_s=dec,
                    tau_s=_num(r.get("tau")), policy=policy,
                    y_delay_s=_num(r.get("total_delay_ref_s")),
                    collision=_num(r.get("collision_any")),
                    ttc_lt3_s=_num(r.get("ttc_lt3_veh_s")),
                    hard_brake_s=_num(r.get("hard_brake_s")),
                    n_vehicles=_num(r.get("n_vehicles_present")),
                    features={"arm": it.get("arm"), "chunk": it.get("chunk"),
                              "pack_split": it.get("split")}))
        self.sign_threshold = max(1.0, self._median_abs_contrast(rows))
        return rows

    def _median_abs_contrast(self, rows) -> float:
        by = {}
        for r in rows:
            key = (r.world, r.decision_t_s, r.tau_s)
            by.setdefault(key, {})[r.policy] = r.y_delay_s
        d = [abs(v[DAMPING] - v[NATIVE]) for v in by.values()
             if DAMPING in v and NATIVE in v]
        return float(np.median(d)) if d else 0.0

    def rollout_rows(self, split: str):
        RolloutRow = self.protocol.RolloutRow
        dt = 0.5
        veh = self._array("veh_step")
        mask = self._array("veh_mask")
        n_ticks = int(veh.shape[0])
        h = len(self.horizons_s)
        step = [int(round(float(x) / dt)) for x in self.horizons_s]
        rows = []
        for it in self._items_for(split):
            world = str(it["world"])
            tick = int(it["tick"])
            policy = str(it.get("policy", it.get("arm", "unknown")))
            snap = mask[tick].astype(bool)
            slots = np.nonzero(snap)[0]
            v = int(slots.size)
            if v == 0:
                continue
            pos = np.empty((h, v), float)
            spd = np.empty((h, v), float)
            for i, s in enumerate(step):
                j = min(tick + s, n_ticks - 1)
                state = veh[j][slots].astype(np.float32)
                pos[i] = state[:, 0]
                spd[i] = state[:, 4]
            true_dist = float(np.mean(np.abs(pos[-1] - pos[0])))
            true_speed_scale = float(np.mean(spd[0]))
            rows.append(RolloutRow(
                world=world, t0_s=_num(it["t0_s"]), policy=policy,
                true_pos=pos, true_speed=spd, horizons_s=self.horizons_s,
                true_dist=true_dist, true_speed_scale=true_speed_scale,
                features={"arm": it.get("arm"), "tick": tick}))
        return rows

    @classmethod
    def from_pack(cls, pack_dir, **kwargs) -> "PackedEvalSource":
        return cls(pack_dir, **kwargs)
