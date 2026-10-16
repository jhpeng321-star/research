"""Mixed training data: original C1 packs + switching / direct-output packs.

The switching packs (``hetpack switching``, branch ``wms/swpack``) are ordinary
WM-N1 v3 packs (K=192) whose items are *busy-tick snapshots* of runs in which
every CAV follows a per-1 s controller schedule over
``{native, damping, pi-external-u, direct}``.  Each pack directory additionally
carries the action timelines in the world model's own format::

    controller_timeline.npy         int8    [n_items, 120, K]   (CONTROLLERS index)
    issued_acceleration_mps2.npy    float32 [n_items, 120, K]   (0 unless direct)
    issued_target_speed_mps.npy     float32 [n_items, 120, K]   (0 unless direct)

indexed by the item's ORIGINAL manifest position and joined to the pack through
``ACTION_TIMELINE_INDEX_<split>.json`` (``position/world/arm/tick``).  The CAV
axis is the item's stable t0 slot columns; a vehicle absent at a step is
``native``.

This module (all new; the trainer only gets small hooks) provides

* :class:`SwitchTimelines` - mmap-backed, join-verified binding of those arrays
  to the pack items (``timeline(position, n_control)`` per window);
* :class:`HeldOutGuard` + :class:`SwitchItemFilter` - the FAIL-CLOSED zero-shot
  guard: the worlds in ``ZERO_SHOT_HELDOUT.json`` may never appear in training
  and, by default, items with a CAV direct segment of the held-out direct-profile
   family are dropped (policy ``refuse`` stops the run instead; no bypass);
* :class:`MixedTrajectoryDataset` - original + switching sources with a
  configurable sampling mix ratio and over-sampling of windows that contain a
  controller switch or a direct segment WITHIN the training stage's horizon;
* :func:`preflight` / :func:`check_receipt` - the launch-pitfall checks on the data.

TRAIN only: nothing here reads CAL / HOLDOUT / sealed / chunk-5 / J4B / INC1
bytes, an outcome, or a pack array other than one item's mmap slice at a time.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from wms.core.errors import TrainDataRefusal
from wms.data.action_timeline import CONTROLLERS, ActionTimeline
from wms.data import controller_hygiene as hygiene
from wms.data import jam as jamdata
from wms.data.dataset import TrajectoryDataset
from wms.data.sidecars import read_identity_rows
from wms.data.sources import validate_train_source
from wms.data.windows import ControllerExclusion

INDEX_SCHEMA = "hetpack_switching_actions/1"
HELDOUT_SCHEMA = "hetpack_switching_heldout/1"
DATA_CONFIG_SCHEMA = "wms_train_main_data/1"
TIMELINE_ARRAYS = ("controller_timeline", "issued_acceleration_mps2",
                   "issued_target_speed_mps")
NATIVE = CONTROLLERS.index("native")
DIRECT = CONTROLLERS.index("direct")
FAMILY_POLICIES = ("drop", "refuse", "release")
_WORLD_RE = re.compile(r"^w(\d+)m")


class MixDataRefusal(TrainDataRefusal):
    """A mixed-data contract violation (fail closed)."""


# --------------------------------------------------------------- held-out ----
def direct_family(seed: int, vehicle_id: str, *, n_families: int = 4) -> int:
    """Direct-profile family of ``(world seed, CAV id)``.

    Bit-identical re-implementation of ``hetdata_kit.hetpack.switching.direct_family``
    (branch ``wms/swpack``), which buckets the weight-mean angular frequency of the
    generator's four-sinusoid mixture.  Pinned by golden values in the tests
    (seed 8859 -> family 3 for the held-out vehicles of ZERO_SHOT_HELDOUT.json).
    """
    if n_families < 2:
        raise MixDataRefusal("need at least two direct families")
    rng = random.Random(f"direct_profile_v1:{int(seed)}:{vehicle_id}")
    terms = [(rng.uniform(0.004, 0.025), rng.uniform(0, 2 * math.pi),
              rng.uniform(0.2, 1.0)) for _ in range(4)]
    total = sum(weight for _o, _p, weight in terms)
    mean_omega = sum(omega * weight for omega, _p, weight in terms) / total
    lo, hi = 0.004, 0.025
    frac = min(1.0, max(0.0, (mean_omega - lo) / (hi - lo)))
    return min(n_families - 1, int(frac * n_families))


def world_seed(world: str) -> int:
    """Seed encoded in a HETDATA world id (``w8859m2700p10`` -> 8859)."""
    match = _WORLD_RE.match(str(world))
    if not match:
        raise MixDataRefusal(f"cannot parse a seed from world id {world!r}")
    return int(match.group(1))


class HeldOutGuard:
    """Parsed ``ZERO_SHOT_HELDOUT.json`` (fails closed on any inconsistency)."""

    def __init__(self, doc: dict, *, source: str = "<memory>"):
        if not isinstance(doc, dict) or doc.get("schema") != HELDOUT_SCHEMA:
            raise MixDataRefusal(f"{source}: not a {HELDOUT_SCHEMA} document")
        guard = doc.get("overlap_guard")
        if (not isinstance(guard, dict) or guard.get("overlap_guard_ok") is not True
                or guard.get("problems") or guard.get("held_subset_of_test") is not True
                or guard.get("disjoint") is not True):
            raise MixDataRefusal(f"{source}: the builder's overlap guard is not clean")
        worlds = doc.get("held_worlds")
        if not isinstance(worlds, list) or not worlds:
            raise MixDataRefusal(f"{source}: no held_worlds")
        self.held_worlds = frozenset(str(w) for w in worlds)
        if (len(self.held_worlds) != len(worlds)
                or int(doc.get("n_held_worlds", -1)) != len(worlds)
                or set(doc.get("worlds", {})) != set(self.held_worlds)):
            raise MixDataRefusal(f"{source}: held-out world bookkeeping is inconsistent")
        self.held_family = int(doc["held_family"])
        some = next(iter(doc["worlds"].values()))
        self.n_families = int(some.get("n_families", 4))
        if any(int(value.get("n_families", 4)) != self.n_families or
               int(value.get("held_family", self.held_family)) != self.held_family
               for value in doc["worlds"].values()):
            raise MixDataRefusal(f"{source}: inconsistent direct family definitions")
        if not 0 <= self.held_family < self.n_families:
            raise MixDataRefusal(f"{source}: held family out of range")
        sel = doc.get("family_selection") or {}
        # The builder records that the held family is NOT absent from TRAIN/VAL
        # (``clean: false``): family exclusion is therefore enforced here, per item.
        self.family_clean = bool(sel.get("clean"))
        self.family_train_val_overlap = int(sel.get("train_val_overlap", 0))
        self.source = source

    @classmethod
    def load(cls, path: str | Path) -> "HeldOutGuard":
        p = Path(path)
        if not p.is_file():
            raise MixDataRefusal(f"ZERO_SHOT_HELDOUT file is required and missing: {p}")
        return cls(json.loads(p.read_text(encoding="utf-8")), source=str(p))

    def refuse_worlds(self, worlds: Sequence[str], *, where: str) -> None:
        bad = sorted(set(map(str, worlds)) & self.held_worlds)
        if bad:
            raise MixDataRefusal(
                f"zero-shot held-out worlds present in {where}: {bad[:5]} "
                f"({len(bad)} total); refusing")

    def is_held_family(self, world: str, vehicle_id: str) -> bool:
        return direct_family(world_seed(world), vehicle_id,
                             n_families=self.n_families) == self.held_family


class ReleasedGuard:
    """The switching pack's zero-shot held-out family is
    RELEASED for training.  No guard file is required or opened; nothing is refused or
    dropped.  (The held-out WORLDS live outside the TRAIN pack and are still never read.)"""

    held_worlds = frozenset()
    held_family = -1
    n_families = 0
    family_clean = True
    family_train_val_overlap = 0
    source = "released (C4)"

    def refuse_worlds(self, worlds, *, where: str) -> None:
        return None

    def is_held_family(self, world: str, vehicle_id: str) -> bool:
        return False


class IdentityBook:
    """Slot -> ``(world, vehicle_id)`` through a ``wm_n1_train_identity_sidecar/1``."""

    def __init__(self, identity_dir: str | Path):
        root = validate_train_source(identity_dir)
        receipt = json.loads((root / "TRAIN_IDENTITY_RECEIPT.json").read_text(encoding="utf-8"))
        if receipt.get("split") != "TRAIN" or receipt.get("schema") != \
                "wm_n1_train_identity_sidecar/1":
            raise MixDataRefusal(f"identity sidecar is not TRAIN: {root}")
        self.ids = np.load(root / "identities.npy", mmap_mode="r", allow_pickle=False)
        positions = np.load(root / "global_positions.npy", mmap_mode="r", allow_pickle=False)
        self.row_of = {int(p): i for i, p in enumerate(positions.tolist())}
        self.vehicle_map = json.loads((root / "vehicle_map.json").read_text(encoding="utf-8"))
        if self.ids.ndim != 2 or len(self.row_of) != self.ids.shape[0]:
            raise MixDataRefusal(f"identity sidecar shape mismatch: {root}")
        read_identity_rows(root, corpus_k=self.ids.shape[1])

    def vehicle(self, position: int, slot: int) -> Optional[tuple[str, str]]:
        row = self.row_of.get(int(position))
        if row is None:
            return None
        idx = int(self.ids[row, int(slot)])
        if idx < 0:
            return None
        entry = self.vehicle_map[idx]
        return str(entry[1]), str(entry[2])


# -------------------------------------------------------------- timelines ----
def event_steps(controller: np.ndarray, continuity: Optional[np.ndarray] = None
                ) -> tuple[int, int]:
    """``(first_switch_step, first_direct_step)`` of ``[T,K]`` controller ids.

    A *switch* is a controller change of the same vehicle between consecutive
    control steps; ``continuity[c, k]`` (``[T-1, K]``) marks that the slot's
    vehicle is present at both steps, so a departure (which resets the column to
    ``native``) is never counted.  ``T`` (= no event) is returned when absent.
    """
    ctrl = np.asarray(controller)
    steps = int(ctrl.shape[0])
    change = ctrl[1:] != ctrl[:-1]
    if continuity is not None:
        change &= np.asarray(continuity, dtype=bool)
    rows = np.flatnonzero(change.any(axis=1)) if steps > 1 else np.empty(0, np.int64)
    first_switch = int(rows[0]) + 1 if rows.size else steps
    drows = np.flatnonzero((ctrl == DIRECT).any(axis=1))
    first_direct = int(drows[0]) if drows.size else steps
    return first_switch, first_direct


class SwitchTimelines:
    """mmap-backed, join-verified action timelines of one switching pack."""

    def __init__(self, pack_dir: str | Path, index_path: str | Path, *,
                 split: str = "TRAIN"):
        self.pack_dir = validate_train_source(pack_dir)
        validate_train_source(index_path)
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if index.get("schema") != INDEX_SCHEMA:
            raise MixDataRefusal(f"{index_path}: not a {INDEX_SCHEMA} index")
        if str(index.get("split", "")).upper() != split.upper():
            raise MixDataRefusal(f"{index_path}: split {index.get('split')!r} != {split}")
        if list(index.get("controllers", ())) != list(CONTROLLERS):
            raise MixDataRefusal(f"{index_path}: controller vocabulary != {CONTROLLERS}")
        if float(index.get("control_dt_s", 0.0)) != 1.0:
            raise MixDataRefusal(f"{index_path}: control_dt_s must be 1.0")
        self.control_dt_s = 1.0
        self.items = list(index["items"])
        if len(self.items) != int(index["n_items"]):
            raise MixDataRefusal(f"{index_path}: n_items does not match its items")
        for expected, item in enumerate(self.items):
            if int(item["position"]) != expected:
                raise MixDataRefusal(f"{index_path}: item {expected} has position "
                                     f"{item['position']}")
        self.control_steps = int(index["control_steps"])
        self.k_max = int(index["k_max"])
        if self.control_steps < 1 or self.k_max < 1:
            raise MixDataRefusal("timeline dimensions must be positive")
        if any(not 0 <= int(item["valid_steps"]) <= self.control_steps for item in self.items):
            raise MixDataRefusal("timeline valid_steps outside stored bounds")
        self.arrays = {}
        for name in TIMELINE_ARRAYS:
            f = self.pack_dir / f"{name}.npy"
            if not f.is_file():
                raise MixDataRefusal(f"timeline array missing: {f}")
            self.arrays[name] = np.load(f, mmap_mode="r", allow_pickle=False)
        shape = (len(self.items), self.control_steps, self.k_max)
        for name, arr in self.arrays.items():
            if tuple(arr.shape) != shape:
                raise MixDataRefusal(f"{self.pack_dir / name}.npy shape {arr.shape} != {shape}")
        if self.arrays["controller_timeline"].dtype.kind not in "iu":
            raise MixDataRefusal("controller_timeline must be an integer array")
        self.worlds = sorted({str(i["world"]) for i in self.items})
        self.ticks_per_run: Optional[dict] = None

    def __len__(self) -> int:
        return len(self.items)

    def check_item(self, position: int, world: str, arm: str, tick: int) -> None:
        """Fail closed unless pack item ``position`` is the indexed ``(world, arm, tick)``."""
        if not 0 <= int(position) < len(self.items):
            raise MixDataRefusal(f"timeline index has no position {position}")
        item = self.items[int(position)]
        if (str(item["world"]), str(item["arm"]), int(item["tick"])) != \
                (str(world), str(arm), int(tick)):
            raise MixDataRefusal(
                f"timeline join mismatch at position {position}: index "
                f"({item['world']}, {item['arm']}, {item['tick']}) vs pack "
                f"({world}, {arm}, {tick})")

    def controller(self, position: int) -> np.ndarray:
        return np.asarray(self.arrays["controller_timeline"][int(position)]).astype(np.int64)

    def raw(self, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.controller(position),
                np.asarray(self.arrays["issued_acceleration_mps2"][int(position)], np.float32),
                np.asarray(self.arrays["issued_target_speed_mps"][int(position)], np.float32))

    def timeline(self, position: int, n_control: int) -> dict:
        """``wms_action_timeline/1`` arrays ``[n_control, K]`` for one window.

        Steps beyond ``valid_steps`` are ``native`` with zero commands for
        departed vehicles. Requests longer than stored coverage are refused.
        """
        ctrl, accel, target = self.raw(position)
        ActionTimeline(ctrl, accel, target)             # vocabulary / finite checks
        n = int(n_control)
        if n > self.control_steps:
            raise MixDataRefusal("requested timeline horizon exceeds stored control steps")
        out_c = np.full((n, ctrl.shape[1]), NATIVE, np.int64)
        out_a = np.zeros((n, ctrl.shape[1]), np.float32)
        out_t = np.zeros((n, ctrl.shape[1]), np.float32)
        m = min(n, ctrl.shape[0], int(self.items[int(position)]["valid_steps"]))
        out_c[:m], out_a[:m], out_t[:m] = ctrl[:m], accel[:m], target[:m]
        return {"controller": out_c, "acceleration_mps2": out_a,
                "target_speed_mps": out_t, "control_dt_s": self.control_dt_s}

    def full_scan(self, chunk: int = 64) -> dict:
        """Chunked value scan of every array (vocabulary, finite, direct has commands)."""
        n_direct = n_switch_items = 0
        for a in range(0, len(self.items), chunk):
            c = np.asarray(self.arrays["controller_timeline"][a:a + chunk]).astype(np.int64)
            ac = np.asarray(self.arrays["issued_acceleration_mps2"][a:a + chunk])
            tg = np.asarray(self.arrays["issued_target_speed_mps"][a:a + chunk])
            if c.min() < 0 or c.max() >= len(CONTROLLERS):
                raise MixDataRefusal(f"controller id outside vocabulary in items {a}..")
            if not (np.isfinite(ac).all() and np.isfinite(tg).all()):
                raise MixDataRefusal(f"non-finite issued command in items {a}..")
            n_direct += int((c == DIRECT).any(axis=(1, 2)).sum())
            n_switch_items += int((c[:, 1:] != c[:, :-1]).any(axis=(1, 2)).sum())
        return {"items_with_direct": n_direct, "items_with_any_change": n_switch_items}


def held_family_slots(timelines: SwitchTimelines, book: Optional[IdentityBook],
                      guard: HeldOutGuard, position: int, world: str
                      ) -> tuple[list[int], int]:
    """Slots with a direct segment of the held family (+ unidentified direct slots)."""
    ctrl = timelines.controller(position)
    direct_slots = np.flatnonzero((ctrl == DIRECT).any(axis=0)).tolist()
    held, unknown = [], 0
    for slot in direct_slots:
        who = None if book is None else book.vehicle(position, slot)
        if who is None:
            unknown += 1                    # cannot prove it is not held -> held
            held.append(slot)
        elif guard.is_held_family(who[0], who[1]):
            held.append(slot)
        if who is not None and who[0] != world:
            raise MixDataRefusal("direct slot identity world does not match timeline world")
    return held, unknown


class SwitchItemFilter:
    """``item_filter`` hook for :meth:`TrajectoryDataset.from_real_packs`.

    Metadata + one mmap slice per item: verifies the timeline join for every
    kept item, refuses held-out worlds, applies the held-family policy and
    records each kept item's first switch / direct step (for over-sampling).
    """

    def __init__(self, timelines: SwitchTimelines, guard: HeldOutGuard,
                 book: Optional[IdentityBook], *, policy: str = "drop",
                  max_drop_frac: float = 0.85, leave_pi_wall_out: bool = False):
        if policy not in FAMILY_POLICIES:
            raise MixDataRefusal(f"heldout family policy must be one of {FAMILY_POLICIES}")
        self.timelines, self.guard, self.book = timelines, guard, book
        self.policy, self.max_drop_frac = policy, float(max_drop_frac)
        if not 0 <= self.max_drop_frac <= 1:
            raise MixDataRefusal("max_drop_frac must be in [0,1]")
        self.leave_pi_wall_out = leave_pi_wall_out
        self.events: dict[int, tuple[int, int]] = {}
        self.receipt: dict = {}

    def __call__(self, corpus, kept_orig: list[int]):
        offsets = corpus.tick_offsets()
        kept, dropped, unknown_total, held_items, pi_dropped = [], 0, 0, 0, 0
        tm = corpus.array("target_mask")
        for i in kept_orig:
            meta = corpus.meta(i)
            pack = corpus.pack_index(i)
            gpos = corpus.manifest_position(i)
            if pack != 0:
                raise MixDataRefusal("a switching timeline source must be pack 0")
            tick = int(meta["tick"]) - (0 if offsets is None else int(offsets[pack]))
            world, arm = str(meta["world"]), str(meta["arm"])
            self.timelines.check_item(gpos, world, arm, tick)
            self.guard.refuse_worlds([world], where="the switching training pack")
            ctrl = self.timelines.controller(gpos)
            ActionTimeline(*self.timelines.raw(gpos))
            if self.leave_pi_wall_out and (ctrl == CONTROLLERS.index("pi-external-u")).any():
                pi_dropped += 1
                continue
            held, unknown = held_family_slots(self.timelines, self.book, self.guard,
                                              gpos, world)
            unknown_total += unknown
            if held:
                held_items += 1
                if self.policy == "refuse":
                    raise MixDataRefusal(
                        f"item {gpos} ({world}) has held-out family-"
                        f"{self.guard.held_family} direct CAV segments (policy refuse)")
                if self.policy == "drop":
                    dropped += 1
                    continue
            kept.append(i)
            steps = ctrl.shape[0]
            lo, hi = tick, tick + 2 * (steps - 1) + 1
            rows = np.asarray(tm[lo:min(hi + 1, tm.shape[0])], dtype=bool)
            # Continuity is cumulative from t0: a reused slot never returns.
            rows = np.cumprod(rows, axis=0).astype(bool)
            cont = np.zeros((steps - 1, ctrl.shape[1]), bool)
            for c in range(steps - 1):
                if 2 * c + 1 < rows.shape[0]:
                    cont[c] = rows[2 * c + 1]
            self.events[gpos] = event_steps(ctrl, cont)
        frac = dropped / max(1, len(kept_orig))
        self.receipt = {
            "policy": self.policy, "held_family": self.guard.held_family,
            "held_family_clean_in_builder": self.guard.family_clean,
            "builder_train_val_family_overlap": self.guard.family_train_val_overlap,
            "n_considered": len(kept_orig), "n_kept": len(kept),
            "n_items_with_held_family_direct": held_items,
            "n_dropped": dropped, "drop_fraction": round(frac, 4),
            "leave_pi_wall_out": self.leave_pi_wall_out, "n_pi_windows_dropped": pi_dropped,
            "n_direct_slots_without_identity": unknown_total,
            "n_held_worlds_checked": len(self.guard.held_worlds),
            "guard_source": self.guard.source}
        if frac > self.max_drop_frac:
            raise MixDataRefusal(
                f"held-family exclusion would drop {frac:.1%} of the switching items "
                f"(> {self.max_drop_frac:.0%}); refusing instead of training on a sliver")
        return kept, self.receipt


def scan_held_family(timelines: SwitchTimelines, book: Optional[IdentityBook],
                     guard: HeldOutGuard) -> dict:
    """Corpus-free held-family census of a whole timeline pack (preflight)."""
    held_items = unknown = 0
    for position, item in enumerate(timelines.items):
        guard.refuse_worlds([item["world"]], where="the switching pack index")
        slots, unk = held_family_slots(timelines, book, guard, position, str(item["world"]))
        held_items += bool(slots)
        unknown += unk
    n = len(timelines.items)
    return {"n_items": n, "n_items_with_held_family_direct": held_items,
            "drop_fraction_if_policy_drop": round(held_items / max(1, n), 4),
            "n_direct_slots_without_identity": unknown}


# ------------------------------------------------------------ DATAHYG strict --
#: strict hygiene disables the switching source when it would drop more than this fraction of
#: its items (drop unless > 60 % is removed; otherwise decided case by case).
HYGIENE_MAX_DROP_FRAC = 0.6


class HygieneSourceExhausted(MixDataRefusal):
    """Strict hygiene removed too much of the switching source; it is left out of the mix."""

    def __init__(self, receipt: dict):
        super().__init__(f"controller hygiene left {receipt['n_kept']}/{receipt['n_considered']} "
                         "switching items")
        self.receipt = receipt


class DirectMixHygiene:
    """``data.controller_hygiene = "strict"`` filter of the switching (direct-mix) items.

    An item is dropped when its window ``[t0, t0 + horizon_steps]`` (= its CAV-teacher span)
    holds a ``damping`` control step of ANY CAV on the road
    (:func:`wms.data.controller_hygiene.directmix_damping_ticks`: recorded t0-cohort labels
    united with the generator's deterministic per-vehicle schedule; post-t0 entrants named by
    the run's next snapshot item; unresolved CAVs count as damping).  Metadata + mmap slices.
    """

    def __init__(self, timelines: SwitchTimelines, book: IdentityBook, *, horizon_steps: int,
                 max_drop_frac: float = HYGIENE_MAX_DROP_FRAC):
        self.timelines, self.book = timelines, book
        self.horizon = int(horizon_steps)
        self.max_drop_frac = float(max_drop_frac)
        self._slots: dict[int, dict[int, str]] = {}
        self.runs: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for item in timelines.items:
            self.runs.setdefault((str(item["world"]), str(item["arm"])), []).append(
                (int(item["tick"]), int(item["position"])))
        self.receipt: dict = {}

    def slots(self, position: int) -> dict[int, str]:
        if position not in self._slots:
            row = self.book.row_of.get(int(position))
            ids = {} if row is None else {
                int(k): str(self.book.vehicle_map[int(v)][2])
                for k, v in enumerate(np.asarray(self.book.ids[row])) if v >= 0}
            self._slots[position] = ids
        return self._slots[position]

    def check(self, corpus, i: int) -> dict:
        """Damping census of dataset position ``i`` (single switching pack = pack 0)."""
        meta = corpus.meta(i)
        gpos = corpus.manifest_position(i)
        world, arm = str(meta["world"]), str(meta["arm"])
        tick = int(meta["tick"])
        start, end = corpus.run_bounds(tick)
        hi = min(end, tick + self.horizon + 121)
        vm = np.asarray(corpus.array("veh_mask")[tick:hi]).astype(bool)
        tm = np.asarray(corpus.array("target_mask")[tick:hi]).astype(bool)
        cav = (np.asarray(corpus.array("veh_step")[tick:hi, :, 7], np.float32) > 0.5) & vm
        snaps = [(t, self.slots(p)) for t, p in sorted(self.runs[(world, arm)])]
        cohort = {k: v for k, v in self.slots(gpos).items() if cav[0, k]}
        return hygiene.directmix_damping_ticks(
            world, tick, self.horizon, veh_mask=vm, target_mask=tm, is_cav=cav,
            cohort_vid=cohort, recorded_ctrl=self.timelines.controller(gpos),
            snapshots=snaps, run_start_tick=start)

    def __call__(self, corpus, kept_orig: list[int]):
        if corpus.tick_offsets() is not None:
            raise MixDataRefusal("the direct-mix hygiene filter needs a single switching pack")
        kept, n_damping, n_unknown, stats = [], 0, 0, {}
        for i in kept_orig:
            res = self.check(corpus, i)
            for key, value in res["stats"].items():
                stats[key] = stats.get(key, 0) + int(value)
            if res["damping"].any():
                n_damping += 1
            elif res["unknown"].any():
                n_unknown += 1
            else:
                kept.append(i)
        n = len(kept_orig)
        frac = 1.0 - len(kept) / max(1, n)
        self.receipt = {
            "mode": "strict", "rule": "drop items whose horizon holds a damping CAV step",
            "horizon_steps": self.horizon, "n_considered": n, "n_kept": len(kept),
            "n_dropped_damping": n_damping, "n_dropped_unresolved_cav": n_unknown,
            "drop_fraction": round(frac, 4), "max_drop_frac": self.max_drop_frac,
            "identity_and_schedule_check": stats,
            "source_disabled": bool(frac > self.max_drop_frac or not kept)}
        if self.receipt["source_disabled"]:
            raise HygieneSourceExhausted(self.receipt)
        return kept, self.receipt


# ------------------------------------------------------------------- mixing --
class MixSource:
    """One source of a :class:`MixedTrajectoryDataset`."""

    def __init__(self, name: str, dataset: Dataset, *, share: float,
                 events: Optional[np.ndarray] = None,
                 jam: Optional[np.ndarray] = None):
        n = len(dataset)
        if n < 1 or not math.isfinite(share) or share <= 0.0:
            raise MixDataRefusal(f"source {name!r} needs items and a positive share")
        if events is not None and (events.shape != (n, 2)):
            raise MixDataRefusal(f"source {name!r} events must be [n_items, 2]")
        if jam is not None and (np.asarray(jam).shape != (n,)):
            raise MixDataRefusal(f"source {name!r} jam flags must be [n_items]")
        self.name, self.dataset, self.share = name, dataset, float(share)
        self.events = events
        self.jam = None if jam is None else np.asarray(jam, dtype=bool)


class MixedTrajectoryDataset(Dataset):
    """Source-mixed lazy dataset with per-stage sampling weights.

    ``stage_weights(stage_s)`` gives each item a weight so that source ``s`` is
    drawn with probability ``share_s`` (normalised) and, inside a source,
    items whose first switch / direct step lies within the stage's horizon are
    ``event_oversample`` times more likely than the rest.  Every record gets an
    ``action_timeline`` (constant ``native`` when its source has none) so batches
    from different sources always collate with their timelines.
    """

    def __init__(self, sources: Sequence[MixSource], *, event_oversample: float = 1.0,
                 jam_oversample: float = 1.0):
        if not sources:
            raise MixDataRefusal("a mixed dataset needs at least one source")
        if not math.isfinite(float(event_oversample)) or float(event_oversample) < 1.0:
            raise MixDataRefusal("event_oversample must be >= 1")
        if not math.isfinite(float(jam_oversample)) or float(jam_oversample) < 1.0:
            raise MixDataRefusal("jam_oversample must be >= 1")
        self.sources = list(sources)
        self.event_oversample = float(event_oversample)
        self.jam_oversample = float(jam_oversample)
        self.offsets, total = [], 0
        for s in self.sources:
            self.offsets.append(total)
            total += len(s.dataset)
        self._n = total
        norm = sum(s.share for s in self.sources)
        self.shares = [s.share / norm for s in self.sources]
        self.receipt = {"mode": "mixed", "n_records": total,
                        "sources": [{"name": s.name, "n": len(s.dataset), "share": sh,
                                     "receipt": getattr(s.dataset, "receipt", {})}
                                    for s, sh in zip(self.sources, self.shares)],
                        "event_oversample": self.event_oversample}
        if self.jam_oversample > 1.0:
            self.receipt["jam_oversample"] = self.jam_oversample

    def __len__(self) -> int:
        return self._n

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(index)
        for k in range(len(self.sources) - 1, -1, -1):
            if index >= self.offsets[k]:
                return k, index - self.offsets[k]
        raise IndexError(index)

    def __getitem__(self, index: int) -> dict:
        k, local = self.locate(index)
        rec = dict(self.sources[k].dataset[local])
        if "action_timeline" not in rec:
            horizon = int(rec["states"].shape[0])
            n_control = max(1, int(np.ceil((horizon - 1) * 0.5)))
            fixed = ActionTimeline.constant("native", n_control, rec["states"].shape[1])
            rec["action_timeline"] = {
                "controller": fixed.controller, "acceleration_mps2": fixed.acceleration_mps2,
                "target_speed_mps": fixed.target_speed_mps,
                "control_dt_s": fixed.control_dt_s}
            rec["action_timeline_synthetic"] = True
        rec["mix_source"] = self.sources[k].name
        return rec

    def stage_event_mask(self, source: int, stage_s: float) -> np.ndarray:
        s = self.sources[source]
        if s.events is None:
            return np.zeros(len(s.dataset), bool)
        steps = max(1, int(math.ceil(float(stage_s))))
        return (s.events < steps).any(axis=1)

    def stage_weights(self, stage_s: float) -> list[float]:
        out: list[float] = []
        for k, s in enumerate(self.sources):
            w = np.where(self.stage_event_mask(k, stage_s), self.event_oversample, 1.0)
            if self.jam_oversample > 1.0 and s.jam is not None:
                w = w * np.where(s.jam, self.jam_oversample, 1.0)
            out.extend((w / w.sum() * self.shares[k]).tolist())
        return out

    def describe_stages(self, stages_s: Sequence[float]) -> list[dict]:
        """Expected per-stage draw statistics (receipt / READY evidence)."""
        rows = []
        for stage in stages_s:
            row = {"stage_s": float(stage), "sources": {}}
            for k, s in enumerate(self.sources):
                mask = self.stage_event_mask(k, stage)
                n_ev = int(mask.sum())
                n = len(mask)
                raw = n_ev / n
                eff = (n_ev * self.event_oversample) / (n_ev * self.event_oversample
                                                         + (n - n_ev))
                entry = {
                    "share": round(self.shares[k], 4), "n_items": n,
                    "n_event_items": n_ev, "event_fraction_uniform": round(raw, 4),
                    "event_fraction_sampled": round(eff, 4)}
                if s.jam is not None:
                    n_jam = int(s.jam.sum())
                    entry["n_jam_items"] = n_jam
                    entry["jam_fraction_uniform"] = round(n_jam / max(1, n), 4)
                    # INFRAV5: expected jam share of this source's draws at this stage
                    # without / with the jam factor (event weights included).
                    w_event = np.where(mask, self.event_oversample, 1.0)
                    w_jam = w_event * np.where(s.jam, self.jam_oversample, 1.0)
                    entry["jam_fraction_sampled_before"] = round(
                        float(w_event[s.jam].sum() / w_event.sum()), 4)
                    entry["jam_fraction_sampled"] = round(
                        float(w_jam[s.jam].sum() / w_jam.sum()), 4)
                row["sources"][s.name] = entry
            if all(s.jam is not None for s in self.sources):
                # INFRAV5: share of ALL ordinary draws that are jam windows (paired batches,
                # drawn separately, are not part of this sampler).
                srcs = row["sources"]
                row["jam_share_total"] = {
                    "before": round(sum(self.shares[k] * srcs[s.name]
                                        ["jam_fraction_sampled_before"]
                                        for k, s in enumerate(self.sources)), 4),
                    "after": round(sum(self.shares[k] * srcs[s.name]["jam_fraction_sampled"]
                                       for k, s in enumerate(self.sources)), 4),
                    "jam_oversample": self.jam_oversample}
            rows.append(row)
        return rows


def write_world_split(window_list: str | Path, out_dir: str | Path, *, frac: float,
                      seed: int) -> tuple[str, str]:
    """TRAIN-scoped world split of a window list -> ``(train_list, val_list)`` paths.

    Same split as :func:`wms.data.val_split.split_document`, but written under names
    that do not contain the forbidden ``HOLDOUT`` token (``validate_train_source``
    refuses ``windows_train_holdout.json``, which breaks ``--val-holdout-frac``).
    """
    from wms.data.val_split import split_document

    doc = json.loads(Path(window_list).read_text(encoding="utf-8"))
    train_doc, val_doc = split_document(doc, frac=float(frac), seed=int(seed))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out / "windows_train_valsplit.json", out / "windows_val_valsplit.json"
    train_path.write_text(json.dumps(train_doc, indent=1, sort_keys=True), encoding="utf-8")
    val_path.write_text(json.dumps(val_doc, indent=1, sort_keys=True), encoding="utf-8")
    return str(train_path), str(val_path)


def build_switch_datasets(*, pack: str, identities: str, window_list: str,
                          index_path: str, heldout_path: str, horizon_steps: int,
                          entries_v2: bool, family_policy: str = "drop",
                          max_drop_frac: float = 0.85, val_holdout_frac: float = 0.1,
                          val_holdout_seed: int = 0, val_window_list: Optional[str] = None,
                           work_dir: str | Path = ".", leave_pi_wall_out: bool = False,
                           history_s=None, history_stride_s=1.0,
                           junction_fix: str = "off", controller_hygiene: str = "off") -> dict:
    """Switching-pack TRAIN (and TRAIN-scoped monitoring) datasets + events.

    Returns ``{"train": TrajectoryDataset, "train_events": [n,2] array,
    "val": TrajectoryDataset | None, "receipt": dict}``.

    ``controller_hygiene = "strict"`` (DATAHYG) drops items whose horizon holds a damping CAV
    (:class:`DirectMixHygiene`); when that removes more than ``HYGIENE_MAX_DROP_FRAC`` of the
    items the source is left out: ``train`` (and ``val``) are ``None`` and the receipt says so.
    """
    if controller_hygiene not in hygiene.HYGIENE_MODES:
        raise MixDataRefusal(f"controller_hygiene must be one of {hygiene.HYGIENE_MODES}")
    guard = (ReleasedGuard() if family_policy == "release"
             else HeldOutGuard.load(heldout_path))
    timelines = SwitchTimelines(pack, index_path)
    guard.refuse_worlds(timelines.worlds, where="the switching pack timeline index")
    book = IdentityBook(identities)
    if not 0 <= float(val_holdout_frac) < 1:
        raise MixDataRefusal("switch validation fraction must be in [0,1)")
    train_list, val_list = str(window_list), val_window_list
    if val_window_list is None and float(val_holdout_frac) > 0.0:
        train_list, val_list = write_world_split(
            window_list, Path(work_dir) / "switch_valsplit",
            frac=float(val_holdout_frac), seed=int(val_holdout_seed))
    if val_list:
        train_worlds = {w["world"] for w in json.loads(Path(train_list).read_text())["windows"]}
        val_worlds = {w["world"] for w in json.loads(Path(val_list).read_text())["windows"]}
        if train_worlds & val_worlds:
            raise MixDataRefusal("switch monitoring worlds overlap training worlds")
    for path in (train_list, val_list):
        if path:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            if doc.get("split", "TRAIN") != "TRAIN":
                raise MixDataRefusal("switch monitoring/training lists must be TRAIN-scoped")
            guard.refuse_worlds([w["world"] for w in doc.get("windows", [])],
                                where=f"window list {path}")

    def one(listfile: str):
        flt = SwitchItemFilter(timelines, guard, book, policy=family_policy,
                               max_drop_frac=max_drop_frac, leave_pi_wall_out=leave_pi_wall_out)
        hyg = (DirectMixHygiene(timelines, book, horizon_steps=int(horizon_steps))
               if controller_hygiene == "strict" else None)

        def filtered(corpus, positions):
            packed_guard = None
            if leave_pi_wall_out:
                positions, packed_guard = ControllerExclusion(horizon_steps=horizon_steps,
                    heldout=guard, leave_pi_wall_out=True)(corpus, positions)
            kept, receipt = flt(corpus, positions)
            receipt["packed_controller_guard"] = packed_guard
            if hyg is not None:
                kept, receipt["controller_hygiene"] = hyg(corpus, kept)
            return kept, receipt
        if hyg is None:
            return one_dataset(listfile, filtered, flt)
        try:
            return one_dataset(listfile, filtered, flt)
        except HygieneSourceExhausted as exc:
            return None, None, {**flt.receipt, "controller_hygiene": exc.receipt}

    def one_dataset(listfile, filtered, flt):
        ds = TrajectoryDataset.from_real_packs(
            [pack], identity_dirs=[identities], horizon_steps=int(horizon_steps),
            start_shift_prob=0.0, window_list=listfile, entries_v2=bool(entries_v2),
            timelines={0: timelines}, item_filter=filtered,
            # The frozen ``train.identity_sidecar`` loader rejects the per-item
            # (18 busy ticks per run) switching sidecar; ``""`` disables it for
            # THIS corpus only.  The trainer still reads + hash-verifies the
            # sidecar itself (``wms.data.sidecars.read_identity_rows``) for ``vehicle_ids``.
            identity_env_dir="", history_s=history_s, history_stride_s=history_stride_s,
            **({} if junction_fix == "off" else {"junction_fix": junction_fix}))
        ev = np.asarray([flt.events[int(gpos)] for _p, gpos in ds.positions], np.int64)
        return ds, ev, flt.receipt

    train, train_events, train_receipt = one(train_list)
    val = val_receipt = None
    if val_list:
        val, _ev, val_receipt = one(val_list)
    if train is None:
        val = None                       # strict hygiene left the whole source out
    receipt = {"pack": str(pack), "identities": str(identities),
               "timeline_index": str(index_path), "train_window_list": train_list,
               "val_window_list": val_list, "heldout": train_receipt,
               "heldout_val": val_receipt, "start_shift_prob": 0.0,
               "n_timeline_items": len(timelines)}
    if controller_hygiene != "off":
        receipt["controller_hygiene"] = {
            "mode": controller_hygiene, "source_disabled": train is None,
            "train": train_receipt.get("controller_hygiene"),
            "val": None if val_receipt is None else val_receipt.get("controller_hygiene")}
    return {"train": train, "train_events": train_events, "val": val, "receipt": receipt}


def mix_datasets(orig: Dataset, switch: dict, *, switch_ratio: float,
                 event_oversample: float, orig_val: Optional[Dataset] = None,
                 jam_oversample: float = 1.0, jam_speed_mps: float = 5.0,
                 jam_cache_dir: Optional[str | Path] = None,
                 jam_min_fraction: float = 0.0) -> tuple:
    """``(train, val)`` mixed datasets from the original and switching datasets.

    ``jam_oversample > 1`` marks each source's jam windows (recorded HDV speed
    below ``jam_speed_mps``) and reweights them, composing with the source mix
    ratio and the switch/direct event oversampling.  ``1`` is the OFF path.
    """
    if not 0.0 < float(switch_ratio) < 1.0:
        raise MixDataRefusal("--switch-mix-ratio must be in (0, 1)")
    orig_jam = switch_jam = None
    jam_block = None
    if float(jam_oversample) > 1.0:
        cache = None if jam_cache_dir is None else Path(jam_cache_dir)
        orig_jam = jamdata.compute_jam_flags(
            orig, speed_mps=float(jam_speed_mps),
            cache_path=None if cache is None else cache / "jam_original.json",
            source_key=jamdata.dataset_source_key(orig), min_fraction=jam_min_fraction)
        switch_jam = jamdata.compute_jam_flags(
            switch["train"], speed_mps=float(jam_speed_mps),
            cache_path=None if cache is None else cache / "jam_switching.json",
            source_key=jamdata.dataset_source_key(switch["train"]),
            min_fraction=jam_min_fraction)
        jam_block = {
            "schema": jamdata.JAM_SCHEMA, "oversample": float(jam_oversample),
            "speed_mps": float(jam_speed_mps),
            "sources": {
                "original": jamdata.jam_receipt({**orig_jam, "oversample":
                                                 float(jam_oversample)}),
                "switching": jamdata.jam_receipt({**switch_jam, "oversample":
                                                  float(jam_oversample)})},
            "n_windows": int(len(orig) + len(switch["train"])),
            "n_jam": int(orig_jam["n_jam"] + switch_jam["n_jam"])}
    train = MixedTrajectoryDataset(
        [MixSource("original", orig, share=1.0 - float(switch_ratio),
                   jam=None if orig_jam is None else orig_jam["flags"]),
         MixSource("switching", switch["train"], share=float(switch_ratio),
                   events=switch["train_events"],
                   jam=None if switch_jam is None else switch_jam["flags"])],
        event_oversample=event_oversample,
        jam_oversample=float(jam_oversample))
    val_sources = []
    if orig_val is not None:
        val_sources.append(MixSource("original_val", orig_val, share=1.0))
    if switch["val"] is not None:
        val_sources.append(MixSource("switching_val", switch["val"], share=1.0))
    val = MixedTrajectoryDataset(val_sources) if val_sources else None
    train.receipt["switching"] = switch["receipt"]
    if jam_block is not None:
        train.receipt["jam"] = jam_block
    return train, val


# --------------------------------------------------------------- preflight ---
def preflight(config: dict, sw_root: str | Path, *, full_scan: bool = False,
              max_drop_frac: float = 0.85) -> dict:
    """Data-side launch checks for the switching root; raises on any failure."""
    if config.get("schema") != DATA_CONFIG_SCHEMA:
        raise MixDataRefusal(f"data config schema must be {DATA_CONFIG_SCHEMA}")
    root = Path(sw_root)
    sw = config["switching"]
    checks: dict = {}
    for rel in sw["required_train_files"]:
        p = root / rel
        if not p.exists():
            raise MixDataRefusal(f"missing required switching input: {p}")
        validate_train_source(p)
    checks["required_files"] = len(sw["required_train_files"])
    pack = root / sw["train_pack"]
    for name in TIMELINE_ARRAYS:
        if not (pack / f"{name}.npy").is_file():
            raise MixDataRefusal(f"missing timeline sidecar {(pack / name)}.npy")
    index_path = root / sw["train_timeline_index"]
    timelines = SwitchTimelines(pack, index_path)
    if len(timelines) != int(sw["items"]["train"]):
        raise MixDataRefusal(f"timeline items {len(timelines)} != expected {sw['items']['train']}")
    if timelines.k_max != int(sw["k_max"]) or timelines.control_steps != int(sw["control_steps"]):
        raise MixDataRefusal("timeline k_max/control_steps differ from the data config")
    checks["timeline_items"] = len(timelines)
    guard = HeldOutGuard.load(root / sw["heldout"])
    guard.refuse_worlds(timelines.worlds, where="switching TRAIN timeline index")
    checks["heldout_worlds"] = len(guard.held_worlds)
    checks["heldout_family"] = guard.held_family
    wl = json.loads((root / sw["train_window_list"]).read_text(encoding="utf-8"))
    guard.refuse_worlds([w["world"] for w in wl["windows"]], where="switching window list")
    if int(wl["n_windows"]) != int(sw["runs"]["train"]):
        raise MixDataRefusal(f"window list n_windows {wl['n_windows']} != {sw['runs']['train']}")
    checks["train_windows"] = int(wl["n_windows"])
    book = IdentityBook(root / sw["train_identities"])
    census = scan_held_family(timelines, book, guard)
    if census["drop_fraction_if_policy_drop"] > max_drop_frac:
        raise MixDataRefusal(f"held-family census {census} exceeds drop cap {max_drop_frac}")
    checks["held_family_census"] = census
    if full_scan:
        checks["full_scan"] = timelines.full_scan()
    return {"status": "passed", "root": str(root), "checks": checks}


def check_receipt(path: str | Path) -> dict:
    """Assert a ``TRAIN_RECEIPT.json`` records a real two-source mix + the guard."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    data = doc.get("data") or {}
    if data.get("mode") != "mixed":
        raise MixDataRefusal("receipt data.mode is not 'mixed'")
    sources = {s["name"]: s for s in data.get("sources", [])}
    if not {"original", "switching"} <= set(sources) or any(
            int(s["n"]) < 1 for s in sources.values()):
        raise MixDataRefusal(f"receipt lacks both non-empty sources: {sorted(sources)}")
    switching = data.get("switching") or {}
    guard = switching.get("heldout") or {}
    released = guard.get("policy") == "release"
    if (not released and int(guard.get("n_held_worlds_checked", 0)) < 1)             or int(guard.get("n_kept", 0)) < 1:
        raise MixDataRefusal("receipt lacks the held-out guard census")
    if not data.get("stage_sampling"):
        raise MixDataRefusal("receipt lacks per-stage sampling statistics")
    return {"status": "passed", "mode": "mixed",
            "n_records": int(data["n_records"]),
            "sources": {k: {"n": int(v["n"]), "share": v["share"]} for k, v in sources.items()},
            "heldout": {k: guard[k] for k in ("policy", "n_dropped", "drop_fraction",
                                              "n_kept") if k in guard},
            "cav_drive": (doc.get("config") or {}).get("cav_drive"),
            "event_oversample": data.get("event_oversample")}


MixedDataset = MixedTrajectoryDataset  # architecture name (docs/ARCHITECTURE_V4.md 2.3)
