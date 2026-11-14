"""Sealed extra CAL-tune *feature* pack (``--extra-cal-pack``).

The E4 direct-speed trajectory arms (``arm0``..``arm3``) are featurized into a
separate sealed pack (``dspeed.build_cal_pack``) because their features are not
part of the released A8 CAL shards.  The registered scorer pass may score them
only by naming that pack explicitly; this module is the minimal, fail-closed
reader:

* the pack directory must carry a ``SHA256SUMS``; EVERY listed file is verified
  against its recorded SHA-256 before any array is opened, and no file outside
  the manifest is read;
* the pack's ``manifest.json`` / ``INDEX.json`` must carry the expected schemas
  and a ``sealed == "CAL_TUNE"`` tag;
* only items whose ``(world, arm)`` is on the explicitly supplied CAL-tune world
  list and registered trajectory arm set are exposed; anything else is refused.

The pack holds GLOBAL arrays (one memmap per channel, run offsets in
``INDEX.json`` runs).  :class:`ExtraPack` rewrites each run to LOCAL offsets and
materializes per-run array slices on demand (a derived cache inside the pack
directory) so :func:`evalh.scorer.predictors.build_item_world` reads exactly the
released-source shape.  No outcome/label value is ever read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

PACK_SCHEMA = "wm_n1_pack/1"
INDEX_SCHEMA = "wm_n1_corpus_pack/2"
#: Positions are offset far above any real pack manifest index so extra entries
#: can never collide with the released ``source.selected`` positions.
POSITION_BASE = 1_000_000

#: Channels sliced per run (all others are not consumed by the predictor).
_TICK_ARRAYS = ("veh_step", "veh_mask", "veh_kin", "hdv_base", "cav_plan_ch",
                "counter", "accel_next", "target_mask", "lane_next",
                "wall", "wall_mask")
_DEC_ARRAYS = ("plan",)


class ExtraPackRefusal(PermissionError):
    """The extra feature pack is absent, unverified, or unregistered."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_sha256sums(directory) -> dict:
    manifest = Path(directory) / "SHA256SUMS"
    if not manifest.is_file():
        raise ExtraPackRefusal(f"no SHA256SUMS in {directory}")
    entries: dict = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or len(parts[0]) != 64:
            raise ExtraPackRefusal(f"malformed SHA256SUMS line in {manifest}")
        entries[parts[-1].lstrip("*")] = parts[0].lower()
    if not entries:
        raise ExtraPackRefusal(f"empty SHA256SUMS in {directory}")
    return entries


def verify_pack_manifest(directory, *, expected_sha256=None) -> dict:
    """Verify the pack SHA-256 (optional) and every listed file (fail closed)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ExtraPackRefusal(f"extra pack path is not a directory: {directory}")
    if expected_sha256 is not None:
        pin = str(expected_sha256).strip().lower()
        if len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            raise ExtraPackRefusal("extra pack sha must be 64 lowercase hex")
        sums = directory / "SHA256SUMS"
        if not sums.is_file():
            raise ExtraPackRefusal(f"no SHA256SUMS in {directory}")
        if _sha256_file(sums) != pin:
            raise ExtraPackRefusal(
                f"extra pack SHA256SUMS sha256 {_sha256_file(sums)} != pin {pin}")
    entries = _read_sha256sums(directory)
    for name, sha in entries.items():
        path = directory / name
        if not path.is_file():
            raise ExtraPackRefusal(f"{name} listed in SHA256SUMS is absent")
        if _sha256_file(path) != sha:
            raise ExtraPackRefusal(f"{name} differs from its SHA256SUMS pin")
    return entries


class ExtraPack:
    """Validated sealed feature pack exposing selected trajectory items."""

    def __init__(self, root, *, manifest, index, entries, worlds, arms):
        self.root = Path(root)
        self.manifest = manifest
        self.index = index
        self.worlds = set(worlds)
        self.arms = set(arms)
        self._by_key = {}
        self.selected = []
        self.positions = {}
        self._slice_cache = self.root / "_slices"
        run_map = {str(r.get("key")): r for r in index.get("runs", [])}
        pos = POSITION_BASE
        for item in manifest.get("items", []):
            world, arm = str(item.get("world")), str(item.get("arm"))
            if world not in self.worlds or arm not in self.arms:
                continue
            key = f"{item.get('batch') or item.get('chunk')}/{world}__{arm}"
            run = run_map.get(key)
            if run is None:
                raise ExtraPackRefusal(f"extra pack item {key} lacks an INDEX run")
            local_item = dict(item)
            # ``_array_path`` slices the GLOBAL pack arrays using the run's
            # global offsets (``*_offset``); the build path
            # (``build_item_world``/``build_schedule``/``training_summary_raw``)
            # then indexes the returned per-run slice with the RUN-LOCAL
            # snapshot/decision offsets, exactly like a released per-run shard.
            local_item["_extra_tick_offset"] = int(run["tick_start"])
            local_item["_extra_dec_offset"] = int(run["dec_start"])
            local_item["tick"] = int(item["tick"]) - int(run["tick_start"])
            local_item["plan_row"] = int(item["plan_row"]) - int(run["dec_start"])
            local_item["n_ticks"] = int(run["n_ticks"])
            local_item["_extra_pack"] = True
            local_item["policy"] = str(item.get("policy") or "trajectory")
            local_run = dict(run)
            local_run["tick_start"] = 0
            local_run["tick_end"] = int(run["n_ticks"])
            local_run["dec_start"] = 0
            local_run["dec_end"] = int(run["n_decision"])
            entry = (pos, local_item, local_run)
            self.selected.append(entry)
            self._by_key[(world, arm)] = entry
            self.positions[(world, arm)] = pos
            pos += 1
        if not self.selected:
            raise ExtraPackRefusal(
                "extra pack holds no item on the CAL-tune world/arm list")

    def entry_for(self, world, arm):
        return self._by_key.get((str(world), str(arm)))

    def _array_path(self, item, name):
        """A per-run array file, materialized from the global memmap if absent."""
        if item.get("_extra_pack") is not True:
            raise ExtraPackRefusal("not an extra-pack item")
        stem = name[:-4] if str(name).endswith(".npy") else str(name)
        world, arm = str(item["world"]), str(item["arm"])
        src = self.root / f"{stem}.npy"
        if not src.is_file():
            raise ExtraPackRefusal(f"extra pack lacks {stem}.npy")
        out_dir = self._slice_cache / f"{world}__{arm}"
        out = out_dir / f"{stem}.npy"
        if out.is_file():
            return out
        if stem in _DEC_ARRAYS:
            lo = int(item["_extra_dec_offset"]) + int(item["plan_row"])
            hi = lo + 1
        else:
            lo = int(item["_extra_tick_offset"])
            hi = lo + int(item["n_ticks"])
        out_dir.mkdir(parents=True, exist_ok=True)
        src_arr = np.load(src, mmap_mode="r")
        shape = (hi - lo, *src_arr.shape[1:])
        dst = np.lib.format.open_memmap(out, mode="w+", dtype=src_arr.dtype,
                                        shape=shape)
        dst[...] = np.asarray(src_arr[lo:hi])
        dst.flush()
        del dst, src_arr
        return out

    def checked_file(self, item, name):
        return self._array_path(item, name)


class ExtraPackSource:
    """Released CAL source overlaid with a sealed extra feature pack.

    ``selected`` exposes the released items plus the extra-pack items (with
    offset positions), so :func:`evalh.scorer.predictors._entry` resolves an
    extra row's ``pack_item_position``; ``_checked_file`` routes each item to
    its pack.  ``rows_loader`` must be called on the *released* source
    (``.base``) so the released shard reader never sees an extra item.
    """

    def __init__(self, source, extra: ExtraPack):
        self.base = source
        self.extra = extra
        self.selected = tuple(source.selected) + tuple(extra.selected)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _checked_file(self, item, name):
        if item.get("_extra_pack") is True:
            return self.extra.checked_file(item, name)
        return self.base._checked_file(item, name)


def load_extra_pack(path, *, expected_sha256, worlds, arms) -> ExtraPack:
    """Validate and load a sealed extra feature pack (fail closed)."""
    import json

    directory = Path(path)
    entries = verify_pack_manifest(directory, expected_sha256=expected_sha256)
    for required in ("manifest.json", "INDEX.json"):
        if required not in entries:
            raise ExtraPackRefusal(f"extra pack SHA256SUMS lists no {required}")
    manifest = json.loads((directory / "manifest.json").read_text(
        encoding="utf-8"))
    index = json.loads((directory / "INDEX.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PACK_SCHEMA or index.get("schema") != INDEX_SCHEMA:
        raise ExtraPackRefusal("extra pack manifest/INDEX schema mismatch")
    sealed = str(index.get("sealed") or manifest.get("sealed") or "")
    if sealed.upper().replace("-", "_") != "CAL_TUNE":
        raise ExtraPackRefusal("extra pack is not tagged sealed CAL-tune")
    return ExtraPack(directory, manifest=manifest, index=index, entries=entries,
                     worlds=worlds, arms=arms)


__all__ = ["ExtraPackRefusal", "ExtraPack", "ExtraPackSource",
           "load_extra_pack", "verify_pack_manifest", "POSITION_BASE"]
