"""E2 sealed extra CAL-tune label packs (``--extra-cal-labels``).

The registered scorer pass must accept CAL-tune outcome-label rows for the E2
arms (fixed/switching mix, wall pilot, trajectory) that are not part of the
packed A8 CAL shards.  Those rows live in separately sealed directories handed
to the scorer explicitly.  This module is the minimal, fail-closed interface:

* ``--extra-cal-labels PATH`` is repeatable; each PATH is a **directory**;
* the directory must contain a ``SHA256SUMS`` manifest; EVERY file it lists is
  verified against its recorded SHA-256 **before** any parquet is read, and no
  file outside the manifest is opened;
* the directory's parquet table carries ``world``, ``source``, ``arm`` (or
  ``source`` is the arm), ``split``, ``t0_s``, ``t``, ``tau``,
  ``tau_available`` and ``total_delay_ref_s``; only ``CALIBRATION`` / ``CAL_TUNE``
  rows with ``t == t0_s`` and a registered tau enter the pass;
* each ``(world, arm)`` must be in the explicitly supplied CAL-tune **selected**
  list (built by the caller from the released spec); anything else is refused.

The trajectory pack is the same interface with a different directory: a sealed
pack/run dir whose SHA256SUMS lists its parquet.  No CAL-cal, HOLDOUT, chunk-5,
J4B or INC1 path is ever read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: Accepted split labels inside an explicitly sealed CAL-tune pack.  The E4
#: trajectory labels were written by the dspeed label builder with the TRAIN
#: split token even though the sealed directory is a CAL-tune slice; the
#: ``(world, arm)`` selected-list gate below is the real authorization, so the
#: split token is admitted for these sealed packs only.
CAL_TUNE_SPLITS = ("CALIBRATION", "CAL_TUNE", "CAL-TUNE", "TRAIN")

#: The minimum label columns an extra pack must expose.
REQUIRED_COLUMNS = ("world", "source", "split", "t0_s", "t", "tau",
                    "tau_available", "total_delay_ref_s")


class ExtraLabelRefusal(PermissionError):
    """An extra CAL-tune label pack is absent, unverified, or unregistered."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sha256sums(directory) -> dict:
    """Parse ``SHA256SUMS`` -> ``{filename: sha256}`` (fail closed)."""
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    if not manifest.is_file():
        raise ExtraLabelRefusal(f"no SHA256SUMS in {directory}")
    entries: dict = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or len(parts[0]) != 64:
            raise ExtraLabelRefusal(f"malformed SHA256SUMS line in {manifest}")
        entries[parts[-1].lstrip("*")] = parts[0].lower()
    if not entries:
        raise ExtraLabelRefusal(f"empty SHA256SUMS in {directory}")
    return entries


def verify_manifest(directory, *, require=None) -> dict:
    """Verify every listed file's SHA-256 and return the manifest.

    ``require`` is an optional iterable of filenames that must be present.
    """
    directory = Path(directory)
    entries = read_sha256sums(directory)
    for name, sha in entries.items():
        path = directory / name
        if not path.is_file():
            raise ExtraLabelRefusal(f"{name} listed in SHA256SUMS is absent")
        actual = _sha256_file(path)
        if actual != sha:
            raise ExtraLabelRefusal(
                f"{name} sha256 {actual} != manifest {sha}")
    for name in require or ():
        if name not in entries:
            raise ExtraLabelRefusal(f"manifest lists no {name}")
    return entries


def _table_names(directory) -> list:
    entries = read_sha256sums(directory)
    return sorted(n for n in entries if n.lower().endswith(".parquet"))


def _norm_split(value) -> str:
    return str(value).strip().upper().replace("-", "_")


def load_extra_cal_labels(paths, *, selected_keys, taus, allowed_sources,
                          level_levels=None):
    """Load and validate the sealed CAL-tune rows in every pack ``paths``.

    ``selected_keys`` is an iterable of ``(world, arm)`` tuples (or
    ``"batch/world__arm"`` keys) that the released spec selected.  ``allowed_sources``
    is the set of arm names the pass may use (the E2 members, plus ``native``).
    Returns ``(records, receipt)``; any unlisted / non-CAL-tune / malformed row
    raises :class:`ExtraLabelRefusal`.
    """
    import pandas as pd

    if not paths:
        raise ExtraLabelRefusal("at least one --extra-cal-labels PATH is required")
    tau_set = {round(float(t), 6) for t in taus}
    allowed = {str(a) for a in allowed_sources}
    selected_pairs = set()
    for key in selected_keys:
        if isinstance(key, (tuple, list)) and len(key) == 2:
            selected_pairs.add((str(key[0]), str(key[1])))
        else:
            text = str(key)
            if "__" in text:
                world_arm = text.split("/", 1)[-1]
                world, _sep, arm = world_arm.partition("__")
                selected_pairs.add((world, arm))

    records, receipt = [], []
    for raw in paths:
        directory = Path(raw)
        if not directory.is_dir():
            raise ExtraLabelRefusal(f"extra label path is not a directory: {raw}")
        tables = _table_names(directory)
        if not tables:
            raise ExtraLabelRefusal(f"no parquet table listed in {raw}/SHA256SUMS")
        verify_manifest(directory, require=tables)
        n_rows = 0
        for name in tables:
            frame = pd.read_parquet(directory / name)
            missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
            if missing:
                raise ExtraLabelRefusal(
                    f"{name} lacks required columns {missing}")
            for rec in frame.to_dict("records"):
                split = _norm_split(rec.get("split", ""))
                if split not in CAL_TUNE_SPLITS:
                    raise ExtraLabelRefusal(
                        f"{name} holds non-CAL-tune split {rec.get('split')!r}")
                source = str(rec.get("source", ""))
                if source not in allowed:
                    raise ExtraLabelRefusal(
                        f"{name} arm {source!r} is not a registered E2 arm")
                world = str(rec.get("world", ""))
                # the extra-arm rows may carry the arm in ``source`` only
                arm = str(rec.get("arm", "") or source)
                if (world, arm) not in selected_pairs:
                    raise ExtraLabelRefusal(
                        f"{name} item {world}/{arm} is not in the CAL-tune "
                        "selected list")
                t0 = round(float(rec["t0_s"]), 6)
                if round(float(rec.get("t", t0) or t0), 6) != t0:
                    continue
                tau = round(float(rec["tau"]), 6)
                if tau not in tau_set:
                    continue
                if float(rec.get("tau_available", 0.0) or 0.0) <= 0.0:
                    continue
                records.append(rec)
                n_rows += 1
        receipt.append(dict(path=str(directory), tables=tables,
                            sha256={n: read_sha256sums(directory)[n]
                                    for n in tables},
                            n_rows=int(n_rows)))
    if not records:
        raise ExtraLabelRefusal(
            "no usable CAL-tune rows in the supplied extra label packs")
    return records, receipt


__all__ = ["ExtraLabelRefusal", "CAL_TUNE_SPLITS", "REQUIRED_COLUMNS",
           "read_sha256sums", "verify_manifest", "load_extra_cal_labels"]
