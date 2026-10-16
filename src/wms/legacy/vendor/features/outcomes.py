# ruff: noqa
# VENDORED from output/wd01/wm_n1/features/outcomes.py (last changed in commit ed9c656; copied at baseline bf7d4cb).
# Copied byte-for-byte apart from this header.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""Outcome targets REUSED read-only from the v0.4a label parquets (item 6).

Key join: one row per ``(world, source, t, tau)`` in
``<RD>/output/wd01_partB_v0data_v04a/worlds/<chunk>/<world>__<source>_labels
.parquet``, where ``source`` is the resolved v0.4a source suffix and ``t`` is
a decision time, ``tau`` in {30, 60, 120, 300, 570}.  The WM-N1 featurizer
keeps decision times that are on the 30 s stride (``t - t0`` a multiple of
30) and joins them to this table; the label window is the v0.4a one (base t0
row uses the branch interval convention), which the join preserves verbatim.

Resolving the source (WM-N1 labelfix)
-------------------------------------
A pack item's ``arm`` is NOT always the v0.4a source file: for the
chunk-2/chunk-3 CAMPAIGN per-CAV switching runs the pack carries the run-dir
arm ``native``/``pi`` with policy ``switching``, while the v0.4a producer
writes those labels as ``<world>__swnative`` / ``<world>__swpi`` and keeps
``<world>__native`` / ``<world>__pi`` for the separate J3 TRUE fleet arms
(``partB_v0data/v04a_sources.py:168-182``; file naming
``partB_v0data/build_v04a.py:250,273``).  :func:`label_source` resolves the
item ``(batch, chunk, arm, policy)`` to the v0.4a source suffix; a ``None``
result means no v0.4a labels (zero-label, as today).

Source registry (cited from ``partB_v0data/v04a_sources.py``)::

    chunk4            base, native_base, native, pi, fixed1, fixed2,
                      switching1..3                      (:147-164)
    chunk2/chunk3     base (:165-167);
                      truearms native, pi (label native_fleet / pi_fleet,
                      kind native_fleet / pi_fleet)       (:168-177);
                      campaign swnative, swpi (label switching_mix, kind
                      switching; run dirs __native / __pi) (:178-182)
    chunk1/fix27      base (:165-167); swnative, swpi       (:178-182)

:data:`EXPECTED_LABEL_IDENTITY` maps every resolved source to the frozen
v0.4a label ``policy`` class it must carry (``v04a_sources.py:148-182``).

The parquet is copied as-is into the shard (``outcomes.parquet``); no target
is recomputed here, so sign conventions stay the frozen v0.4a ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

V04A_ROOT = Path("output/wd01_partB_v0data_v04a")
# WM-N1 label coverage build: fresh root holding
# the label parquets that the frozen v0.4a root lacks (M/N classes).  The
# frozen root is always consulted FIRST and is never written.
V04A_COV_ROOT = Path(
    "output/wd01_partB_v0data_v04a_cov_20261004")
LABELS_SUFFIX = "_labels.parquet"
TARGET_COLUMNS = (
    "time_on_section_s", "free_flow_ref_s", "traversed_m",
    "wait_mainline_s", "wait_ramp_s", "total_delay_ref_s",
    "ttc_lt3_veh_s", "hard_brake_s", "collision_any", "collision_n_veh",
    "teleports", "n_queued_at_horizon", "n_vehicles_present",
    "n_unmatched_insertions", "tau_available")
KEY_COLUMNS = ("world", "source", "chunk_id", "split", "t0_s", "policy",
               "tau", "t", "role", "aux", "arm_kind")

#: Resolved v0.4a source suffix -> expected label ``policy`` class
#: (``partB_v0data/v04a_sources.py:147-182``).  A loaded label row whose
#: ``policy`` differs is a mis-bound source and must fail closed.
EXPECTED_LABEL_IDENTITY = {
    "base": "damping_fleet",
    "native_base": "native_fleet",
    "native": "native_fleet",
    "pi": "pi_fleet",
    "fixed1": "fixed_mix",
    "fixed2": "fixed_mix",
    "switching1": "switching_mix",
    "switching2": "switching_mix",
    "switching3": "switching_mix",
    "swnative": "switching_mix",
    "swpi": "switching_mix",
    "w-lo-early": "wall",
    "w-lo-late": "wall",
    "w-mid-early": "wall",
    "w-mid-late": "wall",
}

# chunk-4 branch arms stored as their own source name (v04a_sources.py:154-164).
_CHUNK4_ARMS = ("native_base", "native", "pi", "fixed1", "fixed2",
                "switching1", "switching2", "switching3")
# walls-pilot branch arms stored as their own source name (WM-N1 label
# coverage 2026-10-04; v04a_sources.WALL_ARMS).  The pack records them under
# batch ``walls_pilot_chunk4`` / chunk ``chunk4`` with policy ``wall``.
_CHUNK4_WALL_ARMS = ("w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")
# Campaign fixed / switching branch arm names (WM-N1 label coverage
# 2026-10-04; v04a_sources.FIXED_ARMS / SWITCH_ARMS): their raw runs are
# ``runs/campaign_stage1_chunk*/<world>__<arm>`` and the v0.4a producer
# names the label file exactly ``<world>__<arm>_labels.parquet``.
_COV_BRANCH_ARMS = ("fixed1", "fixed2",
                    "switching1", "switching2", "switching3")
# Batches whose campaign ``__native``/``__pi`` runs are the switching arms
# (v04a_sources.py:178-182).
_CAMPAIGN_SWITCH_BATCHES = ("chunk1", "chunk1_fix27", "chunk2", "chunk3")
_TRUEARM_BATCHES = ("truearms_chunk2", "truearms_chunk3")


class OutcomeSourceError(ValueError):
    """The matching v0.4a label source is missing or malformed."""


def label_source(batch, chunk, arm, policy) -> str | None:
    """Resolve a pack item to its v0.4a label source suffix (or ``None``).

    Derived verbatim from the v0.4a source registry
    (``partB_v0data/v04a_sources.py:145-182``):

    * ``arm == 'base'`` -> ``'base'`` (every chunk; :148-149, 166-167);
    * ``chunk == 'chunk4'`` -> the arm as-is for the chunk-4 branch arms
      ``native_base / native / pi / fixed1 / fixed2 / switching1..3``
      (:151-164) and for the walls-pilot arms ``w-lo-early / w-lo-late /
      w-mid-early / w-mid-late`` (WM-N1 label coverage 2026-10-04), else
      ``None``;
    * ``batch in truearms_chunk2/3`` with arm ``native``/``pi`` -> the arm
      (the J3 TRUE fleet arms; :168-177);
    * ``batch in chunk1 / chunk1_fix27 / chunk2 / chunk3`` with arm
      ``native``/``pi`` and policy ``switching`` -> ``swnative``/``swpi``
      (the campaign per-CAV switching arms; :178-182);
    * ``batch in chunk1 / chunk1_fix27 / chunk2 / chunk3`` with a campaign
      fixed / switching branch arm ``fixed1 / fixed2 / switching1..3``
      -> the arm as-is (WM-N1 label coverage 2026-10-04);
    * anything else -> ``None``: no v0.4a labels, zero-label.
    """
    if arm is None:
        return None
    arm = str(arm)
    batch = None if batch is None else str(batch)
    chunk = None if chunk is None else str(chunk)
    policy = None if policy is None else str(policy)
    if arm == "base":
        return "base"
    if chunk == "chunk4":
        return arm if arm in _CHUNK4_ARMS or arm in _CHUNK4_WALL_ARMS \
            else None
    if batch in _TRUEARM_BATCHES and arm in ("native", "pi"):
        return arm
    if batch in _CAMPAIGN_SWITCH_BATCHES:
        if arm in ("native", "pi") and policy == "switching":
            return "swnative" if arm == "native" else "swpi"
        if arm in _COV_BRANCH_ARMS:
            return arm
    return None


def labels_path(chunk: str, world: str, source: str,
                root: Path | str | None = None) -> Path:
    """Path of the v0.4a label parquet for a RESOLVED ``source``."""
    base = Path(root) if root else V04A_ROOT
    return base / "worlds" / str(chunk) / f"{world}__{source}{LABELS_SUFFIX}"


def resolve_labels_file(chunk: str, world: str, source: str, *,
                        frozen_root: Path | str | None = None,
                        cov_root: Path | str | None = None) -> Path | None:
    """First EXISTING label parquet for a RESOLVED ``source``.

    The frozen v0.4a root wins; the WM-N1 label-coverage root is consulted
    only for sources the frozen root lacks.  ``None`` = present in neither
    (the caller must not guess).
    """
    for root in (frozen_root if frozen_root is not None else V04A_ROOT,
                 cov_root if cov_root is not None else V04A_COV_ROOT):
        p = labels_path(chunk, world, source, root)
        if p.is_file():
            return p
    return None


def available(chunk: str, world: str, source: str,
              root: Path | str | None = None) -> bool:
    return labels_path(chunk, world, source, root).is_file()


def _check_identity(df, source: str, item: dict) -> None:
    """Fail-closed identity check of the selected rows against one item.

    Every row must satisfy ``world == item.world``, ``source == resolved
    source``, ``chunk_id == item.chunk``, ``policy ==
    EXPECTED_LABEL_IDENTITY[source]``, ``t0_s == item.t0_s`` (within 1e-6) and
    ``split == item.split`` when the item records a split.  Any mismatch
    raises :class:`OutcomeSourceError`; the caller never falls back.
    """
    if not isinstance(item, dict):
        raise OutcomeSourceError("identity check requires an item mapping")
    expected_policy = EXPECTED_LABEL_IDENTITY.get(source)
    if expected_policy is None:
        raise OutcomeSourceError(f"unknown resolved label source {source!r}")
    for key in ("world", "chunk", "t0_s"):
        if key not in item:
            raise OutcomeSourceError(f"item lacks {key!r} for identity check")
    expected = {
        "world": str(item["world"]),
        "source": str(source),
        "chunk_id": str(item["chunk"]),
        "policy": expected_policy,
    }
    split = item.get("split")
    if split is not None:
        expected["split"] = str(split)
    for col, want in expected.items():
        if col not in df.columns:
            raise OutcomeSourceError(
                f"label parquet for {source!r} lacks column {col!r}")
        if len(df) and not (df[col].astype(str) == want).all():
            got = sorted(set(df[col].astype(str)))
            raise OutcomeSourceError(
                f"label identity mismatch for {source!r}: column {col!r} "
                f"expected {want!r}, observed {got!r}")
    if len(df):
        t0 = float(item["t0_s"])
        t0s = df["t0_s"].astype(float).to_numpy()
        if not np.allclose(t0s, t0, rtol=0.0, atol=1e-6):
            raise OutcomeSourceError(
                f"label identity mismatch for {source!r}: t0_s expected "
                f"{t0!r}, observed {sorted(set(t0s.tolist()))!r}")


def load_outcome_rows(chunk: str, world: str, source: str, decision_ts,
                      taus, root: Path | str | None = None, *,
                      item: dict | None = None):
    """Read-only reuse of the matching label rows for a RESOLVED ``source``.

    Returns a pandas DataFrame restricted to the requested decision times
    and taus, preserving the frozen v0.4a column values.  When ``item`` is
    supplied the loaded rows are hard-checked against that item
    (:func:`_check_identity`); any mismatch raises :class:`OutcomeSourceError`
    (never a silent fall-back).  Raises ``OutcomeSourceError`` if the labels
    file is absent.
    """
    import pandas as pd

    path = labels_path(chunk, world, source, root)
    if not path.is_file():
        raise OutcomeSourceError(
            f"no v0.4a labels for {world}__{source} at {path}")
    df = pd.read_parquet(path)
    want_t = {round(float(t), 6) for t in np.asarray(decision_ts).ravel()}
    want_tau = {round(float(x), 6) for x in np.asarray(taus).ravel()}
    keep = df["t"].round(6).isin(want_t) & df["tau"].round(6).isin(want_tau)
    out = df.loc[keep].copy()
    out = out.sort_values(["t", "tau"]).reset_index(drop=True)
    if item is not None:
        _check_identity(out, source, item)
    return out
