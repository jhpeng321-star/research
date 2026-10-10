"""WM-N1 honest TRAIN climatology for the CAL-tune scorer baselines (E9).

The owner-chosen comparator option (c) scores model A against two fixed,
non-tuned baselines on the registered paired-dV label: (i) ZERO-EFFECT (dV = 0
point mass) and (ii) TRAIN-CLIMATOLOGY, the per-``(contrast, tau)`` empirical
distribution of paired dV taken from TRAIN labels only.  This module builds the
second baseline's artifact; :mod:`evalh.scorer.baselines` consumes it.

Pairing mirrors the scorer's paired-dV construction
(``evalh.metrics.build_delta_pairs`` / ``evalh.scorer.metrics_loop``): the
reference is the all-native arm and the contrasts are ``damping-native`` /
``pi-native`` / ``wall-native`` (``evalh.metrics.CONTRASTS``).  A pair is the
``treatment`` minus ``native`` delay on the SAME ``(world, t0, tau)`` window,
using the TRAIN label row whose decision time ``t`` equals the item ``t0`` (the
``_t0_row_pick`` mapper of ``train/dataset.py:186``) with ``tau_available > 0``
and ``tau`` in the registered set.  Policy names are canonicalised exactly like
the scorer: ``arm_kind.split("_")[0]`` (the v0.4a fleet/schedule action token,
``train/label_time_audit.py:55``) through ``evalh.idea_a.canon_action``,
keeping only ``{native, damping, pi, wall}``.

Owner-registered exclusions (fail closed, never substituted):

* ``train.dataset.OUTCOME_SUPERVISION_EXCLUDED_ARMS`` (``train/dataset.py:124``,
  ``("base", "native_base")``) -- these arms' labels may not enter the retrain's
  outcome supervision, so they never enter the climatology either;
* the switching-native reference exclusion enforced by
  ``train.dataset.is_reference`` (``train/dataset.py:335``): a reference must
  carry policy ``native``, so the per-CAV ``swnative`` / ``swpi`` arms (policy
  ``switching``) are never references.

The output JSON records the exact TRAIN ``(world, arm/source, t0)`` items used
per contrast plus per-cell ``n_pairs`` / ``n_worlds`` and the 19
``evalh.protocol.QUANTILE_LEVELS`` quantiles per ``(contrast, tau)``.  Missing
cells are OMITTED, never filled with zeros: the consuming baseline refuses a
missing cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evalh.metrics import CONTRASTS
from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS

SCHEMA = "wm_n1_train_climatology/1"

TRAIN_OUTCOMES_FILENAME = "train_outcomes.parquet"

#: Pinned SHA-256 of the corrected TRAIN-only label sidecar this artifact is
#: built from (labels_fix1_20261003T141730Z/train_outcomes.parquet).
EXPECTED_TRAIN_OUTCOMES_SHA256 = (
    "1e0ec6de0e2cdfd386fd84165879cac88b21d8cf16277d71108668d2306839ef")

TAUS_S = (30.0, 60.0, 120.0, 300.0, 570.0)

#: Columns read from the sidecar (a projection; the raw table has 35).
_COLUMNS = ("world", "source", "arm_kind", "policy", "split", "t0_s", "t", "tau",
            "tau_available", "total_delay_ref_s")

#: Canonical plan actions that may enter a registered contrast.
_KNOWN = ("native", "damping", "pi", "wall")


class ClimatologyBuildRefusal(RuntimeError):
    """The TRAIN climatology could not be built (fail closed)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _excluded_arms() -> tuple:
    """The registered outcome-supervision exclusions (lazy train import)."""
    from train.dataset import OUTCOME_SUPERVISION_EXCLUDED_ARMS

    return tuple(OUTCOME_SUPERVISION_EXCLUDED_ARMS)


def _is_reference(source: str, canonical: str) -> bool:
    """``train.dataset.is_reference`` on a reconstructed item (lazy import)."""
    from train.dataset import is_reference

    return bool(is_reference({"arm": str(source), "policy": str(canonical)}))


def _canonical_policy(arm_kind) -> str:
    """Canonical plan action for a v0.4a arm kind (fail closed on empty)."""
    from evalh.idea_a import canon_action

    token = str(arm_kind or "").strip().lower().split("_")[0]
    return canon_action(token)


def _require_train(frame) -> None:
    split = frame["split"].astype(str).str.strip().str.upper()
    bad = sorted(set(split[split != "TRAIN"].tolist()))
    if bad:
        raise ClimatologyBuildRefusal(
            f"sidecar holds non-TRAIN split rows ({bad[:5]}); refusing to "
            "build a TRAIN climatology from another split")


def compute_climatology(train_dir,
                        *, expected_sha256=EXPECTED_TRAIN_OUTCOMES_SHA256,
                        taus=TAUS_S, levels=QUANTILE_LEVELS,
                        filename=TRAIN_OUTCOMES_FILENAME) -> dict:
    """Build the per-``(contrast, tau)`` paired-dV TRAIN quantiles.

    ``train_dir`` is the sidecar directory holding ``filename`` (default
    :data:`TRAIN_OUTCOMES_FILENAME`); ``expected_sha256`` pins the file and is
    checked BEFORE the parquet is read (``None`` disables the pin for generated
    test fixtures and is never the CLI default).
    """
    train_dir = Path(train_dir)
    path = train_dir / filename
    if not path.is_file():
        raise ClimatologyBuildRefusal(
            f"no {filename} in {train_dir}")
    sha = _sha256_file(path)
    if expected_sha256 is not None and sha != str(expected_sha256).strip().lower():
        raise ClimatologyBuildRefusal(
            f"{filename} sha256 {sha} != pinned "
            f"{expected_sha256}; refusing before any label read")

    import pandas as pd

    frame = pd.read_parquet(path, columns=list(_COLUMNS))
    _require_train(frame)

    lv = np.asarray(levels, dtype=float)
    if lv.shape != (N_QUANTILES,):
        raise ClimatologyBuildRefusal(
            f"levels must be the {N_QUANTILES} registered quantiles")
    tau_set = {round(float(t), 6) for t in taus}
    excluded_lower = {a.strip().lower() for a in _excluded_arms()}

    n_read = int(len(frame))
    index: dict = {}
    n_excl_arm = n_excl_tau = n_excl_unavail = n_excl_unknown = 0
    for rec in frame.to_dict("records"):
        source = str(rec["source"])
        if source.strip().lower() in excluded_lower:
            n_excl_arm += 1
            continue
        tau = round(float(rec["tau"]), 6)
        if tau not in tau_set:
            n_excl_tau += 1
            continue
        if float(rec["tau_available"]) <= 0.0:
            n_excl_unavail += 1
            continue
        t0 = round(float(rec["t0_s"]), 6)
        # t0 mapper: only the label row at the item's own decision time.
        if round(float(rec["t"]), 6) != t0:
            n_excl_tau += 1
            continue
        canon = _canonical_policy(rec.get("arm_kind"))
        if canon not in _KNOWN:
            n_excl_unknown += 1
            continue
        key = (str(rec["world"]), t0, tau, canon)
        if key in index:
            raise ClimatologyBuildRefusal(
                f"duplicate TRAIN label row for {key} (fail closed)")
        index[key] = (source, float(rec["total_delay_ref_s"]))

    cells: dict = {}
    items_by_contrast: dict = {}
    for name, (pol_a, pol_b) in CONTRASTS.items():
        by_tau: dict = {}
        used: dict = {}
        for (world, t0, tau, canon) in sorted(index):
            if canon != pol_a:
                continue
            ka = (world, t0, tau, pol_a)
            kb = (world, t0, tau, pol_b)
            if ka not in index or kb not in index:
                continue
            src_a, y_a = index[ka]
            src_b, y_b = index[kb]
            if pol_b == "native" and not _is_reference(src_b, pol_b):
                # switching-native arms must never serve as the reference.
                continue
            by_tau.setdefault(tau, []).append((world, float(y_a) - float(y_b)))
            used[(world, src_a, t0)] = "treatment"
            used[(world, src_b, t0)] = "reference"
        if not by_tau:
            continue
        cells[name] = {}
        for tau in sorted(by_tau):
            pairs = by_tau[tau]
            dv = np.asarray([d for _w, d in pairs], dtype=float)
            cells[name][f"{tau:g}"] = dict(
                n_pairs=int(dv.size),
                n_worlds=int(len({w for w, _d in pairs})),
                quantiles=[float(x) for x in np.quantile(dv, lv)])
        items_by_contrast[name] = [
            dict(world=w, source=s, t0_s=float(t0), role=role)
            for (w, s, t0), role in sorted(used.items())]

    doc = dict(
        schema=SCHEMA,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        train_outcomes=dict(path=str(path), sha256=sha, n_rows=n_read),
        input_sha256s={filename: sha},
        taus=[float(t) for t in taus],
        quantile_levels=[float(x) for x in lv],
        reference_policy="native",
        contrasts=sorted(cells),
        exclusions=dict(
            outcome_supervision_excluded_arms=list(_excluded_arms()),
            cite="train/dataset.py:124 OUTCOME_SUPERVISION_EXCLUDED_ARMS",
            switching_reference_cite=(
                "train/dataset.py:335 is_reference (policy 'native' only)")),
        cells=cells,
        items_by_contrast=items_by_contrast,
        counts=dict(
            n_rows_read=n_read,
            n_rows_used=int(len(index)),
            n_excluded_arm=n_excl_arm,
            n_excluded_tau=n_excl_tau,
            n_excluded_tau_unavailable=n_excl_unavail,
            n_excluded_unknown_policy=n_excl_unknown),
    )
    return doc


def run_build(train_dir, out_path, *,
              expected_sha256=EXPECTED_TRAIN_OUTCOMES_SHA256,
              taus=TAUS_S, levels=QUANTILE_LEVELS,
              filename=TRAIN_OUTCOMES_FILENAME) -> dict:
    """Build the climatology and write it (refusing an existing output)."""
    out_path = Path(out_path)
    if out_path.exists():
        raise ClimatologyBuildRefusal(
            f"output {out_path} already exists; refusing to overwrite")
    doc = compute_climatology(train_dir, expected_sha256=expected_sha256,
                              taus=taus, levels=levels, filename=filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    return doc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evalh.scorer.build_climatology",
        description="Build the WM-N1 TRAIN paired-dV climatology for the "
                    "CAL-tune scorer baselines.")
    parser.add_argument("--train-outcomes", required=True,
                        help="directory holding train_outcomes.parquet "
                             "(TRAIN-only corrected label sidecar)")
    parser.add_argument("--out", required=True,
                        help="output JSON path (must not exist)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_build(args.train_outcomes, args.out)
    except ClimatologyBuildRefusal as exc:
        print(f"[wm-n1 climatology refused] {exc}", file=sys.stderr)
        return 87
    except Exception as exc:                             # noqa: BLE001
        print(f"[wm-n1 climatology refused] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 87
    print(json.dumps({"schema": SCHEMA, "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
