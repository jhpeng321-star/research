"""TRAIN-only fixed-best V2 for ``--dq-arms e2`` (registration EVALSLICE E2).

The historical ``DECISION_QUALITY_FIXED_BEST.json`` ranked only the registered
arms (``native``/``pi``) from a single TRAIN sidecar.  This builder produces the
same ``wm_n1_decision_quality_fixed_best/1`` schema but ranks the E2 *arm types*
(``native``, ``pi``, ``fixed_mix``, ``switching_mix``, ``wall``, ``trajectory``,
``damping``) from TRAIN labels only, so a ``--dq-arms e2`` scoring pass can use
``fixed_best_arm`` against the same arm names it scores.

Inputs (all TRAIN, every byte SHA-256 pinned before any read):

* ``labels_cov_20261004T072826Z/base_train_outcomes.parquet`` (native/pi/damping
  and the corrected switching overlay);
* ``labels_cov_20261004T072826Z/overlay_outcomes.parquet`` (fixed/switching/wall
  coverage arms);
* ``labels_overlay_dspeed/overlay_outcomes.parquet`` (trajectory arms).

Rows enter a cell exactly like the arm-contrast/climatology path: ``split ==
TRAIN``, ``tau_available > 0``, ``t == t0_s`` and a registered tau; the
corrected (``pre_fix == False``) row wins a duplicate ``(world, source, tau)``.
The arm type comes from the raw ``source`` (member token) via
:func:`evalh.scorer.arm_contrasts.arm_type_of`.  No CAL, HOLDOUT, chunk-5, J4B
or INC1 path is ever touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from evalh.scorer import arm_contrasts as AC
from evalh.scorer import decision_quality as DQ


class FixedBestV2Refusal(RuntimeError):
    """A fixed-best V2 input is absent, unverified, or inconsistent."""


#: Column projection read from every sidecar.
COLUMNS = ("world", "source", "arm", "arm_kind", "policy", "split", "t0_s",
           "t", "tau", "tau_available", "total_delay_ref_s", "hard_brake_s",
           "pre_fix")
#: The subset every sidecar must expose (``arm``/``arm_kind``/``policy`` and
#: ``pre_fix`` are optional; the arm type falls back through the present ones).
REQUIRED = ("world", "source", "split", "t0_s", "t", "tau", "tau_available",
            "total_delay_ref_s", "hard_brake_s")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pinned(path, expected_sha256, label):
    path = Path(path)
    if not path.is_file():
        raise FixedBestV2Refusal(f"{label} is absent: {path}")
    sha = _sha256_file(path)
    pin = str(expected_sha256 or "").strip().lower()
    if not pin or sha != pin:
        raise FixedBestV2Refusal(
            f"{label} sha256 {sha} != pinned {expected_sha256!r}; refusing "
            "before any label read")
    return path, sha


def _read_arm_rows(path):
    """TRAIN t0 arm rows (corrected duplicate wins), as plain records."""
    import pandas as pd

    frame = pd.read_parquet(path)
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise FixedBestV2Refusal(f"{path} lacks required columns {missing}")
    frame = frame[frame["split"].astype(str).str.strip().str.upper() == "TRAIN"]
    chosen: dict = {}
    for rec in frame.to_dict("records"):
        if float(rec.get("tau_available", 0.0) or 0.0) <= 0.0:
            continue
        try:
            t0 = round(float(rec["t0_s"]), 6)
            t = round(float(rec["t"]), 6)
            tau = round(float(rec["tau"]), 6)
        except (TypeError, ValueError):
            continue
        if t != t0 or tau not in {round(float(x), 6) for x in AC.TAUS_S}:
            continue
        source = str(rec.get("source", "") or "")
        if not source:
            continue
        key = (str(rec["world"]), t0, source, tau)
        fixed = not bool(rec.get("pre_fix", False))
        prior = chosen.get(key)
        if prior is None or (fixed and not prior[0]):
            chosen[key] = (fixed, rec)
    return [rec for _fixed, rec in chosen.values()]


def _arm_type(rec) -> str:
    for name in ("source", "arm", "arm_kind", "policy"):
        value = rec.get(name)
        if value is not None and str(value).strip():
            return AC.arm_type_of(value)
    return ""


def _decision_points(rows, *, costs=DQ.COSTS, taus=AC.TAUS_S):
    """Label-only decision points over E2 arm types (>=2 types per window)."""
    from evalh.scorer.decision_quality import (DecisionArm, DecisionPoint)

    raw_attr = {"delay_s": "total_delay_ref_s", "hard_brake_s": "hard_brake_s"}
    tau_set = {round(float(x), 6) for x in taus}
    points = []
    for cost in costs:
        attr = raw_attr[cost]
        groups: dict = {}
        for rec in rows:
            arm = _arm_type(rec)
            if arm not in AC.E2_ARM_TYPES:
                continue
            tau = round(float(rec["tau"]), 6)
            if tau not in tau_set:
                continue
            t0 = round(float(rec["t0_s"]), 6)
            dec = round(float(rec.get("t", t0) or t0), 6)
            try:
                realized = float(rec[attr])
            except (KeyError, TypeError, ValueError):
                continue
            if not (realized == realized):        # NaN guard
                continue
            key = (str(rec["world"]), t0, dec, tau)
            groups.setdefault(key, {})[arm] = realized
        for (world, t0, dec, tau), by_arm in sorted(groups.items()):
            if len(by_arm) < 2:
                continue
            points.append(DecisionPoint(
                world=world, t0_s=t0, decision_t_s=dec, tau_s=tau, cost=cost,
                arms=tuple(DecisionArm(arm=a, realized=by_arm[a])
                           for a in sorted(by_arm))))
    return points


def build(paths, *, shas, costs=DQ.COSTS, taus=AC.TAUS_S) -> dict:
    pinned, info = [], []
    for path, sha in zip(paths, shas):
        p, s = _pinned(path, sha, f"TRAIN sidecar {Path(path).name}")
        pinned.append(p)
        info.append(dict(path=str(p), sha256=s, filename=p.name))
    all_rows = []
    for p in pinned:
        all_rows.extend(_read_arm_rows(p))
    points = _decision_points(all_rows, costs=costs, taus=taus)
    if not points:
        raise FixedBestV2Refusal(
            "no TRAIN decision points survived the fail-closed filters")
    doc = DQ.build_fixed_best(
        points, costs=costs, taus=taus,
        train_source=dict(split="TRAIN", inputs=info,
                          n_rows_in=len(all_rows),
                          n_rows_t0=len(all_rows),
                          arm_types=list(AC.E2_ARM_TYPES)))
    return doc


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m evalh.scorer.build_fixed_best_v2",
        description="Build the TRAIN-only e2 fixed-best V2 from pinned TRAIN "
                    "labels (native/pi/fixed_mix/switching_mix/wall/trajectory/"
                    "damping).")
    p.add_argument("--base", required=True)
    p.add_argument("--base-sha256", required=True)
    p.add_argument("--coverage", required=True)
    p.add_argument("--coverage-sha256", required=True)
    p.add_argument("--trajectory", required=True)
    p.add_argument("--trajectory-sha256", required=True)
    p.add_argument("--out", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        doc = build([args.base, args.coverage, args.trajectory],
                    shas=[args.base_sha256, args.coverage_sha256,
                          args.trajectory_sha256])
        sha = DQ.write_fixed_best(doc, args.out)
    except (FixedBestV2Refusal, DQ.DecisionQualityRefusal) as exc:
        print(f"[wm-n1 fixed-best v2 refused] {exc}", file=sys.stderr)
        return 87
    except Exception as exc:                              # noqa: BLE001
        print(f"[wm-n1 fixed-best v2 refused] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 87
    print(json.dumps({"schema": DQ.SCHEMA_FIXED_BEST, "out": str(args.out),
                      "sha256": sha, "n_cells": sum(
                          len(v) for v in doc["cells"].values())},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
