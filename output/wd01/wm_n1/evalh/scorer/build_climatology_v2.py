"""E2 TRAIN_CLIMATOLOGY_V2: V1 pi-native plus the arm contrasts.

Builds the registration ``WM-N1-EVALSLICE-20261004`` E2 climatology from
TRAIN labels only:

* the corrected base sidecar ``base_train_outcomes.parquet``
  (``labels_cov_20261004T072826Z``) supplies the ``native`` reference and the
  ``pi`` treatment, so the ``pi-native`` cell is produced by the SAME V1 code
  path and must be byte-identical in value to the frozen
  ``TRAIN_CLIMATOLOGY.json`` (sha256 ``970f5eff...``);
* the same directory's ``overlay_outcomes.parquet`` supplies the coverage
  fixed/switching arms and the wall pilot arms;
* the direct-speed pack ``labels_overlay_dspeed/overlay_outcomes.parquet``
  supplies the trajectory arms ``arm0``..``arm3``.

The arm contrasts (pooled per group plus a per-member contrast) are added by
:mod:`evalh.scorer.arm_contrasts` under the SAME
``wm_n1_train_climatology/1`` schema, so
:class:`evalh.scorer.baselines.TrainClimatologyBaseline` consumes V2 unchanged.

Every input is SHA-256 pinned BEFORE it is read.  No CAL, HOLDOUT, chunk-5,
J4B or INC1 path is ever touched; only the three registered TRAIN sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from evalh.scorer import arm_contrasts as AC
from evalh.scorer import build_climatology as BC

BASE_FILENAME = "base_train_outcomes.parquet"
COVERAGE_FILENAME = "overlay_outcomes.parquet"
TRAJECTORY_FILENAME = "overlay_outcomes.parquet"

#: Columns the arm-cell builder reads from a sidecar (a projection).  ``pre_fix``
#: disambiguates the coverage pack's duplicated pre/post-fix rows.
ARM_COLUMNS = ("world", "source", "split", "t0_s", "t", "tau", "tau_available",
               "total_delay_ref_s", "pre_fix")


class ClimatologyV2Refusal(RuntimeError):
    """The V2 climatology could not be built (fail closed)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pinned_path(path, expected_sha256, label):
    path = Path(path)
    if not path.is_file():
        raise ClimatologyV2Refusal(f"{label} is absent: {path}")
    sha = _sha256_file(path)
    pin = str(expected_sha256 or "").strip().lower()
    if not pin or sha != pin:
        raise ClimatologyV2Refusal(
            f"{label} sha256 {sha} != pinned {expected_sha256!r}; refusing "
            "before any label read")
    return path, sha


def sha256sums_pin(directory, filename) -> str:
    """The SHA-256 a sealed directory's ``SHA256SUMS`` records for ``filename``.

    Used by callers/tests to pin a sidecar without recomputing a known hash.
    Raises when the file is absent from the manifest (fail closed).
    """
    manifest = Path(directory) / "SHA256SUMS"
    if not manifest.is_file():
        raise ClimatologyV2Refusal(f"no SHA256SUMS in {directory}")
    wanted = Path(filename).name.lower()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*").lower() == wanted:
            return parts[0].strip().lower()
    raise ClimatologyV2Refusal(f"SHA256SUMS lists no {filename}")


def _read_rows(path):
    """Read a pinned sidecar's arm-label projection.

    Only the relevant sources (native + a registered arm) are projected; the
    coverage pack may hold a ``pre_fix`` and a corrected row at the same
    ``(world, source, t0, tau)``, and the corrected (``pre_fix == False``) row
    wins.
    """
    import pandas as pd

    wanted = {AC.NATIVE} | set(AC.MEMBER_TO_GROUP)
    frame = pd.read_parquet(path)
    missing = [c for c in ARM_COLUMNS if c not in frame.columns]
    if missing not in ([], ["pre_fix"]):
        raise ClimatologyV2Refusal(
            f"{path} lacks required columns {missing}")
    frame = frame[frame["source"].astype(str).isin(wanted)]
    splits = set(frame["split"].astype(str).str.strip().str.upper())
    if splits - {"TRAIN"}:
        raise ClimatologyV2Refusal(
            f"{path} holds non-TRAIN split rows {sorted(splits)}; refusing")
    chosen: dict = {}
    for rec in frame.to_dict("records"):
        key = (str(rec["world"]), round(float(rec["t0_s"]), 6),
               str(rec["source"]), round(float(rec["tau"]), 6))
        fixed = not bool(rec.get("pre_fix", False))
        prior = chosen.get(key)
        if prior is None or (fixed and not prior[0]):
            chosen[key] = (fixed, rec)
    return [rec for _fixed, rec in chosen.values()]


def build_document(base_path, coverage_path, trajectory_path, *,
                   base_sha256, coverage_sha256, trajectory_sha256,
                   taus=AC.TAUS_S, levels=None) -> dict:
    """Build the merged V2 document (V1 pi-native + E2 arm contrasts)."""
    from evalh.protocol import QUANTILE_LEVELS

    levels = QUANTILE_LEVELS if levels is None else levels
    base_path, base_sha = _pinned_path(
        base_path, base_sha256, "base TRAIN sidecar")
    coverage_path, cov_sha = _pinned_path(
        coverage_path, coverage_sha256, "coverage TRAIN sidecar")
    trajectory_path, traj_sha = _pinned_path(
        trajectory_path, trajectory_sha256, "trajectory TRAIN sidecar")

    # V1 pi-native from the base sidecar, same algorithm/sha as the frozen file.
    doc = BC.compute_climatology(
        base_path.parent, expected_sha256=base_sha, taus=taus, levels=levels,
        filename=base_path.name)

    base_rows = _read_rows(base_path)
    coverage_rows = _read_rows(coverage_path)
    trajectory_rows = _read_rows(trajectory_path)

    treatment_rows = []
    for row in base_rows + coverage_rows + trajectory_rows:
        if str(row.get("source", "")) in AC.MEMBER_TO_GROUP:
            treatment_rows.append(row)
    reference_rows = [r for r in base_rows
                      if str(r.get("source", "")) == AC.NATIVE]

    cells, items = AC.build_arm_cells(
        treatment_rows, reference_rows, taus=taus, levels=levels)
    doc["cells"].update(cells)
    doc["contrasts"] = sorted(doc["cells"])
    doc.setdefault("items_by_contrast", {}).update(items)
    doc["e2"] = AC.e2_metadata(
        input_sha256s={BASE_FILENAME: base_sha,
                       "coverage/" + COVERAGE_FILENAME: cov_sha,
                       "trajectory/" + TRAJECTORY_FILENAME: traj_sha},
        contrasts=tuple(doc["contrasts"]))
    doc["input_sha256s"] = dict(doc.get("input_sha256s", {}),
                                **{BASE_FILENAME: base_sha,
                                   "coverage/" + COVERAGE_FILENAME: cov_sha,
                                   "trajectory/" + TRAJECTORY_FILENAME: traj_sha})
    return doc


def pi_parity(doc, frozen_path) -> dict:
    """Compare V2's ``pi-native`` cell with the frozen V1 document.

    Returns ``{"identical": bool, "frozen": path, "detail": ...}``.  The
    comparison is on the decoded cell value (worlds, pair counts and 19
    quantiles), which is the reproducibility contract the owner asked for.
    """
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    mine = (doc.get("cells") or {}).get("pi-native")
    theirs = (frozen.get("cells") or {}).get("pi-native")
    identical = mine == theirs
    return dict(identical=bool(identical), frozen=str(frozen_path),
                detail=None if identical else {
                    "v2": mine, "frozen": theirs})


def run_build(base_path, coverage_path, trajectory_path, out_path, *,
              base_sha256, coverage_sha256, trajectory_sha256,
              frozen_path=None, taus=AC.TAUS_S, allow_parity_diff=False) -> dict:
    out_path = Path(out_path)
    if out_path.exists():
        raise ClimatologyV2Refusal(
            f"output {out_path} already exists; refusing to overwrite")
    doc = build_document(
        base_path, coverage_path, trajectory_path,
        base_sha256=base_sha256, coverage_sha256=coverage_sha256,
        trajectory_sha256=trajectory_sha256, taus=taus)
    if frozen_path is not None:
        parity = pi_parity(doc, frozen_path)
        doc["pi_native_parity"] = parity
        if not parity["identical"] and not allow_parity_diff:
            raise ClimatologyV2Refusal(
                f"pi-native cell differs from {frozen_path}; refusing to "
                "write a V2 whose V1 cell is not reproduced (pass "
                "--allow-parity-diff to record the difference instead)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    out_path.write_text(text, encoding="utf-8")
    sha = _sha256_file(out_path)
    out_path.with_name(out_path.name + ".sha256").write_text(
        f"{sha}  {out_path.name}\n", encoding="utf-8")
    return doc


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m evalh.scorer.build_climatology_v2",
        description="Build the E2 TRAIN_CLIMATOLOGY_V2 (V1 pi-native + arm "
                    "contrasts) from pinned TRAIN labels only.")
    p.add_argument("--base", required=True,
                   help="base_train_outcomes.parquet (native + pi)")
    p.add_argument("--base-sha256", required=True)
    p.add_argument("--coverage", required=True,
                   help="overlay_outcomes.parquet (fixed/switching + walls)")
    p.add_argument("--coverage-sha256", required=True)
    p.add_argument("--trajectory", required=True,
                   help="labels_overlay_dspeed/overlay_outcomes.parquet")
    p.add_argument("--trajectory-sha256", required=True)
    p.add_argument("--frozen", default=None,
                   help="frozen V1 TRAIN_CLIMATOLOGY.json for pi-native parity")
    p.add_argument("--allow-parity-diff", action="store_true")
    p.add_argument("--out", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        doc = run_build(
            args.base, args.coverage, args.trajectory, args.out,
            base_sha256=args.base_sha256,
            coverage_sha256=args.coverage_sha256,
            trajectory_sha256=args.trajectory_sha256,
            frozen_path=args.frozen,
            allow_parity_diff=args.allow_parity_diff)
    except (ClimatologyV2Refusal, BC.ClimatologyBuildRefusal,
            AC.ArmContrastRefusal) as exc:
        print(f"[wm-n1 climatology v2 refused] {exc}", file=sys.stderr)
        return 87
    except Exception as exc:                              # noqa: BLE001
        print(f"[wm-n1 climatology v2 refused] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 87
    print(json.dumps({"schema": AC.SCHEMA, "out": str(args.out),
                      "contrasts": len(doc["contrasts"]),
                      "pi_native_parity": doc.get("pi_native_parity")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
