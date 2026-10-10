"""Fail-closed entry point for a separately released real WM-N1 evaluation.

The only implemented operation is a metadata-only dry run.  It authenticates
the externally pinned release specification and selects its CAL half through
``ReleasedCalSource`` without opening a selected shard.  It does not import
the legacy synthetic evaluation CLI, construct predictors, score rows, or
consume a CAL/final-test use marker.

Real scoring remains disabled until the M1 temporal-target/summary contract,
controller action parity, real comparator artifacts, checkpoint/source
identities, and an explicit half release are independently proven and wired to
a separately reviewed entry point.  A release specification by itself is not
authorization to score.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HALVES = ("cal_tune", "cal_cal")


class RealEvaluationRefusal(PermissionError):
    """The real evaluation has not satisfied its release or parity gates."""


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealEvaluationRefusal(f"{label} must be explicit")
    return value.strip()


def metadata_dry_run(args, *, source_factory=None) -> dict:
    """Validate only the registered release and selected CAL metadata.

    ``source_factory`` is injectable for synthetic refusal tests.  The real
    factory's constructor hashes/parses metadata and performs pure selection;
    its data-reading ``outcome_rows`` and ``rollout_rows`` methods are never
    called here.  The returned summary deliberately omits world names/paths.
    """
    sha = _required(args.release_sha256, "raw release SHA-256")
    if not _SHA256.fullmatch(sha):
        raise RealEvaluationRefusal("raw release SHA-256 must be 64 lowercase hex digits")
    allocation_id = _required(args.allocation_id, "allocation ID")
    release_id = _required(args.data_release_id, "data-release ID")
    half = _required(args.cal_half, "CAL half")
    if half not in _HALVES:
        raise RealEvaluationRefusal("CAL half must be cal_tune or cal_cal")
    spec_path = _required(args.release_spec, "release specification path")
    if source_factory is None:
        from .real_source import ReleasedCalSource

        source_factory = ReleasedCalSource
    source = source_factory(spec_path, expected_release_sha256=sha)
    spec = source.spec
    if not isinstance(spec, dict):
        raise RealEvaluationRefusal("released source has no validated specification")
    if (spec.get("allocation_id") != allocation_id
            or spec.get("data_release_id") != release_id
            or spec.get("half") != half
            or source.half != half):
        raise RealEvaluationRefusal(
            "explicit allocation/data-release/CAL-half identity differs from pinned release")
    selected = source.selected
    if not isinstance(selected, (tuple, list)) or not selected:
        raise RealEvaluationRefusal("released CAL half has no selected runs")
    return {
        "schema": "wm_n1_real_eval_metadata_dry_run/1",
        "status": "metadata_valid_scoring_disabled",
        "allocation_id": allocation_id,
        "data_release_id": release_id,
        "cal_half": half,
        "release_spec_sha256": sha,
        "selected_run_count": len(selected),
        "use_marker_consumed": False,
        "outcome_rows_read": False,
        "prediction_or_scoring_performed": False,
    }


def run(args, *, source_factory=None) -> dict:
    """Refuse scoring before any metadata, data, checkpoint, or marker read."""
    if not args.dry_run:
        raise RealEvaluationRefusal(
            "real scoring disabled: M1 target-time binding and correction "
            "summary parity (a matched 10 s inference may be valid; the "
            "current default is 120 s), controller action parity, pinned "
            "real hybrid T and Part A comparator artifacts, checkpoint/source "
            "identities, and explicit CAL-half release are not all proven; "
            "no data or use marker was opened")
    return metadata_dry_run(args, source_factory=source_factory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WM-N1 released CAL metadata preflight; scoring is disabled")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="validate release/selection metadata only")
    mode.add_argument("--score", action="store_true",
                      help="explicit scoring request, currently refused")
    parser.add_argument("--release-spec", required=True,
                        help="externally registered release JSON path")
    parser.add_argument("--release-sha256", required=True,
                        help="externally registered raw SHA-256 of release JSON")
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--data-release-id", required=True)
    parser.add_argument("--cal-half", choices=_HALVES, required=True)
    return parser


def main(argv=None, *, source_factory=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args, source_factory=source_factory)
    except RealEvaluationRefusal as exc:
        print(f"[wm-n1 real eval refused] {exc}", file=sys.stderr)
        return 87
    except Exception as exc:
        # Source construction may refuse via RealSourceRefusal or a malformed
        # metadata file.  Neither case is permission to fall back to legacy
        # synthetic scoring, and the diagnostic must not echo file contents.
        print(f"[wm-n1 real eval refused] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 87
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
