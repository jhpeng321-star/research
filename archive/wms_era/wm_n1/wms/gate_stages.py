"""Multi-stage WM-S smoke gate (Run 3).

Run 2's smoke gate only ran the 10 s stage, so it could not see the traffic-loss
blow-up that opened when the 30-120 s ramp activated at stage 60 (see
``_out/RUN2_TRAFFIC_LOSS_DIAG.md``).  This gate runs a short curriculum that
passes through **every** stage (10/30/60/120/300 s) with every loss term active
and fails closed on:

* any non-finite numeric field (or a missing required term) at any step;
* a per-stage term magnitude more than ``--dominance``x the position term
  (default 100x): a single term dominating the update;
* a term growing more than ``--growth``x between consecutive stages (default
  10x).  The bounded per-car terms (position/speed/lane) are given the position
  term's own horizon growth as their baseline, so their expected growth with the
  unroll length is not a false positive; the traffic term is exempt while its
  ramp is still opening (``stage_s <= --ramp-end-s``).

Logged loss terms are per-chunk means (horizon-invariant), so comparing
magnitudes across stages is meaningful.  ``loss_share`` (each term's share of
the total gradient norm) is read from the trainer's ``loss_share.jsonl`` (or
``val_log.jsonl`` when a validation set was used) next to the train log when
present and printed per stage.

The default preserves the full five-stage smoke contract.  Trainsetup's
four-stage curriculum is selected explicitly with ``--stages 10,30,60,120``
(or ``stages=(10, 30, 60, 120)`` through the Python API).

Usage::

    python -m wms.gate_stages smoke_stages/train_log.jsonl \
        --stages 10,30,60,120,300 \
        --require-terms position,speed,lane,traffic,stability,conservation
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .brakefix import required_log_terms

#: Loss terms the smoke must exercise (per-car + traffic + stability + conservation).
DEFAULT_TERMS = ("position", "speed", "lane", "traffic", "stability", "conservation")
#: Braking-fix loss terms; append the ENABLED ones to ``--require-terms``.
BRAKEFIX_TERMS = ("asym_speed", "decel_weight", "propagation")
#: Long-horizon loss terms; require only the ENABLED or explicitly requested ones.
LONG_HORIZON_TERMS = ("residual_mag", "lane_rate")
#: Curriculum stages the smoke must cover.
DEFAULT_STAGES = (10.0, 30.0, 60.0, 120.0, 300.0)


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _required_long_horizon_terms(log_path: Path,
                                 config_path: Path | None = None) -> tuple[str, ...]:
    """Enabled losses from config/receipt metadata, independent of log keys.

    Prefer the pre-update config so interrupted runs retain their contract.
    Historical logs without metadata keep the original requirements. An explicit
    path may contain either a flat config or a receipt's nested ``config``.
    """
    path = config_path
    if path is None:
        for name in ("train_config.json", "TRAIN_RECEIPT.json"):
            candidate = Path(log_path).with_name(name)
            if candidate.is_file():
                path = candidate
                break
    if path is None:
        return ()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = data.get("config", data)
    return tuple(term for term in LONG_HORIZON_TERMS
                 if float(cfg.get(f"{term}_weight", 0.0)) > 0.0)


def evaluate(path: Path, *, stages=DEFAULT_STAGES, terms=DEFAULT_TERMS,
             position_term: str = "position", dominance: float = 100.0,
             growth: float = 10.0, floor: float = 1.0e-6,
             ramp_end_s: float = 120.0, config_path: Path | None = None) -> dict:
    """Check a multi-stage smoke log; return ``{passed, failures, per_stage, ...}``.

    No exception on failure (use :func:`check` for the CLI fail-closed wrapper).
    """
    terms = tuple(dict.fromkeys((*terms, *required_log_terms(path, config_path),
                                *_required_long_horizon_terms(path, config_path))))
    rows = _read_rows(path)
    failures: list[str] = []
    if not rows:
        return {"passed": False, "failures": ["empty train log"], "per_stage": {},
                "loss_share": {}}

    # Base contract: every numeric field finite; gate stays well-formed.
    for row in rows:
        step = row.get("step")
        if row.get("finite") is not True:
            failures.append(f"finite flag at step {step}")
        if row.get("gate_monotone") is not True:
            failures.append(f"gate not monotone at step {step}")
        if abs(float(row.get("gate_g0", 0.0))) > 0.0:
            failures.append(f"g(0) != 0 at step {step}")
        for key, value in row.items():
            if _numeric(value) and not math.isfinite(float(value)):
                failures.append(f"nonfinite {key} at step {step}")
        for term in set(terms) & set((*BRAKEFIX_TERMS, *LONG_HORIZON_TERMS)):
            if not _numeric(row.get(term)):
                failures.append(f"required term {term!r} absent/non-numeric at step {step}")

    # Group by curriculum stage.
    by_stage: dict[float, list[dict]] = {}
    for row in rows:
        if "stage_s" not in row:
            failures.append("row without stage_s")
            continue
        by_stage.setdefault(round(float(row["stage_s"]), 3), []).append(row)

    ordered = [float(s) for s in stages]
    missing_stages = [s for s in ordered if round(s, 3) not in by_stage]
    if missing_stages:
        failures.append(f"curriculum stages absent from log: {missing_stages}")

    per_stage: dict[float, dict[str, float]] = {}
    for stage_s in ordered:
        block = by_stage.get(round(stage_s, 3))
        if not block:
            continue
        magnitudes: dict[str, float] = {}
        for term in terms:
            values = [float(r[term]) for r in block if term in r and _numeric(r[term])]
            if not values:
                failures.append(f"stage {stage_s}: required term {term!r} absent")
                continue
            magnitudes[term] = sum(values) / len(values)
        per_stage[stage_s] = magnitudes

    # Dominance: no non-position term may exceed dominance x position.
    for stage_s, mags in per_stage.items():
        pos = mags.get(position_term)
        if pos is None:
            continue
        denom = max(abs(pos), floor)
        for term, value in mags.items():
            if term == position_term:
                continue
            if abs(value) > dominance * denom:
                failures.append(
                    f"stage {stage_s}: {term}={value:.6g} exceeds {dominance:g}x "
                    f"{position_term}={pos:.6g}")

    # Growth: consecutive stages.  The per-car terms (position/speed/lane) are
    # bounded Huber/CE losses that grow with the unroll horizon by construction,
    # so their allowed factor is scaled by the position term's own growth (the
    # horizon baseline); a per-car term that decouples from that baseline is
    # still flagged.  The unbounded auxiliary terms (traffic/stability/
    # conservation) use the literal ``growth`` factor.  The traffic term is
    # exempt while its 30-120 s ramp is still opening (intended schedule).
    percar = {position_term, "speed", "lane", *BRAKEFIX_TERMS}
    for term in terms:
        for prev_s, cur_s in zip(ordered, ordered[1:]):
            prev_mags, cur_mags = per_stage.get(prev_s), per_stage.get(cur_s)
            if not prev_mags or not cur_mags:
                continue
            prev_v, cur_v = prev_mags.get(term), cur_mags.get(term)
            if prev_v is None or cur_v is None:
                continue
            if term == "traffic" and cur_s <= float(ramp_end_s):
                continue  # intentional 30-120 s ramp, not a blow-up
            threshold = float(growth)
            if term in percar:
                pos_prev = prev_mags.get(position_term)
                pos_cur = cur_mags.get(position_term)
                if pos_prev is not None and pos_cur is not None and pos_prev > floor:
                    threshold = float(growth) * max(1.0, abs(pos_cur) / abs(pos_prev))
            if prev_v > floor and cur_v > threshold * prev_v:
                failures.append(
                    f"{term} grew {cur_v / prev_v:.1f}x from stage {prev_s} to "
                    f"{cur_s} (> {threshold:g}x)")

    # Optional gradient-norm shares from the trainer's dedicated per-stage log
    # (``loss_share.jsonl``) or, if that is absent, the validation log.
    shares: dict[str, dict[str, float]] = {}
    for name in ("loss_share.jsonl", "val_log.jsonl"):
        candidate = Path(path).with_name(name)
        if not candidate.is_file():
            continue
        for row in _read_rows(candidate):
            if "loss_share" in row and "stage_s" in row:
                shares[float(row["stage_s"])] = row["loss_share"]

    return {"passed": not failures, "failures": failures,
            "records": len(rows), "stages": ordered, "terms": list(terms),
            "per_stage": per_stage, "loss_share": shares,
            "dominance_limit": dominance, "growth_limit": growth,
            "ramp_end_s": ramp_end_s}


def check(path: Path, **kwargs) -> dict:
    """Fail-closed CLI wrapper around :func:`evaluate` (prints the per-stage table)."""
    result = evaluate(path, **kwargs)
    print("records %d stages %s" % (result["records"],
                                    ",".join("%g" % s for s in result["stages"])))
    print("nonfinite 0 missing 0" if result["passed"] else
          "nonfinite/missing failures %d" % len(result["failures"]))
    header = "stage_s\t" + "\t".join(result["terms"])
    print(header)
    for stage_s in result["stages"]:
        mags = result["per_stage"].get(stage_s)
        if mags is None:
            print("%g\tMISSING" % stage_s)
            continue
        print("%g\t" % stage_s + "\t".join("%.6g" % mags.get(t, float("nan"))
                                           for t in result["terms"]))
    for stage_s, share in sorted(result["loss_share"].items()):
        print("loss_share stage %g: %s" % (
            stage_s, json.dumps(share, sort_keys=True)))
    print(json.dumps({"status": "passed" if result["passed"] else "failed",
                      "records": result["records"], "stages": result["stages"],
                      "failures": result["failures"],
                      "per_stage": {str(k): v for k, v in result["per_stage"].items()},
                      "loss_share": {str(k): v for k, v in result["loss_share"].items()}},
                     sort_keys=True))
    if not result["passed"]:
        raise SystemExit("GATE FAIL " + "; ".join(result["failures"]))
    return result


def _float_list(spec: str) -> tuple[float, ...]:
    return tuple(float(x) for x in spec.replace(",", " ").split() if x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--stages", default=",".join("%g" % s for s in DEFAULT_STAGES))
    parser.add_argument("--require-terms", default=",".join(DEFAULT_TERMS))
    parser.add_argument("--position-term", default="position")
    parser.add_argument("--dominance", type=float, default=100.0)
    parser.add_argument("--growth", type=float, default=10.0)
    parser.add_argument("--floor", type=float, default=1.0e-6)
    parser.add_argument("--ramp-end-s", type=float, default=120.0)
    parser.add_argument("--train-config", type=Path, default=None,
                        help="config/receipt override; enabled braking and long-horizon "
                             "terms are automatic")
    args = parser.parse_args()
    check(args.log, stages=_float_list(args.stages),
          terms=tuple(t for t in args.require_terms.replace(",", " ").split() if t),
          position_term=args.position_term, dominance=args.dominance,
          growth=args.growth, floor=args.floor, ramp_end_s=args.ramp_end_s,
          config_path=args.train_config)
