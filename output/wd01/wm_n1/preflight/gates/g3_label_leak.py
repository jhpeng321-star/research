"""G3 ``label-leak``.

The main-config HDV network input schema must hold no action, strategy or
CAV-flag channel (asserted on the tensor schema).  A perturbation test changes
the labels, flags and strategies and requires an identical output.

Entry points (P2): ``hdv.input_schema.main_config_schema()`` and
``hdv.input_schema.perturbation_check()``.

``main_config_schema()`` returns either a list of field names or
``{"fields": [...]}``.  ``perturbation_check()`` returns
``{"identical": bool, "max_abs_delta": float, "n_perturbations": int}``.
"""
from __future__ import annotations

from ..imports import PieceMissing
from ..result import fail, ok

NAME = "label-leak"
TITLE = "G3 label-leak (no action/strategy/CAV channel; perturbation invariant)"
ENTRY_POINTS = ("hdv.input_schema.main_config_schema",
                "hdv.input_schema.perturbation_check")
FORBIDDEN_TOKENS = ("action", "strategy", "cav", "one_hot", "onehot")


def _fields(schema):
    if isinstance(schema, dict):
        fields = schema.get("fields") or schema.get("channels") or []
    else:
        fields = schema
    return [str(f) for f in (fields or [])]


def evaluate(schema, perturb):
    fields = _fields(schema)
    forbidden = sorted({f for f in fields
                        if any(t in f.lower() for t in FORBIDDEN_TOKENS)})
    p = perturb if isinstance(perturb, dict) else {}
    try:
        delta = float(p.get("max_abs_delta", float("inf")))
    except (TypeError, ValueError):
        delta = float("inf")
    n_pert = int(p.get("n_perturbations", 0))
    same = p.get("identical") is True and delta == 0.0
    checks = {"n_fields": len(fields), "forbidden_channels": forbidden,
              "n_perturbations": n_pert, "max_abs_delta": delta,
              "perturbation_identical": bool(p.get("identical"))}
    problems = []
    if not fields:
        problems.append("empty input schema")
    if forbidden:
        problems.append(f"forbidden input channel(s): {forbidden}")
    if n_pert < 1:
        problems.append("no perturbation was applied")
    if not same:
        problems.append(f"perturbation changed the output (max_abs_delta={delta:g})")
    passed = not problems
    detail = "kinematics-only schema; output invariant to labels/flags/strategies" \
        if passed else "; ".join(problems)
    return passed, detail, checks


def run(settings, resolve):
    try:
        schema_fn = resolve(ENTRY_POINTS[0])
        perturb_fn = resolve(ENTRY_POINTS[1])
    except PieceMissing as e:
        return fail(NAME, TITLE, str(e), ENTRY_POINTS, error="piece-missing")
    try:
        schema = schema_fn()
        perturb = perturb_fn()
    except Exception as e:                                # noqa: BLE001
        return fail(NAME, TITLE, f"entry point raised "
                    f"{type(e).__name__}: {e}", ENTRY_POINTS, error="gate-error")
    passed, detail, checks = evaluate(schema, perturb)
    return (ok if passed else fail)(NAME, TITLE, detail, ENTRY_POINTS, checks)
