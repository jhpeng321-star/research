"""Horizon tables, blow-up rates and paired world bootstraps over per-state results.

Inputs are the state results of :mod:`wms.eval.score` (one list per model and set).  The
headline statistic is the HDV ``|ds|`` (m) / ``|dv|`` (m/s) *per-state median*, summarised
across states by median / IQR / p90.  A *blow-up* is a state whose HDV median ``|ds|``
exceeds 50 m at 60 s **and** 100 m at 120 s; states missing a threshold horizon are
excluded from the denominator and counted, never treated as survived.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Mapping, Sequence

from .errors import EvalError
from .stats import summarize, world_bootstrap_median

HORIZONS_S = (10.0, 30.0, 60.0, 120.0)
BLOWUP_THRESHOLDS = ((60.0, 50.0), (120.0, 100.0))
REPORT_SCHEMA = "wms_eval_horizon_report/1"
StateResults = Sequence[dict]


def tested_arm(set_name: str, arms_present: Sequence[str]) -> str | None:
    """The controller arm under test: pi for unseen-PI sets, wall for wall sets, else native."""
    arms = [str(a) for a in arms_present]
    if not arms:
        return None
    name = str(set_name).lower()
    if name.startswith(("u_", "unseen")) and "pi" in arms:
        return "pi"
    if name.startswith("wall") and "wall" in arms:
        return "wall"
    return "native" if "native" in arms else sorted(arms)[0]


def car_value(state: Mapping, arm: str, horizon: float, field: str,
              stratum: str = "hdv") -> float | None:
    """One finite scalar of a state result, or ``None`` when absent or the arm errored."""
    block = (state.get("arms") or {}).get(arm)
    if not isinstance(block, dict) or block.get("error"):
        return None
    value = ((block.get("car") or {}).get(str(float(horizon))) or {}).get(stratum, {}).get(field)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def normalize_seen(states: StateResults) -> list[dict]:
    """Alias the single recorded arm of a seen-controls state to ``native``."""
    out = []
    for state in states:
        arms = state.get("arms") or {}
        if "native" in arms or len(arms) != 1:
            out.append(state)
            continue
        (own,) = arms.keys()
        out.append({**state, "arms": {"native": arms[own]}, "source_arm": own})
    return out


def horizon_summary(states: StateResults, arm: str, horizon: float, field: str = "ds_med"
                    ) -> dict:
    """Median / IQR / p90 / sd across states of the per-state HDV scalar ``field``."""
    values = [v for v in (car_value(s, arm, horizon, field) for s in states) if v is not None]
    return summarize(values)


def blowup_rate(states: StateResults, arm: str,
                thresholds: Sequence[tuple[float, float]] = BLOWUP_THRESHOLDS) -> dict:
    """Fraction of states whose HDV median ``|ds|`` exceeds every threshold."""
    n_eval = n_blowup = 0
    for state in states:
        values = [car_value(state, arm, h, "ds_med") for h, _t in thresholds]
        if any(v is None for v in values):
            continue
        n_eval += 1
        if all(v > t for v, (_h, t) in zip(values, thresholds)):
            n_blowup += 1
    return {"rate": (n_blowup / n_eval) if n_eval else None, "n_blowup": n_blowup,
            "n_eval": n_eval, "n_excluded": len(states) - n_eval,
            "rule": " and ".join(f"|ds|>{t:g}m @{h:g}s" for h, t in thresholds)}


def state_key(state: Mapping) -> tuple:
    """Pairing identity of a state across models: world, key, source arm and t0."""
    return (str(state.get("world")), str(state.get("key")), str(state.get("source_arm")),
            round(float(state["t0_s"]), 3))


def paired_differences(model_states: StateResults, ref_states: StateResults, arm: str,
                       horizon: float, field: str = "ds_med") -> dict[str, list[float]]:
    """Per-world lists of ``model - reference`` for states scored by both."""
    def keyed(states):
        out = {}
        for state in states:
            value = car_value(state, arm, horizon, field)
            if value is not None:
                out[state_key(state)] = value
        return out

    mine, theirs = keyed(model_states), keyed(ref_states)
    grouped: dict[str, list[float]] = {}
    for key in sorted(set(mine) & set(theirs)):
        grouped.setdefault(key[0], []).append(mine[key] - theirs[key])
    return grouped


def paired_bootstrap(model_states: StateResults, ref_states: StateResults, arm: str,
                     horizon: float, *, field: str = "ds_med", n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Median paired difference (model - reference) with a CI that resamples whole worlds.

    Negative favours the model.
    """
    grouped = paired_differences(model_states, ref_states, arm, horizon, field)
    return world_bootstrap_median(grouped, n_boot=n_boot, seed=seed)


def model_report(states_by_set: Mapping[str, StateResults], *,
                 horizons: Sequence[float] = HORIZONS_S,
                 thresholds: Sequence[tuple[float, float]] = BLOWUP_THRESHOLDS) -> dict:
    """Per set: tested arm, |ds| and |dv| summaries per horizon, and the blow-up rate."""
    out: dict[str, dict] = {}
    for set_name, raw in states_by_set.items():
        states = normalize_seen(raw) if str(set_name).startswith("seen") else list(raw)
        arms = sorted({str(a) for s in states for a in (s.get("arms") or {})})
        arm = tested_arm(set_name, arms)
        if arm is None:
            raise EvalError(f"set {set_name!r} carries no scored arm")
        out[set_name] = {
            "tested_arm": arm,
            "n_states": len(states),
            "n_worlds": len({str(s.get("world")) for s in states}),
            "n_errors": sum(1 for s in states if ((s.get("arms") or {}).get(arm) or {}).get("error")),
            "ds": {f"{h:g}": horizon_summary(states, arm, h, "ds_med") for h in horizons},
            "dv": {f"{h:g}": horizon_summary(states, arm, h, "dv_med") for h in horizons},
            "blowup": blowup_rate(states, arm, thresholds),
        }
    return out


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_ci(block: Mapping | None, digits: int = 2) -> str:
    if not block or block.get("point") is None:
        return "n/a"
    return f"{fmt(block['point'], digits)} [{fmt(block.get('lo'), digits)}, {fmt(block.get('hi'), digits)}]"


def horizon_table(name: str, report: Mapping[str, Mapping]) -> list[str]:
    """Markdown rows (median / IQR / p90 of |ds| and |dv|) for one model's report."""
    lines = [f"### {name}", "",
             "| set | arm | h | n | ds median | ds IQR | ds p90 | dv median | blow-up |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for set_name, block in report.items():
        blow = block["blowup"]
        for horizon, ds in block["ds"].items():
            dv = block["dv"][horizon]
            lines.append(
                f"| {set_name} | {block['tested_arm']} | {horizon}s | {ds['n']} | "
                f"{fmt(ds['median'])} | {fmt(ds['iqr'])} | {fmt(ds['p90'])} | "
                f"{fmt(dv['median'])} | {fmt(blow['rate'], 3)} ({blow['n_blowup']}/{blow['n_eval']}) |")
    lines.append("")
    return lines


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
