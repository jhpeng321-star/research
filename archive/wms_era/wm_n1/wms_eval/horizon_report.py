"""Standard horizon report over per-state C1 result files.

Every future world model must be scored the same way, so this module turns the
per-state result files of a C1 scoring run (``states__<model>__<set>.json``,
schema ``wms_eval_c1v2_states/1``) into one fixed horizon report:

* for each model x set x ARM and each horizon (10/30/60/120 s, 300 s when the
  run produced it) the HDV ``|ds|`` median across states with IQR, p90 and sd
  (population) across states, plus the median of the within-state p90;
* a blow-up rate: the fraction of states whose HDV median ``|ds|`` exceeds
  50 m at 60 s **and** 100 m at 120 s (states missing either horizon are
  excluded from the denominator and reported);
* a paired bootstrap over worlds against a chosen reference model: whole
  worlds are resampled, states are paired by ``(world, t0_s)``, and the
  statistic is the median paired difference of the HDV ``|ds|`` median.

The arm that is scored per set is the controller arm actually under test: the
PI arm for the unseen PI sets (``u_*`` / ``unseen*``), the native arm for the
PI-free seen sets.  Every arm present in a file is still summarised in the
JSON, so nothing is hidden; the markdown headline uses the tested arm.

Inputs are small per-state summaries (never raw packs), files are read one at
a time and one state at a time, so memory stays far below the 6 GB ceiling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

HORIZONS_S = (10.0, 30.0, 60.0, 120.0, 300.0)
BLOWUP_THRESHOLDS = ((60.0, 50.0), (120.0, 100.0))
UNSEEN_PREFIXES = ("u_", "unseen")
DEFAULT_REFERENCE = "keep_speed"
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 0


class HorizonReportError(ValueError):
    """A per-state result file or report request is invalid."""


# --------------------------------------------------------------------------- #
# input loading
# --------------------------------------------------------------------------- #
def load_results(path: Path) -> tuple[str, str, list[dict]]:
    """Return ``(model, set, states)`` from one ``states__<model>__<set>.json``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        states = raw.get("states")
        model, set_name = raw.get("model"), raw.get("set")
    elif isinstance(raw, list):
        states, model, set_name = raw, None, None
    else:
        raise HorizonReportError(f"{path} is neither a states object nor a list")
    if not isinstance(states, list) or not states:
        raise HorizonReportError(f"{path} carries no per-state results")
    if not all(isinstance(item, dict) for item in states):
        raise HorizonReportError(f"{path} has non-object state entries")
    if model is None:
        model = Path(path).name.split("__")[1] if "__" in Path(path).name else "unknown"
    if set_name is None:
        parts = Path(path).stem.split("__")
        set_name = parts[2] if len(parts) > 2 else "unknown"
    return str(model), str(set_name), states


def tested_arm(set_name: str, arms_present: Sequence[str]) -> str | None:
    """The controller arm actually under test for this set."""
    arms = [str(a) for a in arms_present]
    if not arms:
        return None
    unseen = str(set_name).lower().startswith(UNSEEN_PREFIXES)
    if unseen and "pi" in arms:
        return "pi"
    if "native" in arms:
        return "native"
    return sorted(arms)[0]


# --------------------------------------------------------------------------- #
# per-state value extraction
# --------------------------------------------------------------------------- #
def _car_value(state: dict, arm: str, horizon: float, field: str) -> float | None:
    block = (state.get("arms") or {}).get(arm)
    if not isinstance(block, dict) or block.get("error"):
        return None
    car = block.get("car") or {}
    strata = car.get(str(float(horizon))) or {}
    hdv = strata.get("hdv") or {}
    value = hdv.get(field)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def car_medians(states: Sequence[dict], arm: str, horizon: float) -> list[float]:
    """Per-state HDV median |ds| at one horizon (states with a value only)."""
    out = []
    for state in states:
        value = _car_value(state, arm, horizon, "ds_med")
        if value is not None:
            out.append(value)
    return out


# --------------------------------------------------------------------------- #
# across-state summary
# --------------------------------------------------------------------------- #
def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def summarize(values: Sequence[float]) -> dict:
    """Median / IQR / p90 / sd (population) across states."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None,
                "p90": None, "sd": None}
    n = len(xs)
    q1, med, q3 = (_quantile(xs, q) for q in (0.25, 0.5, 0.75))
    mean = sum(xs) / n
    var = sum((v - mean) ** 2 for v in xs) / n
    return {"n": n, "median": med, "q1": q1, "q3": q3, "iqr": (q3 - q1),
            "p90": _quantile(xs, 0.9), "sd": math.sqrt(var)}


def horizon_block(states: Sequence[dict], arm: str, horizon: float) -> dict:
    medians, p90s = [], []
    for state in states:
        med = _car_value(state, arm, horizon, "ds_med")
        if med is not None:
            medians.append(med)
            p90 = _car_value(state, arm, horizon, "ds_p90")
            if p90 is not None:
                p90s.append(p90)
    block = summarize(medians)
    block["p90_within_median"] = summarize(p90s)["median"]
    return block


def blowup_rate(states: Sequence[dict], arm: str,
                thresholds: Sequence[tuple[float, float]] = BLOWUP_THRESHOLDS) -> dict:
    """Fraction of states exceeding every horizon threshold (fail-closed count).

    A state enters the denominator only when it has an HDV median ``|ds|`` at
    *every* threshold horizon; states truncated before the longest horizon are
    excluded and counted, never treated as survived.
    """
    n_eval = n_blowup = 0
    for state in states:
        values = [_car_value(state, arm, h, "ds_med") for h, _t in thresholds]
        if any(v is None for v in values):
            continue
        n_eval += 1
        if all(v > t for v, (_h, t) in zip(values, thresholds)):
            n_blowup += 1
    return {"rate": (n_blowup / n_eval) if n_eval else None,
            "n_blowup": n_blowup, "n_eval": n_eval,
            "n_excluded": len(states) - n_eval,
            "rule": " and ".join(f"|ds|>{t:g}m @{h:g}s" for h, t in thresholds)}


# --------------------------------------------------------------------------- #
# paired bootstrap over worlds
# --------------------------------------------------------------------------- #
def _pair_by_world(model_states: Sequence[dict], ref_states: Sequence[dict],
                   arm: str, horizon: float) -> dict[str, list[float]]:
    """Paired per-world differences of the HDV |ds| median at one horizon."""
    def keyed(states):
        out = {}
        for state in states:
            value = _car_value(state, arm, horizon, "ds_med")
            if value is None:
                continue
            key = (str(state.get("world", "unknown")),
                   round(float(state.get("t0_s", 0.0)), 3))
            out[key] = value
        return out

    mine, theirs = keyed(model_states), keyed(ref_states)
    paired: dict[str, list[float]] = {}
    for key in sorted(set(mine) & set(theirs)):
        paired.setdefault(key[0], []).append(mine[key] - theirs[key])
    return paired


def paired_bootstrap(model_states: Sequence[dict], ref_states: Sequence[dict],
                     arm: str, horizon: float, *, n_boot: int = DEFAULT_N_BOOT,
                     seed: int = DEFAULT_SEED) -> dict:
    """Bootstrap CI of the median paired difference, resampling whole worlds."""
    grouped = _pair_by_world(model_states, ref_states, arm, horizon)
    worlds = sorted(grouped)
    diffs = [d for world in worlds for d in grouped[world]]
    point = summarize(diffs)["median"]
    if not diffs or n_boot <= 0:
        return {"point": point, "lo": None, "hi": None, "n_boot": 0,
                "n_worlds": len(worlds), "n_pairs": len(diffs)}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        pick = rng.choices(worlds, k=len(worlds))
        values = [d for world in pick for d in grouped[world]]
        if values:
            samples.append(summarize(values)["median"])
    samples.sort()
    lo, hi = _quantile(samples, 0.025), _quantile(samples, 0.975)
    return {"point": point, "lo": lo, "hi": hi, "n_boot": len(samples),
            "n_worlds": len(worlds), "n_pairs": len(diffs)}


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
def _present_horizons(states: Sequence[dict], arm: str,
                      horizons: Sequence[float]) -> list[float]:
    out = []
    for horizon in horizons:
        if car_medians(states, arm, horizon):
            out.append(float(horizon))
    return out


def build_report(sources: Mapping[str, Mapping[str, Path]], *, reference: str =
                 DEFAULT_REFERENCE, n_boot: int = DEFAULT_N_BOOT,
                 seed: int = DEFAULT_SEED,
                 horizons: Sequence[float] = HORIZONS_S,
                 thresholds: Sequence[tuple[float, float]] = BLOWUP_THRESHOLDS) -> dict:
    """One report dict from ``{model: {set: path}}`` per-state result files."""
    loaded: dict[str, dict[str, list[dict]]] = {}
    origins: dict[str, dict[str, dict]] = {}
    for model, sets in sources.items():
        for set_name, path in sets.items():
            file_model, file_set, states = load_results(Path(path))
            if file_model != model or file_set != set_name:
                raise HorizonReportError(
                    f"{path} declares model/set {file_model!r}/{file_set!r}, "
                    f"requested {model!r}/{set_name!r}")
            loaded.setdefault(model, {})[set_name] = states
            origins.setdefault(model, {})[set_name] = {
                "path": str(path),
                "n_states": len(states),
                "n_worlds": len({str(s.get("world")) for s in states}),
            }

    arms_seen: dict[str, dict[str, list[str]]] = {}
    for model, sets in loaded.items():
        for set_name, states in sets.items():
            arms = sorted({str(a) for s in states
                           for a in (s.get("arms") or {})})
            arms_seen.setdefault(model, {})[set_name] = arms

    models_block: dict[str, dict] = {}
    for model, sets in loaded.items():
        entry: dict[str, dict] = {}
        for set_name, states in sets.items():
            arms_block = {}
            for arm in arms_seen[model][set_name]:
                n_error = sum(
                    1 for s in states
                    if ((s.get("arms") or {}).get(arm) or {}).get("error"))
                horizons_here = _present_horizons(states, arm, horizons)
                arms_block[arm] = {
                    "n_states": len(states),
                    "n_errors": n_error,
                    "horizons_present": horizons_here,
                    "summary": {f"{h:g}": horizon_block(states, arm, h)
                                for h in horizons_here},
                    "blowup": blowup_rate(states, arm, thresholds),
                }
            entry[set_name] = {
                "tested_arm": tested_arm(set_name, arms_seen[model][set_name]),
                "arms_present": arms_seen[model][set_name],
                "arms": arms_block,
            }
        models_block[model] = entry

    paired_block: dict[str, dict] = {}
    for model, sets in loaded.items():
        if model == reference:
            continue
        ref_sets = loaded.get(reference) or {}
        model_paired: dict[str, dict] = {}
        for set_name, states in sets.items():
            ref_states = ref_sets.get(set_name)
            if not ref_states:
                continue
            info = models_block[model][set_name]
            arm = info["tested_arm"]
            ref_arm = (models_block.get(reference, {}).get(set_name, {})
                       .get("tested_arm"))
            if arm is None or ref_arm is None:
                continue
            per_horizon = {}
            for horizon in info["arms"][arm]["horizons_present"]:
                if not _present_horizons(ref_states, ref_arm, [horizon]):
                    continue
                per_horizon[f"{horizon:g}"] = paired_bootstrap(
                    states, ref_states, arm, horizon,
                    n_boot=n_boot, seed=seed)
            if per_horizon:
                model_paired[set_name] = {"arm": arm, "reference": reference,
                                          "reference_arm": ref_arm,
                                          "horizons": per_horizon}
        if model_paired:
            paired_block[model] = model_paired

    return {
        "schema": "wms_eval_horizon_report/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metric": "HDV |ds| (m): per-state median, summarised across states",
        "blowup_rule": blowup_rate([], "native", thresholds)["rule"],
        "reference": reference,
        "horizons_s": [float(h) for h in horizons],
        "sources": origins,
        "models": models_block,
        "paired_vs_reference": paired_block,
    }


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def _fmt(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(report: dict) -> str:
    thresholds = report.get("blowup_rule", "")
    lines = ["# Horizon report (HDV |ds| by horizon, per-state result files)",
             "",
             f"Generated {report['generated_utc']}. Metric: {report['metric']}.",
             f"Blow-up rule: {thresholds}. Reference model for the paired "
             f"bootstrap: `{report['reference']}` (negative difference favours "
             "the model over the reference).",
             "The scored arm per set is the controller arm under test "
             "(PI for the unseen PI sets, native for the PI-free seen sets).",
             ""]
    lines += ["## Sets", ""]
    for model, sets in report["sources"].items():
        for set_name, info in sets.items():
            arms = report["models"][model][set_name]
            lines.append(f"- `{model}` / `{set_name}`: {info['n_states']} states / "
                         f"{info['n_worlds']} worlds, tested arm "
                         f"`{arms['tested_arm']}` (arms present: "
                         f"{', '.join(arms['arms_present'])})")
    for horizon in report["horizons_s"]:
        key = f"{float(horizon):g}"
        rows = []
        for model, sets in report["models"].items():
            for set_name, info in sets.items():
                for arm, block in info["arms"].items():
                    summary = block["summary"].get(key)
                    if summary and summary["n"]:
                        rows.append((model, set_name, arm, block, summary))
        if not rows:
            continue
        lines += ["", f"## HDV |ds| at {key} s", "",
                  "| model | set | arm | n | median | IQR | p90 | sd | p90 within (median) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for model, set_name, arm, block, summary in rows:
            lines.append(f"| {model} | {set_name} | {arm} | {summary['n']} "
                         f"| {_fmt(summary['median'])} | {_fmt(summary['iqr'])} "
                         f"| {_fmt(summary['p90'])} | {_fmt(summary['sd'])} "
                         f"| {_fmt(summary['p90_within_median'])} |")
    lines += ["", "## Blow-up rate (HDV median |ds|)", "",
              "| model | set | arm | rate | blowups / evaluated | excluded |",
              "|---|---|---|---|---|---|"]
    for model, sets in report["models"].items():
        for set_name, info in sets.items():
            for arm, block in info["arms"].items():
                blow = block["blowup"]
                lines.append(f"| {model} | {set_name} | {arm} "
                             f"| {_fmt(blow['rate'], 3)} | {blow['n_blowup']} / "
                             f"{blow['n_eval']} | {blow['n_excluded']} |")
    lines += ["", f"## Paired model - reference (`{report['reference']}`), "
              "median HDV |ds| difference, bootstrap over worlds", ""]
    any_paired = False
    for model, sets in report["paired_vs_reference"].items():
        for set_name, info in sets.items():
            any_paired = True
            for horizon, ci in info["horizons"].items():
                lines.append(f"- {model} / {set_name} (arm `{info['arm']}`) "
                             f"@ {horizon} s: {_fmt(ci['point'])} "
                             f"[{_fmt(ci['lo'])}, {_fmt(ci['hi'])}] "
                             f"(worlds={ci['n_worlds']}, pairs={ci['n_pairs']}, "
                             f"n_boot={ci['n_boot']})")
    if not any_paired:
        lines.append("- not available (no shared set with the reference)")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_sources(entries: Sequence[str]) -> dict[str, dict[str, Path]]:
    """Parse ``MODEL=SET=PATH`` entries into ``{model: {set: path}}``."""
    sources: dict[str, dict[str, Path]] = {}
    for entry in entries:
        model, sep1, rest = entry.partition("=")
        set_name, sep2, path = rest.partition("=")
        if not sep1 or not sep2 or not model.strip() or not set_name.strip() \
                or not path.strip():
            raise HorizonReportError(f"expected MODEL=SET=PATH, got {entry!r}")
        sources.setdefault(model.strip(), {})[set_name.strip()] = Path(path.strip())
    if not sources:
        raise HorizonReportError("at least one --states MODEL=SET=PATH is required")
    return sources


def run(args) -> int:
    sources = parse_sources(args.states)
    report = build_report(sources, reference=args.reference, n_boot=args.bootstrap,
                          seed=args.seed)
    out_md = Path(args.out)
    out_json = out_md.with_suffix(".json")
    if out_md.suffix != ".md":
        raise HorizonReportError(f"--out must name the markdown file, got {args.out}")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if out_md.exists() or out_json.exists():
        raise HorizonReportError(f"refusing to overwrite existing report: {out_md}")
    out_md.write_text(markdown(report), encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"out_md": str(out_md), "out_json": str(out_json),
                      "models": sorted(report["models"]),
                      "reference": report["reference"]}))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wms_eval horizon",
        description="Standard horizon report over per-state C1 result files")
    p.add_argument("--states", action="append", required=True,
                   help="MODEL=SET=PATH to states__<model>__<set>.json (repeatable)")
    p.add_argument("--reference", default=DEFAULT_REFERENCE,
                   help=f"reference model for the paired bootstrap "
                        f"(default {DEFAULT_REFERENCE!r})")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", required=True, help="markdown output path (.json sibling)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (HorizonReportError, OSError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
