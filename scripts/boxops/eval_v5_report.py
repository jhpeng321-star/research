#!/usr/bin/env python3
"""Report completed C4 eval summaries without importing the simulator.

Usage: python scripts/boxops/eval_v5_report.py runs/active/my_eval --out runs/active/report
Writes report.json and report.md. All input/output paths must be inside this checkout.
Only summary.json and optional sibling manifest.json are read; restricted names,
incomplete attempts, symlinks and Windows reparse points are never traversed.

Recorder manifests must say complete when present. Relayed trees omit manifests:
their summary schema and sim_end_s >= warmup_s + horizon_s establish completion.
Policy identity is manifest.policy, summary.policy, or the collector directory tag
(qualified by controller_NNN for direct policies); finally summary.controller.
Directory-derived identities are recorded explicitly, not inferred from checkpoints.

Driver population comes from explicit driver_population metadata or the recorder's
manifest.scenario (a recognized legacy scenario omits driver_population). Numeric
summary.population is only the requested vehicle count, never a population model.
Current env summaries omit driver population and collector policy metadata. For
summary-only inputs lacking population provenance, --driver-population is an explicit
caller assertion, recorded in the report; it must agree with any available metadata.
Unidentified or mixed driver populations are rejected before deduplication.

Statistics use equally weighted seed means, not episode-level standard errors.
Grouped pairs average differences on matching (seed, demand, penetration) cells
within each seed before computing a CI. Missing/nonfinite values are unavailable,
never zero. CIs are two-sided 95% Student t intervals (sample SD, df=n-1), unavailable
for n<2. SciPy is optional; the stdlib fallback uses a published t-critical table,
rounding df DOWN conservatively between entries and capping df at 120. No dependency
is added. Mixed warmup/horizon schedules are rejected; report them separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics

WORKSPACE = Path(__file__).absolute().parents[2]
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
SEEN = (0.05, 0.1, 0.2, 0.4)
UNSEEN = (0.15, 0.3, 0.6)
METRICS = {
    "system_seconds_per_requested_vehicle": "episode.system_seconds_per_requested_vehicle",
    "source_wait_vehicle_seconds": "episode.source_wait_vehicle_seconds",
    "window_source_wait_vehicle_seconds": "window.source_wait_vehicle_seconds",
    "window_speed_deficit_vehicle_seconds": "window.speed_deficit_vehicle_seconds",
    "arrivals": "window.arrivals",
    "episode_arrivals": "episode.arrived",
    "braking_vehicle_seconds": "window.braking_vehicle_seconds",
    "minimum_closing_ttc_s": "minimum_closing_ttc_s",
    "collisions": "collisions",
    "teleports": "teleports",
    "wall_seconds": "wall_seconds",
}
PAIRED_INTERPRETATION = (
    "Differences remain treatment - native. Negative is better for cost metrics "
    "(system seconds, source wait, speed deficit, braking, collisions, teleports); "
    "positive is better for arrivals (window and episode) and minimum closing TTC. "
    "Negative wall time means faster execution, not improved traffic performance."
)
# t(.975, df), rounded upward at six decimals, so fallback never narrows the CI.
T_CRITICAL = dict(enumerate((
    12.706205, 4.302653, 3.182447, 2.776446, 2.570582, 2.446912,
    2.364625, 2.306005, 2.262158, 2.228139, 2.200986, 2.178813,
    2.160369, 2.144787, 2.131450, 2.119906, 2.109816, 2.100923,
    2.093025, 2.085964, 2.079614, 2.073874, 2.068658, 2.063899,
    2.059539, 2.055530, 2.051831, 2.048408, 2.045230, 2.042273,
), start=1))
T_CRITICAL.update({40: 2.021076, 60: 2.000298, 120: 1.979931})


def forbidden(path: Path) -> bool:
    return any(token in str(path).lower() for token in FORBIDDEN)


def safe_path(path: str | Path) -> Path:
    """Validate lexically BEFORE filesystem access; do not resolve through links."""
    path = Path(os.path.abspath(path))
    if forbidden(path) or not path.is_relative_to(WORKSPACE):
        raise ValueError("restricted or outside-workspace path refused")
    current = WORKSPACE
    for part in ("", *path.relative_to(WORKSPACE).parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("symlink/reparse path refused")
    return path


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def metric(summary: dict, dotted: str):
    value = summary
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return number(value)


class ConfidenceIntervals:
    def __init__(self, use_scipy: bool = True):
        self.ppf = None
        if use_scipy:
            try:
                from scipy.stats import t
            except ImportError:
                pass
            else:
                self.ppf = t.ppf
        self.method = ("Student t (scipy, df=n-1)" if self.ppf else
                       "Student t (stdlib conservative table; df rounded down, capped at 120)")

    def summarize(self, seed_values: dict) -> dict:
        values = {str(seed): value for seed, value in sorted(seed_values.items())
                  if number(value) is not None}
        n = len(values)
        mean = statistics.mean(values.values()) if n else None
        ci = None
        if n > 1:
            df = n - 1
            critical = (float(self.ppf(0.975, df)) if self.ppf else
                        T_CRITICAL[max(key for key in T_CRITICAL if key <= df)])
            half_width = critical * statistics.stdev(values.values()) / math.sqrt(n)
            ci = [mean - half_width, mean + half_width]
        return {"n_seeds": n, "mean": mean, "ci95": ci, "seed_means": values}


def read_object(path: Path) -> dict:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def collector_identity(path: Path):
    """Retain controller index: two checkpoints can share a policy basename."""
    parts = path.parts
    indices = [i for i, part in enumerate(parts[:-3]) if part == "episodes"]
    if not indices:
        return None
    index = indices[-1]
    tag = parts[index + 1]
    prefix = parts[index - 1] if index else ""
    return f"{prefix}/{tag}" if re.fullmatch(r"controller_\d+", prefix) else tag


class PopulationError(ValueError):
    """A population ambiguity must fail the report, not silently drop an episode."""


def population_metadata(summary: dict, manifest: dict | None, declared: str | None):
    sources = []
    scenario = (manifest or {}).get("scenario")
    for label, obj in (("summary", summary), ("manifest", manifest),
                       ("manifest.scenario", scenario)):
        if not isinstance(obj, dict):
            continue
        model = obj.get("driver_population")
        source = f"{label}.driver_population"
        if (label == "manifest.scenario" and "driver_population" not in obj
                and obj.get("schema") == "native_damping_explicit_requests/1"):
            # scenarios.build_scenario deliberately leaves legacy manifests unchanged.
            model = "legacy3"
            source = "manifest.scenario legacy schema without driver_population"
        if model is None:
            continue
        if not isinstance(model, str) or not model.strip():
            raise PopulationError(f"invalid driver population metadata: {source}")
        version = obj.get("driver_population_version")
        if version is not None and (not isinstance(version, str) or not version.strip()):
            raise PopulationError(f"invalid driver population version: {label}")
        sources.append({"source": source, "driver_population": model,
                        "driver_population_version": version})
    if declared is not None:
        if declared not in ("textbook", "legacy3"):
            raise PopulationError("declared driver population must be textbook or legacy3")
        sources.append({"source": "caller declaration (--driver-population)",
                        "driver_population": declared, "driver_population_version": None})
    if not sources:
        raise PopulationError(
            "driver population unavailable; provide full manifests or an explicit "
            "--driver-population declaration (numeric population is a vehicle count)")
    models = {item["driver_population"] for item in sources}
    versions = {item["driver_population_version"] for item in sources
                if item["driver_population_version"] is not None}
    if len(models) != 1 or len(versions) > 1:
        raise PopulationError("conflicting driver population metadata/declaration")
    return {"driver_population": next(iter(models)),
            "driver_population_version": next(iter(versions), None),
            "population_provenance": sources}


def discover(roots: list[Path], counts: Counter):
    """Prune restricted directories before opening them, including explicit roots."""
    found = set()
    for root in roots:
        root = safe_path(root)
        if not root.is_dir():
            raise ValueError(f"input is not a directory: {root}")
        if any(".incomplete" in part.lower() for part in root.parts):
            raise ValueError("incomplete input root refused")

        def walk_error(error):
            raise OSError(f"cannot traverse input: {error}")

        for directory, dirs, files in os.walk(root, topdown=True, followlinks=False,
                                               onerror=walk_error):
            parent = Path(directory)
            keep = []
            for name in sorted(dirs):
                if forbidden(Path(name)) or ".incomplete" in name.lower() or name == ".git":
                    counts["pruned_directories"] += 1
                    continue
                try:
                    safe_path(parent / name)
                except ValueError:
                    counts["pruned_links"] += 1
                else:
                    keep.append(name)
            dirs[:] = keep
            if "summary.json" in files:
                path = parent / "summary.json"
                try:
                    safe_path(path)
                except ValueError:
                    counts["pruned_links"] += 1
                    continue
                if path in found:
                    counts["overlapping_paths"] += 1
                found.add(path)
    return sorted(found)


def episode_from(path: Path, *, driver_population=None) -> dict:
    summary = read_object(path)
    if summary.get("schema") != "c4_sumo_episode_summary/1":
        raise ValueError("not a per-episode C4 summary")
    manifest_path = safe_path(path.with_name("manifest.json"))
    manifest = read_object(manifest_path) if manifest_path.exists() else None
    if manifest is not None and manifest.get("status") != "complete":
        raise ValueError("manifest is not complete")
    if "status" in summary and summary["status"] != "complete":
        raise ValueError("summary is not complete")
    fields = ("seed", "demand_vph", "penetration", "warmup_s", "horizon_s", "sim_end_s")
    values = {key: number(summary.get(key)) for key in fields}
    if any(value is None for value in values.values()):
        raise ValueError("missing/nonfinite episode identity or completion timing")
    seed, demand, pen, warmup, horizon, end = (values[key] for key in fields)
    if (seed != int(seed) or demand != int(demand) or seed < 0 or demand <= 0
            or not 0 <= pen <= 1 or warmup < 0 or horizon <= 0):
        raise ValueError("invalid episode identity or timing")
    if end + 1e-6 < warmup + horizon:
        raise ValueError("episode has not reached its horizon")
    if manifest:
        for key in (*fields[:5], "controller"):
            if key in manifest and manifest[key] != summary.get(key):
                raise ValueError(f"manifest/summary mismatch: {key}")
    controller = (manifest or {}).get("policy") or summary.get("policy")
    identity_source = "policy"
    if not controller:
        controller = summary.get("controller")
        identity_source = "summary.controller"
        if controller == "direct":
            controller = collector_identity(path)
            if controller is None:
                raise ValueError("direct controller needs policy metadata or episodes/<policy>/<job>")
            identity_source = "collector directory (policy metadata unavailable)"
    if not isinstance(controller, str) or not controller.strip():
        raise ValueError("missing controller identity")
    return {
        "path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
        "controller": controller, "controller_identity_source": identity_source,
        "collector_identity": collector_identity(path),
        "seed": int(seed), "demand_vph": int(demand), "penetration": round(pen, 10),
        "warmup_s": warmup, "horizon_s": horizon,
        "population": number(summary.get("population")),
        **population_metadata(summary, manifest, driver_population),
        "completion": "manifest" if manifest is not None else "summary_timing",
        "metrics": {name: metric(summary, dotted) for name, dotted in METRICS.items()},
    }


def cell_key(episode):
    return episode["controller"], episode["demand_vph"], episode["penetration"], episode["seed"]


def load_episodes(roots: list[Path], *, driver_population=None):
    counts = Counter()
    rejected = []
    candidates = []
    for path in discover(roots, counts):
        counts["summary_files"] += 1
        try:
            episode = episode_from(path, driver_population=driver_population)
        except PopulationError as error:
            raise PopulationError(f"{path.relative_to(WORKSPACE)}: {error}") from error
        except (OSError, ValueError) as error:
            counts["rejected_summaries"] += 1
            rejected.append({"path": str(path.relative_to(WORKSPACE)), "reason": str(error)})
            continue
        candidates.append(episode)
    # Check BEFORE duplicate/conflict filtering, which might otherwise hide a
    # second population with the same controller/demand/penetration/seed identity.
    populations = {ep["driver_population"] for ep in candidates}
    if len(populations) > 1:
        raise PopulationError("mixed driver populations; generate separate reports: "
                              + ", ".join(sorted(populations)))
    versions = {ep["driver_population_version"] for ep in candidates
                if ep["driver_population_version"] is not None}
    if len(versions) > 1:
        raise PopulationError("mixed driver population versions; generate separate reports")
    # A full tree and its summary-only relay may be supplied together. Resolve
    # directory aliases from available policy metadata before deduplicating them.
    policies = defaultdict(set)
    for episode in candidates:
        alias = episode["collector_identity"]
        if alias and episode["controller_identity_source"] == "policy":
            policies[alias].add(episode["controller"])
    by_identity = defaultdict(list)
    for episode in candidates:
        if episode["controller_identity_source"].startswith("collector directory"):
            names = policies[episode["collector_identity"]]
            if len(names) > 1:
                counts["ambiguous_policy_episodes"] += 1
                rejected.append({"path": episode["path"],
                                 "reason": "directory identity maps to multiple policies"})
                continue
            if names:
                episode["controller"] = next(iter(names))
                episode["controller_identity_source"] = "directory matched to policy metadata"
        by_identity[cell_key(episode)].append(episode)
    episodes = []
    for copies in by_identity.values():
        # Exact mirrors and reruns with identical outcomes count once. Wall time is
        # diagnostic only. Conflicting outcomes/schedules exclude the entire identity.
        signatures = {
            json.dumps([ep["warmup_s"], ep["horizon_s"],
                        {key: value for key, value in ep["metrics"].items()
                         if key != "wall_seconds"}], sort_keys=True)
            for ep in copies
        }
        if len(signatures) > 1:
            counts["conflicting_episodes"] += len(copies)
            rejected.extend({"path": ep["path"], "reason": "conflicting duplicate identity"}
                            for ep in copies)
            continue
        copies.sort(key=lambda ep: (ep["completion"] != "manifest", ep["path"]))
        counts["duplicate_episodes"] += len(copies) - 1
        episodes.append(copies[0])
    counts["accepted_episodes"] = len(episodes)
    return sorted(episodes, key=cell_key), {"counts": dict(counts), "rejected": rejected}


def demand_group(demand):
    return {1900: "light", 2100: "2100", 2400: "heavy", 2700: "heavy"}.get(
        demand, f"other:{demand}")


def penetration_group(pen):
    return "seen" if pen in SEEN else "unseen" if pen in UNSEEN else f"other:{pen:g}"


def aggregate(episodes: list[dict], ci: ConfidenceIntervals, *, grouped=False,
              native: dict | None = None):
    buckets = defaultdict(list)
    for episode in episodes:
        if native is not None and episode["controller"] == "native":
            continue
        d, p = episode["demand_vph"], episode["penetration"]
        key = (episode["controller"], demand_group(d), penetration_group(p)) if grouped else (
            episode["controller"], d, p)
        buckets[key].append(episode)
    rows = []
    for (controller, demand, penetration), members in sorted(buckets.items()):
        row = {"controller": controller,
               "demand_group" if grouped else "demand_vph": demand,
               "penetration_group" if grouped else "penetration": penetration,
               "n_episodes": len(members), "metrics": {}}
        if native is not None:
            row["reference"] = "native"
            row["difference"] = "treatment - native"
        for name in METRICS:
            per_seed = defaultdict(list)
            for ep in members:
                value = ep["metrics"][name]
                if value is None:
                    continue
                if native is not None:
                    reference = native.get((ep["demand_vph"], ep["penetration"], ep["seed"]))
                    other = reference["metrics"][name] if reference else None
                    if other is None:
                        continue
                    value -= other
                per_seed[ep["seed"]].append(value)
            result = ci.summarize({seed: statistics.mean(values)
                                   for seed, values in per_seed.items()})
            result["n_observations"] = sum(map(len, per_seed.values()))
            result["observations_per_seed"] = {str(seed): len(values)
                                                for seed, values in sorted(per_seed.items())}
            row["metrics"][name] = result
        rows.append(row)
    return rows


def build_report(roots: list[Path], *, use_scipy=True, driver_population=None) -> dict:
    episodes, diagnostics = load_episodes(roots, driver_population=driver_population)
    if not episodes:
        raise ValueError(f"no completed, unambiguous episodes: {diagnostics['counts']}")
    schedules = {(ep["warmup_s"], ep["horizon_s"]) for ep in episodes}
    if len(schedules) != 1:
        raise ValueError("mixed warmup/horizon schedules; generate separate reports")
    ci = ConfidenceIntervals(use_scipy)
    native = {(ep["demand_vph"], ep["penetration"], ep["seed"]): ep
              for ep in episodes if ep["controller"] == "native"}
    return {
        "schema": "c4_eval_v5_report/1",
        "methodology": {
            "ci": ci.method, "confidence": 0.95, "unit": "seed",
            "aggregation": "available observations averaged within seed, then equal-weight seeds",
            "pairing": "treatment - native on identical seed/demand/penetration; then within-seed mean",
            "paired_interpretation": PAIRED_INTERPRETATION,
            "driver_population": episodes[0]["driver_population"],
            "population_metadata": "per-episode model/version/provenance retained; population is a vehicle count",
            "population_declaration": driver_population,
            "missing": "null/nonfinite/absent values excluded per metric; n<2 CI is null",
            "duplicates": "one copy per controller/demand/penetration/seed; conflicts excluded",
            "coverage": "available-case means; inspect observations_per_seed for unbalanced grids",
            "completion": "complete manifest when present; otherwise summary schema and end timing",
            "wall_time": "summary.wall_seconds (environment episode runtime, not collector overhead)",
            "schedule": dict(zip(("warmup_s", "horizon_s"), next(iter(schedules)))),
        },
        "metric_sources": METRICS,
        "group_definitions": {"penetration": {"seen": SEEN, "unseen": UNSEEN},
                              "demand_vph": {"light": [1900], "2100": [2100],
                                             "heavy": [2400, 2700]}},
        "diagnostics": diagnostics,
        "cells": aggregate(episodes, ci),
        "groups": aggregate(episodes, ci, grouped=True),
        "paired_cells": aggregate(episodes, ci, native=native),
        "paired_groups": aggregate(episodes, ci, grouped=True, native=native),
        "episodes": episodes,
    }


def md_text(value) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def fmt(value) -> str:
    return "NA" if value is None else f"{value:.6g}"


def estimate(result) -> str:
    interval = result["ci95"]
    ci = "CI unavailable" if interval is None else f"[{fmt(interval[0])}, {fmt(interval[1])}]"
    return f"{fmt(result['mean'])} {ci}; n={result['n_seeds']}"


def render_markdown(report: dict) -> str:
    lines = ["# C4 eval v5 report", "", "## Method", ""]
    lines.extend(f"- **{key}**: {md_text(value)}" for key, value in report["methodology"].items())
    lines += ["", "Seen: 0.05, 0.1, 0.2, 0.4. Unseen: 0.15, 0.3, 0.6.",
              "Light: 1900 vph. Heavy: 2400, 2700 vph. 2100 is separate.",
              "All intervals are 95%; n counts seeds with available metric values.",
              "Minimum TTC is the mean of per-episode minima, not a pooled minimum.",
              "", "## Input diagnostics", ""]
    lines.extend(f"- {key}: {value}" for key, value in report["diagnostics"]["counts"].items())
    for item in report["diagnostics"]["rejected"]:
        lines.append(f"- Rejected `{md_text(item['path'])}`: {md_text(item['reason'])}")
    lines += ["", "## Metric schema", ""]
    lines.extend(f"- `{name}`: `{source}`" for name, source in METRICS.items())
    lines += ["", "## Compact smoke table", "",
              "Means over seeds; detailed intervals and metric-specific n follow below.", "",
              "| Controller | Demand | Penetration | n (system) | System s/request | Speed deficit | Arrivals | Collisions | Teleports | Wall s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["cells"]:
        metrics = row["metrics"]
        values = [md_text(row["controller"]), str(row["demand_vph"]), fmt(row["penetration"]),
                  str(metrics["system_seconds_per_requested_vehicle"]["n_seeds"])]
        values += [fmt(metrics[name]["mean"]) for name in (
            "system_seconds_per_requested_vehicle", "window_speed_deficit_vehicle_seconds",
            "arrivals", "collisions", "teleports", "wall_seconds")]
        lines.append("| " + " | ".join(values) + " |")
    for section, title in (("cells", "Controller × demand × penetration"),
                           ("groups", "Grouped estimates"),
                           ("paired_cells", "Paired cell differences (treatment − native)"),
                           ("paired_groups", "Paired grouped differences (treatment − native)")):
        lines += ["", f"## {title}", ""]
        if section.startswith("paired_"):
            lines += [PAIRED_INTERPRETATION, ""]
        if not report[section]:
            lines.append("No treatment rows available.")
        for row in report[section]:
            d = row.get("demand_vph", row.get("demand_group"))
            p = row.get("penetration", row.get("penetration_group"))
            lines += [f"### {md_text(row['controller'])} / {d} / {p}", "",
                      "| Metric | Mean [95% CI]; n seeds | Observations |",
                      "|---|---|---:|"]
            for name, result in row["metrics"].items():
                lines.append(f"| {name} | {estimate(result)} | {result['n_observations']} |")
            lines.append("")
    lines += ["## Per-episode wall time", "",
              "| Controller | Demand | Penetration | Seed | Wall seconds | Summary |",
              "|---|---:|---:|---:|---:|---|"]
    for ep in report["episodes"]:
        lines.append(f"| {md_text(ep['controller'])} | {ep['demand_vph']} | {ep['penetration']} | "
                     f"{ep['seed']} | {fmt(ep['metrics']['wall_seconds'])} | {md_text(ep['path'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="workspace-local eval output trees")
    parser.add_argument("--out", required=True, type=Path, help="directory for report.json/report.md")
    parser.add_argument("--no-scipy", action="store_true", help="force conservative stdlib t table")
    parser.add_argument("--driver-population", choices=("textbook", "legacy3"),
                        help="explicit population assertion for metadata-poor summaries; "
                             "must agree with all available population metadata")
    args = parser.parse_args(argv)
    try:
        out = safe_path(args.out)
        paths = [safe_path(out / name) for name in ("report.json", "report.md")]
        report = build_report(args.roots, use_scipy=not args.no_scipy,
                              driver_population=args.driver_population)
        payloads = [json.dumps(report, indent=2, allow_nan=False) + "\n", render_markdown(report)]
        out.mkdir(parents=True, exist_ok=True)
        for path, payload in zip(paths, payloads):
            path.write_text(payload, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Reported {len(report['episodes'])} episodes: {paths[0]} ; {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
