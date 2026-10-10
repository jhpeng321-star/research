"""Prespecified development selection; safety/service cannot be averaged away."""

from dataclasses import asdict, dataclass
import math

from idm.damping_campaign import DEVELOPMENT_CASES


@dataclass(frozen=True)
class SelectionContract:
    lower_mean_composite_improvement: float = 0.05
    lower_mean_component_worsening: float = 0.0
    overload_mean_component_worsening: float = 0.02
    maximum_case_composite_worsening: float = 0.10
    system_time_margin_s_per_vehicle: float = 2.0
    source_wait_margin_s_per_vehicle: float = 2.0
    arrival_shortfall_margin: int = 1
    ttc_relative_margin: float = 0.05
    ttc_absolute_margin_vehicle_seconds: float = 5.0

    def manifest(self):
        return dict(
            schema="native_damping_development_selection/1",
            **asdict(self),
            ranking="eligible first; lower mean composite, then system time, then earlier checkpoint, then optimizer seed",
            ttc_meaning="longitudinal closing-time proxy; chosen noninferiority margin, not a safety theorem",
        )


def case_metrics(actual, native):
    for key in ("scenario_identity", "seed", "population", "global_control_steps", "team_terminal"):
        if actual[key] != native[key]:
            raise ValueError(f"paired selection identity mismatch: {key}")
    if not actual["team_terminal"] or actual["global_control_steps"] != 2400:
        raise ValueError("selection requires complete team episodes")
    improvements = {}
    for name, key in (("braking", "braking_vehicle_seconds"), ("energy", "acceleration_energy")):
        a, b = actual["totals"][key], native["totals"][key]
        if b <= 0:
            raise ValueError("selection requires positive native physical references")
        improvements[name] = 1 - a / b
    population = native["population"]
    result = dict(
        **improvements,
        composite=(improvements["braking"] + improvements["energy"]) / 2,
        system_time=actual["system_seconds_per_requested_vehicle"]
        - native["system_seconds_per_requested_vehicle"],
        source_wait=(actual["source_wait_vehicle_seconds"] - native["source_wait_vehicle_seconds"])
        / population,
        arrived=actual["arrived"] - native["arrived"],
        uninserted=actual["uninserted_due"] - native["uninserted_due"],
        unfinished=actual["unfinished_requested"] - native["unfinished_requested"],
        ttc=actual["ttc_below_3_vehicle_seconds"] - native["ttc_below_3_vehicle_seconds"],
        native_ttc=native["ttc_below_3_vehicle_seconds"],
        collisions=actual["totals"]["added_collision_events"]
        - native["totals"]["added_collision_events"],
        teleports=actual["totals"]["added_teleport_events"]
        - native["totals"]["added_teleport_events"],
    )
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite physical selection outcome")
    return result


def assess_candidate(rows, contract=SelectionContract()):
    if len(rows) != len(DEVELOPMENT_CASES) or {tuple(row["case"]) for row in rows} != set(
        DEVELOPMENT_CASES
    ):
        raise ValueError("selection needs every registered development case exactly once")
    failures, metrics = [], []
    for row in rows:
        m = case_metrics(row["policy"], row["native"])
        metrics.append({"case": row["case"], **m})
        checks = dict(
            hard_events=m["collisions"] <= 0 and m["teleports"] <= 0,
            system_time=m["system_time"] <= contract.system_time_margin_s_per_vehicle,
            source_wait=m["source_wait"] <= contract.source_wait_margin_s_per_vehicle,
            completions=m["arrived"] >= -contract.arrival_shortfall_margin,
            unfinished=m["unfinished"] <= contract.arrival_shortfall_margin,
            source_backlog=m["uninserted"] <= 0,
            ttc=m["ttc"]
            <= max(
                contract.ttc_absolute_margin_vehicle_seconds,
                contract.ttc_relative_margin * m["native_ttc"],
            ),
            harmful_tail=m["composite"] >= -contract.maximum_case_composite_worsening,
        )
        failures.extend(
            {"case": row["case"], "constraint": name} for name, ok in checks.items() if not ok
        )

    def mean(group, key):
        return sum(row[key] for row in group) / len(group)

    lower = [row for row in metrics if row["case"][1] <= 2200]
    overload = [row for row in metrics if row["case"][1] >= 2600]
    group_means = {
        name: {key: mean(group, key) for key in ("braking", "energy", "composite", "system_time")}
        for name, group in (("lower", lower), ("overload", overload))
    }
    if group_means["lower"]["composite"] < contract.lower_mean_composite_improvement:
        failures.append({"constraint": "lower_mean_improvement"})
    for group, margin in (
        ("lower", contract.lower_mean_component_worsening),
        ("overload", contract.overload_mean_component_worsening),
    ):
        for component in ("braking", "energy"):
            if group_means[group][component] < -margin:
                failures.append({"constraint": f"{group}_{component}_noninferiority"})
    return dict(
        eligible=not failures,
        failures=failures,
        case_metrics=metrics,
        group_means=group_means,
        accepted_checkpoint=False,
        prospective=False,
    )


def select_candidate(candidates):
    eligible = [row for row in candidates if row["assessment"]["eligible"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -row["assessment"]["group_means"]["lower"]["composite"],
            row["assessment"]["group_means"]["lower"]["system_time"],
            row["optimized_global_steps"],
            row["optimizer_seed"],
        ),
    )
