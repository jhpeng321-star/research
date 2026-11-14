"""Prospective damping criteria and paired uncertainty, with no simulation API.

This analysis cannot authorize seed generation or mark inference weights accepted.
It must be called on a complete, provenance-verified prospective cohort.
"""

from dataclasses import asdict, dataclass

import numpy as np

from idm.damping_campaign import ACCEPTANCE_CASES
from idm.damping_selection import SelectionContract, case_metrics


@dataclass(frozen=True)
class AcceptanceContract(SelectionContract):
    bootstrap_draws: int = 20000
    bootstrap_seed: int = 37
    interval_tail_probability: float = 0.025
    analytical_composite_noninferiority_margin: float = 0.02

    def manifest(self):
        return dict(
            schema="native_damping_prospective_acceptance/1",
            **asdict(self),
            uncertainty="paired bootstrap of seeds, stratified by loading, fixed equal stratum weights",
            analytical_margin_units="fraction of native equal-weight braking/energy burden",
            inference="nominal bootstrap interval, not a probability of controller safety",
        )


def stratified_interval(values, loadings, contract=AcceptanceContract()):
    values, loadings = np.asarray(values, dtype=float), np.asarray(loadings)
    if values.ndim != 1 or loadings.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("invalid paired bootstrap inputs")
    if (
        not len(values)
        or contract.bootstrap_draws < 1000
        or not 0 < contract.interval_tail_probability < 0.5
    ):
        raise ValueError("invalid bootstrap specification")
    groups = [values[loadings == loading] for loading in sorted(set(loadings.tolist()))]
    if any(len(group) < 2 for group in groups):
        raise ValueError("each loading needs independent paired seeds")
    rng = np.random.default_rng(contract.bootstrap_seed)
    means = np.zeros(contract.bootstrap_draws)
    for group in groups:
        sample = rng.integers(0, len(group), size=(contract.bootstrap_draws, len(group)))
        means += group[sample].mean(axis=1) / len(groups)
    lower, upper = np.quantile(
        means, [contract.interval_tail_probability, 1 - contract.interval_tail_probability]
    )
    return dict(
        mean=float(np.mean([group.mean() for group in groups])),
        lower=float(lower),
        upper=float(upper),
        seed_pairs=len(values),
        strata=len(groups),
        draws=contract.bootstrap_draws,
    )


def assess_acceptance(rows, contract=AcceptanceContract()):
    if len(rows) != len(ACCEPTANCE_CASES) or {tuple(row["case"]) for row in rows} != set(
        ACCEPTANCE_CASES
    ):
        raise ValueError("acceptance requires all 32 reserved cases exactly once")
    metrics, failures = [], []
    for row in rows:
        seed, loading = row["case"]
        if any(row[arm]["seed"] != seed for arm in ("policy", "native", "analytical")):
            raise ValueError("case label does not match trajectory seed")
        actual = case_metrics(row["policy"], row["native"])
        analytical = case_metrics(row["analytical"], row["native"])
        metrics.append(
            dict(
                case=[seed, loading],
                **actual,
                analytical_composite=analytical["composite"],
                learned_minus_analytical_improvement=actual["composite"] - analytical["composite"],
            )
        )
        checks = dict(
            hard_events=actual["collisions"] <= 0 and actual["teleports"] <= 0,
            system_time=actual["system_time"] <= contract.system_time_margin_s_per_vehicle,
            source_wait=actual["source_wait"] <= contract.source_wait_margin_s_per_vehicle,
            completions=actual["arrived"] >= -contract.arrival_shortfall_margin,
            unfinished=actual["unfinished"] <= contract.arrival_shortfall_margin,
            source_backlog=actual["uninserted"] <= 0,
            ttc=actual["ttc"]
            <= max(
                contract.ttc_absolute_margin_vehicle_seconds,
                contract.ttc_relative_margin * actual["native_ttc"],
            ),
            harmful_tail=actual["composite"] >= -contract.maximum_case_composite_worsening,
        )
        failures.extend(
            dict(case=[seed, loading], constraint=name) for name, ok in checks.items() if not ok
        )
    lower = [row for row in metrics if row["case"][1] <= 2200]
    overload = [row for row in metrics if row["case"][1] >= 2600]
    estimates = {}
    for name, group in (("lower", lower), ("overload", overload)):
        loads = [row["case"][1] for row in group]
        estimates[name] = {
            key: stratified_interval([row[key] for row in group], loads, contract)
            for key in (
                "braking",
                "energy",
                "composite",
                "system_time",
                "learned_minus_analytical_improvement",
            )
        }
    if estimates["lower"]["composite"]["mean"] < contract.lower_mean_composite_improvement:
        failures.append(dict(constraint="lower_mean_improvement"))
    if estimates["lower"]["composite"]["lower"] <= 0:
        failures.append(dict(constraint="lower_improvement_uncertainty"))
    if (
        estimates["lower"]["learned_minus_analytical_improvement"]["lower"]
        < -contract.analytical_composite_noninferiority_margin
    ):
        failures.append(dict(constraint="analytical_comparator_noninferiority"))
    for name, margin in (
        ("lower", contract.lower_mean_component_worsening),
        ("overload", contract.overload_mean_component_worsening),
    ):
        for component in ("braking", "energy"):
            if estimates[name][component]["mean"] < -margin:
                failures.append(dict(constraint=f"{name}_{component}_noninferiority"))
    return dict(
        schema="native_damping_acceptance_analysis/1",
        contract=contract.manifest(),
        passes_statistical_and_physical_rules=not failures,
        failures=failures,
        estimates=estimates,
        case_metrics=metrics,
        accepted_checkpoint=False,
        requires="complete trajectory/bundle provenance verification before freezing D*",
    )
