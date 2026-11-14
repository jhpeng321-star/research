"""Versioned decision-time safety envelopes for moving-wall exploration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


NO_SUPERVISOR = "none"
LOCAL_SUPPORT_DENSITY_COHERENCE_V1 = "local_support_density_coherence_v1"
SUPERVISOR_PROFILES = (
    NO_SUPERVISOR,
    LOCAL_SUPPORT_DENSITY_COHERENCE_V1,
)


@dataclass(frozen=True)
class LocalSupportDensityCoherenceV1:
    min_local_cav_count_100_200m: int = 2
    max_lag10_density_scaled: float = 0.80
    max_soft_band_cav_speed_std_mps: float = 7.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_predecision_supervisor(
    features: Mapping[str, Any],
    profile: str = NO_SUPERVISOR,
) -> dict[str, Any]:
    """Evaluate one declared supervisor using decision-time features only."""
    if profile == NO_SUPERVISOR:
        return {
            "profile": NO_SUPERVISOR,
            "admitted": True,
            "checks": {},
            "refusal_reasons": [],
            "thresholds": {},
        }
    if profile != LOCAL_SUPPORT_DENSITY_COHERENCE_V1:
        raise ValueError(f"unknown predecision supervisor profile {profile!r}")

    thresholds = LocalSupportDensityCoherenceV1()
    names = (
        "local_bin_100_200m_cav_count",
        "traffic_lag_10_density_scaled",
        "formation_soft_cav_speed_std_mps",
    )
    values = {}
    for name in names:
        try:
            value = float(features[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"supervisor feature {name!r} is unavailable") from error
        if not math.isfinite(value):
            raise ValueError(f"supervisor feature {name!r} is non-finite")
        values[name] = value

    checks = {
        "local_cav_support": (
            values["local_bin_100_200m_cav_count"] >= thresholds.min_local_cav_count_100_200m
        ),
        "density_not_deeply_saturated": (
            values["traffic_lag_10_density_scaled"] <= thresholds.max_lag10_density_scaled
        ),
        "cav_speed_field_coherent": (
            values["formation_soft_cav_speed_std_mps"] <= thresholds.max_soft_band_cav_speed_std_mps
        ),
    }
    refusal_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "profile": LOCAL_SUPPORT_DENSITY_COHERENCE_V1,
        "admitted": all(checks.values()),
        "checks": checks,
        "refusal_reasons": refusal_reasons,
        "thresholds": thresholds.to_dict(),
        "observed": values,
        "development_provenance": (
            "declared after excluded seeds 701--716 and before seeds 801--820"
        ),
    }
