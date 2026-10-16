"""INFRAV5 optional driver-estimate prior: a population anchor for the driver encoder.

The v4 diagnosis (runs/scratch/v4speed, MORNING_WM_DIAG_NOTES.md) found the encoded
settings drifting conservative (a ~1.8 vs SUMO ~2.5, T ~1.4 vs ~1.17, b ~3.3 vs ~4.2).  This
term pulls only the BATCH MEAN of the encoded settings (over valid, observed t0 HDV slots)
toward population targets; deviations of individual drivers from the batch mean are not
penalised, so per-driver identification stays free.  Weight 0 (default) = never built.

    prior = weight * sum_p ((mean_batch(x_p) - target_p) / target_p) ** 2
    p in {accel, time_headway, decel, min_gap, v0 / speed_limit}
"""
from __future__ import annotations

from typing import Mapping

from torch import Tensor

from wms.model.driver_encoder import PARAMETER_NAMES

#: Encoded settings anchored by the prior (``v0`` enters as v0 / speed limit).
PRIOR_PARAMETERS = ("accel", "time_headway", "decel", "min_gap", "v0")
#: Medians of the legacy SUMO driver population (scenarios/idm/driver_routes.rou.xml:
#: normal_car 70 % / aggressive 10 % / cautious 20 % -> every median is the normal_car value).
LEGACY_POPULATION_MEDIANS = {"accel": 2.6, "time_headway": 1.1, "decel": 4.5, "min_gap": 2.5,
                             "v0": 1.0}
#: Medians of the textbook population (c4/popfix).  TODO(c4/popfix): import
#: ``TEXTBOOK_POPULATION_MEDIANS`` from ``traffic_shepherd/c4/population.py`` once merged
#: (``speed_factor`` there = the v0 / speed-limit ratio here); tests/unit/test_infrav5.py checks
#: this copy against it when importable.
TEXTBOOK_POPULATION_MEDIANS = {"accel": 1.2, "time_headway": 1.2, "decel": 2.0, "min_gap": 2.0,
                               "v0": 1.0}
POPULATION_MEDIANS = {"legacy": LEGACY_POPULATION_MEDIANS,
                      "textbook": TEXTBOOK_POPULATION_MEDIANS}


def prior_targets(settings) -> dict[str, float]:
    """Targets ``{name: value}`` from TrainSettings (``v0`` is the v0 / speed-limit ratio)."""
    target = getattr(settings, "driver_prior_target", "explicit")
    if target != "explicit":
        return dict(POPULATION_MEDIANS[target])
    return {"accel": float(settings.driver_prior_accel),
            "time_headway": float(settings.driver_prior_time_headway),
            "decel": float(settings.driver_prior_decel),
            "min_gap": float(settings.driver_prior_min_gap),
            "v0": float(settings.driver_prior_v0_ratio)}


def driver_prior_loss(values: Tensor, mask: Tensor, targets: Mapping[str, float], *,
                      weight: float, speed_limit_mps: float) -> tuple[Tensor, dict[str, float]]:
    """``(weighted prior, batch means)`` of encoded settings ``values [B, K, 9]`` over ``mask``.

    Returns a zero (graph-connected) tensor when no slot is valid.
    """
    m = mask.to(values.device).bool()
    count = m.sum()
    total = values.new_zeros(())
    means: dict[str, float] = {}
    if int(count) == 0:
        return total + 0.0 * values.sum(), means
    w = m.to(values.dtype)
    for name in PRIOR_PARAMETERS:
        column = values[..., PARAMETER_NAMES.index(name)]
        if name == "v0":
            column = column / float(speed_limit_mps)
        mean = (column * w).sum() / count.to(values.dtype)
        target = float(targets[name])
        total = total + ((mean - target) / target) ** 2
        means["v0_ratio" if name == "v0" else name] = float(mean.detach())
    return float(weight) * total, means


def prior_mask(present0: Tensor, is_cav0: Tensor, observed_mask=None) -> Tensor:
    """Valid t0 HDV slots with at least one observed history step (when a mask exists)."""
    mask = present0.bool() & ~is_cav0.bool()
    if observed_mask is not None:
        mask = mask & observed_mask.bool().reshape(*mask.shape, -1).any(-1)
    return mask

