"""Continuous "textbook" human-driver population.

Replaces the three discrete legacy HDV vTypes (aggressive_car 10 % / normal_car 70 % /
cautious_car 20 %, ``scenarios/idm/driver_routes.rou.xml``) with one vType PER HDV whose IDM and
LC2013 parameters are drawn from continuous distributions centred on the literature IDM highway
values.  The legacy mix stays the default everywhere (``population="legacy3"``); this module is
used only when ``population="textbook"`` is passed explicitly.

IDM centre values -- Treiber, M. & Kesting, A. (2013) *Traffic Flow Dynamics: Data, Models and
Simulation*, Springer, Table 11.2 (IDM, highway): desired speed v0 = speed limit, time gap
T = 1.0-1.5 s (1.2 s used here), max acceleration a = 1.0-1.5 m/s^2 (1.2), comfortable
deceleration b = 2.0 m/s^2, minimum gap s0 = 2 m, acceleration exponent delta = 4; see also
Treiber, Hennecke & Helbing (2000), Phys. Rev. E 62, 1805.  The spreads below are chosen
(2026-10-14) to give a plausible continuous heterogeneity around those medians:

=====================  ============================================  ===============
parameter (SUMO attr)  distribution                                  truncation
=====================  ============================================  ===============
speedFactor            N(1.0, 0.1)                                   [0.8, 1.2]
T (tau)                lognormal, median 1.2 s, 10-90 % 0.9-1.6 s    [0.7, 2.0]
a (accel)              lognormal, median 1.2, 10-90 % 0.8-1.8        [0.6, 2.5]
b (decel)              lognormal, median 2.0, 10-90 % 1.5-2.8        [1.2, 3.5]
s0 (minGap)            N(2.0, 0.3)                                   [1.2, 3.0]
delta                  4 (SUMO IDM default)                          --
lcCooperative          U[0.3, 1.0]
lcAssertive            U[0.5, 1.5]
lcSpeedGain            U[0.5, 1.5]
impatience             U[0.0, 0.6]
=====================  ============================================  ===============

Lognormal sigma = ln(q90 / q10) / (2 * 1.2816) (0.2245 for T, 0.3164 for a, 0.2435 for b).
Correlation: (ln a, ln 1/T) is bivariate normal with rho = 0.4 before truncation (an aggressive
driver both accelerates harder and follows closer); all other parameters are independent.
Truncation = rejection resampling from the same per-vehicle stream (the (a, T) pair is resampled
jointly), so medians and the correlation sign are preserved.  Desired speed = route speed limit x
speedFactor (``speedDev`` frozen to 0 as in the legacy fixture, so SUMO adds no extra draw).

SUMO details: ``carFollowModel="IDM"`` ignores ``sigma`` (no dawdling in MSCFModel_IDM; sigma
is written as 0 anyway); ``emergencyDecel`` = 9 m/s^2 (SUMO passenger default), so a vehicle may
brake beyond its comfortable ``decel`` to avoid a collision.  CAVs run speedMode 27 (bit 2 = 0:
deceleration bounds are NOT enforced on commands), so a CAV's b = 2.0 never limits its braking.

CAVs use one vType ``cav_typical`` with the population
MEDIANS (:data:`TEXTBOOK_CAV_PARAMS`).  The per-vehicle parameters are recorded as a truth table
(``scenario/hetdata_truth.csv``, the ``wms.data.driver_truth`` format) for EVALUATION ONLY --
never a model input, never in a training loss.

Draws are seeded per vehicle identity (``textbook_driver_v1:{seed}:{identity}``), independent of
the departure, CAV-identity and legacy type streams, so departures and CAV identities are
identical to the legacy population for the same seed.

Stdlib only (importable from both the simulator side and ``wms`` without numpy).
"""
from __future__ import annotations

import math
import random

POPULATIONS = ("legacy3", "textbook")
POPULATION_VERSION = "textbook_v1"
TEXTBOOK_CAV_VTYPE = "cav_typical"
#: vType id prefix of textbook HDVs: ``tb.<vehicle_id>`` (e.g. ``tb.mainline.12``).
TEXTBOOK_HDV_PREFIX = "tb."

#: Population medians for the WM driver prior (keys = wms driver-encoder parameter names;
#: ``speed_factor`` multiplies the route speed limit to give v0).
TEXTBOOK_POPULATION_MEDIANS = {
    "accel": 1.2,          # m/s^2   IDM a
    "time_headway": 1.2,   # s       IDM T (SUMO tau)
    "decel": 2.0,          # m/s^2   IDM b (SUMO decel)
    "min_gap": 2.0,        # m       IDM s0 (SUMO minGap)
    "speed_factor": 1.0,   # -       v0 = speed limit x speed_factor
    "delta": 4.0,          # -       IDM exponent (fixed)
}

#: Numeric parameters of the single CAV vType in textbook mode (population medians).  The CAV
#: execution cap ("idm_normal") must use exactly these values in textbook mode.
TEXTBOOK_CAV_PARAMS = {
    "speed_factor": 1.0,
    "tau_s": 1.2,
    "min_gap_m": 2.0,
    "accel_mps2": 1.2,
    "decel_mps2": 2.0,
    "delta": 4.0,
    "lc_strategic": 1.0,
    "lc_cooperative": 0.65,
    "lc_speed_gain": 1.0,
    "lc_assertive": 1.0,
    "impatience": 0.3,
}

#: Truth-table parameter order (= ``wms.data.driver_truth.PARAM_ORDER``).
PARAM_ORDER = ("speed_factor", "tau_s", "min_gap_m", "accel_mps2", "decel_mps2",
               "delta", "lc_strategic", "lc_cooperative", "lc_speed_gain",
               "lc_assertive", "impatience")

EMERGENCY_DECEL_MPS2 = 9.0
_Z90 = 1.2815515655446004            # standard-normal 90 % quantile
RHO_ACCEL_INV_TAU = 0.4
BOUNDS = {
    "speed_factor": (0.8, 1.2),
    "tau_s": (0.7, 2.0),
    "accel_mps2": (0.6, 2.5),
    "decel_mps2": (1.2, 3.5),
    "min_gap_m": (1.2, 3.0),
}
_LOGNORMAL = {   # name -> (median, q10, q90)
    "tau_s": (1.2, 0.9, 1.6),
    "accel_mps2": (1.2, 0.8, 1.8),
    "decel_mps2": (2.0, 1.5, 2.8),
}
UNIFORM = {
    "lc_cooperative": (0.3, 1.0),
    "lc_assertive": (0.5, 1.5),
    "lc_speed_gain": (0.5, 1.5),
    "impatience": (0.0, 0.6),
}
_MAX_TRIES = 1000
_DIGITS = 4      # values are rounded once; the truth table equals what SUMO receives


def _sigma(name: str) -> float:
    _, lo, hi = _LOGNORMAL[name]
    return math.log(hi / lo) / (2.0 * _Z90)


def _within(name: str, value: float) -> bool:
    lo, hi = BOUNDS[name]
    return lo <= value <= hi


def _truncated_normal(rng: random.Random, name: str, mean: float, sd: float) -> float:
    for _ in range(_MAX_TRIES):
        value = rng.gauss(mean, sd)
        if _within(name, value):
            return value
    lo, hi = BOUNDS[name]
    return min(max(mean, lo), hi)


def _truncated_lognormal(rng: random.Random, name: str) -> float:
    median = _LOGNORMAL[name][0]
    for _ in range(_MAX_TRIES):
        value = median * math.exp(_sigma(name) * rng.gauss(0.0, 1.0))
        if _within(name, value):
            return value
    return median


def _accel_tau(rng: random.Random) -> tuple[float, float]:
    """Jointly truncated (a, T) with corr(ln a, ln 1/T) = rho before truncation."""
    rho = RHO_ACCEL_INV_TAU
    a_med, t_med = _LOGNORMAL["accel_mps2"][0], _LOGNORMAL["tau_s"][0]
    for _ in range(_MAX_TRIES):
        z_a = rng.gauss(0.0, 1.0)
        z_inv_t = rho * z_a + math.sqrt(1.0 - rho * rho) * rng.gauss(0.0, 1.0)
        accel = a_med * math.exp(_sigma("accel_mps2") * z_a)
        tau = t_med * math.exp(-_sigma("tau_s") * z_inv_t)
        if _within("accel_mps2", accel) and _within("tau_s", tau):
            return accel, tau
    return a_med, t_med


def sample_driver(seed: int, identity: str) -> dict:
    """Parameters (:data:`PARAM_ORDER` keys) of one textbook HDV; deterministic per (seed, id)."""
    rng = random.Random(f"textbook_driver_v1:{seed}:{identity}")
    accel, tau = _accel_tau(rng)
    decel = _truncated_lognormal(rng, "decel_mps2")
    speed_factor = _truncated_normal(rng, "speed_factor", 1.0, 0.1)
    min_gap = _truncated_normal(rng, "min_gap_m", 2.0, 0.3)
    lc = {name: rng.uniform(lo, hi) for name, (lo, hi) in UNIFORM.items()}
    raw = dict(speed_factor=speed_factor, tau_s=tau, min_gap_m=min_gap, accel_mps2=accel,
               decel_mps2=decel, delta=4.0, lc_strategic=1.0, **lc)
    return {name: round(float(raw[name]), _DIGITS) for name in PARAM_ORDER}


def vtype_attrs(vtype_id: str, params: dict) -> dict:
    """SUMO ``<vType>`` attributes (strings) for one parameter dict."""
    def fmt(x: float) -> str:
        return repr(round(float(x), _DIGITS))
    return {
        "id": vtype_id,
        "carFollowModel": "IDM",
        "accel": fmt(params["accel_mps2"]),
        "decel": fmt(params["decel_mps2"]),
        "emergencyDecel": fmt(EMERGENCY_DECEL_MPS2),
        "tau": fmt(params["tau_s"]),
        "minGap": fmt(params["min_gap_m"]),
        "delta": fmt(params["delta"]),
        "sigma": "0",
        "speedFactor": fmt(params["speed_factor"]),
        "speedDev": "0",
        "lcStrategic": fmt(params["lc_strategic"]),
        "lcCooperative": fmt(params["lc_cooperative"]),
        "lcSpeedGain": fmt(params["lc_speed_gain"]),
        "lcAssertive": fmt(params["lc_assertive"]),
        # same attribute as the legacy vTypes (SUMO ``impatience``; not ``lcImpatience``)
        "impatience": fmt(params["impatience"]),
    }


def hdv_vtype_id(vehicle_id: str) -> str:
    return TEXTBOOK_HDV_PREFIX + vehicle_id


def check_population(population: str) -> str:
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {POPULATIONS}, got {population!r}")
    return population
