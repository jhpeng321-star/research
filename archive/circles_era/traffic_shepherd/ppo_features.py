"""RSU observation builder for the PPO pilot (acquired evidence only).

Inputs are exactly the policy context a real RSU holds at the decision tick:
timestamped CAV self-state reports (fresh window 1 s) plus the pilot's own
phase bookkeeping. No global density, penetration, hidden driver type, future
sample, matched reference outcome, shadow rollout or hindsight event label
enters any feature. Missing values are encoded as 0.0 together with explicit
validity/coverage features (cohort_fresh_fraction, reporter_count,
feasible_activate_now); phase distinctions are one-hot so damping/assembly/
hold are never merged.

Feature table (name -> formula, unit, source, missing handling):
  cand_*          frozen cohort members' fresh reports (age <= 1 s).
                  Missing member report -> excluded from the statistic;
                  cohort_fresh_fraction records coverage; all-missing -> 0.0.
  all_*           all fresh reporting CAVs (any mapped edge).
  approach_*      fresh reporters with 0 <= distance_to_merge_m <= 400 m.
  downstream_*    fresh reporters with distance_to_merge_m < 0 (past merge).
  braking_reporter_count  reporters with acceleration < -3 m/s^2 (count).
  reporter_count / mean|max_report_age_s  RSU receipt bookkeeping (count, s).
  phase_*         one-hot of the pilot phase (inactive/assembly/hold; taper
                  reports as assembly=0/hold=0 with its own exit bookkeeping).
  elapsed_hold_s / remaining_permitted_hold_s  schedule state (s).
  time_since_start_s  episode clock (s). activation_used in {0,1}.
  *_delta_5s      difference to the previous decision tick's value (m/s);
                  first decision tick -> 0.0.
  feasible_activate_now  the cheap activation mask, also an explicit input.
"""
from statistics import fmean

from . import ppo_contract as C

APPROACH_WINDOW_M = (0.0, 400.0)


def _stats(values):
    if not values:
        return 0.0, 0.0, 0.0
    return fmean(values), min(values), max(values) - min(values)


def _pvariance(values):
    n = len(values)
    if n < 2:
        return 0.0
    m = fmean(values)
    return sum((v - m) ** 2 for v in values) / n


def build_features(policy, context, previous=None):
    """Raw feature dict in ppo_contract.FEATURE_NAMES order (no normalization)."""
    now = context.now_s
    fresh = [r for r in context.reports
             if 0.0 <= now - r.frame.timestamp_s <= 1.0]
    by_id = {r.frame.vehicle_id: r for r in fresh}
    # the cohort the features describe: fixed member_ids for wall_ppo/2;
    # the offered/approved/committed cohort for the activation-time selector
    # (wall_ppo/3) — never zero candidate features while a valid candidate
    # is offered
    member_ids = policy.active_member_ids()
    cohort = [by_id[v] for v in member_ids if v in by_id]

    cand_speeds = [r.frame.speed_mps for r in cohort]
    cand_mean, cand_min, cand_spread = _stats(cand_speeds)
    cand_accel = fmean([r.frame.acceleration_mps2 for r in cohort]) if cohort else 0.0
    distances = [r.frame.distance_to_merge_m for r in cohort
                 if r.frame.distance_to_merge_m is not None]
    cand_span = max(distances) - min(distances) if len(distances) >= 2 else 0.0
    gaps = [r.frame.front.gap_m for r in cohort if r.frame.front is not None]
    rel = [r.frame.front.relative_speed_mps for r in cohort if r.frame.front is not None]

    all_speeds = [r.frame.speed_mps for r in fresh]
    all_mean, all_min, all_spread = _stats(all_speeds)
    all_accel = fmean([r.frame.acceleration_mps2 for r in fresh]) if fresh else 0.0

    def region(lo_hi):
        lo, hi = lo_hi
        return [r.frame.speed_mps for r in fresh if r.frame.distance_to_merge_m is not None
                and lo <= r.frame.distance_to_merge_m <= hi]

    approach = region(APPROACH_WINDOW_M)
    downstream = [r.frame.speed_mps for r in fresh
                  if r.frame.distance_to_merge_m is not None and r.frame.distance_to_merge_m < 0]
    ages = [now - r.frame.timestamp_s for r in fresh]
    phase = policy.phase_now()
    elapsed_hold = 0.0
    if policy.ready_at is not None:
        elapsed_hold = max(0.0, (policy.taper_at if policy.taper_at is not None else now)
                           - policy.ready_at)
    mask, kind = policy.decision_context(context)
    mask = mask or [False, False]
    kind = kind or ""

    def delta(name, value):
        return 0.0 if previous is None else value - previous.get(name, 0.0)

    values = {
        "cand_mean_speed_mps": cand_mean,
        "cand_min_speed_mps": cand_min,
        "cand_speed_spread_mps": cand_spread,
        "cand_mean_accel_mps2": cand_accel,
        "cand_span_m": cand_span,
        "cand_mean_gap_front_m": fmean(gaps) if gaps else 0.0,
        "cand_mean_rel_speed_front_mps": fmean(rel) if rel else 0.0,
        "all_mean_speed_mps": all_mean,
        "all_speed_pvariance_mps2": _pvariance(all_speeds),
        "all_mean_accel_mps2": all_accel,
        "all_min_speed_mps": all_min,
        "approach_mean_speed_mps": fmean(approach) if approach else 0.0,
        "approach_reporter_count": float(len(approach)),
        "downstream_mean_speed_mps": fmean(downstream) if downstream else 0.0,
        "braking_reporter_count": float(sum(1 for r in fresh if r.frame.acceleration_mps2 < -3.0)),
        "reporter_count": float(len(fresh)),
        "mean_report_age_s": fmean(ages) if ages else 0.0,
        "max_report_age_s": max(ages) if ages else 0.0,
        "cohort_fresh_fraction": len(cohort) / len(member_ids) if member_ids else 0.0,        "phase_inactive": 1.0 if phase == "inactive" else 0.0,
        "phase_assembly": 1.0 if phase in ("assembly", "taper", "terminal") else 0.0,
        "phase_hold": 1.0 if phase == "hold" else 0.0,
        "elapsed_hold_s": elapsed_hold,
        "admitted_total_hold_s": float(policy.profile.hold_s)
        if hasattr(policy, "profile") else 0.0,
        "remaining_admitted_hold_s": max(
            0.0, float(policy.profile.hold_s) - elapsed_hold)
        if hasattr(policy, "profile") else 0.0,
        "remaining_extension_capacity_s": max(
            0.0, C.MAX_HOLD_S - float(getattr(policy, "profile").hold_s))
        if hasattr(policy, "profile") else 0.0,
        "hold_deadline_decision": 1.0 if kind == "hold_deadline" else 0.0,
        "time_since_start_s": now,
        "activation_used": float(policy.activation_used),
        "cand_mean_speed_delta_5s": delta("cand_mean_speed_mps", cand_mean),
        "all_mean_speed_delta_5s": delta("all_mean_speed_mps", all_mean),
        "feasible_activate_now": float(mask[C.A_ACTIVATE]) if phase == "inactive" else 0.0,
    }
    assert set(values) == set(C.FEATURE_NAMES)
    return [values[name] for name in C.FEATURE_NAMES], values


class RunningNormalizer:
    """Welford mean/variance over TRAINING decision steps only; frozen for eval."""

    def __init__(self, dim, clamp=8.0):
        self.dim, self.clamp = dim, clamp
        self.count = 0
        self.mean = [0.0] * dim
        self.m2 = [0.0] * dim

    def update(self, vector):
        self.count += 1
        for i, x in enumerate(vector):
            delta = x - self.mean[i]
            self.mean[i] += delta / self.count
            self.m2[i] += delta * (x - self.mean[i])

    def normalize(self, vector):
        if self.count < 2:
            return [0.0] * self.dim
        out = []
        for i, x in enumerate(vector):
            var = self.m2[i] / (self.count - 1)
            std = var ** 0.5 if var > 1e-12 else 1.0
            out.append(max(-self.clamp, min(self.clamp, (x - self.mean[i]) / std)))
        return out

    def state_dict(self):
        return dict(count=self.count, mean=list(self.mean), m2=list(self.m2), clamp=self.clamp)

    @classmethod
    def from_state(cls, state):
        normalizer = cls(len(state["mean"]), clamp=state["clamp"])
        normalizer.count = state["count"]
        normalizer.mean = list(state["mean"])
        normalizer.m2 = list(state["m2"])
        return normalizer
