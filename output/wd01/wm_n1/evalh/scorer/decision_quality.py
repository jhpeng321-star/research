"""Decision-quality (regret) scoring for the WM-N1 scorer (owner-approved).

A **decision point** is a ``(physical_world, t0)`` window at one ``tau`` on which
at least two candidate controller arms carry a realized label.  For a registered
cost (``delay_s`` primary, ``hard_brake_s`` secondary) the module compares five
policies:

* ``wm_mean``  -- choose the arm with the lowest **model-predicted** expected
  cost, where the point prediction is the mean of the 19 predicted quantiles;
* ``wm_risk``  -- the risk-averse variant, the lowest predicted 0.9 quantile;
* ``oracle``   -- the lowest realized cost (regret reference);
* ``fixed_best`` -- the single arm type with the lowest TRAIN-only mean cost
  (frozen artifact, SHA-pinned like ``TRAIN_CLIMATOLOGY.json``); at a decision
  point the best *available* arm by the frozen TRAIN ranking is chosen;
* ``random``   -- the exact expectation over a uniform arm choice;
* ``native``   -- always ``native``.

**Regret** = realized cost of the chosen arm minus the oracle cost.  Means are
reported with world-cluster bootstrap 90% CIs (cluster key = ``physical_world``,
exactly as :func:`evalh.scorer.metrics_loop.physical_world`).  The primary claim
test is WM regret below fixed-best regret at tau 30-120.

Fail-closed: a decision point whose arms disagree on identity (duplicate
canonical arm, non-finite t0) or that has fewer than two finite labelled arms is
refused and counted (never repaired or zero-filled).  The WM path requires a
finite predicted vector per arm; a missing one makes that point unscoreable for
WM and it is counted, not imputed.

This module never opens CAL / HOLDOUT / chunk-5 / J4B / INC1 data; its own tests
and the TRAIN-only dry run use generated fixtures and the corrected TRAIN label
sidecar only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from evalh.metrics import level_index, world_bootstrap_ci
from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS

SCHEMA = "wm_n1_decision_quality/1"
SCHEMA_FIXED_BEST = "wm_n1_decision_quality_fixed_best/1"

#: Primary reporting taus and the owner-secondary taus.
PRIMARY_TAUS = (30.0, 60.0, 120.0)
SECONDARY_TAUS = (300.0, 570.0)
TAUS_S = PRIMARY_TAUS + SECONDARY_TAUS

#: Registered costs -> the ``OutcomeRow`` attribute holding the realized value.
COST_ATTRS = {"delay_s": "y_delay_s", "hard_brake_s": "hard_brake_s"}
COSTS = ("delay_s", "hard_brake_s")

#: Canonical controller arms admitted by DEFAULT, matching the registered
#: scorer universe (``build_climatology`` / the paired-dV metric loop).  The
#: owner brief additionally names fixed / switching arms (and later
#: direct-speed "trajectory" arms); those are listed as :data:`FUTURE_ARMS` and
#: become candidates only when explicitly registered via ``allowed_arms``,
#: because the current TRAIN sidecar labels them through several distinct
#: sources with no canonical-registry entry (they are not one-arm-per-decision).
ALLOWED_ARMS = ("native", "damping", "pi", "wall")
FUTURE_ARMS = ("fixed", "switching", "trajectory")

#: Fix 3 (2026-10-05) ``--dq-arms e2`` candidate construction.
#: * :data:`TRAJECTORY_ARMS` are separate candidates (own realized + own plan
#:   prediction), never pooled into a ``trajectory`` candidate;
#: * :data:`E2_POOL_ARMS` are random mixes pooled by mean;
#: * ``native``/``pi`` duplicates across the released campaign and ``truearms``
#:   shards are de-duplicated (preferring the ``truearms`` true-fleet row);
#: * ``trajectory`` remains a *fixed-best arm type* only.
TRAJECTORY_ARMS = ("arm0", "arm1", "arm2", "arm3")
E2_POOL_ARMS = frozenset({"fixed_mix", "switching_mix", "wall"})
FIXED_BEST_TRAJECTORY_TYPE = "trajectory"

#: Owner rule-(b) arms whose outcome supervision was removed in training
#: (``train/dataset.py:124``); they never enter a decision point.  Applied on
#: the row's ``source``/``arm`` metadata only, exactly like the scorer.
EXCLUDED_SOURCES = ("base", "native_base")

#: Pinned SHA-256 of the corrected TRAIN-only label sidecar used by the
#: zero-simulation dry run (labels_fix1_20261003T141730Z/train_outcomes.parquet).
EXPECTED_TRAIN_OUTCOMES_SHA256 = (
    "1e0ec6de0e2cdfd386fd84165879cac88b21d8cf16277d71108668d2306839ef")
TRAIN_OUTCOMES_FILENAME = "train_outcomes.parquet"

_ALPHA = 0.10                       # 90% world-bootstrap CIs
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20261001


class DecisionQualityRefusal(RuntimeError):
    """A decision-quality input is absent, unverified, or inconsistent."""


# ------------------------------------------------------------------ helpers --
def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def canonical_arm(value) -> str:
    """Canonical controller arm for an ``arm_kind``/``policy`` token.

    ``arm_kind`` values are ``<arm>_fleet`` / ``<arm>_mix``; the legacy policy
    names carry the same suffix, so the first ``_`` token is taken before
    :func:`evalh.idea_a.canon_action` (which resolves ``pi-external-u`` etc.).
    """
    from evalh.idea_a import canon_action

    token = str(value or "").strip().lower().split("_")[0]
    return canon_action(token)


def _row_arm(row) -> str:
    kind = _field(row, "arm_kind", None)
    if kind is None or not str(kind).strip():
        kind = _field(row, "policy", "")
    return canonical_arm(kind)


def _row_arm_type(row) -> str:
    """E2 decision-quality arm *type* of one row (source/arm/arm_kind/policy).

    ``--dq-arms e2`` scores the pooled controller family (``fixed_mix``,
    ``switching_mix``, ...) so the TRAIN-only fixed-best V2 rankings match the
    scored arms; the default registry keeps :func:`_row_arm`.

    The true ARM NAME is preferred when present in ``features['arm']``: a
    released ``OutcomeRow`` carries its ``policy`` as the run's *motion policy*
    (``switching`` / ``fixed`` / ``wall`` ...), NOT the arm, so reading
    ``policy`` first mislabels a ``native``/``pi`` shard whose run policy is
    ``switching`` as ``switching_mix`` and destroys the native reference
    (Fix 2b, 2026-10-05).  Plain records without features fall back to the
    historical ``source``/``arm``/``arm_kind``/``policy`` order.
    """
    from evalh.scorer.arm_contrasts import arm_type_of

    feats = _field(row, "features", None)
    if isinstance(feats, dict):
        arm = feats.get("arm")
        if arm is not None and str(arm).strip():
            return arm_type_of(arm)
    for name in ("source", "arm", "arm_kind", "policy"):
        value = _field(row, name, None)
        if value is not None and str(value).strip():
            return arm_type_of(value)
    return ""


def _row_arm_name(row) -> str:
    """The raw arm token of one row (``features['arm']`` preferred)."""
    feats = _field(row, "features", None)
    if isinstance(feats, dict):
        arm = feats.get("arm")
        if arm is not None and str(arm).strip():
            return str(arm).strip()
    for name in ("source", "arm", "arm_kind", "policy"):
        value = _field(row, name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _e2_candidate_arm(row) -> str:
    """Fix 3 ``--dq-arms e2`` candidate label for one row.

    Every trajectory member (``arm0``..``arm3``) keeps its own identity so its
    realized cost and head prediction are its own plan's; the random-mix
    families stay pooled under their type name (``fixed_mix`` etc.); the arm
    types (``native``, ``pi``, ``damping``) pass through.  A bare ``trajectory``
    token (no member) is returned unchanged and dropped by the caller's allowed
    set, never scored.
    """
    from evalh.scorer.arm_contrasts import arm_type_of

    raw = _row_arm_name(row)
    kind = arm_type_of(raw)
    if kind == FIXED_BEST_TRAJECTORY_TYPE and raw.strip().lower() in TRAJECTORY_ARMS:
        return raw.strip().lower()
    return kind


def _row_is_truearms(row) -> bool:
    """Whether a released row comes from the ``truearms`` true-fleet shard.

    The released source records ``features['run_key'] = '<batch>/<world>__<arm>'``;
    the true-fleet batches are ``truearms_chunk*``.  Used only to break a
    ``native``/``pi`` duplicate deterministically (Fix 3 rule 4).
    """
    feats = _field(row, "features", None)
    if not isinstance(feats, dict):
        return False
    run_key = str(feats.get("run_key", "") or "")
    batch = run_key.split("/", 1)[0] if "/" in run_key else ""
    return batch.startswith("truearms")


def e2_dedup_key(row) -> int:
    """Deterministic duplicate preference: ``truearms`` (1) over released (0)."""
    return 1 if _row_is_truearms(row) else 0


def candidate_type(arm: str) -> str:
    """The fixed-best arm *type* of a candidate label (trajectory members)."""
    return (FIXED_BEST_TRAJECTORY_TYPE
            if str(arm) in TRAJECTORY_ARMS else str(arm))


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class DecisionArm:
    """One candidate arm at a decision point."""

    arm: str
    realized: float
    predicted: Optional[np.ndarray] = None    # [N_QUANTILES] or None


@dataclass
class DecisionPoint:
    """A ``(physical_world, t0, decision_t, tau)`` window with >= 2 arms."""

    world: str
    t0_s: float
    decision_t_s: float
    tau_s: float
    cost: str
    arms: tuple                                # of DecisionArm, name-sorted

    @property
    def cluster(self) -> str:
        from evalh.scorer.metrics_loop import physical_world

        return physical_world(self.world)

    def arm(self, name: str) -> Optional[DecisionArm]:
        for a in self.arms:
            if a.arm == name:
                return a
        return None


@dataclass
class DecisionStats:
    n_rows_input: int = 0
    n_rows_used: int = 0
    n_rows_excluded_arm: int = 0
    n_skipped_tau: int = 0
    n_arms_unknown: int = 0
    n_arms_nonfinite: int = 0
    n_skipped_identity: int = 0
    n_skipped_insufficient_arms: int = 0
    n_decision_points: int = 0
    n_points_without_prediction: int = 0
    #: Fix 3 rule 4: which duplicate was kept for a de-duplicated candidate.
    n_dedup_truearms: int = 0
    n_dedup_released: int = 0
    dedup_choices: dict = field(default_factory=dict)   # label -> count


def decision_points_from_rows(rows: Sequence, quantiles=None, *,
                              cost: str = "delay_s", taus=TAUS_S,
                              levels=QUANTILE_LEVELS,
                               allowed_arms=ALLOWED_ARMS,
                               excluded_sources=EXCLUDED_SOURCES,
                               normalizer=None, pool_arms=None,
                               dedup_key=None):
    """Build decision points from label rows plus aligned predictions.

    ``quantiles`` is an ``[len(rows), N_QUANTILES]`` array aligned with ``rows``
    (or ``None`` for a labels-only baseline, e.g. the fixed-best dry run).  Rows
    are grouped by ``(world, t0_s, decision_t_s, tau_s)``; the distinct canonical
    arms in a group are its candidates.  ``allowed_arms`` fixes the registered
    controller set; ``excluded_sources`` applies the owner rule-(b) arm
    exclusion on the row's ``source`` metadata when present.  ``normalizer`` is
    an optional ``row -> arm`` override.

    ``--dq-arms e2`` (Fix 3, 2026-10-05) passes :func:`_e2_candidate_arm` plus
    ``pool_arms=E2_POOL_ARMS`` and ``dedup_key=e2_dedup_key``:

    * ``arm0``..``arm3`` each keep their OWN candidate identity (own realized
      value and own plan prediction) -- they are NEVER pooled into a
      ``trajectory`` candidate;
    * the random-mix families (``fixed_mix``, ``switching_mix``, ``wall``) ARE
      pooled by mean realized / mean predicted (:data:`E2_POOL_ARMS`);
    * duplicated ``native``/``pi`` rows (released campaign + ``truearms``
      shards at one ``(world, t0)``) are DE-DUPLICATED deterministically,
      preferring the ``truearms`` true-fleet row via ``dedup_key``; the choice
      is recorded in :attr:`DecisionStats.dedup_choices` and the
      ``n_dedup_truearms`` / ``n_dedup_released`` counters.

    Without a ``normalizer``/``pool_arms``/``dedup_key`` the historical
    fail-closed rule is kept: a duplicate canonical arm is an identity conflict
    and the whole group is refused.  A plain ``normalizer`` alone preserves the
    legacy "pool every duplicate label" behaviour.

    Returns ``(points, DecisionStats)``.  A group with a duplicate canonical arm
    (identity conflict, default path only), a non-finite t0, or fewer than two
    finite labelled arms is refused and counted.
    """
    if cost not in COST_ATTRS:
        raise DecisionQualityRefusal(
            f"unknown cost {cost!r}; registered costs are {list(COSTS)}")
    norm = normalizer or _row_arm
    if pool_arms is not None:
        pool_set = {str(a).strip().lower() for a in pool_arms}
    elif normalizer is not None and dedup_key is None:
        pool_set = None                    # legacy: pool every duplicate label
    else:
        pool_set = frozenset()             # default: duplicates are conflicts
    attr = COST_ATTRS[cost]
    rows = list(rows)
    n_rows = len(rows)
    if quantiles is not None:
        q = np.asarray(quantiles, dtype=float)
        if q.shape != (n_rows, N_QUANTILES):
            raise DecisionQualityRefusal(
                f"quantiles has shape {q.shape}; expected "
                f"({n_rows}, {N_QUANTILES})")
    else:
        q = None
    tau_set = {round(float(t), 6) for t in taus}
    allowed_set = {str(a).strip().lower() for a in allowed_arms}
    excluded_lower = {str(a).strip().lower() for a in excluded_sources}
    stats = DecisionStats(n_rows_input=n_rows)

    groups: dict = {}
    for i, row in enumerate(rows):
        source = _field(row, "source", None)
        if source is not None and str(source).strip().lower() in excluded_lower:
            stats.n_rows_excluded_arm += 1
            continue
        tau = _field(row, "tau_s", None)
        if tau is None or not _finite(tau):
            stats.n_skipped_tau += 1
            continue
        tau = round(float(tau), 6)
        if tau not in tau_set:
            stats.n_skipped_tau += 1
            continue
        t0 = _field(row, "t0_s", None)
        if t0 is None or not _finite(t0):
            stats.n_skipped_identity += 1
            continue
        dec = _field(row, "decision_t_s", None)
        dec = float(dec) if _finite(dec) else float(t0)
        world = str(_field(row, "world", "") or "")
        if not world:
            stats.n_skipped_identity += 1
            continue
        groups.setdefault((world, round(float(t0), 6), round(dec, 6), tau),
                          []).append(i)

    def _pooled(name: str) -> bool:
        return pool_set is None or name in pool_set

    points = []
    for (world, t0, dec, tau), idx in sorted(groups.items()):
        kept = []
        for i in idx:
            if norm(rows[i]) not in allowed_set:
                stats.n_arms_unknown += 1
                continue
            kept.append(i)
        by_arm: dict = {}
        for i in kept:
            by_arm.setdefault(norm(rows[i]), []).append(i)
        arms = {}
        identity_conflict = False
        for arm, members in by_arm.items():
            members = sorted(members)
            if _pooled(arm):
                chosen = members
            elif len(members) == 1:
                chosen = members
            elif dedup_key is not None:
                chosen = [max(members, key=lambda i: (dedup_key(rows[i]), -i))]
                if _row_is_truearms(rows[chosen[0]]):
                    stats.n_dedup_truearms += 1
                else:
                    stats.n_dedup_released += 1
                stats.dedup_choices[arm] = stats.dedup_choices.get(arm, 0) + 1
            else:
                identity_conflict = True
                break
            reals, preds = [], []
            bad = False
            for i in chosen:
                realized = _field(rows[i], attr, None)
                predicted = None if q is None else np.asarray(q[i], dtype=float)
                if not _finite(realized):
                    stats.n_arms_nonfinite += 1
                    bad = True
                    break
                if predicted is not None and (
                        predicted.shape != (N_QUANTILES,)
                        or not np.isfinite(predicted).all()):
                    stats.n_arms_nonfinite += 1
                    bad = True
                    break
                reals.append(float(realized))
                if predicted is not None:
                    preds.append(predicted)
            if bad:
                continue
            arms[arm] = DecisionArm(
                arm=arm,
                realized=(reals[0] if len(reals) == 1
                          else float(np.mean(reals))),
                predicted=(None if not preds else
                           (preds[0] if len(preds) == 1
                            else np.mean(np.stack(preds), axis=0))))
        if identity_conflict:
            stats.n_skipped_identity += 1
            continue
        if len(arms) < 2:
            stats.n_skipped_insufficient_arms += 1
            continue
        points.append(DecisionPoint(
            world=world, t0_s=float(t0), decision_t_s=float(dec),
            tau_s=float(tau), cost=cost,
            arms=tuple(arms[name] for name in sorted(arms))))
    stats.n_rows_used = sum(len(p.arms) for p in points)
    stats.n_decision_points = len(points)
    if q is None:
        stats.n_points_without_prediction = len(points)
    else:
        stats.n_points_without_prediction = sum(
            1 for p in points for a in p.arms if a.predicted is None)
    return points, stats


# --------------------------------------------------------------- policies --
def predicted_cost(arm: DecisionArm, *, risk_averse: bool = False,
                   levels=QUANTILE_LEVELS) -> float:
    if arm.predicted is None:
        raise DecisionQualityRefusal(
            f"arm {arm.arm!r} has no predicted quantiles; the WM policy "
            "cannot score a labels-only decision point")
    pred = np.asarray(arm.predicted, dtype=float)
    if risk_averse:
        return float(pred[level_index(0.9, levels)])
    return float(np.mean(pred))


def wm_arm(point: DecisionPoint, *, risk_averse: bool = False,
           levels=QUANTILE_LEVELS) -> str:
    """Lowest model-predicted cost; deterministic name tie-break."""
    scored = [(predicted_cost(a, risk_averse=risk_averse, levels=levels),
               a.arm) for a in point.arms]
    return min(scored, key=lambda t: (t[0], t[1]))[1]


def oracle_arm(point: DecisionPoint) -> str:
    return min((a.realized, a.arm) for a in point.arms)[1]


def native_arm(point: DecisionPoint) -> Optional[str]:
    return "native" if point.arm("native") is not None else None


def fixed_best_arm(point: DecisionPoint, ranking: Sequence) -> str:
    """Best *available* fixed-best arm TYPE by the frozen TRAIN ranking.

    The frozen ``DECISION_QUALITY_FIXED_BEST_V2`` ranking is at the arm-TYPE
    level, so a trajectory member (``arm0``..``arm3``) counts as the type
    ``trajectory`` (Fix 3 rule 5).  The deterministic name tie-break is kept.
    """
    rank = {str(a): i for i, a in enumerate(ranking)}
    available = sorted({candidate_type(a.arm) for a in point.arms})
    return min(available, key=lambda t: (rank.get(t, len(rank) + 1000), t))


def fixed_best_realized(point: DecisionPoint, ranking: Sequence) -> float:
    """Realized cost of the frozen fixed-best choice at one decision point.

    Usually this is the chosen candidate's own realized cost.  When the chosen
    type is ``trajectory`` (Fix 3 rule 5) the frozen ranking names a random
    *plan*, so the realized cost is the MEAN over this decision's trajectory
    arms (``arm0``..``arm3``); this is documented and never treated as an
    oracle candidate.
    """
    label = fixed_best_arm(point, ranking)
    if label == FIXED_BEST_TRAJECTORY_TYPE:
        vals = [a.realized for a in point.arms if a.arm in TRAJECTORY_ARMS]
        if not vals:
            raise DecisionQualityRefusal(
                "fixed-best chose 'trajectory' but the point has no trajectory "
                "arm; refusing to impute a cost")
        return float(np.mean(vals))
    chosen = point.arm(label)
    if chosen is None:
        raise DecisionQualityRefusal(
            f"fixed-best chose arm {label!r} that is absent from the point")
    return chosen.realized


# ---------------------------------------------------------- fixed-best doc --
def build_fixed_best(points: Sequence[DecisionPoint], *,
                     train_source: Optional[dict] = None,
                     costs=COSTS, taus=TAUS_S) -> dict:
    """Per-``(cost, tau)`` TRAIN arm means, ranking and single best arm.

    The single best arm is ``ranking[0]``; the ranking also gives the
    deterministic fallback used when the best arm is unavailable at a decision
    point.  Cells with no TRAIN arms are omitted (fail closed downstream).
    """
    by_cost: dict = {}
    for cost in costs:
        tau_cells: dict = {}
        for tau in taus:
            tau = round(float(tau), 6)
            vals: dict = {}
            for p in points:
                if p.cost != cost or round(p.tau_s, 6) != tau:
                    continue
                for a in p.arms:
                    vals.setdefault(a.arm, []).append(float(a.realized))
            if not vals:
                continue
            arm_mean = {a: float(np.mean(v)) for a, v in sorted(vals.items())}
            ranking = sorted(arm_mean, key=lambda a: (arm_mean[a], a))
            tau_cells[f"{tau:g}"] = dict(
                n_points=int(sum(1 for p in points
                                 if p.cost == cost
                                 and round(p.tau_s, 6) == tau)),
                n_arms=len(arm_mean),
                arm_mean=arm_mean,
                ranking=ranking,
                best_arm=ranking[0])
        by_cost[cost] = tau_cells
    return dict(
        schema=SCHEMA_FIXED_BEST,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        train_source=dict(train_source or {}),
        costs=[str(c) for c in costs],
        taus=[float(t) for t in taus],
        cells=by_cost)


def write_fixed_best(doc: dict, path) -> str:
    """Write the fixed-best JSON and return its SHA-256 (never overwrite)."""
    path = Path(path)
    if path.exists():
        raise DecisionQualityRefusal(
            f"fixed-best output {path} already exists; refusing to overwrite")
    payload = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("utf-8"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_fixed_best(path, *, expected_sha256: str) -> tuple:
    """Load and SHA-pin a fixed-best artifact; returns ``(doc, sha)``."""
    if not isinstance(expected_sha256, str) or not expected_sha256.strip():
        raise DecisionQualityRefusal(
            "fixed_best requires an explicit expected_sha256 pin")
    pin = expected_sha256.strip().lower()
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != pin:
        raise DecisionQualityRefusal(
            f"fixed-best {path} sha256 {sha} != pinned {pin}; refusing")
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("schema") != SCHEMA_FIXED_BEST:
        raise DecisionQualityRefusal(
            f"fixed-best schema {doc.get('schema')!r} != {SCHEMA_FIXED_BEST!r}")
    return doc, sha


# --------------------------------------------------------------- evaluate --
def _ci(values, clusters, *, n_boot, seed) -> list:
    lo, hi, point = world_bootstrap_ci(values, clusters, n_boot=n_boot,
                                       alpha=_ALPHA, seed=seed)
    return [float(lo), float(hi), float(point)]


def _cell_metrics(cell_points, ranking, *, n_boot, seed) -> dict:
    """Regret statistics for one ``(cost, tau)`` cell."""
    clusters = [p.cluster for p in cell_points]
    oracle = {id(p): p.arm(oracle_arm(p)).realized for p in cell_points}

    def realized(p, arm):
        return p.arm(arm).realized

    wm_mean = [p for p in cell_points
               if all(a.predicted is not None for a in p.arms)]
    n_missing = len(cell_points) - len(wm_mean)

    def regrets(choose):
        out = []
        for p in cell_points:
            arm = choose(p)
            if arm is None:
                continue
            out.append(realized(p, arm) - oracle[id(p)])
        return out

    wm_mean_regret = [realized(p, wm_arm(p)) - oracle[id(p)] for p in wm_mean]
    wm_risk_regret = [realized(p, wm_arm(p, risk_averse=True)) - oracle[id(p)]
                      for p in wm_mean]
    fixed_regret = [fixed_best_realized(p, ranking) - oracle[id(p)]
                    for p in cell_points]
    native_regret = regrets(lambda p: native_arm(p))
    random_exp = float(np.mean([
        float(np.mean([a.realized for a in p.arms])) - oracle[id(p)]
        for p in cell_points])) if cell_points else float("nan")

    wm_arms = {id(p): wm_arm(p) for p in wm_mean}
    oracle_arms = {id(p): oracle_arm(p) for p in cell_points}
    oracle_pick = float(np.mean([wm_arms[id(p)] == oracle_arms[id(p)]
                                 for p in wm_mean])) if wm_mean else float("nan")

    fb_cost = {id(p): fixed_best_realized(p, ranking) for p in cell_points}
    win = tie = loss = 0
    paired_diff = []
    for p in wm_mean:
        w = realized(p, wm_arms[id(p)])
        f = fb_cost[id(p)]
        paired_diff.append(w - f)
        if w < f:
            win += 1
        elif w == f:
            tie += 1
        else:
            loss += 1
    n_paired = win + tie + loss

    diff_mean = float(np.mean(paired_diff)) if paired_diff else float("nan")
    diff_ci = _ci(paired_diff, [p.cluster for p in wm_mean],
                  n_boot=n_boot, seed=seed) if paired_diff else \
        [float("nan")] * 3

    return dict(
        n_points=len(cell_points),
        n_points_wm=len(wm_mean),
        n_points_without_prediction=n_missing,
        n_clusters=len(set(clusters)),
        oracle_regret=0.0,
        wm_mean_regret=(float(np.mean(wm_mean_regret))
                        if wm_mean_regret else float("nan")),
        wm_mean_regret_ci=(_ci(wm_mean_regret,
                               [p.cluster for p in wm_mean],
                               n_boot=n_boot, seed=seed)
                           if wm_mean_regret else [float("nan")] * 3),
        wm_risk_regret=(float(np.mean(wm_risk_regret))
                        if wm_risk_regret else float("nan")),
        wm_risk_regret_ci=(_ci(wm_risk_regret,
                               [p.cluster for p in wm_mean],
                               n_boot=n_boot, seed=seed)
                           if wm_risk_regret else [float("nan")] * 3),
        fixed_best_regret=float(np.mean(fixed_regret)),
        fixed_best_regret_ci=_ci(fixed_regret, clusters,
                                 n_boot=n_boot, seed=seed),
        native_regret=(float(np.mean(native_regret))
                       if native_regret else float("nan")),
        native_regret_ci=(_ci(native_regret, clusters,
                              n_boot=n_boot, seed=seed)
                          if native_regret else [float("nan")] * 3),
        native_n_points=len(native_regret),
        random_expected_regret=random_exp,
        wm_picks_oracle_fraction=oracle_pick,
        win_rate_vs_fixed_best=(float(win / n_paired) if n_paired
                                else float("nan")),
        tie_rate_vs_fixed_best=(float(tie / n_paired) if n_paired
                                else float("nan")),
        loss_rate_vs_fixed_best=(float(loss / n_paired) if n_paired
                                 else float("nan")),
        wm_minus_fixed_best_mean=diff_mean,
        wm_minus_fixed_best_ci=diff_ci,
        wm_better_than_fixed_best=(bool(diff_ci[1] < 0.0)
                                   if paired_diff else False))


def evaluate(points: Sequence[DecisionPoint], *, fixed_best: dict,
             costs=None, taus=None, n_boot: int = DEFAULT_N_BOOT,
             seed: int = DEFAULT_SEED, source: str = "marginals") -> dict:
    """Regret table over ``(cost, tau)`` cells with world-cluster CIs.

    ``source`` records the per-arm prediction provenance only
    (``"marginals"`` default, or ``"paired_head"`` dV vs native); it never
    changes the policies, regret definitions, fixed-best, random, native or
    oracle computations.
    """
    points = list(points)
    if not points:
        raise DecisionQualityRefusal("no decision points to evaluate")
    cost_list = list(costs) if costs else sorted({p.cost for p in points})
    tau_list = ([float(t) for t in taus] if taus
                else sorted({round(p.tau_s, 6) for p in points}))
    fb_cells = fixed_best.get("cells") or {}
    cells: dict = {}
    for cost in cost_list:
        cells[cost] = {}
        for tau in tau_list:
            tau = round(float(tau), 6)
            cell_points = [p for p in points
                           if p.cost == cost and round(p.tau_s, 6) == tau]
            if not cell_points:
                continue
            fb = (fb_cells.get(cost) or {}).get(f"{tau:g}")
            if fb is None or not fb.get("ranking"):
                raise DecisionQualityRefusal(
                    f"fixed-best artifact has no cell for cost {cost!r} at "
                    f"tau {tau:g}s; refusing (no ranking substitution)")
            cells[cost][f"{tau:g}"] = _cell_metrics(
                cell_points, fb["ranking"], n_boot=n_boot, seed=seed)

    primary = {}
    claims = []
    for tau in [t for t in tau_list if float(t) in PRIMARY_TAUS]:
        cell = cells.get("delay_s", {}).get(f"{round(float(tau), 6):g}")
        if cell is None or cell["n_points_wm"] == 0:
            continue
        primary[f"{round(float(tau), 6):g}"] = dict(
            wm_minus_fixed_best_mean=cell["wm_minus_fixed_best_mean"],
            wm_minus_fixed_best_ci=cell["wm_minus_fixed_best_ci"],
            holds=cell["wm_better_than_fixed_best"])
        claims.append(cell["wm_better_than_fixed_best"])
    # pooled across the primary taus, clustered by world
    pooled_vals, pooled_clusters = [], []
    for tau in PRIMARY_TAUS:
        cell_points = [p for p in points if p.cost == "delay_s"
                       and round(p.tau_s, 6) == round(float(tau), 6)]
        fb = (fb_cells.get("delay_s") or {}).get(f"{round(float(tau), 6):g}")
        if not fb or not fb.get("ranking"):
            continue
        for p in cell_points:
            if any(a.predicted is None for a in p.arms):
                continue
            oracle = p.arm(oracle_arm(p)).realized
            pooled_vals.append(p.arm(wm_arm(p)).realized - oracle
                               - (fixed_best_realized(p, fb["ranking"])
                                  - oracle))
            pooled_clusters.append(p.cluster)
    pooled = None
    if pooled_vals:
        ci = _ci(pooled_vals, pooled_clusters, n_boot=n_boot, seed=seed)
        pooled = dict(mean=float(np.mean(pooled_vals)), ci=ci,
                      holds=bool(ci[1] < 0.0))
    claim = dict(
        text=("Primary claim: WM (mean predicted cost) regret < fixed-best "
              "regret on delay_s at tau in {30,60,120}"),
        cost="delay_s", taus=[float(t) for t in PRIMARY_TAUS],
        per_tau=primary, pooled=pooled,
        holds=bool(pooled["holds"] if pooled else (claims and all(claims))))
    return dict(
        schema=SCHEMA,
        seed=int(seed), n_boot=int(n_boot), alpha=_ALPHA,
        prediction_source=str(source),
        primary_taus=[float(t) for t in PRIMARY_TAUS],
        secondary_taus=[float(t) for t in SECONDARY_TAUS],
        costs=[str(c) for c in cost_list],
        fixed_best=dict(schema=fixed_best.get("schema"),
                        train_source=fixed_best.get("train_source") or {}),
        cells=cells,
        primary_claim=claim)


# ---------------------------------------------------------------- render ---
def render_table(doc: dict) -> str:
    """A compact markdown table of the regret cells for the score report."""
    header = ("| cost | tau | n | WM mean | WM risk | fixed-best | native | "
              "random | pick-oracle | win vs fb | WM-fb [90% CI] |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = [header]
    for cost in doc.get("costs", []):
        for tau_key in sorted(doc.get("cells", {}).get(cost, {}),
                              key=lambda s: float(s)):
            c = doc["cells"][cost][tau_key]
            ci = c.get("wm_minus_fixed_best_ci", [float("nan")] * 3)

            def f(x):
                return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"

            lines.append(
                f"| {cost} | {tau_key} | {c['n_points']} | "
                f"{f(c['wm_mean_regret'])} | {f(c['wm_risk_regret'])} | "
                f"{f(c['fixed_best_regret'])} | {f(c['native_regret'])} | "
                f"{f(c['random_expected_regret'])} | "
                f"{f(c['wm_picks_oracle_fraction'])} | "
                f"{f(c['win_rate_vs_fixed_best'])} | "
                f"{f(c['wm_minus_fixed_best_mean'])} "
                f"[{f(ci[0])}, {f(ci[1])}] |\n")
    return "".join(lines)


# ------------------------------------------------------------- CLI dry run --
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_train_rows(train_dir, *, expected_sha256=EXPECTED_TRAIN_OUTCOMES_SHA256):
    """TRAIN-only label rows for the dry run (sha-pinned, t0-aligned).

    Returns ``(rows, info)`` where ``rows`` are dicts with the fields
    :func:`decision_points_from_rows` reads.  Only the ``(world, arm_kind,
    policy, t0_s, t, tau, tau_available, total_delay_ref_s, hard_brake_s)``
    projection is read; TRAIN split is asserted.
    """
    train_dir = Path(train_dir)
    path = train_dir / TRAIN_OUTCOMES_FILENAME
    if not path.is_file():
        raise DecisionQualityRefusal(f"no {TRAIN_OUTCOMES_FILENAME} in {train_dir}")
    sha = _sha256_file(path)
    if expected_sha256 is not None and sha != str(expected_sha256).strip().lower():
        raise DecisionQualityRefusal(
            f"{TRAIN_OUTCOMES_FILENAME} sha256 {sha} != pinned {expected_sha256}; "
            "refusing before any label read")
    import pandas as pd

    frame = pd.read_parquet(path, columns=[
        "world", "source", "arm_kind", "policy", "split", "t0_s", "t",
        "tau", "tau_available", "total_delay_ref_s", "hard_brake_s"])
    split = frame["split"].astype(str).str.strip().str.upper()
    if (split != "TRAIN").any():
        raise DecisionQualityRefusal(
            "sidecar holds non-TRAIN rows; refusing a TRAIN baseline")
    rows = []
    n_t0 = 0
    for rec in frame.to_dict("records"):
        if float(rec["tau_available"]) <= 0.0:
            continue
        if round(float(rec["t"]), 6) != round(float(rec["t0_s"]), 6):
            continue
        n_t0 += 1
        rows.append(dict(
            world=str(rec["world"]), source=str(rec["source"]),
            arm_kind=rec.get("arm_kind"),
            policy=rec.get("policy"), t0_s=float(rec["t0_s"]),
            decision_t_s=float(rec["t"]), tau_s=float(rec["tau"]),
            y_delay_s=float(rec["total_delay_ref_s"]),
            hard_brake_s=float(rec["hard_brake_s"])))
    info = dict(path=str(path), sha256=sha, n_rows_in=int(len(frame)),
                n_rows_t0=n_t0, split="TRAIN")
    return rows, info


def run_train_baseline(train_dir, out_dir, *,
                       expected_sha256=EXPECTED_TRAIN_OUTCOMES_SHA256,
                       n_boot: int = DEFAULT_N_BOOT,
                       seed: int = DEFAULT_SEED) -> dict:
    """Zero-simulation TRAIN-only dry run: fixed-best + baselines on TRAIN.

    Writes ``DECISION_QUALITY_FIXED_BEST.json``, ``DECISION_QUALITY_TRAIN_RESULTS.json``
    and ``DECISION_QUALITY_TRAIN_BASELINE.md`` into ``out_dir`` (created if
    needed).  No WM predictions exist on TRAIN, so WM cells are omitted; the
    report gives the oracle / fixed-best / random / native gap a perfect model
    could close.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, info = load_train_rows(train_dir, expected_sha256=expected_sha256)
    points, stats = decision_points_from_rows(rows, None, cost="delay_s")
    cost_points = {c: decision_points_from_rows(rows, None, cost=c)[0]
                   for c in COSTS}
    all_points = [p for c in COSTS for p in cost_points[c]]
    fixed_doc = build_fixed_best(all_points, train_source=info)
    doc = evaluate(all_points, fixed_best=fixed_doc, n_boot=n_boot, seed=seed)
    fb_path = out_dir / "DECISION_QUALITY_FIXED_BEST.json"
    if fb_path.exists():
        fb_sha = _sha256_file(fb_path)
        loaded, fb_sha = load_fixed_best(fb_path, expected_sha256=fb_sha)
        stable = {k: v for k, v in loaded.items() if k != "created_utc"}
        rebuilt = {k: v for k, v in fixed_doc.items() if k != "created_utc"}
        if stable != rebuilt:
            raise DecisionQualityRefusal(
                f"{fb_path} exists and differs from the recomputed artifact; "
                "refusing to overwrite")
    else:
        fb_sha = write_fixed_best(fixed_doc, fb_path)
    doc["fixed_best"] = dict(fixed_best=doc["fixed_best"],
                             path=str(fb_path), sha256=fb_sha)
    doc["train_source"] = info
    doc["counts"] = dict(delay_s=asdict(stats))
    results_path = out_dir / "DECISION_QUALITY_TRAIN_RESULTS.json"
    results_path.write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    md = render_train_baseline_md(doc, fixed_doc)
    (out_dir / "DECISION_QUALITY_TRAIN_BASELINE.md").write_text(
        md, encoding="utf-8")
    return doc


def render_train_baseline_md(doc: dict, fixed_doc: dict) -> str:
    """The standalone TRAIN-only baseline markdown report."""
    info = doc.get("train_source") or {}
    lines = [
        "# WM-N1 DECISION-QUALITY (regret) — TRAIN-only baseline",
        "",
        "**Status: COMPLETE** (zero-simulation dry run; TRAIN labels only).",
        "",
        "A decision point is a `(physical_world, t0)` with >= 2 labelled "
        "controller arms. Regret = realized cost of the chosen arm minus the "
        "oracle (lowest realized) cost. This file computes the fixed-best arm "
        "and the oracle / random / native regrets so we know the gap a perfect "
        "WM policy could close.",
        "",
        f"- Labels: `{info.get('path')}`",
        f"  sha256 `{info.get('sha256')}` ({info.get('n_rows_in')} rows, "
        f"{info.get('n_rows_t0')} t0-aligned).",
        "- No CAL / HOLDOUT / chunk-5 / J4B / INC1 data opened; no simulation, "
        "SUMO, GPU or SSH.",
        "- Costs: `delay_s` = `total_delay_ref_s` (primary), `hard_brake_s` "
        "(secondary). WM predictions do not exist on TRAIN, so no WM row is "
        "reported here.",
        "",
        "## Fixed-best (TRAIN-only, frozen)",
        "",
        "| cost | tau | best arm | arm means (s) |",
        "|---|---|---|---|",
    ]
    for cost in fixed_doc.get("costs", []):
        for tau_key in sorted(fixed_doc["cells"].get(cost, {}),
                              key=lambda s: float(s)):
            cell = fixed_doc["cells"][cost][tau_key]
            means = ", ".join(f"{a}={cell['arm_mean'][a]:.3f}"
                              for a in cell["ranking"])
            lines.append(f"| {cost} | {tau_key} | {cell['best_arm']} | "
                         f"{means} |")
    gaps = []
    for tau in PRIMARY_TAUS:
        cell = doc.get("cells", {}).get("delay_s", {}).get(f"{tau:g}")
        if cell is not None:
            gaps.append(f"tau {tau:g}: {cell['fixed_best_regret']:.3f} s")
    lines += [
        "",
        "Registered candidate arms are `native`, `damping`, `pi`, `wall`. In "
        "this corrected sidecar the damping arm is the rule-(b) excluded "
        "`base` source, and the fixed / switching labels arrive under multiple "
        "non-registered sources, so each tau carries exactly **161** "
        "`{native, pi}` decision points (39 damping-only groups are dropped "
        "with rule (b); 2,555 switching/fixed rows are non-registered).",
        "",
        "## Regret (TRAIN in-sample; note: in-sample fixed-best)",
        "",
        render_table(doc).strip(),
        "",
        f"The perfect-model gap (regret a WM policy could close at the primary "
        f"taus, cost `delay_s`) is the fixed-best regret: "
        f"{'; '.join(gaps)}. `fixed-best` regret here is **in-sample** (the "
        "ranking is built from the same TRAIN labels); `random` and `native` "
        "regrets bracket it. The WM claim test on CAL cannot be run in this "
        "task (no CAL access).",
        "",
        "## Decision-point accounting",
        "",
        "```json",
        json.dumps(doc.get("counts", {}), indent=1, sort_keys=True),
        "```",
        "",
        "## Open issues",
        "",
        "- TRAIN-only and in-sample: the fixed-best arm is selected on the "
        "same labels it is scored on, so its regret is optimistically low.",
        "- No CAL-tune run was performed here (rules forbid CAL access in this "
        "task); `--decision-quality` remains default-off and is exercised only "
        "on synthetic fixtures.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evalh.scorer.decision_quality",
        description="TRAIN-only decision-quality (regret) baseline dry run.")
    parser.add_argument("--train-outcomes", required=True,
                        help="directory holding train_outcomes.parquet")
    parser.add_argument("--out", required=True,
                        help="output directory (created if needed)")
    parser.add_argument("--expected-sha256", dest="expected_sha256",
                        default=EXPECTED_TRAIN_OUTCOMES_SHA256,
                        help="SHA-256 pin of train_outcomes.parquet")
    parser.add_argument("--n-boot", dest="n_boot", type=int,
                        default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", dest="seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        doc = run_train_baseline(
            args.train_outcomes, args.out,
            expected_sha256=args.expected_sha256, n_boot=args.n_boot,
            seed=args.seed)
    except DecisionQualityRefusal as exc:
        print(f"[wm-n1 decision-quality refused] {exc}", file=sys.stderr)
        return 87
    except Exception as exc:                             # noqa: BLE001
        print(f"[wm-n1 decision-quality refused] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 87
    print(json.dumps({"schema": doc["schema"], "out": str(args.out),
                      "fixed_best_sha256": doc["fixed_best"].get("sha256")},
                     sort_keys=True))
    return 0


__all__ = [
    "SCHEMA", "SCHEMA_FIXED_BEST", "PRIMARY_TAUS", "SECONDARY_TAUS", "TAUS_S",
    "COSTS", "COST_ATTRS", "ALLOWED_ARMS", "FUTURE_ARMS", "EXCLUDED_SOURCES",
    "TRAJECTORY_ARMS", "E2_POOL_ARMS", "FIXED_BEST_TRAJECTORY_TYPE",
    "DecisionQualityRefusal", "DecisionArm", "DecisionPoint", "DecisionStats",
    "canonical_arm", "decision_points_from_rows", "predicted_cost", "wm_arm",
    "oracle_arm", "native_arm", "candidate_type", "fixed_best_arm",
    "fixed_best_realized", "_e2_candidate_arm", "e2_dedup_key",
    "build_fixed_best",
    "write_fixed_best", "load_fixed_best", "evaluate", "render_table",
    "load_train_rows", "run_train_baseline", "render_train_baseline_md",
]


if __name__ == "__main__":
    raise SystemExit(main())
