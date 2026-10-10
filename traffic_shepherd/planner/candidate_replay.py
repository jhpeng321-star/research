"""Per-candidate decision logging and the zero-noise selection replay.

Two halves, both pure (no SUMO, no model calls):

1. :class:`CandidateRecorder` -- filled by ``Planner`` (loop.py) with every
   ``Prediction`` the planner priced in ONE decision plus role/switch/bar
   annotations, then turned into ``candidates.jsonl`` rows (schema
   ``pl04_candidate_row/1``). It only READS values the planner already
   computed; the one extra computation is a pure ``safety_check`` per row at
   finalize time (deterministic, no state), so logging can never change a
   decision.
2. :func:`replay_decision` -- re-applies the planner's selection rule
   (steps 1, 3, 4, 5 of loop.py: individual picks, group candidate choice,
   switch rule, hold-wall release rule, combine + safety fallback) to the
   LOGGED candidate values. With no noise it reproduces the logged decision;
   an optional ``noise_fn`` adds noise to every priced candidate's oracle
   cost (plan_delay_s and delay_vs_native_s) before the rule is re-applied
   (foundation of the registered noise-injection study). A replay that needs
   a plan that was never priced raises :class:`UnpricedPlanError` (the logged
   set is only guaranteed complete for the noise-free decision path).

Floats are written losslessly (no rounding) so the zero-noise replay is exact.
"""

import json
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .config import PlannerConfig
from .interface import (
    PlanCandidate,
    Prediction,
    WallPrediction,
    WallSpec,
    plan_key,
    with_actions,
    with_wall,
    with_wall_release,
)
from .rules import candidate_sort_key, median_cells, native_return_check, switch_check
from .safety import safety_check
from .walls import assembly_bar_ok

CANDIDATE_ROW_SCHEMA = "pl04_candidate_row/1"
CANDIDATES_FILE = "candidates.jsonl"

_META_KEYS_SKIPPED = ("min_ttc_pairs",)


class UnpricedPlanError(KeyError):
    """The replay needs a plan that the run never priced (noise moved the
    decision path off the logged candidate set)."""


# ---------------------------------------------------------------- serialise
def _array_fields():
    import dataclasses

    skip = {"trajectories_tracked", "native_trajectories_tracked", "trajectories_hidden",
            "walls", "meta"}
    return [f.name for f in dataclasses.fields(Prediction) if f.name not in skip]


_ARRAY_FIELDS = _array_fields()


def prediction_block(prediction: Prediction) -> dict:
    """JSON-ready block of every [M, K] array of a Prediction (lists; NaN
    becomes null in the writer), the bool-typed field names, the per-wall
    arrays and the scalar meta entries (everything ``safety_check``,
    ``switch_check`` and ``assembly_bar_ok`` read)."""
    arrays, bool_fields = {}, []
    for name in _ARRAY_FIELDS:
        value = getattr(prediction, name, None)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.dtype == np.bool_:
            bool_fields.append(name)
            arrays[name] = arr.tolist()
        else:
            arrays[name] = arr.astype(float).tolist()
    wall_arrays = {}
    for wall_id, block in (prediction.walls or {}).items():
        entry = {}
        for name in ("assembled", "assembled_at_s", "safety_abort", "abort_reason",
                     "p_assembly_completes", "expected_assembly_time_s", "p_assembly_abort"):
            value = getattr(block, name, None)
            if value is not None:
                entry[name] = np.asarray(value).tolist()
        wall_arrays[wall_id] = entry
    meta = {}
    for key, value in (prediction.meta or {}).items():
        if key in _META_KEYS_SKIPPED:
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            meta[str(key)] = value
    return {"arrays": arrays, "bool_fields": bool_fields, "wall_arrays": wall_arrays, "meta": meta}


def _restore_array(values, is_bool: bool) -> np.ndarray:
    if is_bool:
        return np.asarray(values, dtype=bool)
    return np.asarray(
        [[np.nan if v is None else v for v in row] for row in values]
        if values and isinstance(values[0], list)
        else [np.nan if v is None else v for v in values],
        dtype=float,
    )


def prediction_from_row(row: Mapping) -> Prediction:
    """Rebuild a Prediction from one candidates.jsonl row (arrays + meta +
    wall blocks; trajectories are not logged and not needed by the rules)."""
    bool_fields = set(row.get("bool_fields", ()))
    kwargs = {
        name: _restore_array(values, name in bool_fields)
        for name, values in (row.get("arrays") or {}).items()
    }
    walls = {}
    for wall_id, entry in (row.get("wall_arrays") or {}).items():
        block = {}
        for name, values in entry.items():
            is_bool = name in ("assembled", "safety_abort")
            block[name] = (
                np.asarray(values, dtype=bool)
                if is_bool
                else np.asarray(
                    [[np.nan if v is None else v for v in r] for r in values], dtype=float
                )
            )
        block["abort_reason"] = np.asarray(
            block.get("abort_reason", np.zeros_like(block["assembled"], dtype=np.int8))
        ).astype(np.int8)
        walls[wall_id] = WallPrediction(**block)
    return Prediction(walls=walls, meta=dict(row.get("meta") or {}), **kwargs)


def plan_from_row(row: Mapping) -> PlanCandidate:
    walls = tuple(
        WallSpec(
            wall_id=w["wall_id"],
            members=tuple(w["members"]),
            speed_mps=float(w["speed_mps"]),
            phase=w["phase"],
            release_at_s=w.get("release_at_s"),
            hold_s=w.get("hold_s"),
        )
        for w in row.get("walls", ())
    )
    return PlanCandidate(per_cav=dict(row["actions"]), walls=walls)


def _wall_json(spec: WallSpec) -> dict:
    return {
        "wall_id": spec.wall_id,
        "members": list(spec.members),
        "speed_mps": float(spec.speed_mps),
        "phase": spec.phase,
        "release_at_s": spec.release_at_s,
        "hold_s": spec.hold_s,
    }


def plan_label(plan: PlanCandidate) -> str:
    parts = [f"{c}={a}" for c, a in sorted(plan.per_cav.items()) if a != "native"]
    parts += [f"wall:{w.wall_id}/{w.phase}" for w in plan.walls]
    return ",".join(parts) or "all_native"


def _relative_json(verdicts) -> list:
    return [
        {
            "metric": v.metric,
            "mean_increase_s": v.mean_increase_s,
            "fraction_within_tol": v.fraction_within_tol,
            "ok": v.ok,
        }
        for v in verdicts
    ]


def _safety_json(prediction: Prediction, config: PlannerConfig) -> dict:
    try:
        v = safety_check(prediction, config)
    except Exception as exc:  # named refusals are data here, never a crash
        return {"error": f"{type(exc).__name__}: {exc}", "ok": None}
    rules = []
    if not v.abs_ok:
        rules.append("absolute_ttc")
    if not v.rel_ok:
        rules.append("relative")
    if v.p_future_collision_veto:
        rules.append("p_future_collision")
    if v.g1_future_floor_veto:
        rules.append("g1_future_floor")
    if v.window_relative_veto:
        rules.append("window_relative")
    floor = float(config.ttc_floor_s)
    applied = floor
    if v.native_floor_min_s is not None:
        applied = min(floor, float(v.native_floor_min_s))
    return {
        "ok": bool(v.ok),
        "abs_ok": bool(v.abs_ok),
        "rel_ok": bool(v.rel_ok),
        "future_ok": bool(v.future_ok),
        "window_relative_veto": bool(v.window_relative_veto),
        "p_future_collision_veto": bool(v.p_future_collision_veto),
        "g1_future_floor_veto": bool(v.g1_future_floor_veto),
        "abs_safe_cells": int(v.abs_safe_cells),
        "abs_total_cells": int(v.abs_total_cells),
        "abs_safe_fraction": float(v.abs_safe_fraction),
        "vetoed_cells": int(v.vetoed_cells),
        "native_collision_cells": int(v.native_collision_cells),
        "absolute_ttc_rule": v.absolute_ttc_rule,
        "ttc_floor_s": floor,
        "native_floor_min_s": v.native_floor_min_s,
        "rule_c_floor_applied_min_s": applied,
        "shadow_rule_a_ok": v.shadow_rule_a_ok,
        "relative": _relative_json(v.relative),
        "future_relative": _relative_json(v.future_relative),
        "veto_rules": rules,
    }


# ----------------------------------------------------------------- recorder
class CandidateRecorder:
    """Collects one decision's priced candidates (see module docstring)."""

    def __init__(self):
        self._records: list = []
        self._latest: dict = {}
        self.default_stage = "probe"

    def register(self, plan: PlanCandidate, prediction: Prediction) -> None:
        rec = {
            "plan": plan,
            "pred": prediction,
            "stage": self.default_stage,
            "roles": [],
            "switch": None,
            "bar": None,
        }
        self._records.append(rec)
        self._latest[plan_key(plan)] = rec

    def role(self, plan: PlanCandidate, stage: str, subject: str, label: str, **extra) -> None:
        rec = self._latest.get(plan_key(plan))
        if rec is None:
            return
        entry = {"stage": stage, "subject": subject, "label": label}
        entry.update(extra)
        if entry not in rec["roles"]:
            rec["roles"].append(entry)

    def switch(self, plan: PlanCandidate, step: str, verdict, *, wall: bool) -> None:
        rec = self._latest.get(plan_key(plan))
        if rec is not None:
            rec["switch"] = {
                "step": step,
                "wall_bar": bool(wall),
                "median_saving_s": verdict.median_saving_s,
                "margin_s": verdict.margin_s,
                "agreement": verdict.agreement,
                "margin_ok": verdict.margin_ok,
                "agreement_ok": verdict.agreement_ok,
                "passes": verdict.passes,
            }

    def bar(self, plan: PlanCandidate, ok: bool, detail: str) -> None:
        rec = self._latest.get(plan_key(plan))
        if rec is not None:
            rec["bar"] = {"ok": bool(ok), "detail": detail}

    def __len__(self) -> int:
        return len(self._records)

    def finalize(
        self,
        config: PlannerConfig,
        chosen_key: Optional[tuple],
        ttc_veto_log: Optional[Mapping] = None,
    ) -> list:
        rows = []
        chosen_rec = self._latest.get(chosen_key) if chosen_key is not None else None
        for index, rec in enumerate(self._records):
            plan, pred = rec["plan"], rec["pred"]
            key = plan_key(plan)
            safety = _safety_json(pred, config)
            roles = rec["roles"]
            primary = roles[0] if roles else {"stage": rec["stage"], "subject": "*"}
            breach = (ttc_veto_log or {}).get(key)
            bar_failed = rec["bar"] is not None and not rec["bar"]["ok"]
            safety_ok = safety.get("ok")
            vetoed = bool(safety_ok is False or bar_failed)
            veto_rules = list(safety.get("veto_rules", ()))
            if bar_failed:
                veto_rules.append("assembly_bar")
            abs_veto = "absolute_ttc" in veto_rules
            steps = sorted({r["stage"] for r in roles} or {rec["stage"]}) if vetoed else []
            row = {
                "schema": CANDIDATE_ROW_SCHEMA,
                "cand_index": index,
                "plan_key": json.dumps(key, sort_keys=True, default=str),
                "label": plan_label(plan),
                "actions": dict(plan.per_cav),
                "walls": [_wall_json(w) for w in plan.walls],
                "stage": primary["stage"],
                "subject": primary["subject"],
                "roles": roles,
                "veto": {
                    "vetoed": vetoed,
                    "rules": veto_rules,
                    "steps": steps,
                    "rule_c_floor_applied_min_s": safety.get("rule_c_floor_applied_min_s"),
                    "pair_class": (
                        (breach or {}).get("pair_class") if (abs_veto and breach) else None
                    ),
                },
                "safety": safety,
                "ttc_breach": None if breach is None else dict(breach),
                "assembly_bar": rec["bar"],
                "switch": rec["switch"],
                "chosen": rec is chosen_rec,
            }
            row.update(prediction_block(pred))
            rows.append(row)
        return rows


# ------------------------------------------------------------------- replay
@dataclass(frozen=True)
class ReplayResult:
    plan: PlanCandidate
    plan_key: str
    kind: str  # "full" | "trivial"
    changes: tuple  # ("kind", "subject", "label") of every surviving change


def _noisy(pred: Prediction, noise) -> Prediction:
    if noise is None:
        return pred
    from dataclasses import replace

    noise = np.asarray(noise, dtype=float)
    return replace(
        pred,
        plan_delay_s=np.asarray(pred.plan_delay_s, dtype=float) + noise,
        delay_vs_native_s=np.asarray(pred.delay_vs_native_s, dtype=float) + noise,
    )


def replay_decision(
    rows: Sequence[Mapping],
    config: PlannerConfig,
    *,
    noise_fn: Optional[Callable[[Mapping], Optional[np.ndarray]]] = None,
) -> ReplayResult:
    """Re-apply the planner selection rule to ONE decision's logged rows.

    ``noise_fn(row) -> None | [M, K] array`` (default None: no noise) is
    added to the priced candidate's plan_delay_s and delay_vs_native_s
    before any rule runs; the probe/current row is perturbed like any other
    priced candidate. Returns the plan the rule chooses."""
    from .loop import _Change, _release_rule_fires  # late: loop imports this module

    if not rows:
        raise ValueError("no candidate rows")
    plans, preds, by_key = [], [], {}
    for row in rows:
        plan = plan_from_row(row)
        pred = _noisy(prediction_from_row(row), None if noise_fn is None else noise_fn(row))
        plans.append(plan)
        preds.append(pred)
        by_key[plan_key(plan)] = len(plans) - 1

    def role_rows(stage):
        return [
            (i, role) for i, row in enumerate(rows) for role in row["roles"] if role["stage"] == stage
        ]

    def lookup(plan):
        try:
            return preds[by_key[plan_key(plan)]]
        except KeyError:
            raise UnpricedPlanError(plan_label(plan)) from None

    current_idx = [i for i, _ in role_rows("current")]
    replacement = [i for i, _ in role_rows("replacement")]
    individual = role_rows("individual")
    if not current_idx or not (individual or role_rows("wall_release") or role_rows("group")):
        # trivial decision: cadence tick / abstain / zero decision units
        start_i = (current_idx or [i for i, _ in role_rows("probe")] or [0])[0]
        cur_safety = safety_check(preds[start_i], config)
        chosen_i = replacement[-1] if (replacement and not cur_safety.ok) else start_i
        return ReplayResult(plans[chosen_i], rows[chosen_i]["plan_key"], "trivial", ())

    start = plans[current_idx[0]]
    current = preds[current_idx[0]]
    current_safety = safety_check(current, config)
    start_key = plan_key(start)
    decision_ids = sorted({role["subject"] for _, role in individual})

    # -- step 1: individual picks
    picks = {}
    for cav in decision_ids:
        table = {role["action"]: i for i, role in individual if role["subject"] == cav}
        best = None
        for rank, action in enumerate(config.actions):
            if action not in table:
                continue
            entry = preds[table[action]]
            entry_safety = safety_check(entry, config)
            if not entry_safety.ok:
                continue
            key = (median_cells(entry.delay_vs_native_s), rank)
            if best is None or key < best[0]:
                best = (key, action)
        picks[cav] = best[1] if best is not None else start.per_cav[cav]

    # -- steps 3+4 per group
    forced_native = native_return_check(current, config, safety=current_safety).should_return
    group_labels = []
    for _, role in role_rows("group") + role_rows("wall"):
        if role.get("group") is not None and role["group"] not in group_labels:
            group_labels.append(role["group"])
    changes = []
    for glabel in group_labels:
        group = tuple(glabel.split("+"))
        cands = sorted(
            [
                (role["order"], i, role)
                for i, row in enumerate(rows)
                for role in row["roles"]
                if role.get("group") == glabel and role.get("order") is not None
            ],
            key=lambda t: t[0],
        )
        best = None
        for _, i, role in cands:
            cand_plan, cand_pred = plans[i], preds[i]
            if plan_key(cand_plan) == start_key:
                cand_safety = current_safety
                cand_pred = current
            else:
                cand_safety = safety_check(cand_pred, config)
            if not cand_safety.ok:
                continue
            wall_id = role.get("wall_id")
            if wall_id is not None:
                bar_ok, _ = assembly_bar_ok(cand_pred, wall_id, config)
                if not bar_ok:
                    continue
            key = candidate_sort_key(cand_pred, cand_plan)
            if best is None or key < best[0]:
                best = (key, role, cand_plan, cand_pred, cand_safety)
        if best is not None:
            _, role, cand_plan, cand_pred, cand_safety = best
            if plan_key(cand_plan) == start_key:
                continue
            wall_id = role.get("wall_id")
            wall_add = wall_id is not None
            verdict = switch_check(current, cand_pred, config, wall=wall_add, safety=cand_safety)
            if not verdict.passes:
                if forced_native and not wall_add:
                    changes.append(
                        _Change("actions", glabel, {m: "native" for m in group}, None, None,
                                median_cells(current.delay_vs_native_s), 1, "free->native")
                    )
                continue
            changed = {
                cav: action
                for cav, action in cand_plan.per_cav.items()
                if cav in start.per_cav and action != start.per_cav[cav]
            }
            if wall_add:
                spec = next(w for w in cand_plan.walls if w.wall_id == wall_id)
                changes.append(
                    _Change("wall_add", wall_id, changed, spec, None,
                            verdict.median_saving_s, 2, role["label"])
                )
            else:
                changes.append(
                    _Change("actions", glabel, changed, None, None,
                            verdict.median_saving_s, 0, role["label"])
                )
        elif forced_native:
            changes.append(
                _Change("actions", glabel, {m: "native" for m in group}, None, None,
                        median_cells(current.delay_vs_native_s), 1, "free->native")
            )

    # -- hold-wall release-now candidates
    for i, role in role_rows("wall_release"):
        release_pred = preds[i]
        release_safety = safety_check(release_pred, config)
        verdict = switch_check(current, release_pred, config, safety=release_safety)
        if release_safety.ok and _release_rule_fires(verdict, config):
            wall_id = role["subject"]
            changes.append(
                _Change("wall_release", wall_id, {}, None, wall_id, verdict.median_saving_s, 0,
                        "release")
            )

    plan = _combine_replay(start, changes, lookup, config)
    return ReplayResult(
        plan,
        json.dumps(plan_key(plan), sort_keys=True, default=str),
        "full",
        tuple((c.kind, c.subject, c.label) for c in changes),
    )


def _combine_replay(start, changes, lookup, cfg) -> PlanCandidate:
    """Mirror of ``Planner._combine`` (no registry side effects)."""
    actions_applied: dict = {}
    wall_adds: list = []
    wall_releases: list = []
    for change in changes:
        if change.kind == "actions":
            actions_applied.update(change.actions)
        elif change.kind == "wall_add":
            wall_adds.append(change)
        else:
            wall_releases.append(change)
    if not (actions_applied or wall_adds or wall_releases):
        return start
    if cfg.max_walls is not None:
        over = len(start.walls) + len(wall_adds) - cfg.max_walls
        for change in sorted(wall_adds, key=lambda c: (c.saving, c.subject))[: max(0, over)]:
            wall_adds.remove(change)

    def build():
        plan = with_actions(start, actions_applied)
        for change in wall_adds:
            plan = with_wall(with_actions(plan, change.actions), change.wall)
        for change in wall_releases:
            plan = with_wall_release(plan, change.wall_id)
        return plan

    plan = build()
    verdict = safety_check(lookup(plan), cfg)
    if not verdict.ok and cfg.combined_plan_fallback:
        droppable = sorted(
            (c for c in changes if c.drop_class in (0, 1)),
            key=lambda c: (c.drop_class, c.saving, c.subject),
        )
        for change in droppable:
            if verdict.ok:
                break
            if change.kind == "actions":
                for cav in change.actions:
                    actions_applied.pop(cav, None)
            elif change.kind == "wall_release":
                wall_releases = [c for c in wall_releases if c.subject != change.subject]
            plan = build()
            verdict = safety_check(lookup(plan), cfg)
        if not verdict.ok and wall_adds:
            for change in sorted(wall_adds, key=lambda c: (c.saving, c.subject)):
                if verdict.ok:
                    break
                wall_adds = [c for c in wall_adds if c.subject != change.subject]
                plan = build()
                verdict = safety_check(lookup(plan), cfg)
    free_ids = sorted(start.per_cav)
    if not verdict.ok and cfg.combined_plan_fallback:
        plan = with_actions(build(), {c: "native" for c in free_ids})
    return plan


def group_rows_by_decision(rows: Sequence[Mapping]) -> dict:
    """{decision_index: [rows in cand_index order]} of a candidates.jsonl."""
    out: dict = {}
    for row in rows:
        out.setdefault(int(row["decision_index"]), []).append(row)
    for group in out.values():
        group.sort(key=lambda r: r["cand_index"])
    return out
