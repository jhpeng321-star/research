"""Planner decision loop: PL01_PLAN steps 0-5 with wall-v2 group actions.

PL01-B scope on top of PL01-A (steps 0-5 without walls): the proposer, wall
candidates in the group check (folded groups, higher wall bar), the
assemble -> hold -> release/abort lifecycle with the assembly lock, multiple
walls with max_walls, and the planner-owned wall registry. Releases after
hold are encoded v3-style: the wall is OMITTED from the emitted plan and its
members are assigned native in the same plan (never an explicit release
phase, never for an assembling wall). Zero simulation.

Per tick (full cadence decide):
0. Wall housekeeping first (safety is a hard limit): dissolve walls released
   on the previous tick; assembled events flip assemble -> hold; a wall in
   ASSEMBLE ends ONLY by safety abort (per-wall prediction signal, or the
   plan predicted unsafe); a HOLD wall on an unsafe plan releases now.
0b. Abstain on poor merge observation; the owner-approved exception still
   replaces the current plan (free CAVs -> native) when it is predicted
   unsafe.
1. Individual check per free CAV (safe action, lowest median
   delay_vs_native_s over ALL M*K cells; ties -> action order
   native < damping < pi-external-u, then sorted ids).
2. Interaction check from the step-1 predictions (no extra model calls):
   measured trajectory deviation -> edges -> components -> cap splitting.
3. Group check: candidates = joint individual picks, flip-one, keep-current,
   plus wall candidates from the proposer whose members lie in the group
   (measured-affected CAVs fold into the wall's group; cap applies, members
   count, members never dropped). Hold walls also get a hold-vs-release-now
   comparison.
4. Switch rule vs the current plan (paired saving over ALL M*K cells). Wall
   additions must pass the higher bar (wall_agreement / wall_margin_k);
   everything else uses the normal bar. Native-return exception: a group
   returns to native when the current plan is unsafe or worse than native
   (locked wall members structurally excepted).
5. Combine; if unsafe and combined_plan_fallback, drop changes ascending by
   predicted saving -- assembling walls are dropped ONLY by the final safety
   pass (assembly lock); worst case all free CAVs -> native.

The registry (walls.WallRegistry) tracks wall_id -> members/speed/phase/
hold-release/dispatch-assembled times, updated from planner decisions,
controller_state, and prediction wall events.

Zero simulation: no SUMO/traci/libsumo import anywhere in this package.
"""

import json
import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .candidate_replay import CandidateRecorder
from .config import PlannerConfig
from .grouping import build_edges, cap_split, influences
from .interface import (
    ACTIONS,
    PlanCandidate,
    Prediction,
    WallSpec,
    WorldModel,
    locked_entries,
    plan_key,
    with_actions,
    with_wall,
    with_wall_phase,
    with_wall_release,
    without_wall,
)
from .rules import (
    abstain_check,
    candidate_sort_key,
    median_cells,
    native_return_check,
    switch_check,
)
from .safety import (
    enforce_future_gate,
    native_collision_note,
    relative_safety_diagnostic,
    safety_check,
    window_relative_note,
)
from .walls import (
    WallRegistry,
    assembly_bar_ok,
    fold_wall_group,
    feasible_wall_members,
    propose_walls,
    t0_lane_position,
)


#: T-B (#10): the abstain reason of a decision with zero eligible units
NO_DECISION_UNITS = "no_decision_units"


def select_decision_units(
    cav_ids: Sequence[str], state_samples, max_units: int
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """PL-04e2 decision-unit cap: the <= max_units CAVs nearest upstream of
    the merge that are present in the current RSU frames.

    Distance source: the state samples' optional
    ``frames_distance_to_merge_m`` mapping ({vehicle: metres to the merge
    along the route; smaller = closer to the merge}), populated by the real
    RSU observation path from the sensor frames. A CAV is CONSIDERED only if
    the freshest sample that reports it carries a finite, non-negative
    distance; ranking is (distance, id). ``cav_ids`` outside the selection
    keep their current action (returned as the second element, sorted).

    No sample carries any distance information -> everything is held
    (fail closed: nobody is considered, and the caller logs a reason).
    """
    indexed: "list[tuple[float, int, Mapping]]" = []
    for index, sample in enumerate(state_samples):
        distances = getattr(sample, "frames_distance_to_merge_m", None)
        if isinstance(distances, Mapping) and distances:
            meta = getattr(sample, "meta", None) or {}
            time_s = float(meta.get("time_s", 0.0))
            indexed.append((time_s, index, distances))
    if not indexed:
        return (), tuple(sorted(str(c) for c in cav_ids))
    indexed.sort(key=lambda entry: (entry[0], entry[1]))  # oldest first; freshest wins
    merged: "dict[str, float]" = {}
    for _time_s, _index, distances in indexed:
        for cav, distance in distances.items():
            try:
                value = float(distance)
            except (TypeError, ValueError):
                value = None
            if value is not None and math.isfinite(value) and value >= 0.0:
                merged[str(cav)] = value
            else:
                # the freshest report wins even when unusable: a CAV whose
                # latest frame carries no rankable distance is HELD
                merged.pop(str(cav), None)
    ranked = sorted(
        (str(c) for c in cav_ids if str(c) in merged),
        key=lambda cav: (merged[cav], cav),
    )
    units = tuple(ranked[: max(1, int(max_units))])
    chosen = set(units)
    held = tuple(sorted(str(c) for c in cav_ids if str(c) not in chosen))
    return units, held


def _release_rule_fires(verdict, config) -> bool:
    """The post-assembly hold-vs-release rule (PlannerConfig.wall_release_rule).

    "median_saving_gt0": release iff the median paired release-vs-hold
    saving exceeds ``wall_release_margin_s`` (default 0.0 = the historical
    positive-median rule). "switch_rule": release only if the standard
    switch rule passes (median saving > margin_floor_s + margin_k * std AND
    agreement). Any other configured name raises.
    """
    if config.wall_release_rule == "median_saving_gt0":
        return verdict.median_saving_s > config.wall_release_margin_s
    if config.wall_release_rule == "switch_rule":
        return bool(verdict.passes)
    raise ValueError(f"unknown wall_release_rule {config.wall_release_rule!r}")


@dataclass(frozen=True)
class DecisionReason:
    """One reason record per decision point (brief PL01-A requirement).

    ``median_saving_s`` is vs the comparison base of the step (positive =
    improvement): for the individual step vs native, for group/combine steps
    vs the current plan. ``margin_s``/``agreement`` are filled where the
    switch rule ran; ``action`` is one of keep/switch/native_return/veto/
    drop/commit/release/abort/hold, plus the gate-step diagnostics
    override/native_collision.
    """

    step: str
    subject: str
    candidate: str
    action: str
    median_saving_s: Optional[float] = None
    margin_s: Optional[float] = None
    agreement: Optional[float] = None
    safety_ok: Optional[bool] = None
    detail: str = ""


@dataclass(frozen=True)
class PlannerDecision:
    """Result of one ``Planner.decide`` call.

    ``chosen_prediction`` (PL-11, contract v2.1) is the world-model output
    of the plan being dispatched this tick -- the chosen plan's own
    prediction (its ``native_*`` fields are the shared native reference).
    It is LOGGING-ONLY evidence for the closed-loop runner (the optional
    Part-A v2.1 per-cell/meta fields in decisions.jsonl); NO decision rule
    reads it, and it is EXCLUDED from dataclass equality (``compare=False``:
    it carries numpy arrays, and decision determinism is defined by the
    plan + reasons, not by the logged prediction object)."""

    plan: PlanCandidate  # the plan to dispatch this tick
    reasons: Tuple[DecisionReason, ...]
    groups: Tuple[Tuple[str, ...], ...] = ()  # step-2 components (sorted)
    abstain: bool = False
    cadence_skipped: bool = False
    switched: bool = False  # plan changed vs the previous tick's plan
    chosen_prediction: Optional[Prediction] = field(  # logging-only (PL-11)
        compare=False, default=None, repr=False
    )
    # T-B (#7): one record per evaluated candidate plan (this decision) whose
    # PLAN min closing TTC fell below the absolute floor -- the triggering
    # follower/leader pair, its TTC and ``pair_class`` ('cav_involved' when
    # either vehicle is a designated CAV, 'hdv_hdv' otherwise; 'unknown_pair'
    # when the world model reports no pair). LOGGING ONLY: the absolute rule
    # itself (all pairs, floor ttc_floor_s) is unchanged. Excluded from
    # equality like chosen_prediction.
    ttc_vetoes: Tuple[dict, ...] = field(compare=False, default=(), repr=False)


@dataclass(frozen=True)
class _Change:
    """One committed-or-droppable plan change produced by steps 0-4."""

    kind: str  # "actions" | "wall_add" | "wall_release"
    subject: str
    actions: Mapping[str, str]  # per-CAV actions applied with this change
    wall: Optional[WallSpec]  # wall_add: the proposed wall spec
    wall_id: Optional[str]  # wall_release: the wall to flip to release
    saving: float  # predicted median saving vs the current plan
    drop_class: int  # 0 optional, 1 forced-safety, 2 assembling wall
    label: str  # candidate label for reason records


class Planner:
    """Stateful planner: current plan, decide cadence, and wall registry."""

    def __init__(
        self,
        world_model: WorldModel,
        config: Optional[PlannerConfig] = None,
        *,
        current_plan: Optional[PlanCandidate] = None,
        log_candidates: bool = False,
    ):
        self.world_model = world_model
        self.config = config if config is not None else PlannerConfig()
        self._current_plan = current_plan
        self._last_decide_s: Optional[float] = None
        self.wall_registry = WallRegistry()
        # T-B (#7): per-decision log of absolute min-TTC floor breaches,
        # keyed by plan key (reset at the start of every decide())
        self._ttc_veto_log: dict = {}
        self._reverted_ids: tuple = ()
        # PL-05 (contract v2): commit-window prediction kwargs go only to a
        # world model that DECLARES v2; v1/v1.1 models are called as today.
        self._model_is_v2 = str(getattr(world_model, "contract_version", "")) == "wd01_predict/2"
        # Per-candidate logging (candidates.jsonl): PURE logging from values
        # the planner already computed -- no extra prediction, no decision
        # reads any of it. ``last_candidates`` holds the rows of the most
        # recent decide() (empty when off).
        self.log_candidates = bool(log_candidates)
        self._cl: Optional[CandidateRecorder] = None
        self.last_candidates: tuple = ()

    # ------------------------------------------------------------------ api
    def decide(
        self,
        state_samples,
        now_s: float,
        *,
        controller_state: Optional[Mapping[str, Mapping]] = None,
    ) -> PlannerDecision:
        self.last_candidates = ()
        if not self.log_candidates:
            self._cl = None
            return self._decide_impl(state_samples, now_s, controller_state=controller_state)
        self._cl = CandidateRecorder()
        decision = None
        try:
            decision = self._decide_impl(state_samples, now_s, controller_state=controller_state)
            return decision
        finally:
            recorder, self._cl = self._cl, None
            try:
                chosen = None if decision is None else plan_key(decision.plan)
                self.last_candidates = tuple(
                    recorder.finalize(self.config, chosen, self._ttc_veto_log)
                )
            except Exception:
                if decision is not None:
                    raise
                self.last_candidates = ()  # never mask the original refusal

    # -- candidate-log hooks (no-ops unless log_candidates) ------------------
    def _cl_role(self, plan, stage, subject, label, **extra) -> None:
        if self._cl is not None:
            self._cl.role(plan, stage, subject, label, **extra)

    def _cl_stage(self, stage: str) -> None:
        if self._cl is not None:
            self._cl.default_stage = stage

    def _decide_impl(
        self,
        state_samples,
        now_s: float,
        *,
        controller_state: Optional[Mapping[str, Mapping]] = None,
    ) -> PlannerDecision:
        samples = list(state_samples)
        if not samples:
            raise ValueError("state_samples must be non-empty")
        self._ttc_veto_log = {}
        self._reverted_ids = ()
        locked = locked_entries(controller_state)
        if self._current_plan is None:
            self._current_plan = _initial_plan(controller_state, locked)
        if self.config.max_decision_units is not None and not self._in_cadence_window(now_s):
            # T-B (#10, owner decision addendum 3): roster CAVs NOT in the
            # decision set (the max_decision_units nearest upstream of the
            # merge) revert to NATIVE. Applied HERE, before the probe, so the
            # probe, the current-plan prediction, every lookahead candidate
            # and the native reference all carry the same reversion (branch
            # and master stay equivalent; no un-reverted prediction is
            # paid for).
            _units, _held = select_decision_units(
                sorted(self._current_plan.per_cav), samples, int(self.config.max_decision_units)
            )
            _reverted = [
                c for c in _held if self._current_plan.per_cav.get(c, "native") != "native"
            ]
            if _reverted:
                self._current_plan = with_actions(
                    self._current_plan, {c: "native" for c in _reverted}
                )
                self._reverted_ids = tuple(_reverted)
        # Adopt execution reality before validating: walls the dispatcher has
        # already ended (members unlocked in controller_state) are dissolved
        # from both the registry and the current plan.
        dissolved = self.wall_registry.sync_from_controller_state(controller_state)
        for wall_id in dissolved:
            if any(w.wall_id == wall_id for w in self._current_plan.walls):
                self._current_plan = without_wall(self._current_plan, wall_id)
        self.wall_registry.sync_from_plan(self._current_plan)
        _validate_locks(self._current_plan, locked, controller_state)

        # PL-05 (contract v2, G-1/G-2): the mode gate runs BEFORE any
        # decision -- a refusal raises here; a controlled-experiment
        # override is logged into every decision record of this tick. The
        # probe prediction is handed to the step functions so nothing is
        # predicted twice.
        self._cl_stage("probe")
        probe = self._predict([self._current_plan], samples, controller_state)[0]
        self._cl_role(self._current_plan, "probe", "*", "probe")
        override_note = enforce_future_gate(probe, self.config)
        gate_reasons: tuple = ()
        if override_note:
            gate_reasons = (
                DecisionReason("gate", "*", "*", "override", safety_ok=None, detail=override_note),
            )
        if self.config.mode == "offline":
            # G-2 (contract v2 §8): offline never refuses, so an unvalidated
            # relative-safety block is carried into every decision record of
            # this tick as a logged diagnostic (fail closed elsewhere:
            # live/closed_loop refuse above, controlled_experiment logs the
            # override note above).
            diagnostic = relative_safety_diagnostic(probe)
            if diagnostic:
                gate_reasons = gate_reasons + (
                    DecisionReason(
                        "gate",
                        "*",
                        "*",
                        "relative_safety_unvalidated",
                        safety_ok=None,
                        detail=diagnostic,
                    ),
                )
        if not self.config.native_collision_vetoes_plan:
            # PL-06 (owner decision 2026-09-27, option A = the default): a
            # native predicted collision is a logged 'native_collision'
            # diagnostic, never a veto (safety.fail_closed_mask already
            # honours the flag); the plan's own absolute + relative gates
            # decide. native_collision_vetoes_plan=True (option B, the
            # pre-2026-09-27 behaviour) produces NO extra reasons.
            note = native_collision_note(probe)
            if note:
                gate_reasons = gate_reasons + (
                    DecisionReason(
                        "gate",
                        "*",
                        "*",
                        "native_collision",
                        safety_ok=None,
                        detail=note
                        + " (native_collision_vetoes_plan=False; logged "
                        "diagnostic, not a veto)",
                    ),
                )

        if self._in_cadence_window(now_s):
            decision = self._cadence_tick(samples, controller_state, float(now_s), seed=probe)
        else:
            self._last_decide_s = float(now_s)
            decision = self._full_decide(samples, controller_state, float(now_s), seed=probe)
        if gate_reasons:
            decision = replace(decision, reasons=tuple(gate_reasons) + tuple(decision.reasons))
        if self._ttc_veto_log:
            decision = replace(decision, ttc_vetoes=tuple(self._ttc_veto_log.values()))
        if self._reverted_ids and not decision.switched:
            # the plan changed vs the previous tick's plan (the reversion)
            decision = replace(decision, switched=True)
        return decision

    # ------------------------------------------------------------- cadence
    def _in_cadence_window(self, now_s: float) -> bool:
        return (
            self._last_decide_s is not None
            and (now_s - self._last_decide_s) < self.config.decide_every_s
        )

    def _cadence_tick(self, samples, controller_state, now_s, seed=None) -> PlannerDecision:
        """Between cadence ticks: keep the current plan, safety check only."""
        cfg = self.config
        registry = self.wall_registry
        seed_key = plan_key(self._current_plan) if seed is not None else None
        current = self._predict(
            [self._current_plan],
            samples,
            controller_state,
            seed={seed_key: seed} if seed_key else None,
        )[0]
        safety = safety_check(current, cfg)
        self._cl_role(self._current_plan, "current", "*", "current")
        if safety.ok:
            reason = DecisionReason(
                "cadence",
                "*",
                "current",
                "keep",
                safety_ok=True,
                detail=f"within decide_every_s={cfg.decide_every_s}; safety ok",
            )
            return PlannerDecision(
                self._current_plan,
                (reason,),
                cadence_skipped=True,
                switched=False,
                chosen_prediction=current,
            )
        # Unsafe on a skipped tick: free CAVs -> native, assembling walls
        # -> safety abort, holding walls -> release now.
        plan = self._current_plan
        reasons = []
        for wall in [w for w in plan.walls if w.phase == "assemble"]:
            plan = without_wall(plan, wall.wall_id)
            registry.dissolve(wall.wall_id)
            reasons.append(
                DecisionReason(
                    "cadence",
                    wall.wall_id,
                    "safety abort",
                    "abort",
                    safety_ok=False,
                    detail="cadence safety check: plan unsafe; assembly aborted",
                )
            )
        for wall in [w for w in plan.walls if w.phase == "hold"]:
            # release encoding toward predict(): the wall STAYS in the plan
            # as phase="release", release_at_s=0.0 (release now) with its
            # locked members; members rejoin per_cav only after the
            # dispatcher reports the handback done.
            plan = with_wall_release(plan, wall.wall_id)
            registry.mark_release(wall.wall_id)
            reasons.append(
                DecisionReason(
                    "cadence",
                    wall.wall_id,
                    "hold unsafe",
                    "release",
                    safety_ok=False,
                    detail="cadence safety check: plan unsafe; hold released "
                    "(wall recast as release now, release_at_s=0.0)",
                )
            )
        free_ids = sorted(plan.per_cav)
        plan = with_actions(plan, {c: "native" for c in free_ids})
        self._cl_stage("replacement")
        replacement = self._predict([plan], samples, controller_state)[0]
        self._cl_role(plan, "replacement", "*", "free->native")
        replacement_safety = safety_check(replacement, cfg)
        self._current_plan = plan
        reasons.append(
            DecisionReason(
                "cadence",
                "*",
                "free->native",
                "native_return",
                safety_ok=replacement_safety.ok,
                detail="cadence safety check: current plan unsafe",
            )
        )
        return PlannerDecision(
            plan,
            tuple(reasons),
            cadence_skipped=True,
            switched=True,
            chosen_prediction=replacement,
        )

    # ---------------------------------------------------------- full decide
    def _full_decide(self, samples, controller_state, now_s, seed=None) -> PlannerDecision:
        cfg = self.config
        cache = {}
        seed_key = plan_key(self._current_plan) if seed is not None else None
        if seed_key:
            cache[seed_key] = seed

        def pred(plan: PlanCandidate) -> Prediction:
            key = plan_key(plan)
            if key not in cache:
                cache[key] = self._predict([plan], samples, controller_state)[0]
            return cache[key]

        def prefetch(plans_to_predict) -> None:
            """PL-03b (E): predict every uncached plan in ONE predict_many
            call (bucketed by wall-set for the §3.2 state sync) so a step's
            candidates share a single native reference computation."""
            todo = []
            seen = set()
            for plan in plans_to_predict:
                key = plan_key(plan)
                if key not in cache and key not in seen:
                    seen.add(key)
                    todo.append(plan)
            if todo:
                for plan, prediction in zip(todo, self._predict(todo, samples, controller_state)):
                    cache[plan_key(plan)] = prediction

        reasons = []
        # -- step 0a: wall housekeeping (safety is a hard limit) ------------
        self._cl_stage("housekeeping")
        start, current, walls_suppressed = self._wall_housekeeping(pred, reasons, now_s)
        self._current_plan = start
        self._cl_role(start, "current", "*", "current")
        start_key = plan_key(start)
        current_safety = safety_check(current, cfg)
        free_ids = sorted(start.per_cav)

        # -- step 0b: abstain -----------------------------------------------
        merge_observed_reason = None
        for sample in samples:
            meta = getattr(sample, "meta", None)
            candidates = []
            if callable(getattr(meta, "get", None)):
                candidates.append(meta.get("merge_observed_reason"))
            candidates.append(getattr(meta, "merge_observed_reason", None))
            value = next((v for v in candidates if v), None)
            if value:
                merge_observed_reason = str(value)
                break
        abstain = abstain_check(samples, cfg)
        if abstain.should_abstain:
            if cfg.abstain_safety_exception and not current_safety.ok:
                plan = with_actions(start, {c: "native" for c in free_ids})
                self._cl_stage("replacement")
                replacement = pred(plan)
                self._cl_role(plan, "replacement", "*", "free->native")
                safety = safety_check(replacement, cfg)
                self._current_plan = plan
                reasons.append(
                    DecisionReason(
                        "abstain",
                        "*",
                        "free->native",
                        "native_return",
                        safety_ok=safety.ok,
                        detail=(
                            "merge_observed "
                            f"{abstain.observed_fraction:.3f} < "
                            f"{cfg.abstain_min_observed} AND current unsafe "
                            "(owner exception)"
                            + (
                                f"; merge_observed_reason: {merge_observed_reason}"
                                if merge_observed_reason
                                else ""
                            )
                        ),
                    )
                )
                return PlannerDecision(
                    plan,
                    tuple(reasons),
                    abstain=True,
                    switched=plan_key(plan) != start_key,
                    chosen_prediction=replacement,
                )
            reasons.append(
                DecisionReason(
                    "abstain",
                    "*",
                    "current",
                    "keep",
                    safety_ok=current_safety.ok,
                    detail=(
                        f"merge_observed {abstain.observed_fraction:.3f} < "
                        f"{cfg.abstain_min_observed}"
                        + (
                            f"; merge_observed_reason: {merge_observed_reason}"
                            if merge_observed_reason
                            else ""
                        )
                    ),
                )
            )
            return PlannerDecision(
                start,
                tuple(reasons),
                abstain=True,
                switched=plan_key(start) != start_key,
                chosen_prediction=current,
            )

        # -- PL-04e2 decision-unit cap (before the individual stage) ---------
        # max_decision_units caps WHICH free CAVs the decision stages may
        # move: the N nearest upstream of the merge that are present in the
        # current RSU frames. Everyone else keeps the current action (they
        # stay in per_cav, receive no candidate, and are excluded from the
        # interaction groups and wall eligibility).
        decision_ids = free_ids
        if cfg.max_decision_units is not None:
            units, held = select_decision_units(
                free_ids, samples, int(cfg.max_decision_units)
            )
            decision_ids = list(units)
            # T-B (#10): the reversion of the non-selected CAVs already ran in
            # decide() BEFORE the probe (so every prediction of this decision
            # -- the probe, the current plan, every candidate and the native
            # reference -- reflects it); here it is only RECORDED.
            if self._reverted_ids:
                reasons.append(
                    DecisionReason(
                        "individual",
                        "*",
                        "revert_non_decision_units",
                        "native_return",
                        safety_ok=current_safety.ok,
                        detail=(
                            f"max_decision_units={cfg.max_decision_units}: "
                            "non-selected CAVs revert to native: "
                            + "+".join(self._reverted_ids)
                        ),
                    )
                )
            if units:
                detail = (
                    f"max_decision_units={cfg.max_decision_units}: considering "
                    + "+".join(units)
                )
            else:
                detail = (
                    f"max_decision_units={cfg.max_decision_units}: no CAV has a "
                    "frames_distance_to_merge_m entry in any state sample; "
                    "fail closed -- every CAV keeps its current action"
                )
            if held:
                detail += "; holding " + "+".join(held)
            reasons.append(
                DecisionReason(
                    "individual",
                    "*",
                    f"decision_units={'+'.join(units) if units else 'none'}",
                    "cap",
                    detail=detail,
                )
            )
            if not units:
                # T-B (#10): zero eligible units is an ABSTAIN with the
                # named reason, never a decision made
                reasons.append(
                    DecisionReason(
                        "abstain",
                        "*",
                        NO_DECISION_UNITS,
                        "keep",
                        safety_ok=current_safety.ok,
                        detail=(
                            f"{NO_DECISION_UNITS}: max_decision_units="
                            f"{cfg.max_decision_units} but no roster CAV is a "
                            "rankable decision unit at this decision; every CAV "
                            "is native"
                        ),
                    )
                )
                return PlannerDecision(
                    start,
                    tuple(reasons),
                    abstain=True,
                    switched=plan_key(start) != start_key,
                    chosen_prediction=current,
                )

        # -- step 1: individual check (PL-03b: ONE batched predict_many) ----
        alternative_plans = []
        for cav in decision_ids:
            for action in cfg.actions:
                if action != start.per_cav[cav]:
                    alternative_plans.append(with_actions(start, {cav: action}))
        self._cl_stage("individual")
        prefetch(alternative_plans)
        alternatives = {}
        picks = {}
        for cav in decision_ids:
            current_action = start.per_cav[cav]
            table = {}
            best = None
            for rank, action in enumerate(cfg.actions):
                if action == current_action:
                    entry_pred, entry_safety = current, current_safety
                else:
                    entry_pred = pred(with_actions(start, {cav: action}))
                    entry_safety = safety_check(entry_pred, cfg)
                if self._cl is not None:
                    self._cl.role(
                        start if action == current_action else with_actions(start, {cav: action}),
                        "individual",
                        cav,
                        f"{cav}->{action}",
                        action=action,
                    )
                table[action] = entry_pred
                if not entry_safety.ok:
                    continue
                key = (median_cells(entry_pred.delay_vs_native_s), rank)
                if best is None or key < best[0]:
                    best = (key, action, entry_pred, entry_safety)
            alternatives[cav] = table
            if best is None:
                picks[cav] = current_action
                reasons.append(
                    DecisionReason(
                        "individual",
                        cav,
                        f"{cav}:*",
                        "veto",
                        safety_ok=False,
                        detail="no safe action; keep current",
                    )
                )
            else:
                _, action, entry_pred, entry_safety = best
                picks[cav] = action
                reasons.append(
                    DecisionReason(
                        "individual",
                        cav,
                        f"{cav}->{action}",
                        "keep",
                        median_saving_s=-median_cells(entry_pred.delay_vs_native_s),
                        safety_ok=entry_safety.ok,
                        detail="best safe action by median delay_vs_native_s "
                        "(saving vs native shown)",
                    )
                )

        # -- step 2: interaction check (no extra model calls) ----------------
        pairwise = influences(alternatives, current, decision_ids)
        edges = build_edges(decision_ids, pairwise, cfg)
        groups = cap_split(decision_ids, edges, cfg.group_cap)
        reasons.append(
            DecisionReason(
                "interaction",
                "*",
                f"{len(edges)} edges",
                "groups",
                detail="components: " + "; ".join("+".join(g) for g in groups),
            )
        )

        # -- wall candidates merge the groups containing their members ------
        # (side-by-side members often share no step-2 edge; the wall brings
        # them into one evaluation group). A tick that aborted a wall proposes
        # no new walls: the members hand back to native first.
        candidates = ()
        if cfg.max_walls == 0:
            # PL-04e2 --walls off: no wall proposals at all (max_walls=0);
            # the proposer never runs, so no lookahead is spent on wall
            # candidates and no wall can be assembled.
            reasons.append(
                DecisionReason(
                    "wall",
                    "*",
                    "new wall proposals",
                    "veto",
                    safety_ok=False,
                    detail="walls off (max_walls=0): no wall proposals",
                )
            )
        elif not walls_suppressed:
            eligible_ids, excluded = feasible_wall_members(
                decision_ids, controller_state, samples, cfg, t0_s=now_s
            )
            if excluded:
                reasons.append(
                    DecisionReason(
                        "wall",
                        "*",
                        "member liveness/frame filter",
                        "veto",
                        safety_ok=False,
                        detail="wall proposer feasibility filter excluded: " + "; ".join(excluded),
                    )
                )
            stage_stats: dict = {}
            candidates = propose_walls(
                t0_lane_position(current, eligible_ids), eligible_ids, cfg, stats=stage_stats
            )
            reasons.append(
                DecisionReason(
                    "wall",
                    "*",
                    "proposer stage counts",
                    "keep",
                    detail="stages: " + json.dumps(stage_stats, sort_keys=True),
                )
            )
        else:
            reasons.append(
                DecisionReason(
                    "wall",
                    "*",
                    "new wall proposals",
                    "veto",
                    safety_ok=False,
                    detail="wall safety abort this tick; members -> native before any new proposal",
                )
            )
        member_group = {cav: gi for gi, group in enumerate(groups) for cav in group}
        parent = list(range(len(groups)))
        component_size = [len(group) for group in groups]

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        # PL-03b (A): the wall-candidate union-find MUST respect group_cap —
        # chained cross-section candidates used to merge groups far beyond
        # it (observed 8 > 5). A candidate whose merge would exceed the cap
        # is dropped here with a named veto reason.
        walls_dropped_by_cap = []
        for candidate in candidates:
            roots_c = sorted({find(member_group[m]) for m in candidate.spec.members})
            merged = sum(component_size[r] for r in roots_c)
            if len(roots_c) > 1 and merged > cfg.group_cap:
                walls_dropped_by_cap.append(candidate)
                continue
            keep = roots_c[0]
            for other in roots_c[1:]:
                parent[other] = keep
                component_size[keep] += component_size[other]
        for candidate in walls_dropped_by_cap:
            reasons.append(
                DecisionReason(
                    "wall",
                    candidate.spec.wall_id,
                    "wall candidate",
                    "veto",
                    safety_ok=False,
                    detail=(
                        f"wall candidate dropped: merging its members would "
                        f"push the evaluation group beyond group_cap="
                        f"{cfg.group_cap}"
                    ),
                )
            )
        roots = sorted({find(gi) for gi in range(len(groups))})
        groups = tuple(
            tuple(
                sorted(cav for gi in range(len(groups)) if find(gi) == root for cav in groups[gi])
            )
            for root in roots
        )
        member_group = {cav: gi for gi, group in enumerate(groups) for cav in group}
        group_walls: "dict[int, list]" = {gi: [] for gi in range(len(groups))}
        for candidate in candidates:
            group_walls[member_group[candidate.spec.members[0]]].append(candidate)

        # PL-03b (E): prefetch every step-3 candidate plan (joint, flips,
        # keep, plain and folded wall plans) in ONE predict_many call per
        # wall-set bucket, so the per-group evaluation below is served from
        # the cache
        step3_plans: list = []
        for gi, group in enumerate(groups):
            joint = {m: picks[m] for m in group}
            step3_plans.append(with_actions(start, joint))
            for member in group:
                for action in cfg.actions:
                    if action != picks[member]:
                        flipped = dict(joint)
                        flipped[member] = action
                        step3_plans.append(with_actions(start, flipped))
            step3_plans.append(with_actions(start, {m: start.per_cav[m] for m in group}))
            for candidate in group_walls[gi]:
                step3_plans.append(with_wall(start, candidate.spec))
                wall_actions = {m: picks[m] for m in group if m not in set(candidate.spec.members)}
                step3_plans.append(with_wall(with_actions(start, wall_actions), candidate.spec))
        self._cl_stage("group")
        prefetch(step3_plans)

        # -- steps 3+4: group check, switch rule, hold-vs-release -------------
        forced_native = native_return_check(current, cfg, safety=current_safety).should_return
        changes = []
        for gi, group in enumerate(groups):
            change = self._decide_group(
                group,
                start,
                current,
                current_safety,
                picks,
                pred,
                reasons,
                forced_native,
                group_walls[gi],
                decision_ids,
            )
            if change is not None:
                changes.append(change)
        # PL-03b (E): batch the hold-wall release candidates (one call)
        self._cl_stage("wall_release")
        prefetch([with_wall_release(start, w.wall_id) for w in start.walls if w.phase == "hold"])
        changes.extend(self._decide_hold_walls(start, current, current_safety, pred, reasons))

        # -- step 5: combine, max_walls, safety fallback ----------------------
        self._cl_stage("combined")
        plan = self._combine(start, changes, pred, reasons, now_s)
        self._current_plan = plan
        # PL-11 (contract v2.1): the dispatched plan's own prediction rides
        # along for LOGGING ONLY (pred() is the plan-keyed cache, so this
        # re-reads the prediction the combine step already evaluated).
        return PlannerDecision(
            plan,
            tuple(reasons),
            groups=groups,
            switched=plan_key(plan) != start_key,
            chosen_prediction=pred(plan),
        )

    # --------------------------------------------------- wall housekeeping
    def _wall_housekeeping(self, pred, reasons, now_s):
        """Safety-first wall events; returns (plan, current, aborted_any).

        Applies, in order: dissolution of walls released on the previous
        tick; assembled events (phase -> hold); safety aborts of ASSEMBLE
        walls (per-wall prediction signal, or the plan predicted unsafe);
        release of HOLD walls while the plan is still predicted unsafe.
        ``aborted_any`` suppresses new wall proposals for the same tick.
        """
        cfg = self.config
        registry = self.wall_registry
        plan = self._current_plan

        # 1. releases already handed back are dissolved at decide() start
        # (dispatcher reports the members unlocked). A release still pending
        # handback STAYS in the plan as phase="release" (re-asserted every
        # tick; a locked CAV may never move into per_cav, section 3.3).
        for record in registry.release_ready():
            if any(w.wall_id == record.wall_id for w in plan.walls):
                reasons.append(
                    DecisionReason(
                        "wall",
                        record.wall_id,
                        "release pending",
                        "keep",
                        detail="release candidate re-asserted; awaiting "
                        "dispatcher handback (members still locked)",
                    )
                )
        current = pred(plan)

        # 2. assembled events: assembly complete -> phase hold
        assembled = registry.assembled_ids(plan, current, cfg)
        for wall_id in assembled:
            plan = with_wall_phase(plan, wall_id, "hold")
            registry.mark_assembled(wall_id, now_s)
            reasons.append(
                DecisionReason(
                    "wall",
                    wall_id,
                    "assembled",
                    "hold",
                    detail=f"assembly complete at t={now_s:g}; hold phase",
                )
            )
        if assembled:
            current = pred(plan)

        # 3. safety aborts: the ONLY way an ASSEMBLE wall ends; never for
        # benefit reasons.
        current_safety = safety_check(current, cfg)
        abort_ids = set(registry.abort_ids(plan, current, cfg))
        if not current_safety.ok:
            abort_ids |= {w.wall_id for w in plan.walls if w.phase == "assemble"}
        for wall_id in sorted(abort_ids):
            plan = without_wall(plan, wall_id)
            registry.dissolve(wall_id)
            reasons.append(
                DecisionReason(
                    "wall",
                    wall_id,
                    "safety abort",
                    "abort",
                    safety_ok=False,
                    detail="assembly aborted for safety; members -> native",
                )
            )
        if abort_ids:
            current = pred(plan)
            current_safety = safety_check(current, cfg)

        # 4. still unsafe: holding walls release now (hold unsafe -> release;
        # released walls are omitted, members -> native in the same plan)
        if not current_safety.ok:
            for wall in [w for w in plan.walls if w.phase == "hold"]:
                plan = with_wall_release(plan, wall.wall_id)
                registry.mark_release(wall.wall_id)
                reasons.append(
                    DecisionReason(
                        "wall",
                        wall.wall_id,
                        "hold unsafe",
                        "release",
                        safety_ok=False,
                        detail="hold predicted unsafe; release now (wall recast "
                        "as release, release_at_s=0.0)",
                    )
                )
            current = pred(plan)
        return plan, current, bool(abort_ids)

    # ------------------------------------------------------- group decisions
    def _decide_group(
        self,
        group,
        start,
        current,
        current_safety,
        picks,
        pred,
        reasons,
        forced_native,
        wall_candidates,
        free_ids,
    ):
        """Steps 3+4 for one component; returns a _Change or None."""
        cfg = self.config
        label = _group_label(group)
        current_actions = {m: start.per_cav[m] for m in group}
        # (label, plan, wall_spec_or_None, fold_detail)
        candidates = []
        if len(group) == 1:
            cav = group[0]
            candidates.append(
                (f"{cav}->{picks[cav]}", with_actions(start, {cav: picks[cav]}), None, "")
            )
        else:
            joint = {m: picks[m] for m in group}
            candidates.append(
                ("joint " + _actions_label(joint), with_actions(start, joint), None, "")
            )
            for member in group:
                for action in cfg.actions:
                    if action != picks[member]:
                        flipped = dict(joint)
                        flipped[member] = action
                        candidates.append(
                            (f"flip {member}->{action}", with_actions(start, flipped), None, "")
                        )
            candidates.append(("keep_current", with_actions(start, current_actions), None, ""))
        for candidate in wall_candidates:
            plain = with_wall(start, candidate.spec)
            plain_pred = pred(plain)
            self._cl_role(
                plain,
                "wall",
                label,
                f"wall_plain {candidate.spec.wall_id}",
                group=label,
                wall_id=candidate.spec.wall_id,
            )
            fold = fold_wall_group(candidate.spec, plain_pred, current, free_ids, cfg)
            if fold is None:
                reasons.append(
                    DecisionReason(
                        "group",
                        label,
                        f"wall {candidate.spec.wall_id}",
                        "veto",
                        safety_ok=False,
                        detail=(f"wall members alone exceed group_cap={cfg.group_cap}"),
                    )
                )
                continue
            folded, dropped = fold
            detail = "folded: " + ("+".join(folded) if folded else "none")
            if dropped:
                detail += "; dropped weakest: " + "+".join(dropped)
            wall_actions = {m: picks[m] for m in group if m not in set(candidate.spec.members)}
            candidates.append(
                (
                    f"wall {candidate.spec.wall_id}",
                    with_wall(with_actions(start, wall_actions), candidate.spec),
                    candidate.spec,
                    detail,
                )
            )

        best = None
        for cand_order, (cand_label, cand_plan, wall_spec, detail) in enumerate(candidates):
            if plan_key(cand_plan) == plan_key(start):
                cand_pred, cand_safety = current, current_safety
            else:
                cand_pred = pred(cand_plan)
                cand_safety = safety_check(cand_pred, cfg)
            self._cl_role(
                cand_plan,
                "wall" if wall_spec is not None else "group",
                label,
                cand_label,
                group=label,
                order=cand_order,
                wall_id=None if wall_spec is None else wall_spec.wall_id,
            )
            if not cand_safety.ok:
                veto_detail = detail or "candidate predicted unsafe"
                # PL-07: name the additive window-only relative veto when it
                # fired (SafetyVerdict.window_relative_veto)
                window_note = window_relative_note(cand_safety)
                if window_note:
                    veto_detail += "; " + window_note
                reasons.append(
                    DecisionReason(
                        "group",
                        label,
                        cand_label,
                        "veto",
                        safety_ok=False,
                        detail=veto_detail,
                    )
                )
                continue
            if wall_spec is not None:
                # PL-05 (contract v2): the wall's higher bar additionally
                # requires the future-assembly heads (median
                # p_assembly_completes / p_assembly_abort); pre-v2
                # predictions pass vacuously.
                bar_ok, bar_detail = assembly_bar_ok(cand_pred, wall_spec.wall_id, cfg)
                if self._cl is not None:
                    self._cl.bar(cand_plan, bar_ok, bar_detail)
                if not bar_ok:
                    reasons.append(
                        DecisionReason(
                            "group",
                            label,
                            cand_label,
                            "veto",
                            safety_ok=cand_safety.ok,
                            detail=((detail + "; ") if detail else "")
                            + f"v2 assembly bar failed: {bar_detail}",
                        )
                    )
                    continue
            key = candidate_sort_key(cand_pred, cand_plan)
            if best is None or key < best[0]:
                best = (key, cand_label, cand_plan, cand_pred, cand_safety, wall_spec, detail)

        if best is not None:
            _, cand_label, cand_plan, cand_pred, cand_safety, wall_spec, detail = best
            if plan_key(cand_plan) == plan_key(start):
                reasons.append(
                    DecisionReason(
                        "group",
                        label,
                        cand_label,
                        "keep",
                        safety_ok=current_safety.ok,
                        detail="best safe candidate is the current plan",
                    )
                )
                return None
            wall_add = wall_spec is not None
            verdict = switch_check(current, cand_pred, cfg, wall=wall_add, safety=cand_safety)
            if self._cl is not None:
                self._cl.switch(cand_plan, "group", verdict, wall=wall_add)
            record = DecisionReason(
                "group",
                label,
                cand_label,
                "switch" if verdict.passes else "keep",
                median_saving_s=verdict.median_saving_s,
                margin_s=verdict.margin_s,
                agreement=verdict.agreement,
                safety_ok=cand_safety.ok,
                detail=((detail + "; ") if detail else "")
                + (
                    "wall bar passed"
                    if wall_add and verdict.passes
                    else "wall bar failed"
                    if wall_add
                    else f"switch rule {'passed' if verdict.passes else 'failed'}"
                ),
            )
            reasons.append(record)
            if not verdict.passes:
                if forced_native and not wall_add:
                    native_actions = {m: "native" for m in group}
                    reasons.append(
                        DecisionReason(
                            "group",
                            label,
                            "free->native",
                            "native_return",
                            median_saving_s=median_cells(current.delay_vs_native_s),
                            safety_ok=current_safety.ok,
                            detail="switch rule failed AND current plan is unsafe "
                            "or worse than native",
                        )
                    )
                    return _Change(
                        "actions",
                        label,
                        native_actions,
                        None,
                        None,
                        median_cells(current.delay_vs_native_s),
                        1,
                        "free->native",
                    )
                return None
            changed_actions = {
                cav: action
                for cav, action in cand_plan.per_cav.items()
                if cav in start.per_cav and action != start.per_cav[cav]
            }
            if wall_add:
                return _Change(
                    "wall_add",
                    wall_spec.wall_id,
                    changed_actions,
                    wall_spec,
                    None,
                    verdict.median_saving_s,
                    2,
                    cand_label,
                )
            return _Change(
                "actions",
                label,
                changed_actions,
                None,
                None,
                verdict.median_saving_s,
                0,
                cand_label,
            )

        if forced_native:
            native_actions = {m: "native" for m in group}
            reasons.append(
                DecisionReason(
                    "group",
                    label,
                    "free->native",
                    "native_return",
                    median_saving_s=median_cells(current.delay_vs_native_s),
                    safety_ok=current_safety.ok,
                    detail="no safe candidate AND current plan is unsafe or worse than native",
                )
            )
            return _Change(
                "actions",
                label,
                native_actions,
                None,
                None,
                median_cells(current.delay_vs_native_s),
                1,
                "free->native",
            )
        reasons.append(
            DecisionReason(
                "group",
                label,
                "*",
                "keep",
                safety_ok=current_safety.ok,
                detail="no safe candidate; current stays",
            )
        )
        return None

    def _decide_hold_walls(self, start, current, current_safety, pred, reasons):
        """Hold-vs-release-now for every assembled (hold) wall.

        The decision is the named ``PlannerConfig.wall_release_rule``:
        "median_saving_gt0" releases when the median paired release-vs-hold
        saving exceeds ``wall_release_margin_s`` (default 0.0: the
        historical positive-median behavior); "switch_rule" releases only
        when the standard switch rule passes (median saving > margin_floor_s
        + margin_k * std AND agreement). The fired rule is recorded in the
        wall reason record. Release encoding toward predict() (contract
        3.3/5): the release candidate KEEPS the wall as phase="release",
        release_at_s=0.0 (release now, offset from t0) with its locked
        members -- locked CAVs may never move into per_cav -- and members
        return to per_cav as native once the dispatcher reports the handback
        done. This is the per-wall form of "hold is worse than native" under
        the N1 reference. An unsafe hold
        is already force-released by housekeeping regardless of the rule;
        an unsafe release candidate is vetoed (never switch into unsafe).
        """
        cfg = self.config
        changes = []
        for wall in sorted((w for w in start.walls if w.phase == "hold"), key=lambda w: w.wall_id):
            release_plan = with_wall_release(start, wall.wall_id)
            release_pred = pred(release_plan)
            release_safety = safety_check(release_pred, cfg)
            verdict = switch_check(current, release_pred, cfg, safety=release_safety)
            if self._cl is not None:
                self._cl.switch(release_plan, "wall_release", verdict, wall=False)
                self._cl.role(release_plan, "wall_release", wall.wall_id, "release-now")
            if not release_safety.ok:
                reasons.append(
                    DecisionReason(
                        "wall",
                        wall.wall_id,
                        "release-now",
                        "veto",
                        median_saving_s=verdict.median_saving_s,
                        margin_s=verdict.margin_s,
                        agreement=verdict.agreement,
                        safety_ok=False,
                        detail="release predicted unsafe; hold stays",
                    )
                )
            elif _release_rule_fires(verdict, cfg):
                reasons.append(
                    DecisionReason(
                        "wall",
                        wall.wall_id,
                        "release-now",
                        "release",
                        median_saving_s=verdict.median_saving_s,
                        margin_s=verdict.margin_s,
                        agreement=verdict.agreement,
                        safety_ok=True,
                        detail=f"release rule {cfg.wall_release_rule} fired: "
                        "hold worse than native (N1: the wall is recast "
                        "as phase=release, release_at_s=0.0, members stay "
                        "locked until the dispatcher hands back)",
                    )
                )
                changes.append(
                    _Change(
                        "wall_release",
                        wall.wall_id,
                        {},
                        None,
                        wall.wall_id,
                        verdict.median_saving_s,
                        0,
                        "release",
                    )
                )
            else:
                reasons.append(
                    DecisionReason(
                        "wall",
                        wall.wall_id,
                        "release-now",
                        "keep",
                        median_saving_s=verdict.median_saving_s,
                        margin_s=verdict.margin_s,
                        agreement=verdict.agreement,
                        safety_ok=release_safety.ok,
                        detail=f"hold stays; release rule "
                        f"{cfg.wall_release_rule} did not fire (median "
                        f"saving {verdict.median_saving_s:g} vs release "
                        f"margin {cfg.wall_release_margin_s:g})",
                    )
                )
        return changes

    # ------------------------------------------------------------- combine
    def _combine(self, start, changes, pred, reasons, now_s):
        """Step 5: combine changes; safety fallback with the assembly lock."""
        cfg = self.config
        registry = self.wall_registry
        actions_applied: "dict[str, str]" = {}
        wall_adds: "list[_Change]" = []
        wall_releases: "list[_Change]" = []
        for change in changes:
            if change.kind == "actions":
                actions_applied.update(change.actions)
            elif change.kind == "wall_add":
                wall_adds.append(change)
            else:
                wall_releases.append(change)
        if not (actions_applied or wall_adds or wall_releases):
            reasons.append(
                DecisionReason(
                    "combine",
                    "*",
                    "current",
                    "keep",
                    safety_ok=safety_check(pred(start), cfg).ok,
                    detail="no group change survived the switch rule",
                )
            )
            return start

        # max_walls caps simultaneous walls (existing walls are never removed
        # to satisfy the cap; the lowest-saving new walls are dropped first).
        if cfg.max_walls is not None:
            over = len(start.walls) + len(wall_adds) - cfg.max_walls
            for change in sorted(wall_adds, key=lambda c: (c.saving, c.subject))[: max(0, over)]:
                wall_adds.remove(change)
                reasons.append(
                    DecisionReason(
                        "wall",
                        change.subject,
                        "max_walls",
                        "veto",
                        median_saving_s=change.saving,
                        detail=(
                            f"simultaneous wall cap {cfg.max_walls} reached; "
                            "lowest-saving proposal dropped"
                        ),
                    )
                )

        def priced(candidate_plan):
            self._cl_role(candidate_plan, "combined", "*", "combined")
            return pred(candidate_plan)

        def build():
            plan = with_actions(start, actions_applied)
            for change in wall_adds:
                plan = with_wall(with_actions(plan, change.actions), change.wall)
            for change in wall_releases:
                # release encoding toward predict(): keep the wall in the
                # plan as phase="release", release_at_s=0.0 (release now).
                plan = with_wall_release(plan, change.wall_id)
            return plan

        plan = build()
        verdict = safety_check(priced(plan), cfg)
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
                verdict = safety_check(priced(plan), cfg)
                reasons.append(
                    DecisionReason(
                        "combine",
                        change.subject,
                        "drop " + (change.label or change.subject),
                        "drop",
                        median_saving_s=change.saving,
                        safety_ok=verdict.ok,
                        detail="combined plan unsafe; reverted change",
                    )
                )
            if not verdict.ok and wall_adds:
                # Assembly lock: an assembling wall is dropped ONLY here,
                # i.e. only for safety.
                for change in sorted(wall_adds, key=lambda c: (c.saving, c.subject)):
                    if verdict.ok:
                        break
                    wall_adds = [c for c in wall_adds if c.subject != change.subject]
                    plan = build()
                    verdict = safety_check(priced(plan), cfg)
                    reasons.append(
                        DecisionReason(
                            "wall",
                            change.subject,
                            "safety abort (fallback)",
                            "abort",
                            median_saving_s=change.saving,
                            safety_ok=verdict.ok,
                            detail="combined plan unsafe; assembly aborted for "
                            "safety (the only benefit-independent drop)",
                        )
                    )
        free_ids = sorted(start.per_cav)
        if not verdict.ok and cfg.combined_plan_fallback:
            plan = with_actions(build(), {c: "native" for c in free_ids})
            verdict = safety_check(priced(plan), cfg)
            reasons.append(
                DecisionReason(
                    "combine",
                    "*",
                    "all free -> native",
                    "commit",
                    safety_ok=verdict.ok,
                    detail="combined fallback exhausted; worst-case native",
                )
            )
        elif not verdict.ok:
            # combined_plan_fallback is False: the combined plan is dispatched
            # as decided by steps 0-4 even though it is predicted unsafe.
            reasons.append(
                DecisionReason(
                    "combine",
                    "*",
                    _actions_label(actions_applied),
                    "commit",
                    safety_ok=False,
                    detail="combined plan unsafe; combined_plan_fallback disabled",
                )
            )
        else:
            walls_note = ""
            if wall_adds:
                walls_note += "; walls +" + ",".join(c.subject for c in wall_adds)
            if wall_releases:
                walls_note += "; walls release:" + ",".join(c.subject for c in wall_releases)
            reasons.append(
                DecisionReason(
                    "combine",
                    "*",
                    _actions_label(actions_applied) + walls_note,
                    "commit",
                    safety_ok=True,
                    detail="combined plan safe",
                )
            )
        self._current_plan = plan
        for change in wall_adds:
            registry.register_spec(change.wall, dispatched_at_s=now_s)
        for change in wall_releases:
            # the release is encoded in the candidate (phase="release",
            # release_at_s=0.0); the registry entry dissolves when the
            # dispatcher reports the handback done (members unlocked).
            registry.mark_release(change.wall_id)
        return plan

    # ------------------------------------------------------------ plumbing
    def _predict(self, plans, samples, controller_state, seed=None):
        """Predict plans, honouring the decide() probe seed (a prediction
        already computed for the mode gate is reused, never recomputed)."""
        if seed:
            remaining = [plan for plan in plans if plan_key(plan) not in seed]
        else:
            remaining = list(plans)
        preds = self._predict_uncached(remaining, samples, controller_state)
        self._note_ttc_vetoes(remaining, preds)
        if self._cl is not None:
            for plan, prediction in zip(remaining, preds):
                self._cl.register(plan, prediction)
        out = []
        iterator = iter(preds)
        for plan in plans:
            key = plan_key(plan)
            if seed and key in seed:
                out.append(seed[key])
            else:
                out.append(next(iterator))
        return out

    def _note_ttc_vetoes(self, plans, preds) -> None:
        """T-B (#7): record the triggering pair of every candidate whose PLAN
        min closing TTC is below the absolute floor in any cell (logging
        only; safety.absolute_safety_mask is the rule and is unchanged)."""
        floor = float(self.config.ttc_floor_s)
        for plan, prediction in zip(plans, preds):
            try:
                ttc = np.asarray(prediction.min_ttc_s, dtype=float).reshape(-1)
            except (TypeError, ValueError, AttributeError):
                continue
            bad = [k for k in range(ttc.size) if np.isfinite(ttc[k]) and ttc[k] < floor]
            if not bad:
                continue
            key = plan_key(plan)
            if key in self._ttc_veto_log:
                continue
            k = min(bad, key=lambda i: (float(ttc[i]), i))
            meta = getattr(prediction, "meta", None) or {}
            pairs = meta.get("min_ttc_pairs") if hasattr(meta, "get") else None
            pair = pairs[k] if pairs is not None and k < len(pairs) else None
            # Rule C: also log which rule applies and the native floor of
            # this cell (min(ttc_floor_s, native TTC); None under "absolute"
            # or when the native TTC is NaN/absent) and whether the cell
            # would veto under the configured rule (report-only).
            rule = self.config.absolute_ttc_rule
            native_ttc = getattr(prediction, "native_min_ttc_s", None)
            native_k = None
            if native_ttc is not None:
                nt = np.asarray(native_ttc, dtype=float).reshape(-1)
                if k < nt.size and np.isfinite(nt[k]):
                    native_k = float(nt[k])
            if rule == "relative_to_native" and native_k is not None:
                applied = min(floor, native_k)
            else:
                applied = floor
            entry = {
                "plan": _actions_label(
                    {c: a for c, a in plan.per_cav.items() if a != "native"}
                ),
                "cell": int(k),
                "absolute_ttc_rule": rule,
                "native_min_ttc_s": native_k,
                "floor_applied_s": float(applied),
                "vetoed_under_rule": bool(
                    float(ttc[k])
                    < applied
                    - (float(self.config.ttc_rule_tol_s) if rule == "relative_to_native" else 0.0)
                ),
                "ttc_s": float(ttc[k]),
                "follower_id": None if not pair else pair.get("follower_id"),
                "leader_id": None if not pair else pair.get("leader_id"),
                "pair_class": "unknown_pair" if not pair else pair.get("pair_class"),
            }
            self._ttc_veto_log[key] = entry

    def _predict_uncached(self, plans, samples, controller_state):
        # PL-03b (E): bucket the batch by wall-id set -- the §3.2 state sync
        # depends only on which walls a plan carries, so each distinct set
        # gets one synced controller_state per predict_many call.
        if not plans:
            return []
        # PL-05 (contract v2): commit-window semantics go only to a model
        # that declares v2; v1/v1.1 models are called exactly as before.
        v2_kwargs = (
            {"commit_s": self.config.commit_s, "continuation": self.config.continuation}
            if self._model_is_v2
            else {}
        )
        if controller_state is None:
            return self.world_model.predict_many(
                samples,
                plans,
                horizon_s=self.config.horizon_s,
                trajectory_s=self.config.trajectory_s,
                **v2_kwargs,
            )
        buckets: dict = {}
        for i, plan in enumerate(plans):
            sig = tuple(sorted(w.wall_id for w in plan.walls))
            buckets.setdefault(sig, []).append(i)
        if len(buckets) == 1 and not any(
            e.get("locked") and e.get("wall_id") not in next(iter(buckets))
            for e in (controller_state or {}).values()
        ):
            return self.world_model.predict_many(
                samples,
                plans,
                controller_state=controller_state,
                horizon_s=self.config.horizon_s,
                trajectory_s=self.config.trajectory_s,
                **v2_kwargs,
            )
        out: list = [None] * len(plans)
        for sig, idxs in buckets.items():
            state = _sync_state_to_plan(plans[idxs[0]], controller_state)
            preds = self.world_model.predict_many(
                samples,
                [plans[i] for i in idxs],
                controller_state=state,
                horizon_s=self.config.horizon_s,
                trajectory_s=self.config.trajectory_s,
                **v2_kwargs,
            )
            for local, i in enumerate(idxs):
                out[i] = preds[local]
        return out


def _sync_state_to_plan(plan: PlanCandidate, controller_state: Mapping[str, Mapping]):
    """controller_state as the dispatcher will see it once ``plan`` executes.

    The plan is the planner's decision for THIS tick; when it frees the
    members of a wall (safety abort / release handback), the incoming
    controller_state still reports them locked. Predictions of the new plan
    must use the post-execution state, so stale locks for walls that the
    plan no longer contains are lifted (members -> native).
    """
    walls = {w.wall_id for w in plan.walls}
    stale = any(
        entry.get("locked") and entry.get("wall_id") not in walls
        for entry in controller_state.values()
    )
    if not stale:
        return controller_state
    out = {}
    for cav, entry in controller_state.items():
        if entry.get("locked") and entry.get("wall_id") not in walls:
            entry = dict(entry)
            # §3.2 hygiene: a native CAV carries NO wall fields at all
            entry["locked"] = False
            entry["action"] = "native"
            entry["wall_id"] = None
            entry["wall_phase"] = None
            entry["wall_speed_mps"] = None
        out[cav] = entry
    return out


def _group_label(group) -> str:
    return "+".join(group)


def _actions_label(actions: Mapping[str, str]) -> str:
    return ",".join(f"{cav}={actions[cav]}" for cav in sorted(actions)) or "none"


def _initial_plan(controller_state, locked) -> PlanCandidate:
    """First current plan: from controller_state (contract section 3.2).

    Locked wall members cannot be adopted from controller_state alone (the
    registry learns hold_s/release_at_s/lane order only from planner
    decisions or an explicit current plan), so a session that starts with
    foreign walls must provide the current_plan that carries them.
    """
    if locked:
        raise ValueError(
            "controller_state has locked wall members; construct the Planner "
            "with a current_plan that contains their walls (the registry "
            "adopts them from there)"
        )
    if controller_state is None:
        raise ValueError("the first decide() needs controller_state or an explicit current_plan")
    per_cav = {}
    for cav, entry in controller_state.items():
        action = entry.get("action")
        if action == "wall" or entry.get("wall_id"):
            raise ValueError(
                f"CAV {cav!r} is in a wall; the initial plan needs an "
                "explicit current_plan (PL01-B)"
            )
        if action not in ACTIONS:
            raise ValueError(f"controller_state[{cav!r}] action {action!r} not in {ACTIONS}")
        per_cav[cav] = action
    return PlanCandidate(per_cav=per_cav)


def _validate_locks(plan: PlanCandidate, locked, controller_state) -> None:
    """Current plan vs controller_state consistency (contract section 3.3)."""
    members = {m: w.wall_id for w in plan.walls for m in w.members}
    for cav, entry in locked.items():
        if cav in plan.per_cav:
            raise ValueError(f"locked CAV {cav!r} must not appear in per_cav")
        if members.get(cav) != entry.get("wall_id"):
            raise ValueError(
                f"locked CAV {cav!r} must be a member of its existing wall "
                f"{entry.get('wall_id')!r} in the current plan"
            )
    if controller_state is not None:
        for cav, wall_id in members.items():
            entry = controller_state.get(cav)
            if entry is None or entry.get("wall_id") != wall_id:
                raise ValueError(
                    f"wall member {cav!r} of {wall_id!r} is missing from controller_state"
                )
