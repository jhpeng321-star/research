"""Expected-delay selection with the owner's registered 0.8 probability rule."""

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Mapping, Sequence

import numpy as np

SWITCH_CONFIDENCE = 0.8  # Owner-registered confidence, never a traffic-behaviour parameter.


@dataclass(frozen=True)
class Future:
    """One action under one shared hidden-driver/arrival draw.

    draw_id must identify the complete common-random-number draw, not just an
    action-local row number. The producer owns actual stream coupling.
    breakdown is the registered detector's event, or None if unscorable.
    """

    draw_id: str
    team_delay: float
    breakdown: bool | None = None


def _align(actions: Mapping[str, Sequence[Future]]):
    if not actions:
        raise ValueError("At least one action is required")
    aligned = {}
    common = None
    for action, futures in actions.items():
        if not isinstance(action, str) or not action:
            raise ValueError("Action names must be nonempty strings")
        by_id = {}
        for future in futures:
            if not isinstance(future.draw_id, str) or not future.draw_id:
                raise ValueError("draw_id must be a nonempty shared-draw identifier")
            if future.draw_id in by_id:
                raise ValueError(f"Duplicate draw_id for {action}: {future.draw_id}")
            if not math.isfinite(future.team_delay):
                raise ValueError("Every team delay must be finite; never drop failed futures")
            if future.breakdown is not None and type(future.breakdown) is not bool:
                raise ValueError("breakdown must be bool or None")
            by_id[future.draw_id] = future
        if not by_id:
            raise ValueError("Each action requires at least one future")
        if common is None:
            common = set(by_id)
        if set(by_id) != common:
            raise ValueError("Actions must contain exactly the same shared draw IDs")
        aligned[action] = by_id
    ids = sorted(common)
    return ids, {a: [rows[i] for i in ids] for a, rows in aligned.items()}


def spread(values):
    """Predictive spread; standard error is Monte Carlo error of the mean only."""
    x = np.asarray(values, dtype=float)
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else None
    return dict(mean=float(x.mean()), sd=sd,
                mean_mc_se=sd / math.sqrt(len(x)) if sd is not None else None,
                minimum=float(x.min()), q10=float(np.quantile(x, .1)),
                median=float(np.median(x)), q90=float(np.quantile(x, .9)),
                maximum=float(x.max()))


def summarize(actions: Mapping[str, Sequence[Future]]):
    """Return marginal delay/event summaries and all ordered paired comparisons.

    P(A beats B) = count(J_A < J_B)/K. Paired saving is J_B-J_A (positive
    helps A). This is not P(E[J_A] < E[J_B]) or a model-error confidence bound.
    """
    ids, aligned = _align(actions)
    arrays = {a: np.array([f.team_delay for f in rows]) for a, rows in aligned.items()}
    marginal = {}
    for action, rows in aligned.items():
        events = [f.breakdown for f in rows]
        known = sum(e is not None for e in events)
        p = sum(e is True for e in events) / len(rows) if known == len(rows) else None
        marginal[action] = dict(team_delay=spread(arrays[action]),
            breakdown_probability=p, breakdown_known=known,
            breakdown_mc_se=math.sqrt(p * (1-p) / len(rows)) if p is not None else None)
    pairwise = []
    for a, b in permutations(aligned, 2):
        saving = arrays[b] - arrays[a]
        p = float(np.mean(saving > 0))
        pairwise.append(dict(action=a, reference=b, probability_beats=p,
            probability_tie=float(np.mean(saving == 0)), wins=int(np.sum(saving > 0)),
            probability_mc_se=math.sqrt(p * (1-p) / len(ids)), saving=spread(saving)))
    return dict(K=len(ids), draw_ids=ids, actions=marginal, pairwise=pairwise,
                probability_semantics="strict_paired_delay_win")


def decide(actions: Mapping[str, Sequence[Future]], incumbent: str):
    """Lowest expected delay among candidates passing the fixed 0.8 gate.

    A candidate must also strictly improve expected delay. This avoids switching
    to a frequent small winner with a rare catastrophic loss. Equal expected
    candidate scores use lexical action order; an incumbent tie always stays.
    """
    if incumbent not in actions:
        raise ValueError("Incumbent must be included in the paired futures")
    result = summarize(actions)
    current_mean = result["actions"][incumbent]["team_delay"]["mean"]
    against = {p["action"]: p for p in result["pairwise"] if p["reference"] == incumbent}
    eligible = [a for a, p in against.items()
                if p["probability_beats"] >= SWITCH_CONFIDENCE
                and result["actions"][a]["team_delay"]["mean"] < current_mean]
    selected = min(eligible, key=lambda a: (result["actions"][a]["team_delay"]["mean"], a)) \
        if eligible else incumbent
    return dict(incumbent=incumbent, selected=selected, switched=selected != incumbent,
                confidence_level=SWITCH_CONFIDENCE, eligible=sorted(eligible),
                reason="expected_delay_and_probability_gate" if eligible else "keep_incumbent",
                summary=result)
