"""When-to-cooperate policy: documented rule, oracle headroom, OOF regret.

Rule (COOPBANDIT_DESIGN.md, identical to executor learn.py choose()):
among AVAILABLE group options take the argmax predicted D at the policy
horizon; execute it only if predicted D minus the fold's calibrated radius is
positive; otherwise execute O0 (MAPPO). The radius is the model's own
world-grouped CV uncertainty - there is NO hand-set behavioural threshold.

Evaluation is strictly out-of-fold: each state is scored with the fold model
and the fold radius, neither of which saw the state's world. Regret is
measured against the per-state oracle max(0, max over available options of
true D); the oracle helpful fraction is the oracle headroom. Uncertainty is
world-bootstrap (resample worlds, 2000 draws, seed 20261031).
"""
from dataclasses import dataclass

import numpy as np

from .cv import CVResult
from .loader import Dataset
from .schema import BOOTSTRAP, BOOTSTRAP_SEED, POLICY_HORIZON


@dataclass
class PolicyEvaluation:
    states: list
    chosen: list
    realized: np.ndarray
    oracle: np.ndarray
    helpful: np.ndarray
    cooperate: np.ndarray
    world_rows: list
    summary: dict
    baseline_summary: dict


def decide(pred_by_option, radius, available_options):
    """The documented choice rule. pred_by_option: {option: predicted D}."""
    best, best_pred = "O0", -np.inf
    for option in available_options:
        if pred_by_option.get(option, -np.inf) > best_pred:
            best, best_pred = option, pred_by_option[option]
    if best != "O0" and best_pred - radius > 0.0:
        return best
    return "O0"


def evaluate(dataset: Dataset, result: CVResult, horizon=POLICY_HORIZON):
    """Out-of-fold policy evaluation with oracle headroom and regret."""
    result.dev_guard()
    state_ids = list(dataset.states.state_id)
    oof = result.oof
    chosen, realized, oracle, helpful, cooperate = [], [], [], [], []
    for sid in state_ids:
        world = int(dataset.world_of([sid])[0])
        radius = result.radius_for_world(world)
        available = dataset.available(sid)
        labels = dataset.labels_for(sid, horizon)
        pred = {o: float(oof[(oof.state_id == sid) & (oof.option == o)].y_pred.iloc[0])
                for o in available}
        pick = decide(pred, radius, available)
        gains = [max(0.0, labels.get(o, 0.0)) for o in available]
        chosen.append(pick)
        realized.append(labels.get(pick, 0.0) if pick != "O0" else 0.0)
        oracle.append(max(gains) if gains else 0.0)
        helpful.append(max(gains) > 0.0 if gains else False)
        cooperate.append(pick != "O0")
    realized, oracle = np.array(realized), np.array(oracle)
    helpful, cooperate = np.array(helpful), np.array(cooperate)
    worlds = np.array([int(w) for w in dataset.world_of(state_ids)])
    world_rows, boot = _world_table(worlds, realized, oracle, helpful, cooperate)
    summary = _summarize(world_rows)
    baseline = _summarize(_world_table(worlds, np.zeros_like(realized), oracle, helpful,
                                       np.zeros_like(cooperate))[0])
    return PolicyEvaluation(states=state_ids, chosen=chosen, realized=realized, oracle=oracle,
                            helpful=helpful, cooperate=cooperate, world_rows=world_rows,
                            summary=summary, baseline_summary=baseline)


def _world_table(worlds, realized, oracle, helpful, cooperate):
    rows = []
    for world in np.unique(worlds):
        m = worlds == world
        rows.append(dict(world_id=int(world), learned=float(realized[m].mean()),
                         oracle=float(oracle[m].mean()), regret=float((oracle[m] - realized[m]).mean()),
                         helpful=float(helpful[m].mean()), cooperate=float(cooperate[m].mean())))
    return rows, None


def _summarize(world_rows):
    values = {k: np.array([r[k] for r in world_rows]) for k in
              ("learned", "oracle", "regret", "helpful", "cooperate")}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = rng.integers(0, len(world_rows), (BOOTSTRAP, len(world_rows)))
    summary = {}
    for k, v in values.items():
        means = v[boot].mean(axis=1)
        summary[k] = dict(mean=float(v.mean()),
                          ci95=[float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))])
    return summary


def per_cell(dataset: Dataset, evaluation: PolicyEvaluation):
    """Descriptive (non-primary) breakdown by demand/penetration cell."""
    cells = dataset.states.set_index("state_id").loc[evaluation.states, ["cell"]].cell.to_numpy()
    out = {}
    for cell in np.unique(cells):
        m = cells == cell
        out[int(cell)] = dict(states=int(m.sum()),
                              learned=float(evaluation.realized[m].mean()),
                              oracle=float(evaluation.oracle[m].mean()),
                              regret=float((evaluation.oracle[m] - evaluation.realized[m]).mean()),
                              helpful=float(evaluation.helpful[m].mean()),
                              cooperate=float(evaluation.cooperate[m].mean()))
    return out
