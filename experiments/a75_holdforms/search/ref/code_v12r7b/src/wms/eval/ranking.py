"""Wall-vs-native ranking and regret for paired states.

A paired state carries predicted and recorded delay for both a ``wall`` and a ``native`` arm
started from the identical snapshot.  The model "chooses" wall iff its predicted network
delay difference (wall - native) is negative; ties go to native.  Regret is the recorded
delay the choice costs relative to the better arm (seconds, >= 0).  Zero-regret baselines
``always_wall`` / ``always_native`` bracket the model.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .errors import EvalError
from .stats import DEFAULT_N_BOOT, DEFAULT_SEED, bootstrap_ci

EPS = 1e-9
RANK_HORIZONS_S = (60.0, 120.0)
WALL, NATIVE = "wall", "native"


def sign(value: float) -> int:
    if value > EPS:
        return 1
    if value < -EPS:
        return -1
    return 0


def rank_pair(predicted_delta: float, recorded_delta: float) -> dict:
    """Wall-minus-native cost; lower wins, predicted ties select native."""
    if not all(math.isfinite(v) for v in (predicted_delta, recorded_delta)):
        raise EvalError("nonfinite paired delta")
    chosen = WALL if predicted_delta < -EPS else NATIVE
    realized_relative = recorded_delta if chosen == WALL else 0.0
    return {"predicted_delta": predicted_delta, "recorded_delta": recorded_delta,
            "sign_agreement": sign(predicted_delta) == sign(recorded_delta),
            "selected_arm": chosen,
            "regret_s": realized_relative - min(0.0, recorded_delta)}


def network_delay(delay_block: Mapping, horizon: float) -> float | None:
    """Sum of section delays at ``horizon`` from a ``{"<h>": {section: delay}}`` block."""
    section = delay_block.get(str(float(horizon))) or {}
    values = [v for v in section.values() if v is not None and math.isfinite(float(v))]
    return float(sum(values)) if values else None


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation (average ranks for ties); ``None`` when undefined."""
    if len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    sy = math.sqrt(sum((v - my) ** 2 for v in ry))
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return float(sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy))


def paired_states(states: Sequence[Mapping]) -> list[Mapping]:
    """States that carry an error-free wall AND native arm and are declared paired."""
    out = []
    for state in states:
        arms = state.get("arms") or {}
        wall, native = arms.get(WALL), arms.get(NATIVE)
        if state.get("paired") is True and wall and native \
                and not wall.get("error") and not native.get("error"):
            out.append(state)
    return out


def ranking(states: Sequence[Mapping], *, horizons: Sequence[float] = RANK_HORIZONS_S,
            n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """Sign agreement, Spearman and regret of the model's wall-vs-native choice per horizon."""
    pairs = paired_states(states)
    out: dict = {"n_paired_states": len(pairs),
                 "worlds": len({str(s["world"]) for s in pairs})}
    for horizon in horizons:
        rows = []
        for state in pairs:
            wall, native = state["arms"][WALL]["delay"], state["arms"][NATIVE]["delay"]
            pn, pp = network_delay(native["predicted"], horizon), \
                network_delay(wall["predicted"], horizon)
            rn, rp = network_delay(native["recorded"], horizon), \
                network_delay(wall["recorded"], horizon)
            if None in (pn, pp, rn, rp):
                continue
            rows.append({"predicted": float(pp) - float(pn), "recorded": float(rp) - float(rn)})
        if not rows:
            continue
        pred = [r["predicted"] for r in rows]
        rec = [r["recorded"] for r in rows]

        def mean_regret(sample) -> float:
            return sum(rank_pair(r["predicted"], r["recorded"])["regret_s"]
                       for r in sample) / len(sample)

        out[str(horizon)] = {
            "n": len(rows),
            "sign_agreement": sum(1 for p, r in zip(pred, rec) if sign(p) == sign(r)) / len(rows),
            "spearman": spearman(pred, rec),
            "model_mean_regret_s": mean_regret(rows),
            "model_regret_ci": bootstrap_ci(rows, mean_regret, n_boot=n_boot, seed=seed),
            "always_wall_regret_s": sum(max(0.0, r) for r in rec) / len(rows),
            "always_native_regret_s": sum(max(0.0, -r) for r in rec) / len(rows),
        }
    return out
