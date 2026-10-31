"""Consume the registered local S-RB-EVAL harness and its validated predictions."""

import hashlib
import importlib
from itertools import combinations
from pathlib import Path
import sys
import types

import numpy as np

from .decision import Future, decide
from .metrics import brier, crps, interval_coverage

BAD_PATH_PARTS = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
                  "hetdata_truth", "vehicle_params", "driver_params", "vtype")


def safe_path(path):
    p = Path(path)
    # Check aliases and resolved targets before opening; no recursive raw-data scan.
    if any(word in str(candidate).lower() for word in BAD_PATH_PARTS
           for candidate in (p, p.resolve())):
        raise ValueError(f"Prohibited input/output path: {p}")
    return p


def sha256(path):
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def load_harness(eval_root):
    """Import trusted local evaluator in its own namespace, independent of worktree."""
    root = safe_path(eval_root).resolve() / "tools" / "rbeval"
    for name in ("core.py", "score.py", "real_observables.json", "references.json"):
        safe_path(root / name)
        if not (root / name).is_file():
            raise ValueError(f"Missing evaluator dependency: {root / name}")
    name = "_wmdecide_rbeval_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    core = importlib.import_module(name + ".core")
    score = importlib.import_module(name + ".score")
    return core, score


def baseline_worlds(real, target, *, cell=False):
    """Development only, entire target world excluded, no reserved-label fallback."""
    by_world = {}
    for row in real:
        if not 969400 <= row["seed"] < 969432 or row["seed"] == target["seed"]:
            continue
        if cell and (row["demand"], row["penetration"]) != (
                target["demand"], target["penetration"]):
            continue
        by_world.setdefault(row["seed"], {})[row["arm"]] = row
    return by_world


def _climate(worlds, action, reference, h, region=None):
    """Return same-arm absolute or same-world paired empirical climatology."""
    values = []
    for arms in worlds.values():
        if action not in arms or (reference is not None and reference not in arms):
            continue
        if region:
            event = arms[action]["bins"][region]["events"][str(h)]["event"]
            if event is not None:
                values.append(int(event))
        else:
            value = arms[action]["clocks"][str(h)]["team_delay_vs"]
            if reference is not None:
                value -= arms[reference]["clocks"][str(h)]["team_delay_vs"]
            values.append(float(value))
    return values


def observations(pred, real, arms, horizons, regions):
    """Produce score rows without altering any recorded or imagined data.

    Pair orientation follows the fixed evaluator arm order, one unordered pair
    per world/horizon. A beats B iff paired A-B delay is strictly negative.
    """
    lookup = {(r["seed"], r["arm"]): r for r in real}
    seeds = sorted({s for s, _ in pred})
    distributions, events, coverage, choices = [], [], [], []
    for seed in seeds:
        first = lookup[seed, arms[0]]
        common = dict(seed=seed, split="reserved8" if seed >= 969432 else "development12",
                      demand=first["demand"], penetration=first["penetration"])
        pools = {"pooled": baseline_worlds(real, first),
                 "cell": baseline_worlds(real, first, cell=True)}
        for h in horizons:
            action_futures = {}
            for arm in arms:
                samples = pred[seed, arm]
                # Validated rbeval output is ordered by future index 0..7.
                ids = [s["future"] for s in samples]
                if ids != list(range(8)) or any(s["seed"] != seed for s in samples):
                    raise ValueError("Expected validated complete K8 futures in shared-index order")
                action_futures[arm] = [Future(f"{seed}:{s['future']}",
                    float(s["clocks"][str(h)]["team_delay_vs"])) for s in samples]
                for region in regions:
                    known = [s["bins"][region]["events"][str(h)]["event"] for s in samples]
                    truth = lookup[seed, arm]["bins"][region]["events"][str(h)]["event"]
                    p = float(np.mean(known)) if all(v is not None for v in known) else None
                    row = dict(common, horizon_s=h, target="breakdown", action=arm,
                               reference="", region=region, probability=p,
                               outcome=int(truth) if truth is not None else None,
                               unscorable_futures=sum(v is None for v in known),
                               preexisting=lookup[seed, arm]["bins"][region]["events"][str(h)]
                               .get("preexisting", False))
                    _event_scores(row, {key: _climate(pool, arm, None, h, region)
                                        for key, pool in pools.items()})
                    events.append(row)
            # Expose each incumbent's decision for later picker reuse, without real-label feedback.
            for incumbent in arms:
                choice = decide(action_futures, incumbent)
                choices.append(dict(common, horizon_s=h, incumbent=incumbent,
                    selected=choice["selected"], switched=choice["switched"],
                    confidence_level=choice["confidence_level"],
                    expected_delay=choice["summary"]["actions"][choice["selected"]]
                    ["team_delay"]["mean"]))
            for arm, reference in [(a, None) for a in arms] + list(combinations(arms, 2)):
                x = np.array([f.team_delay for f in action_futures[arm]])
                y = float(lookup[seed, arm]["clocks"][str(h)]["team_delay_vs"])
                if reference is not None:
                    x -= np.array([f.team_delay for f in action_futures[reference]])
                    y -= lookup[seed, reference]["clocks"][str(h)]["team_delay_vs"]
                meta = dict(common, horizon_s=h, target="paired_delay" if reference else "delay",
                            action=arm, reference=reference or "", region="")
                climate = {key: _climate(pool, arm, reference, h) for key, pool in pools.items()}
                row = dict(meta, outcome=y, expected_delay=float(x.mean()),
                           predictive_sd=float(np.std(x, ddof=1)), K=len(x),
                           crps=crps(x, y), fair_crps=crps(x, y, fair=True))
                for key, values in climate.items():
                    score = crps(values, y) if values else None
                    row.update({key + "_n": len(values), key + "_crps": score,
                                "crps_minus_" + key: row["crps"]-score if score is not None else None})
                distributions.append(row)
                coverage.extend(dict(meta, **c) for c in interval_coverage(x, y))
                if reference is not None:
                    event = dict(meta, target="preference", probability=float(np.mean(x < 0)),
                                 outcome=int(y < 0), real_tie=int(y == 0),
                                 predicted_tie_probability=float(np.mean(x == 0)),
                                 unscorable_futures=0)
                    _event_scores(event, {k: [int(v < 0) for v in vals]
                                          for k, vals in climate.items()})
                    events.append(event)
    return dict(distributions=distributions, events=events, coverage=coverage, decisions=choices)


def _event_scores(row, climate):
    valid = row["probability"] is not None and row["outcome"] is not None
    row["brier"] = brier(row["probability"], row["outcome"]) if valid else None
    for key, values in climate.items():
        p = float(np.mean(values)) if values else None
        score = brier(p, row["outcome"]) if p is not None and row["outcome"] is not None else None
        row.update({key + "_n": len(values), key + "_probability": p,
                    key + "_brier": score,
                    "brier_minus_" + key: row["brier"]-score if valid and score is not None else None})
