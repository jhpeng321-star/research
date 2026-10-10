"""G6 ``heldout-dryrun``.

Held-out configs (i)-(iii) must build the correct train/eval splits and assert
that the excluded arms are ABSENT from training:

* (i) ``heldout-i-j4b`` excludes the J4B arms from training;
* (ii) ``heldout-ii-leave-pi`` excludes every PI arm;
* (iii) ``heldout-iii-leave-highest-wall-speed`` excludes the a-priori highest
  J4 wall-speed factor.

Entry point (P7): ``train.idea_a.heldout_split_report()`` returning::

    {"splits": {"heldout-i-j4b": {"train": [...], "eval": [...],
                                  "excluded": [...]}, ...}}
"""
from __future__ import annotations

from ..imports import PieceMissing
from ..result import fail, ok

NAME = "heldout-dryrun"
TITLE = "G6 heldout-dryrun (3 splits build; excluded arms absent from train)"
ENTRY_POINTS = ("train.idea_a.heldout_split_report",)
#: The exact arm id each held-out config must exclude from its train split.
EXCLUDED_ARM = {
    "heldout-i-j4b": "j4b",
    "heldout-ii-leave-pi": "pi",
    "heldout-iii-leave-highest-wall-speed": "wall-speed-highest",
}


def _names(seq):
    return [str(x) for x in (seq or [])]


def evaluate(report):
    if not isinstance(report, dict):
        return False, f"report is {type(report).__name__}, expected dict", {}
    splits = report.get("splits") or {}
    checks = {"splits": sorted(splits)}
    problems = []
    for config, arm in EXCLUDED_ARM.items():
        entry = splits.get(config)
        if not isinstance(entry, dict):
            problems.append(f"{config}: missing split report")
            continue
        train = _names(entry.get("train"))
        eval_ = _names(entry.get("eval"))
        excluded = _names(entry.get("excluded"))
        train_l = {t.lower() for t in train}
        excl_l = {x.lower() for x in excluded}
        checks[f"{config}_n_train"] = len(train)
        checks[f"{config}_n_eval"] = len(eval_)
        checks[f"{config}_excluded_arm"] = arm
        if not train:
            problems.append(f"{config}: empty train split")
        if not eval_:
            problems.append(f"{config}: empty eval split")
        if arm not in excl_l:
            problems.append(f"{config}: excluded arm {arm!r} not declared")
        leaked = sorted(train_l & excl_l)
        if leaked:
            problems.append(f"{config}: excluded arm(s) leaked into train: {leaked}")
        if arm in train_l:
            problems.append(f"{config}: excluded arm {arm!r} is in train")
    passed = not problems
    detail = "all 3 held-out splits build; excluded arms absent from train" \
        if passed else "; ".join(problems)
    return passed, detail, checks


def run(settings, resolve):
    try:
        fn = resolve(ENTRY_POINTS[0])
    except PieceMissing as e:
        return fail(NAME, TITLE, str(e), ENTRY_POINTS, error="piece-missing")
    try:
        report = fn()
    except Exception as e:                                # noqa: BLE001
        return fail(NAME, TITLE, f"{ENTRY_POINTS[0]} raised "
                    f"{type(e).__name__}: {e}", ENTRY_POINTS, error="gate-error")
    passed, detail, checks = evaluate(report)
    return (ok if passed else fail)(NAME, TITLE, detail, ENTRY_POINTS, checks)
