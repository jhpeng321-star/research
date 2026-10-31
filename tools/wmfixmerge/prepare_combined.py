"""S-WMFIX-ES: combined TRAIN registry over 12 FIT + 40 wmdata_fit groups.

MAIN addendum 2026-10-31 13:38 (S-WMDATA COMPLETE): the ES calibration data is
the combined TRAIN corpus.  This tool runs on a host that can verify BOTH
prepared docs (cpuQ): it fully re-verifies each corpus (data.verify digests
every source file), binds all 12 FIT groups plus all 40 wmdata worlds/all arms and
reuses the FIT doc's scales/anchor_scales/crossing_width so the fitness stays
the SAME pre-registered paired objective, merely evaluated on more groups.

The driver consumes only this registry (recorded digests; it cannot read the
wmdata bytes on gpuA); every worker verifies its own corpus doc directly and
fail-closes on any group-order/arm mismatch (receipt assertions in es.sweep).
"""
from __future__ import annotations

import argparse
import json

from tools.wmfixmerge.data import SEEDS, digest, install_guard, safe, verify

WMDATA_ARMS = ("circles", "ctrl0", "mappo", "pi")
FIT_ARMS = ("coopoff_c1", "ours_c1")


def group_order(doc):
    """Production loader group order = manifest order of the prepared doc."""
    seen, order = set(), []
    for row in doc["episodes"]:
        if row["pair_key"] not in seen:
            seen.add(row["pair_key"])
            order.append(row["pair_key"])
    return sorted(order)


def fitness_subset(fit_groups, wm_groups, seeds_of):
    fitness_groups = [dict(corpus="fit", group=i, pair_key=k, seed=seeds_of[("fit", k)],
                           arm=arm, arm_name=name)
                      for i, k in enumerate(fit_groups)
                      for arm, name in enumerate(FIT_ARMS)]
    fitness_groups += [dict(corpus="wm", group=i, pair_key=k, seed=seeds_of[("wm", k)],
                            arm=arm, arm_name=name)
                       for i, k in enumerate(wm_groups)
                       for arm, name in enumerate(WMDATA_ARMS)]
    return fitness_groups


def build(fit_prepared, wmdata_prepared, out):
    fit = verify(fit_prepared)
    wm = verify(wmdata_prepared)
    if sorted(int(s) for s in fit["seeds"]) != sorted(SEEDS):
        raise ValueError("fit part is not the registered 12-seed corpus")
    fit_groups = group_order(fit)
    wm_groups = group_order(wm)
    if len(fit_groups) != 12 or len(wm_groups) != 40:
        raise ValueError(f"unexpected group counts {len(fit_groups)}/{len(wm_groups)}")
    for part, prepared, expected in (("fit", fit, FIT_ARMS), ("wm", wm, WMDATA_ARMS)):
        for key in group_order(prepared):
            actual = sorted(r["arm"] for r in prepared["labels"]
                            if r["pair_key"] == key and not r["native"])
            if actual != list(expected):
                raise ValueError(f"{part} group {key} arm set differs: {actual}")
    seeds_of = {}
    for part, doc in (("fit", fit), ("wm", wm)):
        for row in doc["labels"]:
            seeds_of[(part, row["pair_key"])] = int(row["seed"])
    fitness_groups = fitness_subset(fit_groups, wm_groups, seeds_of)
    doc = dict(schema="wmfix_train_comb/1",
               parts=dict(fit=dict(prepared=str(safe(fit_prepared)), sha256=digest(fit_prepared),
                                   groups=fit_groups, seeds=sorted(int(s) for s in fit["seeds"])),
                          wm=dict(prepared=str(safe(wmdata_prepared)), sha256=digest(wmdata_prepared),
                                  groups=wm_groups, seeds=sorted(int(s) for s in wm["seeds"]))),
               fitness_groups=fitness_groups,
               scales=fit["scales"], anchor_scales=fit["anchor_scales"],
               crossing_width=fit["crossing_width"],
               rule="FIT 0-11 + all 40 wmdata worlds, all four native contrasts; "
                    f"scales/anchors/crossing_width from the FIT doc (unchanged objective)")
    with safe(out).open("x") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
    return doc


def check_registry(registry, fit_prepared):
    """Driver-side validation: local FIT doc digest + objective binding."""
    doc = json.loads(safe(registry).read_text())
    if doc.get("schema") != "wmfix_train_comb/1":
        raise ValueError("unexpected combined registry")
    if doc["parts"]["fit"]["sha256"] != digest(fit_prepared):
        raise ValueError("registry FIT digest differs from the local prepared doc")
    fit = json.loads(safe(fit_prepared).read_text())
    for key in ("scales", "anchor_scales", "crossing_width"):
        if doc[key] != fit[key]:
            raise ValueError(f"registry {key} differs from the FIT objective")
    if len(doc["parts"]["wm"]["groups"]) != 40:
        raise ValueError("registry wm part is not the 40-world corpus")
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-prepared", required=True)
    parser.add_argument("--wmdata-prepared", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    install_guard()
    build(args.fit_prepared, args.wmdata_prepared, args.out)
