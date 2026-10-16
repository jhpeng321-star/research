#!/usr/bin/env python
"""Choose DEVELOPMENT evaluation seeds for the C4 campaign (zero simulation, metadata only).

Rule (outcome-independent, frozen before any C4 run):
  1. Candidate pool = the exposed/development partitions registered in traffic_shepherd/seeds.py
     (never SEALED_SEEDS, never the sealed chunk-5 / TEST_REPL blocks, never FINAL_TEST).
  2. A (seed, demand) cell is eligible only if ``traffic_shepherd.seeds.require_development_case``
     (the gate used by traffic_shepherd.workflow.evaluate) accepts it for role 'development' or 'exposed'.
  3. Seeds whose world id appears in any world-model TRAIN pack (pack_v3, pack_dspeed, pack_damping,
     pack_switching_train INDEX 'world' field, parsed as w<seed>m...) are removed: the policy/WM must not
     be scored on traffic realisations the world model was fitted on.
  4. For each demand the first N eligible seeds are taken in fixed partition-priority order
     (OE65, WM28, WM34, CV31, CV29, CV28, CV20, OE59, OE62, OE34, OE32), ascending inside a partition,
     skipping seeds already given to another demand (independent worlds per demand cell).
  5. Collection (training-data) seeds are NOT chosen here; they need their own exposure audit.

This is development evidence only.  A final claim needs a freshly audited, registered seed block
(see REGISTRATION.md section 7); this script never certifies freshness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DATA = Path("data_cache")
PACK_INDEXES = ["pack_v3", "pack_dspeed", "pack_damping", "wmn1_switching_train/pack_switching_train"]
DEMANDS = (2400, 2600, 2700)


def training_world_seeds(data_root: Path) -> dict[int, list[str]]:
    seeds: dict[int, list[str]] = {}
    for pack in PACK_INDEXES:
        path = data_root / pack / "INDEX.json"
        if not path.is_file():
            print(f"warning: {path} missing; its worlds cannot be excluded", file=sys.stderr)
            continue
        index = json.loads(path.read_text(encoding="utf-8"))
        for run in index["runs"]:
            m = re.match(r"w(\d+)m", str(run.get("world", "")))
            if m:
                seeds.setdefault(int(m.group(1)), []).append(pack)
    return seeds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-demand", type=int, default=8)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "runs/active/WM-S-C4-20261013/eval_seeds.json")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO))
    from traffic_shepherd import seeds as S

    priority = [
        ("OE65_DATASET_SEEDS", S.OE65_DATASET_SEEDS),
        ("WM28_FRESH_VALIDATION_SEEDS", sorted(S.WM28_FRESH_VALIDATION_SEEDS)),
        ("WM34_FRESH_RETEST_SEEDS", sorted(S.WM34_FRESH_RETEST_SEEDS)),
        ("CV31_COVERAGE_EXTENSION_SEEDS", S.CV31_COVERAGE_EXTENSION_SEEDS),
        ("CV29_CORPUS_EXTENSION_SEEDS", S.CV29_CORPUS_EXTENSION_SEEDS),
        ("CV28_VALIDATION_HOLDOUT_SEEDS", S.CV28_VALIDATION_HOLDOUT_SEEDS),
        ("CV20_MULTISEED_CORPUS_SEEDS", S.CV20_MULTISEED_CORPUS_SEEDS),
        ("OE59_CONDITIONAL_PANEL_SEEDS", S.OE59_CONDITIONAL_PANEL_SEEDS),
        ("OE62_CONDITIONAL_SCREEN_SEEDS", S.OE62_CONDITIONAL_SCREEN_SEEDS),
        ("OE34_RELATIONAL_DEVELOPMENT_SEEDS", S.OE34_RELATIONAL_DEVELOPMENT_SEEDS),
        ("OE32_OBSERVATION_DEVELOPMENT_SEEDS", S.OE32_OBSERVATION_DEVELOPMENT_SEEDS),
    ]
    forbidden = set(S.SEALED_SEEDS) | set(S.WM58_CHUNK5_SEEDS) | set(S.WM_N1_REPL_SEEDS) \
        | set(S.FINAL_TEST_SEEDS)
    trained = training_world_seeds(args.data_root)

    def accepted(seed: int, demand: int) -> str | None:
        for role in ("development", "exposed"):
            try:
                S.require_development_case(int(seed), int(demand), role)
                return role
            except ValueError:
                continue
        return None

    chosen: dict[int, list[dict]] = {d: [] for d in DEMANDS}
    used: set[int] = set()
    skipped = {"forbidden": 0, "in_wm_train_pack": 0, "gate_refused": 0}
    for demand in DEMANDS:
        for part, seeds in priority:
            for seed in seeds:
                if len(chosen[demand]) >= args.per_demand:
                    break
                if seed in used:
                    continue
                if seed in forbidden:
                    skipped["forbidden"] += 1
                    continue
                if seed in trained:
                    skipped["in_wm_train_pack"] += 1
                    continue
                role = accepted(seed, demand)
                if role is None:
                    skipped["gate_refused"] += 1
                    continue
                chosen[demand].append({"seed": int(seed), "partition": part, "role": role})
                used.add(seed)
    doc = {
        "schema": "c4_eval_seeds/1",
        "purpose": "development evaluation seeds for WM-S-C4 (paired across controllers and penetrations)",
        "rule": __doc__.split("Rule (outcome-independent, frozen before any C4 run):")[1].split("This is development")[0].strip(),
        "per_demand": args.per_demand,
        "role": "development evidence only; not acceptance, not a freshness certificate",
        "gate": "traffic_shepherd.seeds.require_development_case (used by traffic_shepherd.workflow.evaluate)",
        "excluded_world_seed_count": len(trained),
        "skipped_counts": skipped,
        "cells": {str(d): chosen[d] for d in DEMANDS},
        "seeds_by_demand": {str(d): [c["seed"] for c in chosen[d]] for d in DEMANDS},
    }
    doc["sha256_of_seeds_by_demand"] = hashlib.sha256(
        json.dumps(doc["seeds_by_demand"], sort_keys=True).encode()).hexdigest()
    for d in DEMANDS:
        print(f"demand {d}: {len(chosen[d])} seeds {[c['seed'] for c in chosen[d]]} "
              f"({sorted({c['partition'] for c in chosen[d]})})")
        if len(chosen[d]) < args.per_demand:
            print(f"  WARNING: only {len(chosen[d])} eligible seeds for demand {d}", file=sys.stderr)
    print("skipped:", skipped, "wm-train world seeds:", len(trained))
    if not args.print_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
