"""C4 SUMO seed guard: exposed controller-development seeds only.

Allowed: ``traffic_shepherd.seeds.EXPOSED_CONTROLLER_SEEDS`` (4107-4148: previously
exposed controller-development identities, any loading in the 1900-2800 veh/h family)
and only through ``require_development_case(seed, loading, "exposed")``.  These seeds are
not WM-S pack worlds (84xx-89xx, 92xxx, 93xxx), so C4 SUMO runs can never regenerate a
WM CAL/HOLDOUT/TEST world.  Sealed acceptance blocks, the chunk-5 TEST block, TEST_REPL,
the planner block, the WM-28/34 fresh-validation blocks and every 'holdout' block are
refused explicitly even if a future registry change would admit them.  Fresh seeds for
the final C4 comparison need their own registered, exposure-audited block.
"""
from __future__ import annotations

from traffic_shepherd import seeds as S

ALLOWED_SEEDS = frozenset(S.EXPOSED_CONTROLLER_SEEDS)
BLOCKED_SEEDS = frozenset(
    set(S.SEALED_SEEDS) | set(S.FINAL_TEST_SEEDS) | set(S.WM58_CHUNK5_SEEDS)
    | set(S.WM_N1_REPL_SEEDS) | set(S.PLWM1_SEEDS) | set(S.CV28_VALIDATION_HOLDOUT_SEEDS)
    | set(S.WM28_FRESH_VALIDATION_SEEDS) | set(S.WM34_FRESH_RETEST_SEEDS))


class SeedRefusal(ValueError):
    pass


def check_case(seed: int, demand_vph: int) -> None:
    if type(seed) is not int or type(demand_vph) is not int:
        raise SeedRefusal("seed and demand must be exact integers")
    if seed in BLOCKED_SEEDS:
        raise SeedRefusal(f"seed {seed} is sealed/test/holdout; refused")
    if seed not in ALLOWED_SEEDS:
        raise SeedRefusal(f"seed {seed} is not an exposed C4 development seed "
                          f"({min(ALLOWED_SEEDS)}-{max(ALLOWED_SEEDS)})")
    S.require_development_case(seed, demand_vph, "exposed")


def parse_seeds(text: str) -> list[int]:
    """``"4107-4110,4120"`` -> [4107, 4108, 4109, 4110, 4120]."""
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out
