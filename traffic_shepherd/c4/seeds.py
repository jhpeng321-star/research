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

#: + the C4 GENRAND undirected random-action TRAINING block (961000-961999, loadings 2400/2700;
#: GENRAND_REGISTRATION.md) - WM training data only.
GENRAND_SEEDS = frozenset(S.C4_GENRAND_SEEDS)
#: + the C4 WALLREGEN unseen-wall EVALUATION block (962000-962199, loadings 1000-2800 step 50;
#: WALLREGEN_NOTES.md) - regenerated wall evaluation states only, never training.
WALLREGEN_SEEDS = frozenset(S.C4_WALLREGEN_EVAL_SEEDS)
#: + the C4 V5 textbook-population TRAINING (963000-963999) and EVALUATION (964000-964199) blocks.
V5_TRAIN_SEEDS = frozenset(S.C4_V5_TRAIN_SEEDS)
V5_EVAL_SEEDS = frozenset(S.C4_V5_EVAL_SEEDS)
#: + the C4 V6 long-branch TRAINING block (965000-965999).
V6_LONGPAIR_SEEDS = frozenset(S.C4_V6_LONGPAIR_SEEDS)
#: + the C4 Dreamer-loop TRAINING block (966000-969999).
LOOP_TRAIN_SEEDS = frozenset(S.C4_LOOP_TRAIN_SEEDS)
ALLOWED_SEEDS = (frozenset(S.EXPOSED_CONTROLLER_SEEDS) | GENRAND_SEEDS | WALLREGEN_SEEDS
                 | V5_TRAIN_SEEDS | V5_EVAL_SEEDS | V6_LONGPAIR_SEEDS | LOOP_TRAIN_SEEDS)
BLOCKED_SEEDS = frozenset(
    set(S.SEALED_SEEDS) | set(S.FINAL_TEST_SEEDS) | set(S.WM58_CHUNK5_SEEDS)
    | set(S.WM_N1_REPL_SEEDS) | set(S.PLWM1_SEEDS) | set(S.CV28_VALIDATION_HOLDOUT_SEEDS)
    | set(S.WM28_FRESH_VALIDATION_SEEDS) | set(S.WM34_FRESH_RETEST_SEEDS))
assert not (GENRAND_SEEDS & BLOCKED_SEEDS), "GENRAND seeds overlap a sealed/test block"
assert not (WALLREGEN_SEEDS & BLOCKED_SEEDS), "WALLREGEN seeds overlap a sealed/test block"
assert not (WALLREGEN_SEEDS & GENRAND_SEEDS), "WALLREGEN eval seeds overlap the GENRAND training block"
assert not ((V5_TRAIN_SEEDS | V5_EVAL_SEEDS) & (BLOCKED_SEEDS | GENRAND_SEEDS | WALLREGEN_SEEDS))
assert not (V5_TRAIN_SEEDS & V5_EVAL_SEEDS)
assert not (LOOP_TRAIN_SEEDS & (BLOCKED_SEEDS | GENRAND_SEEDS | WALLREGEN_SEEDS | V5_TRAIN_SEEDS | V5_EVAL_SEEDS | V6_LONGPAIR_SEEDS))
assert not (V6_LONGPAIR_SEEDS & (BLOCKED_SEEDS | GENRAND_SEEDS | WALLREGEN_SEEDS | V5_TRAIN_SEEDS | V5_EVAL_SEEDS))


class SeedRefusal(ValueError):
    pass


def check_case(seed: int, demand_vph: int) -> None:
    if type(seed) is not int or type(demand_vph) is not int:
        raise SeedRefusal("seed and demand must be exact integers")
    if seed in BLOCKED_SEEDS:
        raise SeedRefusal(f"seed {seed} is sealed/test/holdout; refused")
    if seed not in ALLOWED_SEEDS:
        raise SeedRefusal(f"seed {seed} is not an exposed C4 development seed "
                          "(4107-4148), a C4 GENRAND training seed (961000-961999) nor a C4 "
                          "WALLREGEN evaluation seed (962000-962199) nor a C4 V5/V6 seed (963000-964199, 965000-965999)")
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
