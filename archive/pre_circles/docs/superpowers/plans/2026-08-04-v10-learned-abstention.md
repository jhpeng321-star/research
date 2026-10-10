# v10 Learned Abstention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise per-seed domination vs paired no_op on holdout seeds 101–150 above the bars 27/34/33 (v9 i40 / v3 i90 / v8 i150), measured under the slack rule (G margin 0.1s, loss slack 0.5s: G strictly lower by >0.1s AND loss ≤ +0.5s AND safety ≤), by teaching the v9-architecture policy to abstain on easy seeds — learned via reward + curriculum, no action masks.

**Architecture:** Three code changes plus one training run. (1) A domination scoreboard tool becomes the single selection metric. (2) Reward v10 adds a `fair_win` term (pays strict gap reduction below the paired no_op ref) and an `easy_intervention_tax` (soft penalty for non-coast actions when |gap| ≤ ref), both plumbed through the existing `compute_shepherd_reward` signature — no `env/core.py` reward-call changes needed. (3) Easy-seed oversampling biases paired-pool episode sampling toward low-`fair_gap` seeds. Then a warm-started training run (from v9 iter_40) with peek/holdout evaluation.

**Tech Stack:** Python 3.10, conda env `rl_driving`, SUMO + TraCI, Ray RLlib PPO, pytest. Project root `D:\research-dense`; SUMO net/demand at `D:\research\highway_merge_shepherd`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-04-v10-learned-abstention-design.md` (approved).
- All commands run from `D:\research-dense` with conda env `rl_driving` activated, `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo` on `PATH`, and `PYTHONPATH` including `D:\research-dense` (`;` separator — Windows Python).
- New reward weights MUST default to `0.0` / disabled so existing v9 behavior and frozen eval parity are unchanged unless explicitly enabled.
- `fair_win` weight ≤ 0.2 (spec §4.2). Do NOT touch `delay_budget` (1.10) or `delay_win` (0.30) weights.
- No inference-time action masks or hard gates; `shepherd_fairness_mask` and `shepherd_coast_force` stay `False`.
- No warm-start from v3/v8 (117-D obs incompatible); only from v9 iter_40 (121-D).
- Checkpoint selection by domination scoreboard only, never by TensorBoard reward.
- `python -m pytest -q` must be fully green before the training run starts.
- Git: commit after every task (repo initialized 2026-08-04, baseline commit `5147ff3`). `runs/` is gitignored — eval caches live there and are NOT committed.

---

### Task 1: Domination scoreboard tool

**Files:**
- Create: `tools/domination_scoreboard.py`
- Test: `tests/test_domination_scoreboard.py`

**Interfaces:**
- Produces: `score_eval_dir(eval_dir, noop_dir=None, eps=1e-9) -> dict` with keys `n`, `n_dominate`, `n_winnable`, `n_winnable_dominate`, `per_seed` (list of dicts: `seed`, `dominate`, `d_g`, `d_loss`, `d_collisions`, `d_ttc`, `fail_cause`), `structural` (sub-report for STRUCTURAL_SEEDS). Consumed by Task 5 reporting and by the test below.
- `STRUCTURAL_SEEDS = [106, 117, 119, 120, 129, 136, 137, 143, 146, 147, 148]` (from `docs/PLAN_V9_IMPROVE_PASS.md` §1.2).

Background the implementer needs: eval dirs like `runs/eval/fair_safe_v9_i40_dt05_holdout101_150/` contain per-seed JSONs named `{tag}.p010.seed{SEED}.json` for the policy (tag e.g. `iter_40`) and `noop.p010.seed{SEED}.json` for the paired no_op baseline. Each JSON has float fields `fair_gap`, `loss`, `collisions`, `ttc_events`. Domination = policy `fair_gap` strictly < no_op `fair_gap` (by > eps) AND `loss` ≤ AND `collisions` ≤ AND `ttc_events` ≤.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_domination_scoreboard.py"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_scoreboard():
    spec = importlib.util.spec_from_file_location(
        "domination_scoreboard", ROOT / "tools" / "domination_scoreboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASES = [
    ("fair_safe_v9_i40_dt05_holdout101_150", 27),
    ("fair_safe_i90_dt05_holdout101_150", 30),
    ("fair_safe_v8_i150_dt05_holdout101_150", 33),
]


@pytest.mark.parametrize("dirname,expected", CASES)
def test_reproduces_known_bars(dirname, expected):
    d = ROOT / "runs" / "eval" / dirname
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    result = _load_scoreboard().score_eval_dir(d)
    assert result["n"] == 50
    assert result["n_dominate"] == expected
    # Safety must never bind on these dirs: every fail is G- or loss-driven.
    for row in result["per_seed"]:
        assert row["d_collisions"] <= 0
        assert row["d_ttc"] <= 0
    # Structural seeds are reported separately and none is dominated.
    assert result["structural"]["n"] == 11
    assert result["n_winnable"] == 39
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_domination_scoreboard.py -v`
Expected: FAIL — `tools/domination_scoreboard.py` does not exist (FileNotFoundError from spec_from_file_location).

- [ ] **Step 3: Implement the scoreboard**

```python
"""tools/domination_scoreboard.py — per-seed strict-domination scoreboard.

A seed is DOMINATED iff, vs the paired no_op baseline:
    policy fair_gap <  noop fair_gap   (strict, by > eps)
    policy loss     <= noop loss
    policy collisions <= noop collisions
    policy ttc_events <= noop ttc_events

Single source of truth for checkpoint selection (replaces TB reward and the
<= gate, which coast passes trivially). Usage:

    python tools/domination_scoreboard.py runs/eval/fair_safe_v9_i40_dt05_holdout101_150
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Easy no_op seeds (PLAN_V9_IMPROVE_PASS §1.2): no_op already near-fair,
# nothing to win. Reported separately; ties there are the accepted ceiling.
STRUCTURAL_SEEDS = [106, 117, 119, 120, 129, 136, 137, 143, 146, 147, 148]

_SEED_RE = re.compile(r"seed(\d+)")


def _seed_of(path: Path) -> int:
    return int(_SEED_RE.search(path.name).group(1))


def _load_rows(eval_dir: Path):
    policy, noop = {}, {}
    for p in sorted(eval_dir.glob("*.seed*.json")):
        if p.name == "summary.json" or p.name == "scoreboard.json":
            continue
        row = json.loads(p.read_text(encoding="utf-8"))
        seed = _seed_of(p)
        if p.name.startswith("noop."):
            noop[seed] = row
        else:
            if seed in policy:
                raise SystemExit(
                    f"multiple policy files for seed {seed} in {eval_dir}; "
                    "score one checkpoint tag per directory"
                )
            policy[seed] = row
    return policy, noop


def _fail_cause(d_g, d_loss, d_coll, d_ttc, eps):
    causes = []
    if d_g >= -eps:
        causes.append("G")
    if d_loss > eps:
        causes.append("loss")
    if d_coll > 0 or d_ttc > 0:
        causes.append("safety")
    return "+".join(causes) if causes else "none"


def score_eval_dir(eval_dir, noop_dir=None, eps=1e-9) -> dict:
    eval_dir = Path(eval_dir)
    noop_dir = Path(noop_dir) if noop_dir else eval_dir
    policy, noop = _load_rows(eval_dir)
    if noop_dir != eval_dir:
        _, noop_extra = _load_rows(noop_dir)
        noop = {**noop_extra, **noop}
    seeds = sorted(set(policy) & set(noop))
    if not seeds:
        raise SystemExit(f"no paired seeds found in {eval_dir}")

    per_seed = []
    for s in seeds:
        r, n = policy[s], noop[s]
        d_g = float(r["fair_gap"]) - float(n["fair_gap"])
        d_loss = float(r["loss"]) - float(n["loss"])
        d_coll = int(r["collisions"]) - int(n["collisions"])
        d_ttc = int(r["ttc_events"]) - int(n["ttc_events"])
        dominate = d_g < -eps and d_loss <= eps and d_coll <= 0 and d_ttc <= 0
        per_seed.append(
            {
                "seed": s,
                "dominate": dominate,
                "d_g": d_g,
                "d_loss": d_loss,
                "d_collisions": d_coll,
                "d_ttc": d_ttc,
                "fail_cause": "none" if dominate else _fail_cause(d_g, d_loss, d_coll, d_ttc, eps),
            }
        )

    winnable = [r for r in per_seed if r["seed"] not in STRUCTURAL_SEEDS]
    structural = [r for r in per_seed if r["seed"] in STRUCTURAL_SEEDS]
    return {
        "eval_dir": str(eval_dir),
        "n": len(per_seed),
        "n_dominate": sum(r["dominate"] for r in per_seed),
        "n_winnable": len(winnable),
        "n_winnable_dominate": sum(r["dominate"] for r in winnable),
        "structural": {
            "n": len(structural),
            "n_dominate": sum(r["dominate"] for r in structural),
            "n_tie_g": sum(1 for r in structural if abs(r["d_g"]) <= eps),
            "n_harmed": sum(1 for r in structural if r["fail_cause"] != "none"),
        },
        "per_seed": per_seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--noop-dir", type=Path, default=None)
    args = ap.parse_args()
    result = score_eval_dir(args.eval_dir, noop_dir=args.noop_dir)
    out = Path(args.eval_dir) / "scoreboard.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    s = result["structural"]
    print(
        f"dominate {result['n_dominate']}/{result['n']}  "
        f"(winnable {result['n_winnable_dominate']}/{result['n_winnable']}; "
        f"structural: {s['n_dominate']} dominated, {s['n_tie_g']} ties, "
        f"{s['n_harmed']} harmed)"
    )
    fails = [r for r in result["per_seed"] if not r["dominate"]]
    for r in fails:
        print(f"  seed {r['seed']}: {r['fail_cause']} dG={r['d_g']:+.2f} dL={r['d_loss']:+.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_domination_scoreboard.py -v`
Expected: 3 PASSED (27, 30, 33 reproduced). If a count mismatches, debug the tool, not the test — the bars were hand-verified from the per-seed JSONs.

- [ ] **Step 5: Also run the CLI once for the record**

Run: `python tools/domination_scoreboard.py runs/eval/fair_safe_v9_i40_dt05_holdout101_150`
Expected output first line: `dominate 27/50  (winnable 27/39; structural: 0 dominated, 0 ties, 11 harmed)`

- [ ] **Step 6: Commit**

```bash
git add tools/domination_scoreboard.py tests/test_domination_scoreboard.py
git commit -m "feat: per-seed strict-domination scoreboard (reproduces 27/30/33 bars)"
```

---

### Task 2: Reward v10 — fair_win + easy intervention tax

**Files:**
- Modify: `env/rewards.py` (insert helpers after `_delay_win`, line ~488; wire into `compute_shepherd_reward`)
- Modify: `config/defaults.py` (`reward_weights` dict, lines 29–47)
- Test: `tests/test_reward_v10.py`

**Interfaces:**
- Produces: `env.rewards._fair_win(stream_delay_gap, gap_scale=0.40, gap_ref=0.0) -> float` in [0, 1]; `env.rewards._easy_intervention_tax(mode, stream_delay_gap, gap_ref=0.0) -> float` in {-1.0, 0.0}.
- New reward-weight keys `fair_win` and `easy_intervention_tax`, read via `reward_weights.get(key, 0.0)` — unknown keys are safely ignored by old code (verified: `env/rewards.py:601-614` reads only via `.get`), and missing keys default to 0 so old configs are unaffected.
- `compute_shepherd_reward` already receives `mode` (the SELECTED, pre-remap mode — `env/core.py:735`), `stream_delay_gap`, `stream_fairness_gap_scale`, `stream_delay_gap_ref` (`env/core.py:742-744`). No `env/core.py` changes in this task.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_reward_v10.py — v10 learned-abstention reward terms."""

from env.rewards import (
    _easy_intervention_tax,
    _fair_win,
    compute_shepherd_reward,
)


def test_fair_win_pays_only_strictly_below_ref():
    assert _fair_win(None) == 0.0  # unknown gap
    assert _fair_win(0.10, gap_scale=0.40, gap_ref=0.10) == 0.0  # tie: no pay
    assert _fair_win(-0.10, gap_scale=0.40, gap_ref=0.10) == 0.0  # sign-insensitive
    assert _fair_win(0.30, gap_scale=0.40, gap_ref=0.10) == 0.0  # above ref
    assert abs(_fair_win(0.0, gap_scale=0.40, gap_ref=0.20) - 0.5) < 1e-9
    assert _fair_win(0.0, gap_scale=0.40, gap_ref=0.99) == 1.0  # clipped


def test_easy_intervention_tax():
    assert _easy_intervention_tax(0, 0.01, 0.10) == 0.0  # coast never taxed
    assert _easy_intervention_tax(1, None, 0.10) == 0.0  # unknown gap: no tax
    assert _easy_intervention_tax(1, 0.05, 0.10) == -1.0  # easy + act: taxed
    assert _easy_intervention_tax(4, -0.05, 0.10) == -1.0  # |gap| used
    assert _easy_intervention_tax(1, 0.50, 0.10) == 0.0  # hard state: free
    assert _easy_intervention_tax(1, 0.05, 0.0) == 0.0  # ref 0 -> off


def _v9_zero_weights():
    # Every key compute_shepherd_reward reads, zeroed, plus the v10 keys.
    return {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "delay_budget": 0.0,
        "delay_win": 0.0,
        "control_cost": 0.0,
        "mask_penalty": 0.0,
        "ineligible_penalty": 0.0,
        "mode_use_bonus": 0.0,
        "flow": 0.0,
        "tail": 0.0,
        "fair_win": 0.0,
        "easy_intervention_tax": 0.0,
    }


def test_compute_shepherd_reward_v10_terms_additive():
    # All v9 weights zeroed; helpers that would touch traci return 0 without a
    # connection (their exceptions are caught internally), so the total is
    # exactly the two v10 terms.
    w = _v9_zero_weights()
    w["fair_win"] = 0.20
    w["easy_intervention_tax"] = 0.15
    r = compute_shepherd_reward(
        "cav1",
        1,
        35.0,
        w,
        200.0,
        cav_ids={"cav1"},
        stream_delay_gap=0.05,
        stream_fairness_gap_scale=0.40,
        stream_delay_gap_ref=0.10,
    )
    # fair_win = (0.10 - 0.05)/0.40 = 0.125 -> 0.20 * 0.125 = 0.025
    # tax = -1 -> 0.15 * -1 = -0.15 ; total = -0.125
    assert abs(r - (-0.125)) < 1e-9


def test_v10_terms_default_off():
    # Zeroed v9 weights, v10 keys absent -> reward exactly 0 (backward compat).
    w = _v9_zero_weights()
    del w["fair_win"]
    del w["easy_intervention_tax"]
    r = compute_shepherd_reward(
        "cav1",
        1,
        35.0,
        w,
        200.0,
        cav_ids={"cav1"},
        stream_delay_gap=0.05,
        stream_fairness_gap_scale=0.40,
        stream_delay_gap_ref=0.10,
    )
    assert r == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reward_v10.py -v`
Expected: FAIL — `ImportError: cannot import name '_fair_win' from 'env.rewards'`.

- [ ] **Step 3: Implement the helpers in `env/rewards.py`**

Insert immediately after `_delay_win` (ends line ~488), before `_tail_penalty`:

```python
def _fair_win(
    stream_delay_gap: Optional[float],
    gap_scale: float = 0.40,
    gap_ref: float = 0.0,
) -> float:
    """Pay strict stream-balance improvement *below* the paired no_op ref.

    gain = max(gap_ref - |gap|, 0); returns clip(gain/scale, 0, 1) in [0, 1].
    Mirror image of ``_stream_fairness``: that term punishes excess above the
    ref, this one pays wins below it — ties earn nothing. 0 when gap unknown.
    """
    if stream_delay_gap is None:
        return 0.0
    scale = max(float(gap_scale), 1e-6)
    gain = max(float(gap_ref) - abs(float(stream_delay_gap)), 0.0)
    return min(gain / scale, 1.0)


def _easy_intervention_tax(
    mode: int,
    stream_delay_gap: Optional[float],
    gap_ref: float = 0.0,
) -> float:
    """-1 when a non-coast mode is selected while |gap| <= gap_ref.

    "Easy" = already at least as balanced as the paired no_op seed, so there
    is nothing to win and intervention is pure risk. Soft reward signal, not
    a mask: the policy stays free to act; abstention is simply what pays.
    Charged on the SELECTED mode (same convention as ``_control_cost``).
    0 when gap unknown, coast, or |gap| above ref.
    """
    if stream_delay_gap is None or int(mode) == 0:
        return 0.0
    if abs(float(stream_delay_gap)) <= max(float(gap_ref), 0.0):
        return -1.0
    return 0.0
```

- [ ] **Step 4: Wire into `compute_shepherd_reward`**

In `env/rewards.py`, after the line `w_delay_win = reward_weights.get("delay_win", 0.0)` (~line 606) add:

```python
    w_fair_win = reward_weights.get("fair_win", 0.0)
    w_easy_tax = reward_weights.get("easy_intervention_tax", 0.0)
```

After the `r_win = _delay_win(...)` block (~line 635-637) add:

```python
r_fair_win = _fair_win(
    stream_delay_gap,
    stream_fairness_gap_scale,
    stream_delay_gap_ref,
)
c_easy = _easy_intervention_tax(mode, stream_delay_gap, stream_delay_gap_ref)
```

In the final `return float(...)`, add two terms after `+ w_delay_win * r_win`:

```python
+w_fair_win * r_fair_win
+w_easy_tax * c_easy
```

Also update the module docstring term table (top of file, lines ~9-30) with the two new terms, matching the existing comment style.

- [ ] **Step 5: Add the weight keys to `config/defaults.py`**

In `DEFAULT_CONFIG["reward_weights"]`, after `"mode_use_bonus": 0.08,` add:

```python
        # v10 learned abstention (spec 2026-08-04): OFF by default for eval
        # parity with v9 runs; enabled by the v10 recipe commit (Task 4).
        "fair_win": 0.0,
        "easy_intervention_tax": 0.0,
```

- [ ] **Step 6: Run the new tests, then the full suite**

Run: `python -m pytest tests/test_reward_v10.py -v` — Expected: 4 PASSED.
Run: `python -m pytest -q` — Expected: all green (v10 weights are 0.0, so no existing test changes).

- [ ] **Step 7: Commit**

```bash
git add env/rewards.py config/defaults.py tests/test_reward_v10.py
git commit -m "feat: reward v10 — fair_win + easy intervention tax (default off)"
```

---

### Task 3: Easy-seed oversampling of the paired pool

**Files:**
- Modify: `env/core.py` (helper near top-level; init wiring ~lines 174–181; reset line ~491)
- Modify: `config/defaults.py` (after `"density_choices"` block, ~line 193)
- Test: `tests/test_paired_pool_oversample.py`

**Interfaces:**
- Produces: `env.core.paired_pool_sample_probs(pool, refs, easy_quantile=None, easy_boost=1.0) -> Optional[list[float]]` — probabilities aligned to `pool` order, or `None` for uniform. Consumed by `MARLAccelerateEnv.__init__` (stored as `self._paired_pool_probs`) and used in `reset()`.
- New config keys `paired_easy_quantile` (default `None` = uniform, preserves all current behavior) and `paired_easy_boost` (default `1.0`).
- Background: training episodes draw `S = self._episode_seed_rng.choice(self._paired_seed_pool)` uniformly at `env/core.py:491`; per-seed refs live in `self._paired_hdv_refs` (dict keyed by str(seed), each entry has `fair_gap`). Verify the exact attribute name at `env/core.py:176-181` before wiring. `numpy` is already imported in `env/core.py` (used for `np.random.default_rng`).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_paired_pool_oversample.py"""

from env.core import paired_pool_sample_probs

# fair_gap == seed value, 1..10 -> easy seeds are the low ones.
REFS = {str(s): {"fair_gap": float(s)} for s in range(1, 11)}
POOL = list(range(1, 11))


def test_disabled_returns_none():
    assert paired_pool_sample_probs(POOL, REFS, None, 2.0) is None
    assert paired_pool_sample_probs(POOL, REFS, 0.4, 1.0) is None
    assert paired_pool_sample_probs([], REFS, 0.4, 2.0) is None


def test_easy_seeds_boosted_and_probs_normalized():
    probs = paired_pool_sample_probs(POOL, REFS, easy_quantile=0.4, easy_boost=3.0)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
    # np.quantile([1..10], 0.4) = 4.6 -> seeds 1-4 easy: weight 3, others 1.
    assert abs(probs[0] - 3.0 / 18.0) < 1e-9  # seed 1 (easy)
    assert abs(probs[3] - 3.0 / 18.0) < 1e-9  # seed 4 (easy)
    assert abs(probs[4] - 1.0 / 18.0) < 1e-9  # seed 5 (hard)
    assert abs(probs[9] - 1.0 / 18.0) < 1e-9  # seed 10 (hard)


def test_missing_ref_treated_as_hard():
    refs = {"1": {"fair_gap": 1.0}}  # seeds 2,3 missing -> fair_gap 0.0
    probs = paired_pool_sample_probs([1, 2, 3], refs, 0.5, 2.0)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paired_pool_oversample.py -v`
Expected: FAIL — `ImportError: cannot import name 'paired_pool_sample_probs' from 'env.core'`.

- [ ] **Step 3: Implement the helper in `env/core.py`**

At module level in `env/core.py` (after imports, before the env class):

```python
def paired_pool_sample_probs(
    pool: list,
    refs: dict,
    easy_quantile: Optional[float] = None,
    easy_boost: float = 1.0,
) -> Optional[list]:
    """Per-seed sampling probabilities aligned to ``pool`` order.

    Seeds whose paired no_op ``fair_gap`` is at or below the
    ``easy_quantile`` quantile of the pool get ``easy_boost`` x weight, so
    easy episodes (nothing to win, abstention is the lesson) are seen more
    often in training. Returns None (= uniform) when disabled or degenerate.
    """
    if not pool or easy_quantile is None or float(easy_boost) == 1.0:
        return None
    gaps = [float((refs.get(str(s)) or {}).get("fair_gap", 0.0)) for s in pool]
    cutoff = float(np.quantile(gaps, float(easy_quantile)))
    w = [float(easy_boost) if g <= cutoff else 1.0 for g in gaps]
    total = sum(w)
    if total <= 0:
        return None
    return [x / total for x in w]
```

(Ensure `Optional` is imported from `typing` at the top of `env/core.py` — add to the existing import if absent.)

- [ ] **Step 4: Wire into `__init__` and `reset()`**

In `MARLAccelerateEnv.__init__`, immediately after the block that sets `self._paired_seed_pool` (~`env/core.py:179-181`; confirm the exact lines and the refs attribute name — expected `self._paired_hdv_refs`), add:

```python
        self._paired_pool_probs = paired_pool_sample_probs(
            self._paired_seed_pool,
            self._paired_hdv_refs,
            easy_quantile=self.config.get("paired_easy_quantile"),
            easy_boost=float(self.config.get("paired_easy_boost", 1.0)),
        )
```

In `reset()` (~`env/core.py:491`), change:

```python
S = int(self._episode_seed_rng.choice(self._paired_seed_pool))
```

to:

```python
S = int(self._episode_seed_rng.choice(self._paired_seed_pool, p=self._paired_pool_probs))
```

(`np.random.Generator.choice` accepts `p=None` = uniform, so disabled behavior is byte-identical.)

- [ ] **Step 5: Add config keys to `config/defaults.py`**

After the `"density_choices"` entry (~line 193) add:

```python
    # Easy-seed oversampling for paired-pool training (v10). None = uniform
    # (all current behavior preserved); the v10 recipe sets 0.25 / 2.0, which
    # lifts easy-episode share from ~25% to ~40% of sampled episodes.
    "paired_easy_quantile": None,
    "paired_easy_boost": 1.0,
```

- [ ] **Step 6: Run the new tests, then the full suite**

Run: `python -m pytest tests/test_paired_pool_oversample.py -v` — Expected: 3 PASSED.
Run: `python -m pytest -q` — Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add env/core.py config/defaults.py tests/test_paired_pool_oversample.py
git commit -m "feat: easy-seed oversampling of paired pool (default off)"
```

---

### Task 4: v10 recipe + training launch

**Files:**
- Modify: `config/defaults.py` (the v10 recipe: enable Task 2/3 knobs)
- Create: `_eval_fair_safe_v10_best_peek101_120.py`
- Create: `_eval_fair_safe_v10_best_dt05_holdout101_150.py`

**Interfaces:**
- Consumes: Task 2 weight keys, Task 3 config keys, v9 checkpoint `runs/rllib/shepherd_fair_safe_v9_dt05/checkpoints/iter_40`.
- Produces: training run dir `runs/rllib/shepherd_fair_safe_v10_abstain/` (checkpoints `best/`, `final/`, `iter_N/`); eval scripts used by Task 5.

- [ ] **Step 1: Enable the v10 recipe in `config/defaults.py`**

Change the Task-2 weights from `0.0` to the recipe values:

```python
        "fair_win": 0.20,
        "easy_intervention_tax": 0.15,
```

Change the Task-3 knobs:

```python
    "paired_easy_quantile": 0.25,
    "paired_easy_boost": 2.0,
```

Leave everything else untouched (delay_budget 1.10, delay_win 0.30, masks False). Update the comments added in Tasks 2/3 to say "v10 recipe active".

- [ ] **Step 2: Full suite must stay green**

Run: `python -m pytest -q` — Expected: all green. If any test asserts on default weights, update that test's expectation and note it in the commit message; do NOT weaken reward tests.

- [ ] **Step 3: Commit the recipe**

```bash
git add config/defaults.py
git commit -m "recipe: v10 learned abstention (fair_win 0.20, tax 0.15, easy x2 @q0.25)"
```

- [ ] **Step 4: Create the v10 peek eval script**

Copy `_eval_fair_safe_v9b_best_peek101_120.py` to `_eval_fair_safe_v10_best_peek101_120.py` with exactly these edits:
- Module docstring: `v10 best holdout PEEK — seeds 101-120 @ dt=0.5.`
- `OUT = ROOT / "runs" / "eval" / "fair_safe_v10_best_peek101_120"`
- `CKPT = ROOT / "runs" / "rllib" / "shepherd_fair_safe_v10_abstain" / "checkpoints" / "best"`
- `run_tag_prefix=f"v10bestpeek_{seed}"`
- Final print header: `=== v10 {TAG} PEEK @ dt=...`
Everything else identical (same `_seeds()`, `_noop()` cache reuse, gate computation).

- [ ] **Step 5: Create the v10 holdout eval script**

Copy `_eval_fair_safe_v9_i40_dt05_holdout101_150.py` to `_eval_fair_safe_v10_best_dt05_holdout101_150.py` with exactly these edits:
- `OUT = ROOT / "runs" / "eval" / "fair_safe_v10_best_dt05_holdout101_150"`
- `CKPT = ROOT / "runs" / "rllib" / "shepherd_fair_safe_v10_abstain" / "checkpoints" / "best"`
- `TAG = "best"`, `run_tag_prefix=f"v10besthold_{seed}"`, docstring/header updated.
(If the v9 holdout script hard-codes `N_PEEK`/`TAG = "iter_40"`, set the seed count to 50 — `eval_seeds[:50]` = 101–150 — and keep the noop cache reuse from `fair_safe_i90_dt05_holdout101_150`.)

- [ ] **Step 6: Commit the eval scripts**

```bash
git add _eval_fair_safe_v10_best_peek101_120.py _eval_fair_safe_v10_best_dt05_holdout101_150.py
git commit -m "eval: v10 peek (101-120) and holdout (101-150) drivers"
```

- [ ] **Step 7: Launch training (background, hours-long)**

```bash
cd /d/research-dense
source /c/Users/jpeng/Miniconda3/etc/profile.d/conda.sh && conda activate rl_driving
export SUMO_HOME='C:\Program Files (x86)\Eclipse\Sumo'
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="D:\research-dense;${PYTHONPATH:-}"
mkdir -p runs/rllib/shepherd_fair_safe_v10_abstain
python -u train.py --baseline shepherd --density dense --cav-penetration 0.1 \
  --paired-refs runs/eval/paired_noop_refs_dense.json \
  --from-checkpoint runs/rllib/shepherd_fair_safe_v9_dt05/checkpoints/iter_40 \
  --total-timesteps 900000 --logdir runs/rllib/shepherd_fair_safe_v10_abstain \
  > runs/rllib/shepherd_fair_safe_v10_abstain/train.out 2>&1
```

Expected: log shows `Resuming from checkpoint: ...iter_40` and `Paired refs: ... (n_refs=200, pool=100)`. Run as a background task; expected duration hours (v9 reached 240k steps at iter_40).

- [ ] **Step 8: Sanity-check the first iterations**

After ~15 minutes, confirm `runs/rllib/shepherd_fair_safe_v10_abstain/metrics.csv` (or `train.out`) shows rising `timesteps_total` and no crash/restart loop. If crashed: read `train.out` tail, fix, relaunch (same command).

---

### Task 5: Peek → holdout → scoreboard report

**Files:**
- Uses: `_eval_fair_safe_v10_best_peek101_120.py`, `_eval_fair_safe_v10_best_dt05_holdout101_150.py`, `tools/domination_scoreboard.py`
- Create: `docs/superpowers/reports/2026-08-05-v10-results.md` (adjust date to run day)

**Interfaces:**
- Consumes: trained checkpoints from Task 4; scoreboard from Task 1.
- Produces: the final verdict vs the 27/34/33 bars (slack rule), committed report.

Peek baseline to beat on seeds 101–120: v9 i40 stock = **11/20** (gate) / strict-domination equivalent from the scoreboard on `runs/eval/fair_safe_v9_i40_dt05_holdout101_150` restricted to seeds 101–120. Each eval seed takes ~80 s wall-clock; 20-seed peek ≈ 10 min at 3 workers; 50-seed holdout ≈ 25 min.

- [ ] **Step 1: Peek on the trained checkpoint**

Run (with `rl_driving` active, from `D:\research-dense`):
`python _eval_fair_safe_v10_best_peek101_120.py`
Expected: writes `runs/eval/fair_safe_v10_best_peek101_120/summary.json` and prints `pass X/20`.

- [ ] **Step 2: Score the peek with the scoreboard**

Run: `python tools/domination_scoreboard.py runs/eval/fair_safe_v10_best_peek101_120`
Compare `n_dominate/20` vs the v9-i40 peek slice: run
`python tools/domination_scoreboard.py runs/eval/fair_safe_v9_i40_dt05_holdout101_150`
and compare the seeds 101–120 subset. Promotion rule (from PLAN_V9 §protocol): proceed to full holdout only if peek strict-domination improves over v9 i40's slice; if flat, one more training segment may be run (resume same command); stop after 3 flat peeks.

- [ ] **Step 3: Full holdout eval (release candidate only)**

Run: `python _eval_fair_safe_v10_best_dt05_holdout101_150.py`
Expected: writes `runs/eval/fair_safe_v10_best_dt05_holdout101_150/summary.json`.

- [ ] **Step 4: Score the holdout**

Run: `python tools/domination_scoreboard.py runs/eval/fair_safe_v10_best_dt05_holdout101_150`
Record: `n_dominate/50` vs bars 27 (v9 i40), 34 (v3 i90), 33 (v8 i150) — bars measured under the slack rule (G margin 0.1s, loss slack 0.5s); winnable-only count vs 39; structural ties/harmed (target: 0 harmed); median dG on winnable dominated seeds (target ≈ −4 s or better — wins must not collapse to noise).

- [ ] **Step 5: Write and commit the results report**

`docs/superpowers/reports/2026-08-05-v10-results.md` containing: the scoreboard numbers above; fail-cause split; per-seed table of changes vs v9 i40 (gained/lost dominations); if below 27/50 — an autopsy in the style of `runs/probe/v9_coastforce_autopsy/` (action-mix per pass/fail seed) plus the next lever (weight retune table: fair_win 0.10/0.30, tax 0.10/0.25, boost 3.0); if ≥33/50 — statement that the v8 bar is beaten and the winnable-seed ceiling is 39/50.

```bash
git add docs/superpowers/reports/
git commit -m "report: v10 learned-abstention holdout results"
```

---

## Self-Review Notes (done at plan time)

- Spec coverage: scoreboard §4.1 → Task 1; reward v10 §4.2 → Task 2; oversampling §4.3 → Task 3; training + eval + reporting §4.3/§4.4 → Tasks 4–5; git process §5 → baseline commit done + per-task commits; testing §6 → per-task tests + full-suite gates. Coast-collapse and distribution-shift risks (§7) → Task 5 Step 4 checks win magnitudes and structural harm, Step 5 autopsy path.
- Type consistency: `score_eval_dir` keys match the test and Task 5 usage; `_fair_win`/`_easy_intervention_tax` signatures identical in test and implementation; `paired_pool_sample_probs` signature identical in test, helper, and call site.
