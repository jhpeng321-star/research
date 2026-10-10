# Metastable-Control Feasibility Probe — Design

**Status**: APPROVED (2026-08-12, user: "implement this for me")
**Verdict (measured 2026-08-12)**: FAIL on both pre-committed legs → idea
closes; B stays champion; the RL retry was NOT launched (decision rule §1).
Full numbers: `runs/probe/FINDINGS.md` + `runs/probe/metastable_feasibility/`.
**Goal**: Before any RL retrain, determine whether ANY controller can actively
hold the merge corridor on the metastable upper branch — and do it without
harming TTC relative to champion B — using the existing (mode, dial) action
space.

**Context**: Experiment C (RSU health-mask) failed because it was reactive
rescue (mask unlocks metering only when h ≤ 0.2) and the policy was never
rewarded for health. This probe tests the two unknowns C left open:
(1) loop gain — does the CAV's actuator move h at all?
(2) sufficiency — can a hand-built controller exploit the lever within the
    existing action space?

---

## 1. Decision rule (pre-committed, agreed with user)

PASS (proceed to an RL retry with health-as-reward + h in obs + protect
gating) requires BOTH, measured on the 8-seed holdout
(105/110/112/114/115/119/126/149) at 2700:

- **Health holds**: mean Δupper-branch-frac ≥ +0.05 vs no-op AND collapse
  onset (first step with h < 0.2) delayed vs no-op on ≥ 1 chaotic seed.
- **TTC ≥ B**: per-seed TTC(rule_full) ≤ TTC(B iter_95) on ≥ 7/8 seeds
  (same-seed runs are deterministic → strict comparison, no tolerance).

FAIL → the idea closes honestly; B stays champion; the negative result (and
the loop-gain measurement) goes into the paper narrative.

## 2. Files

| File | Change |
|---|---|
| `env/core.py` | NEW public `metastable_health_detail() -> (h, rho, v_mean)`; refactor `_merge_metastable_health` to call it (behavior unchanged, cached lane length reused). |
| `policies/metastable_rule.py` | NEW: `MetastableRulePolicy` (h-table controller + ablations), `ScriptedInterventionPolicy` (Stage 1), pure decision function `metastable_rule_decision(...)` (unit-testable, no env), `HealthRecordingWrapper` (records per-step h/rho for ANY policy, incl. no_op). |
| `_probe_metastable_feasibility.py` | NEW probe driver cloned from `_eval_hybrid_meta_v1.py` harness (run_cfg + per-seed spawn timeout + shared no-op cache + row/summary writing). |
| `tests/test_metastable_rule.py` | NEW unit tests for decision logic, hysteresis, smoothness cap, TTC floor, accessor range. |
| `runs/probe/FINDINGS.md` | Verdict appended after the runs. |

No changes to reward, observation dims, trained policies, or the eval driver.

## 3. Stage 1 — loop-gain (is the lever real?)

Seeds 105 (calm) + 112, 115 (chaotic), scene 2700, ~600 steps.

| variant | behavior (all CAVs) |
|---|---|
| `no_op` (cached) | passive baseline |
| `meter_burst` | while h < 0.4: force `fair_balance` dial 0.6 for 10 s, then coast 5 s |
| `pace` | force `protect` dial 0.2 continuously |
| `meter_aggressive` | same trigger, dial 1.0 (amplitude contrast — smoothness is the known antidote) |

Metrics per variant vs no_op: h-mean, frac(steps h ≥ 0.6), first-collapse
step, TTC events, G, loss.

**Go/no-go**: h-mean / upper-branch frac unmoved on 105 AND collapse onset
unshifted on 112/115 → lever dead → stop, write verdict. Else proceed.

## 4. Stage 2 — h-table controller (`MetastableRulePolicy`)

Per step, per CAV, role-aware (ramp CAV vs mainline CAV via distance-to-merge
helpers already used by `RuleBasedMergePolicy`):

| condition | ramp CAV | mainline CAV |
|---|---|---|
| h ≥ 0.6 | coast | coast |
| 0.35 ≤ h < 0.6 | coast | pace: `protect` dial 0.2 |
| h < 0.35 and ρ ≥ ρ_crit | meter: `fair_balance`, dial 0.3→0.7 by ρ-excess | pace: `protect` dial 0.2 |
| h < 0.35 and ρ < ρ_crit | coast | coast |

Constraints:
- **Hysteresis**: once metering, release only when h ≥ 0.55 for 5
  consecutive steps (KWG upper/lower branch; prevents flip-flop).
- **TTC floor**: leader TTC < 1.8 s → suppress metering, force `protect`
  hold (never meter into a near-miss).
- **Smoothness budget**: dial rate capped ±0.05/step; ≥ 3-step dwell on mode
  switches.

Ablations (config flags on the same policy): `rule_full` (all on),
`rule_no_floor` (TTC floor off), `rule_aggressive` (dial full-strength, no
rate cap, floor off). Grid: 8 seeds × {2700, 2400} × 3 variants = 48 runs,
`--workers 1`, per-seed timeout 900 s, no-op cached. Expect ~1–1.5 h;
timeout/XML failures re-run individually (handoff §8).

## 5. Metrics & PASS rule

Per seed: h-mean, frac(h ≥ 0.6), first-collapse step (from
`HealthRecordingWrapper` history), TTC, G, loss — vs no_op (Δ) and vs B
(existing `runs/eval/hybrid_meta_iter_95_light_m2700r400/summary.json`).
PASS rule as in §1. Output: per-variant JSONs + `summary.json` in
`runs/probe/metastable_feasibility/`; verdict appended to
`runs/probe/FINDINGS.md`.

## 6. Sequencing & risks

Stage 1 (~30 min) → go/no-go → Stage 2. No training anywhere. Only core
change (accessor) covered by existing `pytest` suite. Standard infra risks
(timeouts, transient XML) with standard mitigations.

## 7. Measured results (2026-08-12)

**Stage 1 (8 seeds × 600 steps @2700) — GO**: the lever exists. meter_burst
Δh +0.023 / Δupper-frac +0.028 across all 8 seeds, win rule 5/8 (dTTC +0.4);
dial 0.6 ≡ 1.0 exactly (amplitude invariance reproduces); first-collapse
onset never shifts.

**Stage 2 (8 seeds, full episodes @2700) — FAIL both pre-committed legs**:

| leg | requirement | rule_full measured | verdict |
|---|---|---|---|
| health | Δupper-frac ≥ +0.05 AND collapse delay ≥ 1 chaotic seed | +0.009 (dH −0.001), shifts all 0 | FAIL |
| TTC ≥ B | TTC ≤ B per seed on ≥ 7/8 | 2/8 (105/115/119/126/149 NO) | FAIL |

rule_full: wins 2/8 vs no-op, dTTC +1.5 (B: 4/8, +1.5). The 600-step benefit
evaporates over full episodes; metering for health is not TTC-calibrated
(same root cause as expC), and the ideal rule controller — the upper bound
for any policy in the (mode, dial) space — cannot hold the upper branch.
**Decision-rule consequence: no retry training; B stays champion.** The retry
infrastructure (health-as-reward, h-in-obs, protect gating — committed, all
default-off) remains inert unless new evidence appears. Stage 2 @2400
supplementary leg runs for the paper's negative-result narrative.
