# Final goal: Fair + Safe merge control (travel-time budget)

**Status:** project objective for `research-dense` (dense @ 10% CAV, talk seeds 42–46; extend to hyper later).  
**Baseline:** paired **no_op** (idle CAVs, normal type), same seed — not All-HDV.  
**Evidence:** `runs/probe/fair_safe_scorecard/`, `candidate_actions/`, `FINDINGS.md`, timeLoss re-analysis.

---

## 1. Objective (priority order)

1. **Safety** — collisions and TTC events ≤ no_op (prefer strictly fewer).
2. **Fairness** — reduce imbalance of **delay (timeLoss)**, not raw travel time:
   - report mean/p90 timeLoss for **all / main / ramp**
   - primary fairness metric: \(G = |\bar{d}_{\mathrm{main}} - \bar{d}_{\mathrm{ramp}}|\) (lower is better)
   - no chronic “hold one stream so the other wins”
3. **Efficiency guardrail** — do **not** lose to no_op on total delay:
   - pass requires mean timeLoss ≤ no_op (strict; no +1.5 s slack)
   - beating no_op on delay is a bonus (`delay_win` in training)

**Non-goals (for now):** maximize throughput / beat no_op mean TT as the training target.

---

## 2. Why this goal (probe-backed)

| Candidate | Δloss | Δfair gap \(G\) | Δcoll | Verdict |
|-----------|-------|-----------------|-------|---------|
| recover_when_elig | **−2.4** | **−2.6** | −1.2 | Best overall — clears main delay and **narrows** gap |
| vacate_only (L0) | −0.7 | −1.5 | −0.8 | Keep — safety + mild fairness |
| fair_gap_meter | +1.0 | ~0 | −0.6 | Within budget; needs tuning as fairness lever |
| **alt_priority (fixed turns)** | **+14.5** | **+17** | −0.4 | **Reject** — hurts delay and fairness |
| corridor meter / yield | +5…+6 | worse or one-sided | — | Reject as load-bearing modes |

Raw main vs ramp **TT** is not comparable (different routes). Always use **timeLoss**.

Naive “slow main, then slow ramp by turns” **failed**. Prefer:
- **Recover** when the path ahead is clear (helps the more-delayed main without holding ramp),
- **Vacate** dropped lane,
- **Gap-driven** one-sided tip metering only when \(|G|\) is large,
- **Protect** on imminent conflict (TTC/PET).

---

## 3. Action space — Discrete(5)

| Idx | Name | Intent | Eligibility (sketch) |
|-----|------|--------|----------------------|
| 0 | `coast` | SUMO drives | always |
| 1 | `fair_balance` | Mild tip slowdown of the **advantaged** stream when delay gap is large | near tip; \(\|G\|>\tau\); ego on stream to slow |
| 2 | `recover_speed` | Queue discharge when merge/down clear | pre-merge; ego slow; merge/down healthy |
| 3 | `vacate_drop` | Leave merge L0 → L1 | on merge drop lane |
| 4 | `protect` | Extra headway / soft brake on imminent risk | leader/side TTC or merge PET below threshold |

**Retired as first-class modes:** `meter_inflow` (corridor), `yield_conflict` (open-ended), fixed-phase `alt_priority`.

**Masking:** keep action masking; `fair_balance` / `protect` gated; no “mode use” bonus for intervening.

---

## 4. Reward reshape (v9 — fairness-first do-no-harm)

\[
R = w_s R_{\mathrm{safe}} + w_f R_{\mathrm{fair}} + w_{\mathrm{budget}} R_{\mathrm{delay\_pen}}
  + w_{\mathrm{win}} R_{\mathrm{delay\_win}} + w_e R_{\mathrm{eff}} + w_t R_{\mathrm{stab}}
  + w_c C_{\mathrm{ctrl}} + w_{\mathrm{inelig}} C_{\mathrm{inelig}} + w_{\mathrm{mask}} C_{\mathrm{mask}}
\]

| Term | Role | Default weight (v9) |
|------|------|---------------------|
| \(R_{\mathrm{safe}}\) | TTC/PET/brake/collision | **1.25** |
| \(R_{\mathrm{fair}}\) | excess stream gap vs paired no_op: \(-\mathrm{clip}((|G|-G_{\mathrm{ref}})_+/\kappa,0,1)\) | **1.20** (primary) |
| \(R_{\mathrm{delay\_pen}}\) | one-sided: corridor delay rate above paired `delay_rate_ref` | **1.10** (big do-no-harm) |
| \(R_{\mathrm{delay\_win}}\) | one-sided credit when corridor rate **beats** no_op | **0.30** (modest) |
| \(R_{\mathrm{eff}}\) | light \(F_{\mathrm{flow}}\) | **0.35** |
| \(R_{\mathrm{stab}}\) | comfort | 0.25 |
| \(C_{\mathrm{ctrl}}\) | need-gated intervention tax | 0.25 |
| \(C_{\mathrm{inelig}}\), \(C_{\mathrm{mask}}\) | illegal / masked modes | 0.5 / 0.2 |
| mode_use_bonus | slight + only for **recover** / **vacate** when eligible | **0.08** |

Paired **no_op** refs (`delay_rate_ref`, `stream_delay_gap_ref`, flow refs) are seed-injected via `--paired-refs` (e.g. `runs/eval/paired_noop_refs_dense.json`).

### Zone observation delay slots (v9)

Zone vector is **12-D** (classic 8 + 4). Appended slots (same scales as reward):

| Slot | Meaning |
|------|---------|
| 8 | signed corridor delay excess vs no_op \(\in[-1,1]\) |
| 9 | fair-gap excess vs no_op \(\in[0,1]\) |
| 10–11 | live EMA main / ramp delay rates (abs-normalized) |

**Obs dim change breaks v8 checkpoints** — train a new run (`shepherd_fair_safe_v9_dt05`), from scratch.

---

## 5. Eval / checkpoint selection

Every eval vs paired no_op must print:

- collisions, TTC count  
- timeLoss mean/p90: all, main, ramp  
- fair gap \(G\)  
- mean TT (guardrail only)

**Holdout ruler (dense @ dt=0.5):** seeds **101–150** from `paired_noop_refs_dense.json` `eval_seeds` (train pool is 1–100). Compare to baselines i90 **35/50** and i150 **34/50** pass.

**Select checkpoints** by scorecard, **not** `policy_reward_mean`:

1. collisions ≤ no_op (prefer 0 delta)  
2. maximize **seed-pass** under: \(G \le G_{\mathrm{noop}}\) **and** \(\mathrm{loss} \le \mathrm{loss}_{\mathrm{noop}}\) (fairness + do-no-harm)  
3. among those, improve mean Δloss / ΔG  

Talk seeds 42–46 remain for quick smoke checks only — not for claiming generalization.

---

## 6. Implementation plan

1. Land this doc as the goal (done).  
2. Switch `Mode` / `ShepherdActionEngine` to the table in §3 (done).  
3. Stream fairness + delay budget + delay win + zone delay features (v9).  
4. Unit tests for gates / delay terms / zone dim.  
5. Train `shepherd_fair_safe_v9_dt05` from scratch with `--paired-refs`; select by holdout pass-rate.  
6. Eval holdout 101–150 @ `sumo_step_length=0.5`.

---

## 7. One-line pitch

> Mixed-autonomy shepherd for a highway merge that **improves safety and delay fairness** (timeLoss across main vs ramp) while keeping total delay within a small budget of idle CAVs — not a pure travel-time maximizer.
