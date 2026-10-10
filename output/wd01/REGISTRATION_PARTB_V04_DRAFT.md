# DRAFT: WD-01-PB-V04: Part B value model v0.4 on identifiable policy contrasts

**Status:** DRAFT, 2026-09-30, world model session (docs only). Not frozen.

**Authority:** owner decision (d), 2026-09-30, about 02:05
(OWNER_DECISIONS_WD01.md). Part B v0.4 trains on:

- chunk-4 `__native_base`;
- the wall-pilot branches;
- the corrected true native/PI arms,

so that the native, damping and wall contrasts become identifiable.

**Freezing requires all of:**

- (1) the action/wall input layer (ZCode 67554dd7) reviewed and merged;
- (2) the J4 wall pilot complete, with its receipts verified;
- (3) the owner/main decisions in §8;
- (4) an exposure audit of every source split.

## 1. Question

**Q1.** Can a deployable (CAV-strict) value model V(s, τ, policy) predict the
remaining delay under each fleet policy, with calibrated uncertainty? The
policies are native, damping, PI, switching, fixed mixed assignment, and a
wall episode.

**Q2.** Are its predicted POLICY CONTRASTS skilful on held-out worlds? A
contrast here is ΔV between two policies from the same state at the same t0.
This is the quantity the planner uses, and no earlier Part B version could
test it: v0.2–v0.3.2 saw only damping-fleet and switching-mix data (WM-72).

**Closest prior:** v0.3.2 (CHANGES_V032.md). It used the same model family
(gbm-variant auto, per-τ hybrid) and the same deployable_strict inputs.

**The difference:**

- new sources with true policy labels;
- per-CAV action and wall tokens;
- a policy-conditioned target;
- contrast metrics.

## 2. Sources (true state logs; TRAIN / CAL splits of each plan; HOLDOUT never)

| source | policy label | origin |
| --- | --- | --- |
| chunk-4 `<w>__base` over [t0, t0 + 600] | damping (fleet) | saved state at t0 |
| chunk-4 `<w>__native_base` | native (fleet), [0, t0 + 600] | not restored |
| chunk-4 `<w>__native` | native (fleet) | restored from base@t0 |
| chunk-4 `<w>__pi` | PI external-U (fleet) | restored |
| chunk-4 `<w>__fixed*` (2 arms) | fixed mixed assignment (per-CAV, from switch_schedule) | restored |
| chunk-4 `<w>__switching*` (3 arms) | switching mix (per-CAV, from schedule/commands) | restored |
| wall pilot J4, 44 chunk-4 worlds × 4 arms | wall episode (members) + native (non-members) | restored |

- Chunk-4 plan: output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json
  (sha 65510b27…). There are 7 restored arms per world: native, pi, 2 × fixed,
  3 × switching.
  - **Before build, verify that chunk-4's arms really are the labelled
    policies.** The WM-72 defect was that "__native" in chunks 1–3 was
    switching. Evidence comes from each arm's switch_schedule.json and
    suffix_commands owners. A mismatch refuses the source.
- **Per-CAV actions:** only from switch_schedule.json, suffix_commands.jsonl
  and wall_events.jsonl. Never from dispatcher_rows.jsonl or insertions.jsonl.
- **Splits:**
  - chunk-4 TRAIN is 63 worlds, and CAL is 9 worlds;
  - HOLDOUT is 18 worlds, never opened (tools/holdout_guard.py);
  - wall-pilot worlds keep their chunk-4 labels.
  - Chunk-4 CAL is also C4L1's look set for Part A. Part B's use of it
    (calibration and report only) does not feed Part A, and this is
    disclosed.
- **Chunks 1–3 and fix27:** an optional labelled AUXILIARY pretraining set
  (damping + switching_mix only), reported as an ablation. It is not part of
  the headline (open item §8-2).

## 3. Inputs

- **Feature set:** deployable_strict is the HEADLINE, and deployable_detector
  is a labelled SENSITIVITY. Both carry the per-CAV action tokens (with an
  unknown mask) and the wall tokens from the 67554dd7 layer.
- **Policy conditioning:** the continuation policy π after s. For fleet arms
  it is the fleet label. For mixed/switching/wall it is the per-CAV
  schedule summary known to the planner at t.
- The ideal-sensing premise is stated, and the reconstruction gap is measured
  separately (RG-0).
- The v0.3.2 fixes carry over: no t0 recovery, pre_fix fixed at predict, and
  world-grouped bootstrap and early stopping.

## 4. Targets

- V(s, τ, π) = the remaining tracked-plus-arrival delay over (t, t + τ] under
  π, for τ ∈ {30, 60, 120, 300, 570}.
- Safety heads as in v0.3.x.
- **Contrast target:** ΔV = V(s, τ, a) − V(s, τ, native), on same-world,
  same-t0 restored pairs. It is evaluated from paired predictions and is not
  trained as a separate head unless §8-3 says so.

## 5. Model and selection

- **Model:** gbm-variant auto per τ (ridge, hybrid, pure). The v0.3 net is an
  optional comparator.
- **Selection and calibration:** on TRAIN world-grouped OOB only. The CAL
  guard stays, and CAL is read once, for the report.

## 6. Acceptance (reported per variant; pre-specified before the CAL read)

- **A1:** CAL CRPS beats climatology at every τ.
- **A1:** 10–90 coverage is in [0.75, 0.85].
- **A1:** the HARD per-τ comparison vs ridge on the same inputs (a tie is a
  TIE).
- **A2 (contrast skill, on CAL pairs, per contrast** damping−native,
  PI−native, wall−native, mixed−native):
  - the sign accuracy of the predicted mean ΔV vs the realised ΔV, beating
    0.5 with a world-bootstrap 90 % CI above 0.5;
  - the ΔV CRPS beats a "zero-contrast" baseline and a "TRAIN-mean contrast
    per demand × penetration cell" baseline.
- **A3:** per-policy coverage (the same band), so that no policy is
  systematically miscalibrated.
- **Always reported:**
  - breakdowns by demand and penetration;
  - the wall outcome strata (assembled_released vs aborted);
  - the strict vs detector information price;
  - the auxiliary-pretraining ablation.

## 7. Budget

- Zero SUMO.
- A dataset build (the new root wd01_partB_v0data_v04, copy never hard-link,
  with SHA256SUMS and a backup) plus CPU GBM fits. From v0.3.2, the build is
  about 1–2 h at 4 workers and each refit about 30 min.
- The data volume is the chunk-4 sources plus 176 wall branches. The exact
  row count is computed at build.

## 8. Open items (main / owner)

1. **The wall-pilot shape** (44 worlds; owner sign-off pending) and its
   completion time on the rented box. v0.4 cannot freeze before J4 lands.
2. **Auxiliary use of chunks 1–3:** headline-excluded (recommended) or pooled?
3. **The contrast target:** evaluate ΔV from paired V predictions only
   (recommended), or add a trained Δ head?
4. **Chunk-4 CAL** is shared with C4L1 (Part A). Is that acceptable as
   disclosed (recommended), or should chunk-4 CAL be split between Part A and
   Part B?
