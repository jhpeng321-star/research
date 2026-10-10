# REGISTRATION: replicate campaign — PILOT (WD-01-REP-P)

Status: pilot APPROVED by the owner on 2026-09-28 (OWNER_DECISIONS_20260928.md,
item 3). Finalised from REPLICATE_DESIGN_DRAFT.md by the world model session.

- **Executor:** the reconstructor, who runs the SUMO under its campaign harness
  and owns the allocations and ledger entries.
- **Timing:** after chunk 3, or before it if the reconstructor prefers.
- **Main stage:** NOT approved. It is re-sized from this pilot and goes back to
  the owner.

## 1. Questions

- **Q1 (mechanism).** Can a saved chunk-2 t0 state be replayed with a
  replicate seed s_r, so that native and every arm of replicate r share the
  same random stream (CRN within r), while different r give different
  realisations?
- **Q2 (noise).** Within a pair, what is the replicate noise
  σ_rep(H) = sd over r of the paired delta Δ_r = (arm − native), at
  H = 30, 60, 120 and 600 s? The metrics are:
  - physics delay (contract §4.1);
  - planner-unit delay;
  - TTC<3 exposure;
  - hard-brake seconds.

**Closest prior:** A2 step 2 (ZCode e5265607), which had only a single
realisation per arm; its σ ≈ 15 veh·s at H = 30 mixes between-world
variation with noise. **New difference:** replicated CRN pairs from the same
t0 state, so within-pair noise is measured directly.

## 2. Worlds (chunk-2 TRAIN only; WM-47 plan sha256 75fefc75…49bf)

| world | demand (veh/h) | penetration | t0 (s) |
| --- | --- | --- | --- |
| w8488m2400p02t300 | 2400 | 0.02 | 300 |
| w8514m2600p10t420 | 2600 | 0.10 | 420 |
| w8527m2700p20t240 | 2700 | 0.20 | 240 |
| w8539m2700p40t300 | 2700 | 0.40 | 300 |

- These are one per demand/penetration stratum, with mixed t0.
- The pilot never uses CALIBRATION or HOLDOUT worlds. CALIBRATION is reserved
  for WD-01-C2L1 and Part B.

## 3. Arms and replicates

- **Arms:** `native`, `pi`, `fixed1`. Each arm uses exactly the chunk-2
  branch definition (same suffix controller and the same per-CAV fixed
  assignment as the chunk-2 `fixed1` branch).
- **Excluded:** switching arms (headroom-only) and walls.
- **Replicates:** R = 4, with seeds s_r = 10 × world_seed + r for r = 1..4
  (e.g. w8488 → 84881..84884). The same s_r is used for all three arms of
  replicate r.
- **r = 0 reference:** the existing chunk-2 branches, restored with their
  saved RNG. They are reused, not re-run.
- **CAV speed mode:** speedMode 27, as in chunk 2.
- **Branch length:** 600 s from t0, with the logging of the chunk-2 data
  contract plus `replicate_id` and `seed` in branch_record.json.

## 4. Mechanism and step order (the verification gates the rest)

**Step 1, on w8488 only:**

- **Choose the reseed method.** Preferred: restore the t0 state WITHOUT its
  saved RNG, and start SUMO with `--seed s_r`. The alternative, only if
  SUMO/TraCI supports it: restore with RNG, then reseed deterministically.
  The reconstructor records the method and the exact command line.
- **Run checks (a)–(d):**
  - **(a) Determinism:** (native, s_1) run twice gives byte-identical
    physics_steps.jsonl. This is +1 branch.
  - **(b) Seeds differ:** native s_1 vs native s_2 give different
    trajectories (the first differing step is reported).
  - **(c) Same arrivals:** the insertion schedule (routes and scheduled
    departures) is identical across r.
  - **(d) Per-vehicle RNG:** dawdle and lane-change randomness vary with
    s_r, including under `--thread-rngs` if the campaign uses it.
- **Stop rule:** if (a) or (c) fails, or if (b) shows no difference, STOP
  and report. Do not run steps 2–3.
- **Fallback, if reseeding after restore is impossible:**
  - re-run the base prefix with `--seed s_r` up to t0, save a state per r,
    and branch from it (the t0 states then differ slightly; pairing holds
    within r);
  - this is disclosed;
  - the cost rises by up to about 50 % (prefixes of 240–540 s);
  - it still fits inside the budget cap below.

**Step 2:** the 12 branches of w8488 (3 arms × R = 4).

**Step 3:** the remaining 3 worlds (36 branches).

## 5. Budget (cap, charged at allocation start)

- **Planned:** 48 replicate branches plus 1 determinism repeat = 49 × 600 s
  ≈ 29,400 sim-s, about 1 h at the campaign's throughput.
- **Cap:** 45,000 sim-s, which includes the prefix fallback. Stop at the cap
  and report.
- **No retry after results:** a failed branch is re-run once, only for an
  infrastructure failure (crash, killed process), and the re-run is
  disclosed.

## 6. Analysis (world model session; zero simulation, via ZCode)

- **Δ_r(H):** for each world, arm ∈ {pi, fixed1}, r and H, the paired delta
  Δ_r(H) = metric(arm, r) − metric(native, r).
  - Sign convention: treatment − reference; negative = improvement.
  - Windows are (t0, t0 + H].
- **Noise:**
  - σ_rep(H) = pooled within-(world, arm) sd of Δ_r over r;
  - reported with a bootstrap CI over worlds;
  - reported next to the between-world sd of the replicate means.
- **Reference check:** r = 0 is compared to the replicate distribution, to
  confirm the original branches are ordinary draws.
- **Main-stage sizing (proposal only):**
  R_main = ⌈((1.645 + 0.84) · σ_rep(H) / δ)²⌉, with:
  - δ = 10 veh·s physics delay at H = 30;
  - δ at H = 600 set by the owner from the pilot table. The table also
    shows δ = 5% and 10% of the median |pi − native| saving.
  Main stage: W = 12 chunk-2 TRAIN worlds, stratified. It goes to the owner
  for approval.
- **Outputs:**
  - REPORT_REP_PILOT.md;
  - raw Δ rows (saved before aggregation);
  - LEDGER_DRAFT_REP_PILOT.json, sent to the reconstructor.

## 7. Not in scope

- Scoring the Part A or Part B models on these replicates. That needs the
  main stage and its own registration.
- Any wall arm.
- Any change to chunk 2 or chunk 3.
