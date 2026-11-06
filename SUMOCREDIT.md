# SUMOCREDIT: confident deployment + real-SUMO credit (S-SUMOCREDIT, 2026-10-25/26)

Branch `c4/sumocredit` (from `c4/loop14c` 77d9e574). Everything is flag-gated; with the flags off, behaviour is unchanged. Example loop config: `configs/c4_loop14_sumocredit_example.json`.

## (1) Confident gate with a K-candidate plan check

**Rule.** Each decision imagines every unit (fleet, 150 m CAV chain, or CAV) on the RSU belief.

- **Arms:** `REF` plus K candidates from the learned policy only:
  - k=0 is the deterministic mean plan;
  - odd k are policy samples with independent noise;
  - even k are samples with one acceleration draw shared by every CAV.
- **Eligibility:** calibrated `P(J_cand < J_ref) >= 0.8`. A sample must also pass `P(J_cand < J_mean) >= 0.8`.
- **Choice:** the eligible candidate with the lowest imagined J runs. Otherwise the unit abstains to the reference.
- **Reference:** `native` (default) or `hold`.
- No PI, wall or MAPPO controller is ever a candidate.

**CRN:** shared driver draws and lane uniforms; policy noise is keyed by vehicle id, so SUMO replays the imagined sample.

**Calibration.**

- Raw score (imagined paired road-delay difference) -> isotonic blocks with a Beta(1,1) posterior (1/1 = 0.67, 3/3 = 0.80; gaps map to the lower block). No hand-set probability.
- Fitted on TRAIN records of the loop's own paired SUMO arms (authoritative J).
- Two tables, `reference` and `mean`; without the mean table no sample replaces the mean.

**Paths.**

- **Evaluation:** `deploy --controller policy_gate --gate-table T [--gate-mean-table M --gate-candidates K]`. Paper arms: `policy`, K=1 (gate only), K>1 (gate + check).
- **Loop collection** (`--policy-pairs`): `--gate-horizon/--gate-candidates/...` add calibration arms REF + PLAN_{u,k}; `--gate-table/--gate-mean-table` add `GATED`/`GATED_K` next to the unchanged POLICY and B arms.
- **`loop14.py`:** convert refits both tables on all cycles; extras are budget-charged before SUMO.

**Logged:** open/abstain rates, chosen candidate, chosen-vs-mean improvement, decision ms.

**Decision compute** (median s; cpuX, 2 threads, 13E-A WM, real TRAIN states of 80 slots, fleet; `tools/sumocredit/GATE_TIMING_fleet_cpuX2t.json`):

| H \ K | 1 | 4 | 8 |
|---|---|---|---|
| 10 s | 1.4 | 2.5 | 5.5 |
| 20 s | 2.5 | 4.2 | 10.8 |
| 30 s | 3.4 | 7.0 | 12.1 |

**Chosen:** K=8, H=10 s, decision every 6 s (max 6.2 s); alternative K=4, H=20 s at 5 s. (Laptop numbers, 10x slower under contention, are not used.)

## (2) Real-SUMO cooperation credit

`--credit-horizon H` runs `ONLY:i` (i in G, |G| <= 3), `GROUP`, optional `FLIP:i`; `B`/`ALL` reuse prefixes. Terms (D positive = better, `coop_imagine`/`branching` sign): `D_X = J_B - J_X`, `C_G = D_G - sum D_i`, `M_i = D_G - D_{G\i}`, also imagined with CRN (`wms.control.real_credit14`).

**Correction.**

- Per type, OLS slope b of real on imagined D; `m = clip(b + 1.96 SE, 0, 1)` scales `w_individual/group/coop/team` (`--real-credit`; loop `policy.real_credit`).
- Imprecise evidence leaves m = 1; never amplifies or flips sign.

Not critic targets: ONLY/GROUP returns value mixed policies, and additive offsets leave pathwise gradients unchanged.

**Budget per cycle (29,776 sim-s).**

- **Credit at H=30 s:** 120 sim-s per point, 3,840 sim-s per cycle (13 %).
- **Slope precision:** pulling the bound below 1 when b=0.5 needs about 14 pairs at r=0.5 and about 41 at r=0.3. One cycle resolves D_i (3 pairs per point) at r >= 0.2 and D_G/C_G at r >= 0.34. Pool across cycles.
- **Horizon:** C4-HEADROOM per-state SD of D_all is 300-410 veh-s at 120 s vs about 2,000 at 300 s; keep H <= 60 s.
- **Calibration (fleet, K=8, 10 s):** 70 sim-s per point, 2,240 per cycle. This yields 256 reference records and 224 mean records.
- **Gated arms:** 2 x 300 s per point = 64 % of the budget; run them in evaluation. Example config: 13 worlds = 29,133 sim-s.

## Smoke (registered; cpuX; seeds 967600-967604; 1,997 of 2,000 sim-s)

0 collisions and 0 teleports across all stages.

| Stage | sim-s | What ran |
|---|---|---|
| s1 | 1,081 | Calibration (4 units) + 2 credit points |
| s2 | 451 | GATED arm |
| s3 | 330 | Deploy, K=1 |
| s4 | 75 | K=3 calibration: the replayed samples differ from the mean (71.273/71.220 vs 71.232), and both sample-vs-mean signs match imagination |
| s5 | 60 | Deploy, K=3, 0.44 s per decision |

With n <= 4 the tables reach at most 0.75, so the gate stayed closed (abstained), as designed. Mechanics only; receipts in `tools/sumocredit/smoke_a2`, `smoke_k`.

## Tests

`test_sumocredit.py` + `test_sumocredit_kcheck.py` (23 tests) cover tables, K eligibility on synthetic beliefs, noise replay, executor mixing, real-WmSModel imagination, credit math, loop wiring and flag-off pass-through (identical weights object, unchanged 29,776 charge). Related suites: 454 passed.

## Caveats

- Calibration states are native-base (shift): validate on disjoint TRAIN-dev episodes before any claim.
- Imagined J omits source waiting; jointly open units are not the calibrated event; SUMO keeps LC2013 strategic merges.
