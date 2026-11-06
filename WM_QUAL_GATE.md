# WM_QUAL_GATE - pre-registered gate "WM good enough to PROPOSE plans on wz3to2" (task WM-D)

Registered 2026-11-06 before any wz3to2 world-model prediction was produced or inspected (no fine-tuned
checkpoint existed; the un-fine-tuned WM2 had not been run on wz3to2). Machine-readable twin:
`tools/wzqual/gate_spec.json` (sha256 b013c0da9e9da5c770f9d8fc1b87d52da587cb3200646973f959fe9db54acb1a);
evaluator `tools/wzqual/gate.py`; runner `python -m tools.wzqual.run_qual`. Branch c4/wz-wm-qual.
Seen before registration (no WM involved): SUMO label noise (r0 vs dithered r1 on 83 replicated
candidates: T Spearman 0.68, S Spearman 0.37, S sign agreement 66 %), and a keep-speed persistence
reference on 989200-989209 (used only to check the plumbing).

## 1. Use and owner rules
The WM proposes cooperation moves, SUMO verifies every proposal. Owner 2026-10-31: the WM must act
alike IN DISTRIBUTION (breakdown probability first) and actions are chosen by expectation; the
distributional criteria are primary, per-world MAE is secondary. Comparable merge-map numbers
(WM_ERROR_CHECK_20261105.md): SUMO won 40.7 % of WM-positive predictions, Spearman 0.103.

## 2. Data (zero new SUMO for scoring)
- Truth: WMQUAL-B native prefix tapes (REGISTRATION 16:10, 74 world-cells, 0-1200 s, identity 74/74 vs the
  labelled natives) of the SUMO-labelled worlds 989200-989216, 989350-989353, 989400-989402; optional
  WM-B split=WMQUAL natives (989580-989599). All are WM hold-outs (never WM train/val; WM-A map_guard
  refuses them; WM training worlds are 989100-989199).
- Labels: NEWMAP/COOPKINDS/PLANGEN/LOWCAV `*_MOVES_B/R.jsonl` + `*_NEGATIVES.jsonl` (+ later WM-C
  verifications) in runs/active/C4-V7FT-20261017, member specs from the rows / probe MOMENTS.json
  (else the registered nm2 reconstruction, validated 0/379 mismatches on PLANGEN).
- Starts (distributional): every world-cell tape, t0 = 300, 420, ... while t0 + 300 <= tape end; the WM
  sees the t0 frame, 120 s of observed history and the observed entry log only.
- Starts (action response): each labelled moment (world, cell, t0); arms base / group / each solo
  (solos only for candidates with an S label) in ONE call with common random numbers.
- Samples: K = 8 for a stochastic backend, 1 for a deterministic one; WM value = mean over samples.

## 3. Fixed definitions (WM frame: s = x - 2620.56, drop at s = 75.44; WM lanes 0/1 through, 2 closing)
- Team delay J on [t0, t0+300): sum over 0.5 s steps of dt * (sum_on-road max(0, 1 - v/31.29) + source
  backlog) = the label definition (reproduces the native J to 1e-4 veh-s). Predicted J mode: `total`
  if the backend predicts the source backlog, `road` if it predicts entrants but no backlog, `t0_cohort`
  otherwise (recorded per row). T = J_base - J_group, S = min_i J_solo_i - J_group (expectations).
- Breakdown: approach zone [drop-300 m, drop], all lanes, 10 s bin mean speed; two-regime Gaussian
  mixture on log speed fitted on the fitting tapes (WM training natives if supplied, else the pooled
  qualification truth; NO WM output), init at the 25/75 % quantiles, 200 EM iterations, boundary =
  equal posterior; written and hashed (BREAKDOWN_RULE.json) before any rollout. Onset = first bin
  starting a 3-bin (30 s) run below the boundary (empty zone = free flow). Broken-at-start = the last 3
  OBSERVED pre-t0 bins below (shared by truth and WM; scored separately as recovery). Degenerate rule
  (|mu_hi - mu_lo| < 0.15 or weight < 0.03) => breakdown criteria UNIDENTIFIED.
- Queue: vehicles with v < 5 m/s in [drop-1000 m, drop] (bin means; closing-lane share and extent
  reported). Throughput: crossings of s = drop + 100 m. Zone-speed CRPS: ensemble CRPS per 10 s bin.

## 4. Pass/fail (all primary criteria must PASS; secondary metrics never change the verdict)
Integrity (else INVALID): I1 qualification seeds disjoint from the checkpoint's train+val world list
(missing list = INVALID); I2 tapes identity-gated / split=WMQUAL; I3 rule frozen before rollouts;
I4 backend contract (t0 frame reproduced within 1e-3, 0.5 s steps, CRN, members CLOSED LOOP from the
move rules, never a recorded member path); I5 TRAIN seeds only (never 989600-989699, 964000-964199,
962xxx, 977xxx).

| id | criterion (horizon 300 s) | threshold |
|---|---|---|
| C0 | move-role coverage of T-labelled candidates; roles hold and lane_to executable | >= 80 % |
| D1 | breakdown probability (not-broken starts): mean WM P(breakdown <= 300 s) - real rate | pooled abs <= 0.10; every cell with >= 12 starts abs <= 0.20 |
| D2 | onset time distribution, W1 (WM samples vs real events) | <= 45 s (needs >= 10 real events) |
| D3 | queue (mean queued vehicles over [t0, t0+300)): pooled relative mean error; W1 / real SD | abs <= 0.25 and <= 0.50 |
| D4 | discharge throughput, pooled relative mean error | abs <= 0.08 |
| D5 | zone-speed CRPS at 120 s and 300 s | <= keep-speed persistence (with persistence inflow) |
| A1 | within-moment Spearman (pred T vs SUMO T), mean over moments with >= 3 candidates | >= 0.30 and world-bootstrap CI low > 0 |
| A2 | pooled Spearman (pred T vs SUMO T) | >= 0.30 and CI low > 0.10 (merge map: 0.10) |
| A3 | precision of WM-positive T: P(SUMO T > 0 / pred T > 0) | >= max(0.60, base rate + 0.10) (merge map: 0.41) |
| A4 | synergy precision@k (k = 10 if >= 30 S-labelled else ceil(n/3)): P(SUMO S > 0) in WM top-k by pred S | >= max(0.50, min(0.70, 1.3 x base rate)) |
| A5 | planning regret per moment (WM picks argmax pred T, or no move if all <= 0) | mean regret <= 0.5 x random-choice regret AND mean realised gain CI low > 0 |
| A6 | pooled Spearman T beats the keep-speed persistence backend on the same candidates | strictly greater |

Minimum samples (else UNIDENTIFIED): 40 not-broken starts (D1), 12 per cell for the per-cell check,
10 real breakdown events (D2), 40 starts (D3/D4), 30 ranked moments (A1-A3, A5), 20 S-labelled
candidates (A4). D1-D4 require a backend that predicts entrants (inflow drives the drop).
Verdict: PASS iff integrity holds and all primary PASS; FAIL if any primary FAILs; INCONCLUSIVE if none
fails but one is UNIDENTIFIED; INVALID on an integrity failure. Uncertainty: 2000-resample world-cluster
bootstrap (resample seeds with all their cells/starts), RNG 20261106.

## 5. Secondary (reported, never gating)
Car level at 0.5 s (one step), 60, 120, 300 s: |ds|, |dv| median/p90, lane and presence agreement, HDV
and CAV, WM vs keep-speed; traffic level per section (30 s bins) speed/density MAE, cohort-J error;
Brier/AUC/reliability of P(breakdown); recovery of broken starts; lane-change rates by transition;
per-family ranking; strict precision@k (S lower bound > 0); admitted moves in the WM top-k.

## 6. Planned runs
Baseline: the un-fine-tuned WM2 b8797242 bound to wz3to2 (owner/MAIN 2026-11-06) - reference only.
Main: WM-E's fine-tuned checkpoint (WMTRAIN line), same command with `--checkpoint`; one run, no
threshold or rule change after results; any later variant is a new registered run.
