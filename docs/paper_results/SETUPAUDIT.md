# S-SETUPAUDIT — 2026-10-30
Final audited code/tests/evidence SHA: `b56444ab31d3fb52107e770dba30a311f0e72ede`.
Documentation delivery SHA: `git log -1 --format=%H -- docs/paper_results/SETUPAUDIT.md`.

## Verdict: (b), the training WM/objective does not prefer MAPPO on the measured average
- Shipped mix-OFF, one start: MAPPO−CTRL0 J90 **+77.34**, J300 **+5,324.85 veh-s**; maximised credit **−147.43**.
- Intended mix-ON, two starts: MAPPO−CTRL0 J90 **+8.30**, J300 **+3,278.14 veh-s**; maximised credit **−24.06**.
- Relative to supplied σ90=25.38≈25.4 / σ300=43.91≈43.9: OFF **3.05× / 121.27×**; ON **0.33× / 74.66×**.
- These ratios are descriptive, not significance claims: the qualification metric/population/K mismatch is confirmed below.
- Candidate A equals CTRL0 at both horizons in both setups; every long gate is OFF. ON dense credit prefers MAPPO on one of two starts (+23.83), loses on the other (−71.95).

## Bound inputs and exact value path
WM2 `b8797242`, supplied WM config `f95f7c4c`, MAPPO `aecd7d70`; candidate A update 190 `de235a78` (latest completed when inputs were bound).
Entry config: `configs/c4_policy14_candA_longgrad_skip.toml`; WM frozen, K8, entries/control ON, leader rule ON, always-commanded mode 2, dense 90 s / continuation 300 s.
Next saved TRAIN-sampler draws: `w963758m2400p20t300` @780 s (generic_random), `w963111m2400p20t300` @970 s (native); OFF uses the first, ON both.
Same uniform-size group rule, fixed variant seed 965140 (size 3 here), future seed 190, entry seed 1000003×191; all/base/three solo/group rows, no flips.
`Policy14Trainer.imagine → actor_objective`; MAPPO loaded by `mappo/eval_bridge.py`. Deterministic actor evaluation; same training credit functional, reported at common scale=1.
MAPPO's learned lanes drive its trajectory; categorical REINFORCE log-probabilities are gradient estimators, not part of J. Critic bootstrap/direct real actor credit are OFF.
The audit aligns MAPPO's cap/Euler execution and non-acting native background (`tools/setupaudit/measure.py:47`); native 90 s future arrays and 300 s J match exactly across actors.

## Decisive 90 s table (equal-start means within each setup)
J is accrued team road+source cost (lower better). D columns are pre-q10 K/time means of paired suffix returns (positive gain); C_G=D_G−Σmember D_i.
Dense credit = mean over suffix times of joint-K8-q10[1 D_all +9.9 D_G +8.9 C_G +8.9 ΣD_i], maximised; it need not equal a weighted sum of the pre-q10 columns.

| Setup / actor | J90 | Dense credit | D_all | D_G | C_G | ΣD_i |
|---|---:|---:|---:|---:|---:|---:|
| OFF n=1 / CTRL0 | 15828.35 | -26.26 | 23.14 | 0.21 | -0.01 | 0.21 |
| OFF n=1 / MAPPO | 15905.68 | -173.69 | -37.63 | -0.87 | 0.17 | -1.04 |
| OFF n=1 / A190 | 15828.35 | -26.26 | 23.14 | 0.21 | -0.01 | 0.21 |
| ON n=2 / CTRL0 | 16383.24 | 88.25 | 28.98 | 17.44 | -10.92 | 28.36 |
| ON n=2 / MAPPO | 16391.54 | 64.19 | 22.05 | 16.35 | -14.62 | 30.98 |
| ON n=2 / A190 | 16383.24 | 88.25 | 28.98 | 17.44 | -10.92 | 28.36 |

## 300 s continuation (pre-gate accrued D terms, not suffix averages)
| Setup / actor | J300 | D_all | D_G | C_G | ΣD_i | Long credit | ON / eligible rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| OFF / CTRL0 | 61289.84 | 196.32 | 88.36 | -209.61 | 297.97 | 0 | 0/2 |
| OFF / MAPPO | 66614.69 | -5128.53 | 197.13 | 164.77 | 32.36 | 0 | 0/2 |
| OFF / A190 | 61289.84 | 196.32 | 88.36 | -209.61 | 297.97 | 0 | 0/2 |
| ON / CTRL0 | 63733.72 | -82.55 | 177.55 | 430.49 | -252.94 | 0 | 0/4 |
| ON / MAPPO | 67011.87 | -3360.69 | 65.33 | 78.20 | -12.87 | 0 | 0/4 |
| ON / A190 | 63733.72 | -82.55 | 177.55 | 430.49 | -252.94 | 0 | 0/4 |

Gate is against **learned-native, not CTRL0**: median gain >87.82 AND q10 gain >0 (`src/wms/control/gated300.py:198`). Harm and uncertain/below-margin gains get zero long credit.
ON CTRL0/A start 0 all/group median:q10 = 400.71:−54.13 / 298.08:−70.43; start 1 = 51.61:−3499.85 / 160.55:−1052.00.
ON MAPPO start 0 = −4007.56:−5441.91 / 53.54:−105.78; start 1 = −2506.22:−5019.88 / 366.32:−1052.45. Thus even positive mean/group gains can be rejected; no reversed sign was found.
OFF CTRL0/A all/group median:q10 = 150.11:−646.03 / 67.01:−192.01; MAPPO = −4910.09:−6743.42 / 39.70:−106.72.

## Confirmed setup errors and default-OFF repairs
1. **Mix qualification was not mix execution.** Base 9613 had no entrant sampler connection in `src/wms/control/dreamer.py:1308` / `src/wms/model/wms.py:2553`; qualification used ERRSRC LearnedMix.
   Added `trainer.entrant_learned_mix=false` by default, the same observable-only sampler, per-birth invocation, CRN keys, persistent settings and replay-state restoration. CPU probes confirm actual assignments; no driver truth is used.
2. **Qualification is for a different target.** `tools/wmcombined/e3_score.py:139` selects t0 HDVs, road delay only; CCT-2 provenance is K1. Training uses all present/entrant vehicles + source waiting and K8 joint-q10.
   First ON start J90 includes **13,165.31 veh-s source wait (82.4%)**; this is absent from that qualification metric. `src/wms/control/gated300.py:48` adds default-OFF `require_objective_match`; opt-in refuses long credit until matching population/cost/reference/K/statistic is qualified.
3. **Routine rescore/gate window is 10 s.** `src/wms/control/dreamer.py:917` applies the 90 s curriculum only in training; `src/wms/control/gated300.py:300` retains 10 s unless `score_training_horizon` is enabled.
   Opt-in `configs/c4_policy14_candA_setupaudit.toml` connects mix, enables matched-window scoring, and refuses unmatched long qualification (current authority gives H_long=0 pending requalification). Existing launch configs stay unchanged; this preset was not trained.
4. **Naive MAPPO bridge substitution changes the comparator.** `src/wms/control/bridge.py:313` uses a different absolute-action cap/integrator/native background from residual mode. Audit adapter plus CPU bitwise-native test fixes the comparison; all native references agree.
   A 2 s CPU control showed 47.9% spurious positive MAPPO−CTRL0 deltas with the naive bridge versus 0% after alignment. Full tables use the aligned path.

## Sign, units, inputs and expressivity checks
- `src/wms/control/objective14.py:20` replaces the legacy per-vehicle reward with −(road delay+source wait), **veh-s without a vehicle divisor**; births and queued requests are included. Positive D is improvement; actor loss negates credit (`src/wms/control/dreamer.py:1593`).
- Row coefficients are all=1, group=18.8, member solos=0; C_G subtraction is cancelled by w_ind=8.9. No flip/negative row survives (`src/wms/control/counterfactual.py:371`). Anchor OFF, TTC report-only, residual entropy zero; no hidden opposing penalty.
- All 32 initial CAVs have zero positive and nonzero braking authority under ±1.2. A190 requests +0.0032..+0.0078 m/s² there; full executed delta is **exactly zero** over 167,727 ON and 69,134 OFF CAV physics ticks.
- MAPPO raw absolute range is [−3,+2.6], usually near +2.599; shared physical execution removes the apparent extra positive authority. ON executed−CTRL0: mean −0.0748/−0.0587, p10≈−0.084, median/p90/p99=0, p1=−1.17/−1.32, min −4.004 m/s² (per-start summaries).
- **98.88%** of ON MAPPO longitudinal commands fit the residual execution envelope; the rest need stronger braking. MAPPO emits **786 bridge lane-change requests vs CTRL0/A's 33**; A has no learned lane head. Neither actor switches native/damping/PI controller IDs.
- Initial observations are live: 14/18 CAVs, 50/74 reports, nonzero own speed/acceleration/gaps, group |signal| means .151/.143. Report age/command/intent are deliberately zero for residuals (`src/wms/control/bridge.py:203`); b/c margins are not explicit own inputs.
- Short CPU noisy probe: CTRL0 dense credit −.3548, A −.3375, while deterministic executions coincide; additive exploration can reward undoing exploratory braking. This 2 s probe is not a 90/300 s headroom estimate.
- **Exact-start CTRL0 WM↔SUMO equality is unmeasured:** supplied starts contain native/generic_random tapes, no paired CTRL0 tape; no SUMO was run. Shared-law identity and matched native rows do not establish real-SUMO outcome parity or real-MAPPO mechanism attribution.

## Implication and verification
Owner-supplied real E3FINAL_A has ours−MAPPO +683.5 [256.9,1122.4] at 600 s. The tested late training starts/WM objective do not reproduce that headroom; correcting the missing mix does not restore the ranking.
**(b): fix WM/objective/start-support visibility and matching qualification first.** These measurements do not support treating warm-start/distillation alone as the remedy; the MAPPO lane channel is also outside A's action class.
19 distinct CPU cases passed (birth/CRN/replay gradients/default-OFF parity/signs/objective contracts/native comparator), Ruff and bash syntax passed. Repairs are opt-in; authority generation binds the new adapter and credit selector.
gpuR was idle before each scoring launch; fraction=.35. Owned GPU PIDs 427168/428305/428899/429791/430786 only; completed rows retained across owned restarts. All scorers finished; no compute process remained. No training, SUMO or push.
Full hashes, per-start/K values, gates, distributions, source totals, commands/logs: `tools/setupaudit/evidence/{actual_exact,mix_exact}.json`, `tools/setupaudit/evidence/README.md`, `tools/setupaudit/TRACE.md`; aggregate with `python tools/setupaudit/summarize.py tools/setupaudit/evidence/mix_exact.json --aggregate-only`.
