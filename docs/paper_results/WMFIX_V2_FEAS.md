# PILOT v2 smoothed-objective FD feasibility, 2026-10-31 14:35-15:05 +08

Two receipts. (A) FIT-12 preview below: FAIL, clearly bad. (B) the REGISTERED v2 binding
(combined 52-world TRAIN_PREPARED_V2 84751dc7, HARNESS_CONTRACT_V2 e67850f3) is
`v2/FEASIBILITY_V2c.json` (PID 189176; result pending at handoff - see WMFIX_V2_HANDOFF.md).
In (B) the source-column scale is 53.98 veh-s (S-WMDATA arms change source waiting), so
defect 1 below does NOT apply to the registered binding; defect 2 does.

## (A) FIT-12 preview

Brief: MAIN ADDENDUM 13:50 TRACK 1 — the `--smooth-events` (c4/wmfixc, merged edee986b)
FD check "must now pass"; if FD agreement is clearly bad, stop and report. On the FIT-12
doc it is clearly bad (below). No training, no panel, no SUMO from this receipt.

## Run (gpuA cuda, own PIDs 185187/185188, exited)
`tools.wmfixmerge.feasibility --smooth-events --skip-panel-gate --device cuda` on the
registered 12-FIT TRAIN_PREPARED.json (7bbd41f3; group 0 = w973002, arms native +
coopoff_c1, 60 s), WM2 b8797242, config c4_wm_loop14_r16_ft.toml, W5_COMBINED controllers,
HARNESS_CONTRACT f5548777. Receipt `/root/autodl-tmp/c4/wmfixmerge20261031/v2/FEASIBILITY_SMOOTH_FIT12.json`
sha256 `152e7a038e02b513ed559165dd3d291379f447fc718d779d4c6228df7945b61b` (log
v2/feas_smooth_fit12.log). Gradient columns (1,3,4,2) = SMOOTH_GRADIENT_COLUMNS, width 1.0 m.
(Run on the FIT-12 doc because the combined corpus was still in transfer; group 0 of the
combined corpus is the same FIT world, so the probe is the same rollout.)

| quantity | value |
|---|---|
| analytic grad (merge_response.net.2.weight, max elem) | 0.01258 |
| central FD eps 1e-3 / 1e-4 / 1e-5 | 5.7e17 / -2.1e18 / -5.3e13 |
| finite_difference_ok | **false** |
| loss before -> after one Adam step (lr 3e-4) | 0.0451 -> **1.3e14** |
| paired-J |contrast| before -> after (group 0) | 26.45 -> 29.38 |
| component grad norms J / source / crossing | 199.8 / 207.9 / 1.86 (finite) |
| frozen_bitwise | true |
| source_clock_match | false (expected: runtime.py changed by the wmfixc merge; contract must be rebuilt) |

## Diagnosis (two independent defects, both in the merged fallback, not in the data)
1. **Normalisation blow-up (decisive).** The source column's TRAIN paired-contrast scale is
   the float32 floor `1.19e-7` (TRAIN_PREPARED scales[2]): recorded paired source contrasts
   are identically 0 on the FIT worlds (shared Q0 and recorded demand). SMOOTH_GRADIENT_COLUMNS
   adds column 2 to the loss, so any nonzero surrogate source contrast is divided by 1.2e-7
   (~7e13x in the squared loss). At the unperturbed weights both rows' surrogate source are
   identical (79.58/79.58) so the loss looks sane (0.045); any perturbation (FD eps or one
   Adam step) breaks the tie -> loss 1e14 and FD 1e17. The S-WMFIX-C unit FD test used a
   fixture scale and could not see this.
2. **Surrogate level wrong by ~30x.** Surrogate source 79.6 veh-s vs exact ledger 2375.5
   (recorded 2223) on the same rollout: `Q_{t+1} = (Q_t + due_t)(1 - p_t)` drains the WHOLE
   entrance backlog in one 0.5 s tick when the gate opens, whereas the hard path admits at
   most one vehicle per entrance per tick. Its gradient is therefore not the sensitivity of
   the event chain it replaces, even after fixing (1).

## Consequence / what MAIN must decide
- On FIT-only data the smoothed objective is unusable (1e14 loss). On the registered 52-world
  doc the scale is sane; launch is automatic ONLY if FEASIBILITY_V2c is PASS with
  finite_difference_ok (v2/launch_pilot_v2.sh, pilot_v2.check_feasibility re-checks).
- Candidate repair (needs CPU tests + review; NOT done here): (a) capacity-limited surrogate
  `Q_{t+1} = Q_t + due_t - p_t * min(Q_t + due_t, 1)` per entrance (one insertion/tick), level-
  checked against the exact ledger on TRAIN rollouts; (b) source-column scale from a
  meaningful TRAIN statistic (e.g. the combined corpus scale if S-WMDATA contrasts are
  nonzero — see v2/PREP_SUMMARY.json — or the anchor/level RMS), never the eps floor;
  (c) or run v2 with data change only (smoothing OFF = v1 objective on 52 worlds).
- The combined corpus (change 1) is staged and prepared independently of this failure.

## Infrastructure found/fixed for the 52-world binding (all hash-recorded)
- FEASIBILITY_V2.json BLOCKED: WM closed-loop damping/PI arms need the frozen controller
  sources -> staged `_in/pi_deps` from speedhead e3final/runtime (69 files, 0 mismatches vs
  SOURCE_MANIFEST b3c031ad). FEASIBILITY_V2b BLOCKED: damping needs onnxruntime -> minimal
  `_in/ortdeps` symlinks to speedhead pythondeps (onnxruntime 1.20.1 only; no SUMO module).
- Runtime init on 236 episodes takes ~20 min under the S-WMCALIB GPU/CPU load (12 WMCALIB
  processes, ~37 GB, 100 % util on gpuA from ~14:39) - every panel subprocess pays it.
