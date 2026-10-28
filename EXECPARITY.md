# S-EXECPARITY — recorded-state verification and opt-in execution parity

Branch `c4/execparity`, based on deployment audit `f6531d94`. CPU only; **zero SUMO
executions and zero new sim-s**. Read-only exact-file copies from cpuP, no scenario,
route, assignment or restricted-split data. Input SHA-256s are committed in
`tools/execparity/verification.json`; tapes/checkpoints remain untracked.

## 1. Verification: BLOCKER-1 generalises across this ten-world audit panel

Worlds **977000, 977002, 977004, 977006, 977008, 977010, 977012, 977014,
977016, 977018**. Three recorded arms per world, 300 s warmup + 600 s intervention.
C1 `64f70af4`, coop-off `e1fa4881`, MAPPO `aecd7d70`, all SHA-asserted. C1 and
coop-off decisions replayed on all three arms' observable states. MAPPO statistics
use its recorded requests and observed outcomes, preserving its original command
memory/lane history rather than synthesising a fresh executor state.

Estimates below weight worlds equally; brackets are percentile 95% CIs from 5,000
paired-world bootstrap draws, seed 20261028. They are **not independent-tick CIs**.
Stable CAV identity required through the next two physics steps, post-warmup only.

| Metric | C1 recorded arm | Coop-off replay on C1 states | MAPPO recorded arm |
|---|---:|---:|---:|
| CAV control rows | 64,850 | same 64,850 states | 65,642 |
| Residual positive | 100% [100,100] | 100% [100,100] | n/a (absolute actor) |
| Requested absolute command positive | 90.34% [89.14,91.39] | no observed off command | 93.10% [91.66,94.58] |
| Realised within 0.01 m/s² of request | 1.28% [1.12,1.41] | no observed off outcome | 5.57% [4.29,6.73] |
| Realised below request by >0.01 m/s² | 98.72% [98.59,98.88] | cap-implied, see below | 94.28% [93.08,95.59] |
| Realised ≤ reconstructed follow baseline +0.001 | 100% [100,100] | 99.987% below off request by >0.01 [99.962,100] | baseline not identifiable from absolute request |
| Mean request, m/s² | +0.314 [0.285,0.339] | unobserved | +2.013 [1.956,2.067] |
| Mean realised first-substep acceleration, m/s² | +0.171 [0.144,0.193] | unobserved | +0.164 [0.144,0.181] |
| Realised − reconstructed follow baseline, m/s² | −0.0841 [−0.0951,−0.0720] | unobserved | unidentifiable from request alone |
| Realised − learned native head, m/s² | +0.0303 [0.0243,0.0347] | unobserved | +0.0052 [−0.0005,0.0116] |
| Realised above +1.2 m/s² | 0% [0,0] | unobserved | 0% [0,0] |

The **follow** baseline is reconstructed as `request − replayed residual` ONLY
when C1's logged request is strictly inside its [-3,1.2] composition envelope.
Bound-clipped requests cannot be inverted; their counts remain in the JSON.
This is not SUMO's true-native next acceleration. In particular it does not
include native lane-change yielding. Coop-off has **no E3-final tape in this
panel**: its residual distribution is measured, its capping is a counterfactual
implication, and its actual acceleration is not fabricated. These data confirm
the inert-positive-residual mechanism across ten worlds; they do not causally
allocate E3's −1,094 veh-s benefit between executor switching and learned control.

### What changes even while the residual is capped?

* Native ownership releases longitudinal TraCI influence. C1's nonzero residual
  retains it, leaving SUMO's safe-speed following ceiling in charge. Native lane
  decisions remain enabled, but native lane-change **speed** adjustments/yielding
  do not follow the same path after the external speed command. A recorded native
  acceleration and a recorded C1 acceleration occur at diverged states: their
  difference is not a same-state yielding counterfactual.
* The legacy composer also adds `getMinGap` to `getLeader`'s bumper gap before
  `getFollowSpeed`; IDM adds minGap internally. This overstates the prospective
  composing baseline relative to the actual safety cap. The new opt-in path
  corrects the double addition.
* Dual writers are real in source: `CommandOwnership.slowDown` once per 1 s tick,
  then `setSpeed(-1)` / `setAcceleration` twice per tick. In the observable census,
  **98.73% [98.59,98.88]** of C1 outcomes differ from the issued ledger by >0.01;
  only **0.016% [0.003,0.033]** of issued commands differ from requests. MAPPO's
  corresponding figures are **94.30% [93.11,95.60]** and **23.28% [19.01,28.17]**.
  No stable, in-frame C1 CAV was logged native with a finite request in this panel;
  the fallback-overwrite defect is a source-level risk, not observed incidence.
* First-substep lane-change fractions are C1 **0.633% [0.528,0.736]**, native
  **0.549% [0.462,0.623]**, MAPPO **0.792% [0.684,0.912]**. Lane choice is not
  frozen by the capped residual. These counts cannot isolate lost cooperative
  yielding from the tapes' missing lane-change speed-patch state.
* The native arm's realised − learned-head acceleration is +0.0127
  [0.0068,0.0181] m/s². That head is an approximate reference, not an executor.

Sources: `verification.json` (full per-world distributions and uncertainties),
`ledger_results.json` (metadata, longitudinal ledgers and lane census). The two
tables use slightly different stable-identity requirements: the ledger needs one
following substep, the actor probe needs two; denominators are explicit in JSON.

## Reproduce verification

Use a CPU torch environment, `PYTHONUTF8=1`, `PYTHONPATH=src;.`, deterministic
`C4_POLICY_STOCHASTIC=0 C4_POLICY_ACCEL_NOISE=0`:

```text
python tools/execparity/fetch_recordings.py --out _execparity_data
python tools/execparity/replay.py --data _execparity_data --inputs <audit>/cpuP/e3final/inputs --off <audit>/cpuP/e3long/inputs/coopoff_cycle1/policy_latest.pt --out tools/execparity/verification.json
python tools/execparity/ledger.py --data _execparity_data --out tools/execparity/ledger_results.json
```

Execution-parity implementation, CPU replay gate, runtime ports and the proposed
registered control design are described in the next committed steps.
