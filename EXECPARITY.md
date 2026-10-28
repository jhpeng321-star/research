# S-EXECPARITY — recorded-state verification and opt-in execution parity

**Status: production / approved SUMO check BLOCKED.** Combined-law CPU reference
and27tests PASS; stock SUMO lacks the required same-step native residual hook.
The old cap-only recorded acceptance FAILS for MAPPO. New-law shared-cap proxy
results are not real parity. No new SUMO allocation was made. See sections3–6
and `docs/paper_results/EXECPARITY_SUMO.md` before attempting candidate-A launch.

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

## 2. Learned executor identification (frozen before its E3 replay)

The initial WM3 per-slot-settings prototype failed: C1 RMSE 0.965, MAPPO 0.834,
native-head RMSE 0.199 m/s² (`parity_wm3_initial.json`). Its initial physical/net-gap
conversion was also corrected: SUMO's internal/getLeader gap excludes minGap,
whereas the WM frame supplies physical bumper-to-bumper distance. The latter must
not receive minGap again. This failed prototype is retained, not called parity.

The follow-up is **TRAIN-only system identification**, not E3 calibration: fit
five positive IDM parameters to recorded `genrand_plain` direct commands and
their next observed speed differences, using `min(issued_command, follow_cap)` as
the observation law. This separates safe following from the native lane-change
speed patches. No true driver settings or assignment files are used.

* Existing native-head world partition reused: SHA1(world) modulo 10 <7 for fit,
  otherwise score; only manifest `split_role=val` within the TRAIN corpus.
* 120,760 direct CAV rows fit, 51,740 score, fixed every-tenth-second sampling;
  whole-world split, source array and manifest hashes in the artifact.
* LBFGS, unweighted acceleration MSE, deterministic CPU; no E3/E5 or 964000–964199
  seeds admitted. Initial optimiser scalars are unit values, speed start from the
  observed maximum; all returned driver settings are learned.
* Frozen artifact: `configs/execparity_cav_idm_v1.json`; fit RMSE **0.2287**, score
  RMSE **0.2323** m/s². This is not yet its E3 execution-parity acceptance result.
* Rejected development fits are retained: `rejected_fit_v1.json` (uint8 extraction
  indexing error), `rejected_fit_v2.json` (gap conversion), `native_fit.json`
  (wrong identification target: native yielding contaminates following). None is
  admitted by the runtime artifact loader or was tested to select on E3 outcomes.

Implementation, final CPU replay gate, runtime ports and the registered control
design follow in subsequent commits. Driver artifact and fitting method are
committed **before** their first E3 replay; no evaluation-based parameter tuning.

### Recovery checkpoint: cap-mirroring prototype (superseded by combined decision)

The recovered implementation and JSON receipts are preserved as an engineering
checkpoint. `parity_results.json` tests the **old switch-path** recordings, not the
owner's later continuous law: world-mean RMSE C1 0.1530 [0.1177,0.1902], MAPPO
0.1568 [0.1189,0.1991], native head 0.2090 [0.1852,0.2309] m/s². The runtime
canary replays 600 MAPPO ticks/6,664 CAV rows with bit-identical commands/lanes
and AST-identical frozen BranchController/BranchTracker ports. It is not a
closed-loop trajectory canary. Earlier session CPU receipts report 16 execution
unit tests passing. No task-owned local worker or remote task dispatch receipt
was found on recovery. These results must not be relabelled continuous-law proof.
The cap-only **pooled** acceptance gate actually FAILS: C1 RMSE 0.170708,
MAPPO 0.200415 versus native-head 0.198964 m/s². Passing world-mean comparisons
cannot replace this registered pooled criterion. No parameters were retuned.

## 3. Combined law: CPU implementation PASS, stock-SUMO backend BLOCKED

`C4_EXECUTION_PARITY=1` now selects the **combined**, native-relative law. The
old cap-mirroring algebra remains only as `executed_switch_residual` for historical
replay. New checkpoints bind `execparity_contract/2`; old cap-only receipts cannot
authorize combined-law deployment.

The same-step definition is `a_exec = max(-v/dt, min(a_safe, 1.2,
max(min(-3, a_native), a_native + residual)))`, with `dt=0.5 s`. At exactly zero
there is no deployment write and native is returned bit-exactly. The lower
envelope preserves native emergency braking because speedMode27's comfortable
deceleration bit is OFF. The WM head is projected onto the same feasible envelope
for **both** native and policy rows, so head infeasibility cannot create a
zero-residual switch. Projection error is reported separately. No acting-path
soft `safe_accel` or second `apply_exec_cap` is credited. Imagination uses SUMO's
semi-implicit Euler update and the frozen TRAIN-learned IDM safe-follow cap.

**Critical timing finding:** stock SUMO 1.27.1's `getSpeedWithoutTraCI()` is not a
prospective native-speed oracle. `Vehicle::getSpeedWithoutTraCI` calls
`MSVehicle::getSpeedWithoutTraciInfluence`, which reads the influencer's cached
`originalSpeed`. That cache is assigned by `processTraCISpeedControl(vSafe,vNext)`
*inside executeMove*, just before speed advice is applied. A pre-step query reads
the completed step. A step listener runs too late. Neither the direct cached
speed nor an extrapolation guarantees the upcoming LC-yield speed. Example:
current/cache=10 m/s, upcoming native yield=9.6 m/s. Zero releases to −0.8 m/s²;
cached-base +epsilon requests approximately 0: the discontinuity remains 0.8.
Source links and the deterministic counterexample are in `backend_timing.json`.

Consequently `continuous.write_residuals` requires an explicit **same-step backend
extension**, `native_relative_same_step/1`, whose setter applies the residual
after native CF/LC speed computation and before mode27 safety. These extension
methods are deliberately documented as **absent from stock TraCI**. The sole
CommandOwnership writer sends the residual to that boundary; it never falls back
to stale `getSpeedWithoutTraCI`, `slowDown`, or absolute `setAcceleration`.
Nonzero deployment on the frozen binary fails before a longitudinal write.
This is a built CPU contract/fixture, **not a completed SUMO backend**.

CPU checks: 27 execution tests PASS, including 10,000 independent scalar/tensor
cases, useful positive/negative gradients below the cap, zero gradients when
capped, bit-exact zero/native trajectories with LC yielding, signed-zero tensor
identity, tiny-residual continuity, sole-writer/fallback, and stock-API refusal.
Broader selected nativeinit/execcap/deploy/branching regressions: 81 PASS; the
remaining CIRCLES test initially lacked `onnxruntime` in the test venv and PASSed
separately under the existing Miniconda environment. Eight simulator/dispatch
tests were deliberately excluded. No SUMO was invoked.
The completed-step execution logger additionally records requested residual,
same-step no-TraCI speed (read after stepping), issued acceleration before safety,
actual speed-difference acceleration and accepted/native ownership. Its own
regression PASSes; it does not mistake residuals for absolute issued acceleration.

## 4. New-law recorded-state sensitivity and authority guard

`continuous_results.json` uses the ten **native** recordings, every tenth
post-warmup decision and all stable CAVs: 6,998 rows. The actor sees only the
pre-command state; the observed native next acceleration is an offline target,
never an input. The target scalar backend and differentiable WM law share the
same TRAIN-inferred follow cap. Thus this checks head-replacement sensitivity
and algebra, **not the unobserved true SUMO cap or a new-law SUMO trajectory**.

* Prediction minus native-based executable **proxy**: bias −0.00331, RMSE
  **0.170964**, MAE 0.056481 m/s²; p05/median/p95 −0.13531/0/+0.13070,
  min/max −1.55054/+2.84717.
* Raw native-head error on the same rows: RMSE **0.198964**, MAE 0.090911.
  Equal-world mean proxy RMSE 0.18316 [0.15948,0.20628], versus head
  0.20896 [0.18524,0.23094]; fixed 5,000-draw world bootstrap, seed20261028.
* Feasibility projection is material: WM head projection RMSE0.14277; projection
  of recorded native acceleration onto the inferred envelope RMSE0.15099, worst
  −6.01172 m/s². This exposes inferred-cap/geometry error. It must not be hidden
  by the shared-cap proxy comparison, which can cancel common model error.
* `c1_continuous_authority_full.json` evaluates **all** post-warmup control ticks
  in those ten tapes: 70,228 acting CAV rows, **25.5269%** fully clipped/inert in
  the WM-based counterfactual law; 74.4731% retain some longitudinal effect.
  Another20.5801% are partially clipped: **46.1069% have any clipping**, which is
  the stricter fraction used for the95% admission rule. This is a model-based
  authority admission PASS, not measured SUMO authority.

The `>95%` rule is unchanged. The combined guard counts **any envelope clipping
(partial or full), plus exactly-zero/native requests**, after composing with the
feasible native base, rather than rejecting positive residuals by sign. It
reports partial and fully clipped fractions separately and computes the clip
indicator from the envelope bounds without an arbitrary numeric tolerance.
Positive residuals can now use available headroom; negative residuals may
themselves be clipped. Receipt schema2 binds checkpoint,
recordings, driver SHA and execution contract. Exactly95% is admitted; >95%
refused. `verify_parity` independently refuses old cap-path receipts and new-law
proxy reports, even if someone changes their `passed` boolean. It requires a
real combined-law measurement and zero/native identity. Installed stock-TraCI
introspection in `backend_preflight.json` confirms the required extension methods
are absent without opening a simulator connection or spawning SUMO.

Reproduce (preserve the existing SUMO-tools PYTHONPATH when adding `src;.`):

```text
python tools/execparity/continuous_replay.py --data _execparity_data --policy <audit>/cpuP/e3final/inputs/policy_latest.pt --driver configs/execparity_cav_idm_v1.json --out tools/execparity/continuous_results.json --authority tools/execparity/c1_continuous_authority.json
python tools/execparity/guard.py --policy <candidate.pt> --driver configs/execparity_cav_idm_v1.json --episode <recorded episode.npz> [--episode ...] --out <execution_authority.json>
python tools/execparity/preflight.py --out <backend_preflight.json>
```

The last command intentionally exits2 for stock TraCI. No report in this branch
currently authorizes combined-law SUMO deployment.

## 5. Candidate-A enable contract and integration boundaries

**Not production-ready:** there is no valid current command that enables the
owner's combined law on the frozen stock-SUMO binary. Do not pass the CPU proxy
as the real parity receipt. The following is the exact flag/CLI wiring for
integration and CPU qualification; a real same-step backend and the registered
real-SUMO receipt remain prerequisites for deployment.

In the candidate-A process environment (training **and** deployment):

```powershell
$env:C4_EXECUTION_PARITY = '1'
$env:C4_EXECUTION_DRIVER = '<absolute path>/configs/execparity_cav_idm_v1.json'
$env:C4_POLICY_STOCHASTIC = '0'
$env:C4_POLICY_ACCEL_NOISE = '0'
```

Default is `C4_EXECUTION_PARITY=0`; other values fail. The driver artifact SHA is
`f70c2aaf34c21a801f7d970adad237619bebc954f6acf6dd51407d06aadd60b2`.
Retain candidate-A's registered WM/config, `cav_exec_cap="idm"`, native head,
native lanes,1s decision/0.5s physics, data/credit/selection settings and explicit
resume. `policy14_cli` needs no new CLI switch: the environment selects execution
inside its existing `--native-init` path. Its command shape is:

```text
python -m wms.control.policy14_cli --config <resolved-A/POLICY.toml> --wm-ckpt <adopted-WM.pt> --wm-config <matching-WM.toml> --real-returns <registered-real-returns> --resume <original-C1.pt> --native-init --device <registered-device> --seed <registered-training-seed> --max-updates 150 --output-dir <fresh-phase-dir> <existing registered boundary/credit/qualification arguments>
```

This is enable syntax, not a training launch or a new scientific registration.
An original legacy C1 can migrate into the new phase; saved residual checkpoints
carry execution-contract schema2, and loading them with a different flag/driver
fails. Do not resume cap-only schema1 checkpoints as combined-law checkpoints.
Inspect effective settings after C1 resume; the independent S-AUDITFIX trainer
config/horizon fixes remain separate merge prerequisites.

For eventual deployment, retain the frozen per-world arguments and add:

```text
python -m wms.control.deploy --controller policy --policy <candidate.pt> --population textbook --execution-authority <candidate-execution_authority.json> --execution-parity-receipt <real-combined-execution_parity.json> <frozen seed/demand/penetration/warmup/horizon/out arguments>
```

The usual checkpoint-bound native acceptance receipt is still required by
`load_policy_fn`. Offline authority admission must be generated for the exact
candidate, not reused from C1. Collection/loop callers pass `execution_authority`
and `execution_parity_receipt` in their policy record or use
`C4_EXECUTION_AUTHORITY` / `C4_EXECUTION_PARITY_RECEIPT`; direct deploy uses the
explicit CLI paths. New SUMO verification cannot bootstrap itself using a fake
production PASS: its eventual approved canary wrapper must remain isolated from
production gates and bind the real backend/source/inputs before its first charge.

The frozen E3 runtime ports are implemented on this branch for integration into
`c4/c4prep`; no peer worktree or live runtime was modified. Merge touchpoints:
`bridge.py` execution block (keep POLSPEED batching/shape semantics),
`WmSModel._exec_cap`, checkpoint contract load/save hooks (retain AUDITFIX config
restoration), and env/ownership/recorder (retain LOOPPATCH callbacks and ledgers).
No objective, critic, long-credit/gated300 rule or driver artifact was retuned.
Native/MAPPO deployment use the flag-OFF frozen path; retain the existing CPU
canary receipt and require actual trajectory canaries before adopting new code.

### Backup1 is an additional, explicit integration blocker

`c4/mappores` currently adds a cooperation residual to a **frozen absolute MAPPO
command**, not to the native-head base. Its adapter/checkpoint format is not in
this worktree, and `C4_EXECUTION_PARITY=1` alone does **not** port it. There is no
honest ready-to-run backup1 enable command yet. Keep its actor and group encoder
frozen and preserve its lane path; integrate its total desired longitudinal
command at the same-step native-relative backend boundary, mirror that exact
conversion/cap in its WM controller, then regenerate source-bound zero-residual
MAPPO identity and full-actor CPU/GPU qualification receipts. Reuse the environment,
driver and real-parity/authority gates above only after that adapter is implemented.
Its old build7/90s launch is not combined-law qualification. See peer
`E:/agent-work/c4-mappores/MAPPORES.md`; this task neither changes nor dispatches it.

## 6. Zero-learning control: reuse completed S-CTRLARM, do not duplicate

On the **legacy switch path**, residual exactly0 releases to native, so it is not
the same-executor control. A valid executor-only arm must force residual0 through
the *original* composition/ownership/setAcceleration path, including its original
base-law quirks, bounds, native lanes and mode27. Then C1−CTRL0 isolates learned
residuals from the switch, and CTRL0−native measures the execution change. Merely
using the corrected new cap law would introduce another treatment.

The smallest useful exploratory design would reuse stored native/C1/MAPPO for
three prespecified worlds977000–977002 and add only those three CTRL0 runs:
2,700sim-s, or3,600 including a900s native runtime canary. A one-world900s control
is mechanically informative but cannot support world-level uncertainty. All use
300s warmup+600s intervention and paired equal-world contrasts.

**This work is already finished by S-CTRLARM:** owner22:58 amended panel
977000–977022 plus native977000 canary,24allocations,21,600charged/actual sim-s.
Do not rerun the proposed subset or charge this task for that peer work. Its
scoped CTRL0 forces the exact frozen C1 residual to0 without native release;
543scientific source files remained byte-identical to `b3c031ad`.

Peer report `E:/agent-work/c4-ctrlarm/CTRLARM.md` and published
`E:/research-dense/docs/paper_results/CTRLARM.md`:600s ours−CTRL0 **−45.71
[−464.12,+342.63] veh-s**, CTRL0−native **−1,956.53 [−4,086.66,+144.50]**,
MAPPO−CTRL0 **−623.95 [−2,044.25,+664.51]**;23equal-weight worlds,2,000paired
world-bootstrap draws,seed0. All intervals include0: no incremental learned
benefit or equivalence is established on that reused subset. Only2/23complete
metric payloads were identical; avoid the stronger claim that C1 is universally
inert. These subset numbers do not replace the full E3 −1,093.9veh-s estimate.

On a genuine **continuous** path, residual0 is native by construction and unit
fixture: no separate execution-only arm is needed beyond an actual zero/native
identity canary. The owner23:50 four-job scientific design costs3,600sim-s with
another3,600reserved for at most four infrastructure retries. Worlds977001–003
can compare continuous C1 against their stored native references;977000 is the
byte-identity zero canary. Full design, matching requirements and current
**0allocations/0charged/0actual** accounting are in
`docs/paper_results/EXECPARITY_SUMO.md`. No new dispatch is armed. The missing
same-step backend must be resolved; the conditional SUMO check is still blocked.

### Which option actually meets the requirement?

* **Imagination mirrors the legacy cap:** exact clipping algebra is possible
  given the true same-state cap, but positive residuals above that cap remain
  meaningless and the zero/nonzero native-ownership switch remains. Our inferred
  cap is not exact, and the retained MAPPO pooled acceptance fails. This option
  is neither the owner's chosen production law nor a demonstrated parity PASS.
* **Continuous relative to the same-step native speed, plus mirrored imagination:**
  algebraically preserves yielding, zero identity and useful residual gradients
  wherever safety leaves headroom. Scalar/tensor and deployment-writer fixtures
  establish that contract. Actual parity still includes WM head/cap error; the
  stock cached getter cannot implement the same-step contract. A native-residual
  hook inside SUMO, with its matching client interface, is the concrete missing
  implementation. Replacing it with a stale cache would not be a combined fix.

No option has yet demonstrated the requested real new-law parity on this branch.
