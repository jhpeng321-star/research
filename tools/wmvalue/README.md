# Recorded terminal value experiment

**FINAL: FAIL under MAIN21:12.** All 320 K8 futures / 200 arm cells / 400 pairs completed;
19 existing long trajectories were reused and 301 new 90s prefixes supplied B/C.
Neither short method clears both D/E on sign and MAE; full A is waived by the
registered stopping rule. `docs/paper_results/WMVALUE.md` is the final 38-line table
and recommendation. V/D/E remain frozen; all owned dispatchers/workers finished.
`evidence/short_test.json` is the once-only result; `evidence/final_verification.json`
checks every task hash, all 200 cells, report, frozen artifacts and one verdict line.
Use `python -B -m tools.wmvalue.verify_short_final --out NEW.json` to verify
the artifacts without repeating statistics.

Frozen science: `docs/paper_results/WMVALUE_DESIGN.md`. No simulator dependencies.
Initial status/receipts describe the pre-amendment telemetry gate. MAIN18:12
authorizes audited causal reconstruction; the current modules implement it.
`core.py` supplies the fixed value learner, command-only direct baseline, FIT arm
means, K8 composition, all-pair assembly and exact world-bootstrap decision rule.
`runtime.py`/`restart.py` implement the audited frozen-WM continuation adapter.
Late-start sensing/history/controller/entry/cost/CRN CPU gates passed before test.

## Amended observation adapter

An authorized adapter must supply, per frame, `time_s`, `kin [N,4]` (s, lane, v, a),
`present`, `sensor_mask`, `known_cav [N]`, `coverage [3,5]`,
and `flow_counts [2,2]` (cumulative/trailing60 boundary counts per entrance).
Source queue/request feature keys are rejected. `reconstruct.py` ports the existing
100m same-lane nearest-neighbour and mainline/ramp RSU projection with nominal 5m
length. No hidden length is loaded. Unknown-junction lane=-1 contributes only to
the known self count. Neighbour acceleration outside RSU is not observed.
`observe_prefix` physically slices before access; `observations` yields each
state during a single forward pass. Future poisoning, truncation and existing
CAV/RSU helper parity are covered by `test_reconstruct.py`.

Supervision is the real cumulative fleet+source objective from control t300,
rebuilt on the full half-second grid through t900, reconciled with each
`objective_120/300/600.json`. Label-only independent request RNG is never a feature.
Union-cohort births, carried exit speeds and post-step source clocks match the
observer. Maximum allowed float32 speed integral discrepancy is .01veh-s;
source discrepancies must be <=1e-6. Global outcomes are LABELS, not inputs.
Per-row FIT metadata: seed/arm/role, state, remaining_J, and for D action_summary.
No metadata is concatenated to learned inputs. Equal world/arm weights are used.
`Value(actions=False)` is the terminal V; `Value(actions=True)` is D.
The six-number action summary is present issued acceleration/target-speed
count/mean/std, computed by `action_summary` from known CAV commands only.

## Required frozen WM adapter

Must bind exact WM2 b8797242/config/mix/arm exports and entry/exit/cost semantics,
with K=8 identity-keyed CRNs and correct continuation state/history at every t0.
Do not reuse WMCALIB t300-only predictions for late starts, substitute real
terminal states for imagined ones, or use full-horizon action statistics in D.
`assemble_predictions` takes eight ordered future records per arm, each with
start_time, end_time=900, full J, and windows['60'/'90']={J,state}. It computes
mean(J_H+V(s_H)) before arm differencing. The adapter must preserve native and
controller clock/binding parity. Project the imagined state through `project`,
with prefix boundary counts, before V; no real terminal state substitution.

`evaluate` requires exactly 200 test arm cells (8 worlds x5 times x5 arms),
including real remaining J and predictions A..E. It constructs 400 arm pairs,
resamples whole worlds with common indices for every method, and returns all six
paired contrasts for each B/C. Positive sign-gain AND MAE-reduction lower bounds
against A,D,E are all required. Missing rows are errors, not silently dropped.

## Reproduction (existing environment)

`python -B -m unittest tools.wmvalue.test_cpu tools.wmvalue.test_reconstruct`

Schema audit (read-only inputs; exclusive-create output):
`python -B tools/wmvalue/audit.py --root /root/wmdata2 --corpus wmdata2 --out /root/wmdata2/NEW_RECEIPT.json`
Use `/home/ubuntu/wmdata20261031`, `--corpus wmdata` on cpuQ. Only FIT recordings
are inspected. No recursive file scan or truth sibling is opened.
`report.py` and `status.json` are historical pre-amendment receipts, preserved.
`prepare_causal.py --root ROOT --corpus wmdata[2] --out NEW.npz` exports only FIT
features/labels and a hash-bound JSON audit; no fitting. `--limit 1` is an explicit
mechanical canary, never accepted as a complete training corpus. Existing outputs
are never overwritten. `a_coverage.py` reports missing exact restart/arm/future keys.

`fit_causal.py --wmdata ARRAYS RECEIPT --wmdata2 ARRAYS RECEIPT --out NEW.joblib
--receipt NEW.json` verifies complete FIT grids, seed roles, exporter/code/array
hashes and terminal constraints before learning. Produces `Value` wrappers V/D
and the FIT-only E dictionary. Parameters/weights are frozen; no reserved scoring.
See `evidence/frozen_fit.json` for the model location/hash and exact inputs.
Run `python -B -m unittest tools.wmvalue.test_fit_causal` for export split guards.
`python -B -m tools.wmvalue.verify_receipts` checks the existing model's hash and
serialization roundtrip, frozen sources, both complete exports, and report length
without fitting or scoring reserved worlds. These are historical pre-inference
receipts; use `verify_short_final` for the final closed-test artifacts.

## MAIN21:12 execution

The completed replacement execution used `short_dispatch` (launched through
`launch --short --original ... --prefix-gate ...`). It first harvests the
original long dispatch without repeating live tasks, then executes only missing
90s prefixes using unchanged full-long stochastic plans. `finalize_short`
evaluates the complete B/C/D/E necessary condition once; a failed necessary
condition is the registered overall FAIL, otherwise A must be completed.
See `evidence/resume_2112.md` and `evidence/short_execution.md` for ownership,
CPU parity tests and the immutable snapshot. Preserve both dispatch roots.
