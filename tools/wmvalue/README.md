# Recorded terminal value experiment

Frozen science: `docs/paper_results/WMVALUE_DESIGN.md`. No simulator dependencies.
Current status: **INCOMPLETE before fitting**, because both registered corpora lack
time-resolved source/sensor telemetry. This code deliberately cannot interpret a
global recording as an observed frame. `core.py` supplies the fixed value learner,
command-only direct baseline, FIT arm means, K8 bootstrap composition, all-pair
assembly, metrics and the exact paired world-bootstrap decision rule. It does not
pretend to be a runnable frozen-WM adapter or a raw-to-observable data exporter.

## Required observation adapter (unavailable in audited inputs)

An authorized adapter must supply, per frame, `time_s`, `kin [N,4]` (s, lane, v, a),
`present`, `sensor_mask`, `known_cav [N]`, `coverage [3,5]`, observed `queue [2]`,
and past-only `causal_counts [2,2]` (insertions/requests by entrance). Every actual
or imagined state uses the SAME sensor definition and longitudinal geometry.
The adapter's code hash, sensing range/detection logic, and queue/counter source
must be audited: an all-present mask or deterministic demand/seed reconstruction
is not accepted as evidence that these were the deployed observation channels.
Unobserved kinematics are ignored; no hidden type or true penetration is used.

Supervision is the real cumulative fleet+source objective from control t300,
recorded on the full half-second grid through t900, reconciled with
`objective_600.json`. Global outcomes are legitimate LABELS, not state features.
Per-row FIT metadata: seed/arm/role, state, remaining_J, and for D action_summary.
No metadata is concatenated to learned inputs. Equal world/arm weights are used.
`Value(actions=False)` is the terminal V; `Value(actions=True)` is D.
The six-number action summary is present issued acceleration/target-speed
count/mean/std, computed by `action_summary` from known CAV commands only.

## Required frozen WM adapter (not launched when data gate fails)

Must bind exact WM2 b8797242/config/mix/arm exports and entry/exit/cost semantics,
with K=8 identity-keyed CRNs and correct continuation state/history at every t0.
Do not reuse WMCALIB t300-only predictions for late starts, substitute real
terminal states for imagined ones, or use full-horizon action statistics in D.
`assemble_predictions` takes eight ordered future records per arm, each with
start_time, end_time=900, full J, and windows['60'/'90']={J,state}. It computes
mean(J_H+V(s_H)) before arm differencing. The adapter must preserve native and
controller clock/binding parity and carry predicted observable source queues.

`evaluate` requires exactly 200 test arm cells (8 worlds x5 times x5 arms),
including real remaining J and predictions A..E. It constructs 400 arm pairs,
resamples whole worlds with common indices for every method, and returns all six
paired contrasts for each B/C. Positive sign-gain AND MAE-reduction lower bounds
against A,D,E are all required. Missing rows are errors, not silently dropped.

## Reproduction (existing environment)

`python -B -m unittest tools.wmvalue.test_cpu`

Schema audit (read-only inputs; exclusive-create output):
`python -B tools/wmvalue/audit.py --root /root/wmdata2 --corpus wmdata2 --out /root/wmdata2/NEW_RECEIPT.json`
Use `/home/ubuntu/wmdata20261031`, `--corpus wmdata` on cpuQ. Only FIT recordings
are inspected. No recursive file scan or truth sibling is opened.
`python -B -m tools.wmvalue.report` summarizes committed receipts and writes an
exclusive-create JSON status. Existing receipts are preserved.
