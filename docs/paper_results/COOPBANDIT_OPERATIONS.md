# S-COOPBANDIT execution addendum — owner 23:49

The scientific design and 1,489,200 data / 99,360 planned live sim-s are unchanged.
The resumed session found no charged or running COOPBANDIT simulator attempt.
Prior staging directories and all existing implementation are preserved.

gpuA already has the exact frozen runtime manifest and SUMO 1.27.1 in its existing
speedhead dependency target; use that target, no package installation or new venv.
Run nice 19, single-threaded CPU inference on all admitted hosts. cpuQ performs the
screen/repeat gates; baseline receipts are copied unchanged to disjoint branch shards.
Every branch must match the full baseline prefix bitwise, including on another host;
a cross-host numerical mismatch stops that shard before intervention, without retry.

Before any branch dispatch, write one immutable global host/job allocation whose
unique attempts sum to exactly the registered data budget. Each host accepts only
its allocated ID, duration and phase. Local durable charges precede every reset;
aggregate host charges are the global ledger. An incomplete charged attempt blocks
resumption and is never automatically replayed. Existing completed results are reused.

Initial worker maxima are cpuQ 8, gpuA 8, gpuR 2, subject to measured throughput,
load ceilings 24/16/20 respectively, disk >=2 GiB, memory >=8 GiB and peer load.
gpuR data disk is marginal (~2.45 GiB); give it a small shard and retain the disk gate.
The cpuQ screen starts with one baseline and its exact repeat, then measured ramps.
No process belonging to another experiment is signalled or altered.

At >=100 states with all four completed option labels, export an explicitly partial
EXAM and run nested world-grouped CV. Completion order makes this exploratory;
do not treat it as a representative estimate of the complete registered mixture.
The existing S-COOPBANDIT-LEARN helper receives the same schema and confidence rule.

## Diagnosed infrastructure correction, before further dispatch
cpuQ v1 PID250743 exited before physics because SSH PATH omitted the installed SUMO
binary. CLOSE records actual0 and charged1140. Preserve v1 STOP, source and charge.
Explicit v2 dispatch adds binary-version preflight and exports installed SUMO_HOME/bin
on PATH; scientific runtime, safety and plan are unchanged. Retain the old1140 charge
against the reserve: conservative data1,490,340<=1,500,000, plus live99,360 gives
1,589,700<=1,600,000 (10,300 slack). Successful data remains1,489,200 actual sim-s.

v2 PID257008 stopped at t366 because the optional strict-JSON trace tried to encode
the executor's NaN native-SUMO/no-command sentinel as an acceleration. CLOSE actual366,
charged1140, 66 trace rows retained. v3 maps only this documented sentinel to JSON null
with command_present=false/controller=native_sumo; simulation commands/fingerprint and
safety are unchanged. CPU regression covers sentinel/real commands/infinity rejection.
Both old reservations remain: conservative DATA1,491,480, LIVE99,360, total1,590,840;
actual abandoned physics366, slack9,160. This is an explicit diagnosed correction.

## Resumable dispatch and milestone commands
`python -m tools.coopbandit.dispatch status` reports roots, charges and STOPs.
`plan` waits for all screens/repeats, then freezes SHARDS.json; register/commit its
hash before `install`, `canaries`, and `expand`. Two low-share worlds go to gpuR;
remaining whole worlds alternate cpuQ/gpuA in a 4:5 pattern. No world is split.
Each host executes ONE allocated branch first, requiring the copied full bitwise
prefix, before pool expansion. The canary is reused by the pool, never rerun.
`collect` transports only explicit RESULT/CLOSE receipts and builds immutable
partial/full snapshots; `analyze` runs the registered nested grouped model.
All raw observable tapes remain on their originating hosts with SHA256 receipts.
Transport can be resumed; a physics STOP blocks that shard and is never cleared.

## Python scalar metric compatibility diagnosis (gpuA v3 STOP retained)
All three t300 gates passed; the additional allocated t540 gate passed cpuQ/gpuR
and stopped gpuA before expansion. Its t300 O1 was unavailable (an exact no-op):
600 rows of observable geometry, commands and event traces match cpuQ byte-for-byte,
SHA25611227672bbcefd4861d2d834e29ea2e6e64d2139fe2aac44711aaf5450ffa6f9.
Only road-delay sums differ at ~1e-12; 100 synthetic frozen-executor cases match
bitwise across cpuQ/gpuA. Python3.12 changed built-in float sum to compensation;
the reference is Python3.10. Test a scoped original-order reducer in metric and
observable-summary modules, leaving all policy/safety code and files untouched.
Require ALL full1140s checkpoint fields bitwise equal before admitting gpuA again.
This explicit diagnostic reserves1140; exclude both gpuA v3 labels/attempts, retain
900+1140 reservations and900+540 actual. Total abandoned4320charged/1806actual;
successful data1489200 + abandoned4320 + gate1140=1494660<=1.5M. Live99360 yields
1594020<=1.6M, reserve5980. No metric tolerance or bitwise gate is relaxed.

The full1140s gpuA compatibility gate PASSED at every checkpoint with zero field
differences, including physical/observable/command digests, all metrics/counters,
MAPPO memory, features and observable-only summaries. Fresh gpuA labels_v4 uses
the hash-bound scalar reducer and retains the stopped v3 roots. Both allocated
t300/t540 canaries execute before its eight-worker pool. cpuQ/gpuR completed work
and live jobs are reused; their existing allocations remain unchanged.
