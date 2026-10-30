# Read-only measurement receipts

WM2 b8797242 / config f95f7c4c; MAPPO aecd7d70; candidate A update 190 de235a78.
The two permitted starts are the next TRAIN sampler draws restored from update 190:
w963758m2400p20t300 @ 780 s (generic_random) and w963111m2400p20t300 @ 970 s (native).
State tensors and model checkpoints remain on the GPU hosts and are not committed.

CPU probes are 2 s / K8 plumbing and native-reference checks, not headroom estimates.
`probe.json`: original CTRL0 probe (PID 427061).
`probe_mappo.json`: initial unaligned absolute MAPPO bridge and candidate A (PID 427420).
`probe_matched.json`: all three actors with identical native-background execution (PID 427891).
`probe_mix.json`: opt-in mix actually assigns every newborn before its first WM action (PID 427988).
All probes use the real checkpoint and permitted training start; no update/optimizer step.

Remote root: /root/autodl-tmp/c4/setupaudit20261030 on gpuR; CPU tests use its fix/ subtree.
Full-horizon runs cap GPU memory fraction at .35; the initial GPU PID is 427168.
PID 427168 was terminated by this audit after recording its first complete CTRL0
start (693.64 s), to replace the unmatched MAPPO comparator before measuring it.
No other process was signalled. `initial_ctrl0.json` retains that completed row.
`probe_exact_executor.json` (CPU PID 428327) also aligns the absolute MAPPO bridge
to candidate A's exact cap and Euler integration; native reference arrays match exactly.
Full mix-on paired run: GPU PID 428305; started on an idle card, cap .35.
Completed CTRL0 rows were retained across owned scorer restarts: PID 428305
stopped after start 0 to enable stepK's exact CUDA-CVaR shortcut; PID 428899
completed start 1, then stopped to align the comparator's reported acceleration
round-trip with bridge.py (CPU bitwise-native test added). Final scorer PID 429791
reuses those two immutable CTRL0 rows. All of these processes were this audit's.
The raw long-credit receipt reports gates against the learned-native row, not CTRL0.
Dense values use the actual suffix-time/joint-q10 reduction. All physical J values are veh-s.

CPU verification: 16 targeted mix/sign/credit/dead-zone/native-comparator/objective-contract
cases passed (3.78 s), plus 3 execution-cap/default-OFF bitwise-gradient regressions
passed (5.07 s): 19 distinct CPU cases. No real-data update/optimizer step was run.
Commands: pytest tests/unit/test_setupaudit.py tests/unit/test_gated300.py
tests/unit/test_zerostart.py -k 'setupaudit or sign_gate or gain_only_gate or
executed_acceleration_at_zero or dead_zone_positive or candA_credit_has or
candA_objective_identity'; and test_keyerr.py::test_nonempty_cav_forward_and_actor_gradients_bitwise_legacy
plus test_keyerr.py::test_initial_empty_and_entrant_only_cap_is_real.

`observations_groups.json`: exact t0 inputs, entrance queues and sampled group geometry.
`probe_noisy_mix.json`: training-only additive exploration plumbing, 2 s/K8, CPU PID 429196.
That short noisy receipt is not used as evidence of 90/300 s actor headroom.
The optional objective-contract guard refuses the historical t0-HDV-road/K1
qualification for team+source/K8 long gradients; primary measurements retain the
original guard-OFF semantics so the objective being audited is not changed.

## Completed primary panel
`mix_exact.json`: 2 TRAIN starts x 3 deterministic actors x K8, one 300 s continuation
with the actual dense-90 s prefix/credit. Native references match exactly across
actors at both horizons. Candidate A equals CTRL0 (including 167727 executed CAV ticks).
MAPPO-CTRL0 mean dJ90=+8.3037109375, dJ300=+3278.14453125 veh-s;
maximised dense-credit difference=-24.05652618408203, all long gates OFF.
The sigma comparisons are descriptive: the original authority does not qualify
the same population/cost/K as this objective. No population confidence claim is made.
`mix_first_ctrl0.json` and `mix_progress.json` retain completed intermediate receipts.
Reproduce on the staged inputs: `bash run_measure.sh --out NEW.json --device cuda --mix`.
Local verification/aggregation: `python tools/setupaudit/summarize.py
tools/setupaudit/evidence/mix_exact.json --aggregate-only`.
