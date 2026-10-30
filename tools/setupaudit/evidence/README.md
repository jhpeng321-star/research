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
The raw long-credit receipt reports gates against the learned-native row, not CTRL0.
Dense values use the actual suffix-time/joint-q10 reduction. All physical J values are veh-s.

CPU verification: 13 targeted mix/sign/credit/dead-zone cases passed (4.35 s),
then 3 execution-cap/default-OFF bitwise-gradient regressions passed (5.07 s).
Commands: pytest tests/unit/test_setupaudit.py tests/unit/test_gated300.py
tests/unit/test_zerostart.py -k 'setupaudit or sign_gate or gain_only_gate or
executed_acceleration_at_zero or dead_zone_positive or candA_credit_has or
candA_objective_identity'; and test_keyerr.py::test_nonempty_cav_forward_and_actor_gradients_bitwise_legacy
plus test_keyerr.py::test_initial_empty_and_entrant_only_cap_is_real.
