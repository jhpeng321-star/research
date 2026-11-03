# V2V merged release — Linux CPU qualification PASS

- Integrated `c4/v2v-int`816d045a into this worktree with merge2de40a3c,
  preserving current runner changes and live immutable non-V2V sources.
- Immutable qualification source:
  `/home/ubuntu/coopmwm20261103_v2vcpu2`.
- **764 checks pass** (712 pytest tests +52 subtests), **0 failures,0 skips**;
  one inherited real-SUMO rehearsal test is deselected by the established CPU
  release selection. Wall86.636s; **0sim-s/inference**. Physics subprocess
  launching is forbidden throughout this CPU suite.
- Includes the inherited complete C1 release selection, V2V observation/model/
  bank/end-to-end gates and the seven current runner admission tests. Zero-init
  parity, default-off identity and one imitation step changing V2V weights pass.
- CPU resultSHA256:
  `ab514b5440fcd3f2820a2713573cccbe5fa2277842611019d470311bfbb40eda`.
  All staged source hashes are in `V2V_LINUX_CPU_RESULT_V2.json`.
- Owner3886625/start70523543/bootb8bc1fcb completed; do not dispatch it again.
- V1 result/source/log retained. Its only failure was the launcher's forced
  network binding overriding the missing-network negative fixture. V2 restores
  the existing C1 helper-only test install, with no test exclusions or numeric
  changes; see `V2V_LINUX_CPU_V1_DIAGNOSIS.md`.

**This is not a V2V training launch.** Rehearsal readiness was absent at the
check; runtime adapter/launch gates remain.
No V2V smoke, training, bank branch or evaluation was dispatched by this session.
The concurrently registered17,471,840sim-s cap is preserved. The initial6M
cover-note concern is reconciled by the user's explicit latest-addendum precedence:
the full brief expressly supersedes old resume caps, and MAIN13:45/14:06 funds
the exact +2,971,840 V2V scope within its +3M ceiling. No new budget authority
is created here; all V2V phases remain unspent in the captured ledger snapshot.
