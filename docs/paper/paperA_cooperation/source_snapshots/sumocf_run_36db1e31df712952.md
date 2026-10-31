# S-SUMOCF run receipts

## Fresh-session recovery and amended CPU gate
- Preserved previous uncommitted trainer draft in794956ff. Prior designbf8c1e20
  superseded by cooperation-first amendment3b006b59 and subsequent implementation
  clarification. No inherited SUMOCF running job or simulation charge found.
- CPU amended suite:7 PASS in1.43s on cpuQ's existing environment, zero SUMO.
  Receipt tools/sumocf/evidence/CPU_V2.json. Zero-update policy actions match the
  exact MAPPO export bitwise; general credit, joint scores and snapshots checked.
- New active entry point: tools.sumocf.supervisor {canary,probe,train}; old draft
  train.py/worker.py are preserved in history and are not launch entry points.
- Stable trace root gpuR /root/sumocf20261031/traces; raw primary credits
  and immutable per-update checkpoint/receipt paths are under that run root.
- No measured cooperation or policy gain yet. Canary and fixed SNR probe precede
  training; the learner remains blocked on their receipts and LAUNCH.json binding.
- Final CPU reruns:7 PASS cpuQ3.64s and gpuR2.16s; gpuR frozen source/export
  verification PASS. cpuQ priority workload grew to17+ ES workers plus Track3B's
  24-worker screen. Primary host switched to gpuR CPUs (25quota,load~10), up to12
  nice19 one-thread workers with load admission. No cpuQ SUMO dispatched.

## SUMO canaries (before SNR probe)
- PASS: live-trunk versus RNG saved-state identical-action branch has bitwise
  J600 AND full physical-trajectory SHA identity. Saved-state engine selected;
  full-prefix rerun fallback was not needed.
- PASS: zero-update direct policy versus frozen MAPPO arm J600 bitwise,
  J600=83841.10161736357 veh-s on TRAIN969624(.4,2700).
- Charged=actual2400sim-s inside8M TRAIN cap; dev0. Wall80.156s overall;
  standalone900sim-s MAPPO arm26.045s,124401.96sim-s/worker-hour.
- Receipt tools/sumocf/evidence/CANARY_V2.json. Host load admission reduced its
  capacity as peers arrived (load19.25 at finish); no peer process modified.
