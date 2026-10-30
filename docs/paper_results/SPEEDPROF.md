# S-SPEEDPROF — exploratory B / B-L, 2026-10-30

## Step 1: read-only baseline (22:35 +08)
- B `gpuA`, chain `candB_pilot_final_run_EXPLORATORY_20261030`: still initializing;
  no `updates.jsonl`, rescores or memory trace yet. Do not claim measured s/update.
- Receipt: `tools/candB/evidence/SPEEDPROF_pilot.json`; collector `tools/candB/speedprof.py`.
- Owned dry PID985766/start874675005 used 4562 CPU-s in 4402 elapsed-s (~1.04 cores),
  despite OMP/MKL=2; this supports Python/launch-bound rather than compute-bound work.
- Startup scores one independent start, then the frozen native panel and gate panel
  separately (16+16 starts); each matched score contains dense prefix + 300s K8.
- Per update: K8 unperturbed 300s forwards, M2×2 signs×K8=32 ES 300s forwards,
  K8 dense prefix replays (60/90s plus next-input boundary), one independent rescore.
- Native/CTRL0 rows already share each rollout batch; CTRL0 is NOT an extra whole
  rollout. Its one row per start is still redundantly simulated across ES tasks.
- Every tenth absolute update additionally scores the native panel; phase10 scores
  the gate panel. The per-update rescore is one frozen start, NOT the entire panel.
- Historical `ESLONG.md`: 861s/K8 300s forward, dense90 projection866s; serial core
  5171s/update before rescoring, panel scoring and IO (projection, not B measurement).
- `POLSPEED.md`: earlier 90s run was 0–13% GPU utilization; full-forward284s,
  replay-forward384s, VJP324s; this is supporting historical evidence only.
- B-L branch inspected at `33cfade2`: same M2/K8 forwards; confident-harm dense
  skips may reduce replay cost. No B-L completed-update timing available yet.

## Implementation / measurement
CPU probes use CUDA hidden, nice19 and one thread in owned
`gpuR:/root/autodl-tmp/c4/speedprof_CPU_20261030`; no pilot source/PID changed.
Exact flag implementations, verification and MAIN application diff follow in later commits.
