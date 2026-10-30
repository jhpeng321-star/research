# S-SETUPAUDIT setup trace (2026-10-30)

Base: c4/distill 9613f956; owner requests measurement only, no training/SUMO/push.
Candidate config: configs/c4_policy14_candA_longgrad_skip.toml.
Execution: C4_EXECUTION_PARITY=2, C4_SUMO_LEADER_RULE=1;
configs/execparity_cav_idm_v1.json (executor parameters, not vehicle assignments).
WM2 required sha256: b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436.
MAPPO required sha256: aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564.
Latest completed candidate checkpoint located: gpuA candA_prod_keyerr/policy_000190.pt.
gpuR availability checked with nvidia-smi: 1 MiB / 24564 MiB, 0%, no compute PIDs.
Use CPU first; any GPU scoring is capped at .40 and records its own PID.

Exact objective path: Policy14Trainer.actor_objective -> DreamerTrainer.actor_objective
-> antiexploit.actor_terms/tail_credit -> gated300.actor_term/gate.
Dense return is the time-average of finite-horizon suffix returns, not just t0 J.
Credit is positive for improvement (reward=-road_delay-source_wait); loss is its negative.
Counterfactual reference rows use native_fn, not automatically CTRL0.
Weights: all=1, group=9.9, coop=8.9, individual=8.9, team=0.
K=8; joint q10 selection preserves covariance across counterfactual rows.
Gain-only gate: median improvement > 2*sigma AND q10 improvement > 0.
Independent scorer resolves the base 10 s unless score_training_horizon is enabled.

Potential integration discrepancy to measure: tools/errsrc/inference.py defines LearnedMix,
but no LearnedMix/entrant_driver_sampler reference exists under src/wms at this base.
Do not silently label the actual trainer's prior-entrant path as learned-mix ON.
Only permitted TRAIN windows, observable pre-t0 histories, and command tapes may be read.
Forbidden path tokens and seed ranges are checked before loading selected windows.
