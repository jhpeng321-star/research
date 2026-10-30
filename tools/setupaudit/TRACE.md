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

Confirmed contract discrepancy: the named CCT-2 provenance uses the E3 t0-HDV-only
road-delay scorer (tools/wmcombined/e3_score.py:139-146), K1. Candidate-A long credit
uses all requested vehicles, source wait and K8 (gated300.prefix/actor_term).
At the first mix-on start, J90=15973.92 includes 13165.31 veh-s source wait (82.4%).
The opt-in require_objective_match guard disables long credit until an authority
explicitly qualifies that population/cost/K/statistic; existing defaults are unchanged.
The supplied sigma ratios are descriptive comparisons, not a newly validated
confidence claim for the larger, source-inclusive objective.

## Completed sign/units/observation/expressivity audit
authoritative_reward overrides team_reward's per-vehicle mean with -(delay+source_wait).
Its costs are vehicle-seconds, include births and waiting requests, and have no population divisor.
Positive D means improvement; loss negates credit. C_G = D_G - sum(member D_i).
Actual sampled coefficients are [1,0,0,0,0,18.8]; no negative row or flip credit survives.
The reference is the learned-native head projected onto its WM cap, NOT CTRL0.
Anchor is OFF, direct real-pair actor credit is disabled, and TTC is report-only.
Residual entropy is identically zero; the positive common return scale changes units, not gain signs.
All long gates are OFF (gain-only, median>87.82 AND q10>0; eligible all/group rows only).
At t0 all 32 CAVs have zero positive authority and nonzero braking authority under +/-1.2.
Observations are live: 14/18 CAVs, 50/74 reports, group |signal| means .151/.143.
Report age/command/intent are deliberately zero in the residual path; b/c margins are not own features.
Candidate A's executed delta is zero for 167727 mix-ON and 69134 mix-OFF CAV physics ticks.
MAPPO mix-ON executed-minus-CTRL0 acceleration lies in [-4.0036,0], median/p90/p99=0;
98.8764% is inside the residual execution envelope. Its raw absolute head saturates near +2.6.
MAPPO also emits 786 bridge lane-change requests versus CTRL0's 33; residual A has no lane head.
Neither neural head chooses native/damping/PI controller IDs. Real-MAPPO mechanism attribution is unproven.
Native references match exactly across actors at 90 and 300 s after matching cap/Euler/native background.
No matching CTRL0 SUMO tape exists in the supplied start inputs: exact-start SUMO equality is unmeasured.
Both original mix-OFF and intended mix-ON objectives rank MAPPO worse on their measured average.
Verdict (b): fix WM/objective/start-support visibility and matching qualification before warm-start claims.
The new mix adapter and paired-credit selector are explicitly bound in candA authority manifests.
