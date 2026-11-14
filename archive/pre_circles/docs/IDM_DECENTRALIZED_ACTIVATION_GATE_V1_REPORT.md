# IDM Decentralized Activation Gate V1

Status: pretraining gate passed; resumable development PPO training started.
This is not an accepted checkpoint and contains no independent-seed claim.

## Frozen learned decision

Every CAV runs the same decentralized two-action actor: `abstain` or bid for a
20-second structured damping option. Deterministic V2V contention and hard
safety admission remain rule-based. The treatment is a terminal-speed ceiling
at the detected downstream disturbance p30; it does not increase the allowed
braking rate. Comfort, jerk, leader/follower TTC, secure-gap, SUMO collision
avoidance, and smooth release remain authoritative.

Speed-ceiling bids additionally require at least 5 m of prospective rear
headroom when the follower secure gap is evaluated at the proposed terminal
speed. This fixed robustness margin prevents the short rear-gap aborts found
by the development probes.

## Why this replaced the earlier controller

- Gain and acceleration-cap modes were usually projected to the same command
  and did not provide a learnable action-strength decision.
- The analytical p30 acceleration law often converged to native IDM and had
  little causal authority.
- A p30 terminal-speed ceiling created a sustained gap without raising the
  braking-rate limit.
- A 30-second fixed-member option failed dose because rear safety correctly
  aborted; it was rejected.
- A p70 ceiling was too weak and did not improve the 60-second outcome; it was
  rejected.
- The full 60-second horizon exposed delayed shockwaves hidden by the former
  40-second score and is now mandatory.

## Development-only paired evidence

Artifacts:

- `runs/eval/idm_decentralized_p30_60s_landmark1_pilot_v1`
- `runs/eval/idm_decentralized_p30_60s_landmark2_pilot_v1`
- pooled gate: `runs/eval/idm_decentralized_activation_pretraining_gate_v1`

The pooled protocol contains nine paired landmarks across five development
seeds. Outcomes are 4 helpful, 2 neutral, and 3 harmful. Helpful outcomes occur
in seeds 4107, 4109, 4110, and 4111; nonbeneficial outcomes occur in 4107,
4109, 4110, and 4112. Every treatment completed 20/20 command steps and passed
the collision, teleport, TTC, delay, arrival, and discharge constraints.

The physical effect and cooperative reward delta have correlation 0.997. All
helpful cases have positive reward deltas, all harmful cases have negative
deltas, and neutral deltas are bounded. Every decision observation has the
training-compatible 136-field contract: 134 physical fields plus a two-bit
action mask.

## Reward correction

The prior reward gave corridor cost only to the selected CAV while abstention
received zero, making every intervention look worse. The decentralized path
now gives all active CAV agents the same centered cooperative corridor reward.
Severe braking and acceleration energy are centered on the frozen All-HDV
reference (`1 - observed/reference`). The structured safety shield makes the
legacy ego-motion penalty redundant, so it is disabled only for this centered
decentralized path. The legacy non-decentralized residual reward is unchanged.

## Verification and training

Repository verification after implementation: `569 passed`.

The resumable PPO run is
`runs/rllib/idm_decentralized_activation_smoke_v1`. It uses demand 2200, 10%
scatter CAV penetration, a decentralized local actor, a centralized
training-only critic, two environment runners, 2,000-step iterations, and
per-iteration checkpoints. An explicit restore from `checkpoints/iter_2`
advanced successfully to iteration 3. The run was then resumed toward 100,000
development steps.

Promotion still requires paired checkpoint evaluation and untouched validation
seeds; training reward alone cannot select the paper controller.
