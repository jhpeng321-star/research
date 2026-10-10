# Native-selectable damping v2.0: action/reward repair, damping efficacy fails

**Decision: NO-GO for another expensive campaign. No accepted controller.**
The native action and bounded reward preference are executable and tested.
The later negative-control audit qualifies the physical handback indicator;
see the release-evidence correction below.
The filtered active response and its 12-update short-scene checkpoint do not
deliver reliable damping. All runs are stopped. Acceptance data remain sealed;
wall training remains paused. Do not restart v1 or extend this bounded check.

Protocol: [NATIVE_SELECTABLE_DAMPING_V2_PROTOCOL.md](NATIVE_SELECTABLE_DAMPING_V2_PROTOCOL.md).
Audited result: [completed_physical_audit.json](../runs/probe/native_selectable_damping_v2_check/completed_physical_audit.json).
The protocol was frozen before simulations; this report records outcomes.

## What was built and reused

- [env/damping_v2.py](../env/damping_v2.py): explicit native/active actuator,
  causal ego-speed history and bounded filtered-speed/gap reference. Native
  releases ownership with `setSpeed(-1)`; it never changes tau. Active mode
  with correction zero is explicitly controlled behavior. The approximate
  IDM equation is absent from this new active target.
- [idm/damping_v2_actor.py](../idm/damping_v2_actor.py): shared hybrid policy,
  89 CAV-local inputs, a Bernoulli native/active choice and conditional bounded
  continuous correction. The analytical mode-logit prior persists and can be
  overridden by learned logits; it does not disappear after initialization.
- [idm/damping_v2_reward.py](../idm/damping_v2_reward.py): matched-native
  physical reward, separate source/TTC terms, terminal constraint veto and
  bounded requested-intervention cost. Native outcomes are reward evidence
  only. The critic can use global state during training; the actor cannot.
- [env/damping_v2_rollout.py](../env/damping_v2_rollout.py): versioned complete
  corridor collector. Reuses the request ledger, physical metrics and full
  team clock, with added duty, source and TTC interval records.
- The unchanged PPO update retains signed global-time advantage and downstream
  credit. The old velocity-matching analytical comparator and all v1 source
  files/checkpoints were preserved; they were not replaced by this design.

The historical native cost was real and worth reusing conceptually. The prior
medium i60 records showed -0.9535 s/vehicle mean travel-time change over ten
exposed cases, 7/10 wins, but did not isolate the cost as the cause. Stronger
flat and health-gated cost successors failed. The old per-CAV-per-decision
scaling was therefore not copied. The v1 rewrite had no discrete native action
and omitted the cost; **no recorded justification for that omission was found**.
Zero-initialized continuous output did not provide a reliable native option.

## Reward: a small preference that cannot accumulate without bound

At each global second, the intervention cost is:

`-0.02 / T * requested-active CAV count / live CAV count`, with zero if no CAV
is present. The total is in **[-0.02, 0] dimensionless return units** for the
whole episode, independent of the number of vehicles or objective duration.
Its actual value can vary with CAV presence; only the bound is independent.
Requested actions rejected by the actuator are still charged. Native earns no
positive idle bonus and cannot satisfy the required positive physical benefit.

Physical terms remain primary: normalized severe-braking time B and squared
acceleration integral E, total service J including uninserted demand, separate
source wait W, leader TTC exposure L, and collision/teleport events. J and W
have coefficients 0.1 per second; B/E divide by exposed reference quantities;
L divides by max(5 vehicle-seconds, native L). The full units and equation are
in the protocol. E is acceleration effort, not measured energy or fuel use.

An unsafe/service-violating episode loses any positive aggregate physical
return and pays 2 per failed category. This is a finite empirical constraint
penalty, not a proof of safety. On the actual learned corridor trajectories,
the intervention costs were only **0.004172 and 0.004864**, versus terminal
penalties **2 and 4**. The failures are not explained by an oversized native
preference. Audited returns were **-2.13236 and -4.62687**.

Controlled reward-ordering fixtures give helpful **+0.03**, native **0**,
unnecessary intervention **-0.02**, harmful intervention **-0.07**. Horizon
tests cover 1, 10 and 2,400 intervals. Source, service, TTC, completion and
hard-event penalties are checked separately. Removing the terminal penalties
from saved full corridor rewards changes every earlier actor's return by the
expected penalty; removing duty cost changes it by the exact remaining duty
sum. No time/episode mean subtraction erases either effect.

Actual short-scene ordering also behaves sensibly: learned steady/native earns
0; the beneficial catch-up trajectory earns **+0.000591** after paying a
**0.007** intervention cost; the harmful pulse trajectory earns **-0.091466**
and sharp-braking/recovery earns **-1.024160** under the frozen short scales.
Thus useful measured intervention can beat native even with the preference;
the fixture tests additionally isolate physically identical unnecessary action.

## Verification and the short physical failure

The full suite passed **1,135 tests in 231.18 s**. A subsequently added real
Windows byte-zero launch-lock concurrency test passed separately. Focused
controller/reward tests cover conditional likelihood, native residual gradient
masking, cooldown, release, re-engagement, EMA causality and PPO weight updates.
Ruff checks passed for the new implementation and audit tools.

Actual SUMO checks, not only mocks:

- Explicit always-native exactly reproduced the untouched leader–CAV–follower
  trajectory, including all saved positions, speeds and accelerations.
- Selecting native after intervention executed the `setSpeed(-1)` release path.
  Actual speed equaled SUMO's reported speed without TraCI control on native
  intervals, but a later negative-control audit found equality on active
  intervals too. That equality cannot independently prove actual handback.
- Repeated native choices issued no further override, tau stayed unchanged,
  and active control could re-engage after the five-second cooldown.
- Native release remains immediate. Cooldown guarantees a native residence
  period, not smooth realized jerk at handover or a minimum active duration.

The finite learning check used four 120-second profiles, 12 PPO updates and a
single final checkpoint, without tuning or checkpoint selection. Deterministic
learned effort changes relative to native were:

- Steady flow: exact native behavior; **0% active execution**.
- Two braking pulses: **12.67% worse effort**; CAV/follower endpoint loss
  6.89/9.02 m. The untrained reference was 12.70% worse; the unchanged analytical
  comparator improved 1.17%.
- Catch-up: **13.39% better effort**, negligible endpoint loss. The untrained
  reference improved 13.28%; analytical worsened 0.41%.
- Sharp braking/recovery: **75.84% worse effort**. The untrained reference was
  75.66% worse; analytical worsened 0.18%.

No added short-scene TTC/hard events were observed; this is limited stress
evidence, not broad safety validation. The short task has constant J=120,
W=0 and no arrivals, so it cannot teach corridor service or source protection.
It does exercise the same actuator/sensor contract. The learned response barely
improved upon the untrained reference and failed the pulse-improvement gate.

![Saved short-scene speeds and effort](../output/native_selectable_damping_v2/mechanism_traces.png)

The sharp-braking trace localizes a concrete failure. During recovery at seconds
50, 55, 60 and 61, the CAV speed was 5.63, 6.70, 7.75 and 7.96 m/s; the front
gap grew to 37.5, 60.8, 91.3 and 98.3 m. At second 62 the leader left 100 m
perception, the policy chose native, and speed rose from 8.17 m/s to 22.64 m/s
by second 70 while the leader cruised at 15. Native then closed the gap and
the policy re-engaged, adding braking. In seconds 40–70, combined effort was
52.86 versus native 11.15; another 10.29 versus 0.0014 followed in seconds
70–90. Thus slow active recovery, loss of leader detection and native handover
create the rebound in this fixture. This does not identify all corridor harm
causally, but it is a specific mechanism that must be fixed before more learning.

## Complete corridor results

These are the two deliberately exposed v1 diagnostic cases, not representative
fresh evaluation or acceptance. All four arms completed every request with
zero recorded collisions/teleports; equal arrivals do not imply equal service.
Native repetition reproduced all 2,400 interval costs exactly in both cases.
Two existing analytical trajectories were hash-verified and reused.

**911001, mainline 2000 veh/h, 1,150 requests:**

- Native: B **602.5 vehicle-s**, E **37,489.25**, J **105.8842 s/request**,
  W **0.82677 s/request**, TTC exposure **480 vehicle-s**.
- Learned: B **609.5** (+1.16%), E **38,905.82** (+3.78%), J **106.7224**
  (+0.8383 s/request), unchanged W, TTC **583** (+103; allowance +24).
  **Fails TTC and damping benefit.**
- Untrained filtered reference: B **577.5**, E **36,386.77**, J **106.0555**,
  unchanged W, TTC **523**. It improves B/E but fails TTC.
- Unchanged analytical: B **503.0**, E **32,554.80**, J **104.5268**,
  W **0.82720**, TTC **374.5**. It is the better comparator in this case.

**911003, mainline 2200 veh/h, 1,252 requests:**

- Native: B **1,091.5**, E **64,539.55**, J **117.0839 s/request**,
  W **0.83266 s/request**, TTC **1,150 vehicle-s**.
- Learned: B **1,207.5** (+10.63%), E **70,434.48** (+9.13%), J **121.5407**
  (+4.4569 s/request), unchanged W, TTC **1,267** (+117; allowance +57.5).
  **Fails service, TTC and damping benefit.**
- Untrained filtered reference: B **1,136.5**, E **70,472.00**, J **125.7368**
  (+8.6530 s/request), unchanged W, TTC **1,310.5**. Also fails.
- Unchanged analytical: B **1,023.0**, E **59,743.62**, J **116.8067**,
  unchanged W, TTC **1,123.5**. Again better on this case; its earlier wider
  development assessment still failed, so this is not an accepted fallback.

The learned mean changes across these two cases are braking **5.89% worse**,
effort **6.46% worse**, and service **2.65 s/request worse**. Its intervention
cost is small, but physical returns are negative. Raising the native cost or
running longer would not establish that this mechanism is repaired.

## Budget, integrity and next decision

The old campaign was stopped at verified checkpoints 36/8/8, with no incomplete
successor rollouts; all negative results remain. New work completed 30 short
rollouts (3,600 s) and six full diagnostic rollouts (14,400 s): **18,000 actual
global seconds**, including only **1,440 training seconds**. One full allocation
failed before simulation in the legacy append/text SUMO startup lock and is
charged conservatively as another 2,400 s. Its traceback and started record were
retained. A separately registered recovery used a real binary byte-zero lock
without changing controller/reward/checkpoint sources. **20,400 charged seconds
remain below the frozen 22,800 ceiling.** No remaining allowance is automatically
authorized for a design search.

The completed audit verified source/cache hashes, repeated native intervals,
physical totals, full reward decomposition and **112,063 exact deterministic
action replays**. Acceptance data were not opened. The failed checkpoint is
`runs/probe/native_selectable_damping_v2_check/checkpoint_012.pt`, explicitly
`accepted=False`; it is not a deployed or corridor-trained controller.

The subsequently bounded 480-second recovery mechanism check removed the
sharp-recovery rebound but worsened pulse effort by 18.94%; it also stopped
with no training. See [v2.1 results](NATIVE_DAMPING_RECOVERY_V21_RESULTS_2026-09-06.md).
The owner now prioritizes an existing trained controller as the baseline for
escalation. No bespoke successor campaign is justified by these results.
This report authorizes no parameter search, expensive campaign, wall training
or paper claim that damping works.

## Release-evidence correction, 2026-09-06

[The saved-data negative-control audit](../runs/probe/native_selectable_damping_v2_check/handback_evidence_qualification.json)
finds `getSpeedWithoutTraCI == actual speed` on all active half-steps as well:
10 in the switching trace, 178 in learned pulse, and 110 in learned sharp
braking. Thus the earlier speed-equality check is not discriminative evidence
of release in this setup. Frozen mechanics/results files are preserved; this
report explicitly qualifies their interpretation. Source/API-call evidence,
recorded switching and exact always-native trajectory identity remain valid.
A subsequently completed [paired handback check](../runs/probe/native_selectable_damping_v2_handback_physics/summary.json)
supplies the stronger physical attribution: after an identical prefix, release
halfway through an unfinished command yields +0.098404 m/s2 while retaining
the command yields its requested -0.335789 m/s2. Re-engagement after cooldown
also executes the requested acceleration and native releases again. This used
unchanged v2 source, 20 actual simulated seconds under a separate 40-second
allocation, with no training or corridor rollout. See the detailed
[handback result](NATIVE_DAMPING_RECOVERY_V21_RESULTS_2026-09-06.md).
The paired effort, braking, service, source and safety NO-GO outcomes remain
unchanged; mechanical release does not establish damping efficacy.
