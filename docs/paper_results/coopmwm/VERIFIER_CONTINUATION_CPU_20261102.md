# Verifier continuation / CRN qualification

Frozen source `/home/ubuntu/coopmwm20261102_verifycpu3`: **93 CPU PASS**, zero
skips, **22.509702s**, zero SUMO/GPU inference. Receipt:
`CPU_coopmwm20261102_verifycpu3.json`. First generated-payload indentation
failure is retained separately; it occurred before test execution.

## Engineering correction

`exact_grade.guide_draws` now counts the pre-verification trust draw even when
the proposal subsequently follows the policy or SUMO rejects all candidates.
For R it also retains the matched sampler draw in those cases. The old rule
counted draws only for finally accepted proposals, losing CRN for a later exact
grade after a verified rejection. Legacy non-verification draw schedules are
unchanged. Tests compare actual Torch generator states for both B and R,
accepted and rejected, and preserve the legacy decline schedule.

## Real-smoke support

The isolated branch runner now saves fingerprint hashes from the decision to
the next registered 10s query window. `check_live_continuation` requires:

- a sealed accepted verification receipt;
- the same episode and exact chosen plan actually executed at the decision;
- the same verification receipt hash in the live behaviour tape;
- every chosen-branch/live fingerprint identical through that next window.

Missing fingerprints, a one-bit metric change, wrong plan/receipt or a
nonexecuted decision fail the gate. It cannot launch physics or charge budget.
It intentionally makes no claim that future, not-yet-chosen plans reproduce
the entire duration+120 verification window.

## Activation state

**CPU-only, not deployed or live.** Original training/search/grader owners and
the completed warm-memory qualification remain adopted. Pending: verification
grade reuse, one real five-branch smoke plus live-continuation gate, shared B/R
update-boundary switch, fixed final-endpoint migration and confirmed budget.
No new SUMO/inference or PID control was performed for this qualification.

## Integrated CPU episode-loop gate

Frozen `verifycpu5`: **94 CPU PASS**, zero skips,23.139720s, zero SUMO/inference.
One additional test runs the actual episode loop, live controller and verifier
installation against the deterministic toy plant, with all five branches
(four candidates plus policy) and exact live/chosen fingerprints through t330
after the t320 decision. This checks integration beyond the isolated helpers;
it is still **not a real SUMO smoke**. `verifycpu4` failed because the first test
fixture used stochastic admission and produced no proposal at its sole planned
decision. Its failed receipt remains. The corrected fixture deterministically
admits that one decision, retaining a real generator admission draw; no runtime
rule or search was changed.

## Exact verification-grade reuse

Frozen `verifycpu9`: **100 CPU PASS**, zero skips,27.928785s, zero SUMO/inference.
`verification_grades.py` imports the chosen candidate's exact grade from the
same five SUMO branches, with no new branch or charge. It validates sealed
source identity, episode/world/arm/update/time/plan, five prefix/CRN passes,
the live chosen-plan continuation and the prerequisite five-branch smoke.
Existing exact-only AWR/temperature consumers accept the resulting receipts.
Concurrent rollout/grader importers adopt one receipt; an interrupted
receipt-before-spec publication is adopted without physics. Existing unrelated
queue/claim/attempts cannot be replaced.

The source contrast/window are explicit: **decision-local verification** on the
common max-candidate-duration+120 window (episode capped), with existing plans
but no unknown future proposals. It is not relabelled as a later marginal
replay of the completed live episode. If a plan is later truncated, the source
full-plan grade is not reused: the original executed-portion marginal grader
is queued under its existing allocation. Activation must disclose that rule.

The default path for episodes without `sumo_verification_root` is unchanged.
The real smoke writer requires all four candidates plus policy, all exact
prefix/CRN gates and live equivalence; neither it nor the adapter can launch
SUMO. No live smoke file has been written.

Intermediate receipts remain: CPU6's first SSH execution ended without a final
receipt; read-only stdin-session census confirmed no surviving session before
the CPU-only retry. The retry failed the new fixture's omitted `update` identity
(correctly rejected by the adapter). Adding the real job's update/seed fields
fixed that fixture. CPU7/8 passed98/99 tests; CPU9 adds deterministic concurrent
publication-race coverage. The qualification harness now persists remote
PID/boot/start and result receipts before returning over SSH.

**Still not activated.** Shared B/R boundary deployment, real SUMO smoke,
verification budget confirmation and the final-endpoint supervisor migration
remain necessary. The command-mirror remains OFF/unqualified.
