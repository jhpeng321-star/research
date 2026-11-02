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
