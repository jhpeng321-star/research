# V2V live sampling/scoring adapter — 2026-11-03

The previously merged 764-check release did not connect widened model sampling
to live episode workers. Added an explicit, default-off runtime adapter for
sampling, PPO scoring, checkpoint/Adam/RNG restoration, and spawned branch jobs.

The integration also exposed a causal feature mismatch: bank acceleration used
the same-second issued command (an imitation target), while runtime used speed
differences; runtime intent used the current MAPPO raw command. Both enabled
paths now use observable backward speed differences and the **previous second's
actual post-safety issued command**, captured from the existing observable trace.
The no-current-target-leakage and offline/runtime feature equality tests pass.
The unchanged command/safety/ownership chain remains the sole actuator path.

Linux source `/home/ubuntu/coopmwm20261103_v2vcpu4` passes **769 checks**
(717 tests + 52 subtests), zero failures/skips, 85.941 s, zero SUMO/inference.
Evidence: `V2V_LINUX_CPU_RESULT_V4.json`. Local targeted suite: 24 passed.
Source V3 is preserved: only its overly strict critic bitwise assertion failed
(9.3e-9 difference); the existing integration bound is 1e-7. Sampled actions,
action densities, zero inserted columns, optimizer parameter identity and RNG
restoration are separately checked. No tolerance or model setting was tuned to
traffic outcomes.

This qualifies the core adapter, not the remaining live supervisor/smoke gates.
No V2V physics or training launch is claimed. Existing non-V2V owners and all GPU
workers remain on their frozen sources.
