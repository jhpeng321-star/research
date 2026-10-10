# Recovery v2.1 mechanism result: NO-GO

The single fixed recovery prototype removes the observed sharp-braking recovery
rebound but **does not provide effective damping**. Do not train this version or
extend its check. The owner has since prioritized reuse of an existing trained
controller because escalation is the intended contribution. That supersedes
further bespoke reference design.

The frozen derivation and allocation are in
[NATIVE_DAMPING_RECOVERY_DERIVATION_2026-09-06.md](NATIVE_DAMPING_RECOVERY_DERIVATION_2026-09-06.md).
The completed machine-readable result is
[summary.json](../runs/probe/native_selectable_damping_v21_mechanism/summary.json).

Only the active reference changed to relative-speed/gap feedback with a small
ego-speed filter term. Native release, projection, cooldown, local observation
and correction bounds were retained. All four probes used the existing expert
mode rule and zero residual: **this was not another trained checkpoint**.
The runner checks the actual command against the new formula; a unit test
checks that the actuator apply implementation is unchanged from v2.

The complete allocation was four 120-second profiles: **480 new global seconds,
zero training updates, zero corridor simulations**. The four native records
were reused with identity/hash checks. These are exposed mechanism fixtures,
not acceptance evidence. No additional prototype is implied or authorized.

- Steady: exact native trajectory, no active commands.
- Two braking pulses: acceleration effort **18.9434% worse** than native
  (36.31295 versus 30.52960). Endpoint positions recover to within 0.004 m,
  but equal eventual progress does not excuse extra effort. The pulse gate fails.
- Catch-up: exact native effort and endpoints. Active commands are still issued;
  this is parity, not a benefit that warrants intervention.
- Sharp braking/recovery: no recovery sensor-loss release or speed rebound.
  Effort is 67.695537 versus native 67.695548, a difference of only 0.0000163%.
  Treat this as numerical parity, not a meaningful improvement.

There was no added severe braking, low-TTC exposure or hard event in these four
profiles. Their short objective has constant service, zero source wait and no
arrivals; it cannot establish corridor service/source protection. Ideal linear
stability in the derivation does not establish attenuation in the implemented
sampled, saturated and switching SUMO controller. The failed pulse result is
the relevant physical evidence.

The previous v2 learned policy increased pulse effort 12.67% and sharp-recovery
effort 75.84%. v2.1 resolves the latter mechanism while worsening the former.
This narrows the diagnosis but does not turn the controller into a usable
baseline. Existing analytical, native, v1/v2 checkpoints and negative results
remain preserved. Six recovery-analysis tests and four v2.1 tests passed.

## Qualification of the earlier release evidence

The v2 report previously used equality between actual speed and
`getSpeedWithoutTraCI` as supporting evidence of release. A negative-control
audit finds equality also holds on every saved active half-step in the
switching, learned pulse and learned sharp-braking traces. It is therefore
**not an independent indicator of handback in these captures**. The machine
audit is [handback_evidence_qualification.json](../runs/probe/native_selectable_damping_v2_check/handback_evidence_qualification.json).

The explicit `setSpeed(-1)` source path, API-call tests, recorded release and
re-engagement, unchanged tau, and exact always-native trajectory remain valid
separate evidence. The speed-equality statistic does not by itself distinguish
native release from ongoing influence or ordinary expiration of a timed command.
A same-state branch with release versus an unfinished command supplies the
appropriate stronger physical attribution. That separately bounded check has
now passed, as documented below. This qualification asserts no SUMO bug and
changes none of the measured paired physical outcomes.

## Completed physical handback check

The existing v2 actuator, without a source change, was exercised in two matched
10-second runs under a separate **40-second maximum allocation**. The prefix,
first active command and all measured vehicle states at the branch were exact
matches. At 3.5 s, halfway through the same one-second speed command, one
branch selected native and the other retained the unfinished command. At 4 s:

- Retained command: acceleration **-0.3357893 m/s2**, matching its active target;
  speed **14.8662684 m/s**.
- Native release: acceleration **+0.0984043 m/s2**; speed **15.0833652 m/s**.

The 0.217097 m/s difference demonstrates that selecting native cancels a
still-active command. It is not explained by automatic command expiration or
the nondiscriminative speed-without-TraCI statistic. An attempted early
re-engagement was blocked; re-engagement at the five-second cooldown boundary
produced the requested **-0.3132605 m/s2** measured acceleration, and native
released the command again. Tau remained 1.5 s. A nonzero conditional residual
in the native action was ignored as required.

The [frozen manifest and result](../runs/probe/native_selectable_damping_v2_handback_physics/summary.json)
record **20 actual simulated seconds, 40 charged, zero training/corridor work**;
[re-engagement audit](../runs/probe/native_selectable_damping_v2_handback_physics/reengagement_audit.json)
checks the measured execution. This test establishes mechanical handback and
switching, not damping efficacy or smooth realized jerk at handover. All
physical NO-GO decisions above remain in force.
