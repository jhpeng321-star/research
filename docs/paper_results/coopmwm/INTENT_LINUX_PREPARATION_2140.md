# Intent release qualification plumbing — prepared, not dispatched

`intent_linux_ops` prepares a NEW immutable cpuQ generation after the final
reviewed intent/gate/checker merge. It requires an exact full HEAD and committed,
clean whitelisted inputs. It does not expose training, SUMO gate execution,
budget reservation, GPU release or launch.

It extends the existing durable Linux CPU qualifier with additional tests and
SHA-manifested frozen fixtures. Four exact pre-intent blobs at cf3efc9e support
the option-off identity check without allowing a subprocess through the
CPU-only physics guard. The intent test needs to consume the explicit
`COOPMWM_INTENT_BASELINE_DIR` fixture contract; the active integrator was notified.

Source generations use `/home/ubuntu/coopmwm20261103_intentcpu_i<N>`, never the
live legacy source or previous V2V qualification roots. Existing dispatches and
results can only be adopted with matching pinned commit and staged payload;
otherwise a new generation is required.

Five CPU plumbing tests pass (2.04 s): invalid/dirty commit never dispatches,
exact pre-intent blob provenance, distinct generation validation, generated
remote-program compilation, and fixture path escape refusal. No remote
qualification or physics was launched by this step. Independent audit/merge
readiness still precede qualification and exact-gate execution.
