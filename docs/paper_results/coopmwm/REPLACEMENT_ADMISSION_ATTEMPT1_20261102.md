# Replacement admission attempt 1 — pre-admission module lookup failure

The read-only route/admission preflight failed on import of
`tools.coopmwm.hotadd`: the verifier CPU snapshot does not contain that module.
It failed before RPC, authority append, admission or worker dispatch. Original
training, GPU and broker owners are unchanged. The fix loads ONLY the existing
CPU26-qualified hotadd helper from its original immutable hotadd1 snapshot,
after checking its recorded source SHA. Current recovery broker/health modules
still resolve from verifier CPU9, preserving incident/STOP-release semantics.
