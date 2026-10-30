# Step 1: validation execution provenance

`evidence/RUNTIME.json` was collected read-only on gpuA with CUDA hidden. All six
named frozen source hashes match the original `SOURCE.json` manifest and local
retained runtime. The live controller design contains checkpoint+hash only for
the three validation arms; it supplies no ownership-switch overrides.

- `mappo`, `ours_c1`, `coopoff_c1`: original slots use `TimelineControllers`,
  replaying longitudinal commands on WM states; they do not invoke the actor
  bridge. Synthetic entrant CAVs use `policy_feedback -> make_cav_controller`.
- All three entrant adapters omit `policy_ramp_merge`; the frozen default is
  **False**, with `native_ramp_merge='wm_law'`. Thus acting ramp entrants that
  remain in lane 2 are delegated to native first passage; explicit lane changes
  retain priority. None of these arms exercises SETUPAUDIT2's **True** path.
- There is a separate, shared ownership mismatch: `TimelineControllers` returns
  the current lane and never writes `WM_MERGE_KEY`; `EntrantControllers` preserves
  this all-false original-slot delegation. First passage excludes CAVs without
  delegation. Therefore **original ramp CAVs cannot merge in any arm, including
  the native reference**, if such slots are present. Recorded lane requests are
  not replayed either (the action timeline has longitudinal fields only).

Two exclusion sets must be reported: excluding the specific True-switch bug
removes no arms; excluding every arm exposed to blocked original-ramp ownership
leaves no clean contrasts. A native-lane/speed-only qualification cannot be
claimed from the latter empty set. The CPU probe below quantifies original-ramp
exposure and verifies these paths; do not relabel the shared mismatch as measured
learned-model error. The SETUPAUDIT2 bridge-only repair does not by itself repair
the timeline callback. No affected-arm GPU dispatch is yet justified.

## Step 3: measured CPU confirmation and rescore decision

`evidence/CPU_PROBE.json`: 144 permitted validation records inspected, two CPU
threads, CUDA hidden, 9.87 s elapsed. All four arms share 44 original ramp CAVs
in 21/36 worlds; all 44 merge in each recorded arm within 300 s. Eight actual
callback cohort/arm checks, using the three hash-verified frozen policy actors,
confirm original ramp stays delegate `[False,False,False,False]` and entrant
stays delegate `[False,True,True,True]` for native/MAPPO/ours/coopoff. **Native
entrants have the same missing native ownership as original slots.**

The real records also contain original mainline lane changes (87 native, 107
MAPPO, 98 each ours/coopoff), so lane-held timeline playback is not a measured
native-lane intervention. These counts are vehicles with any recorded change,
not numbers of policy requests or causal lane effects.

No specific-True-path arms exist to rescore with SETUPAUDIT2's bridge repair ON.
That repair alone cannot add ownership to `TimelineControllers` or its native
reference. No repaired K8 outcome is claimed. The existing dual-mix K8 panel
took 14,418.66 s (4.01 GPU h); this is historical elapsed runtime, not a new
benchmark or proof of ON-only runtime. A valid speed-only correction additionally
needs matched native-lane reference execution and matching recorded labels.
No GPU slice was requested or allocated; no pilot PID was inspected or signaled.

Priority correction: explicitly delegate native ramp stays for original and
native entrant callbacks, restore native discretionary lanes for the native
reference, and preserve any explicitly issued policy crossing. Retest the
matched contract before attributing residual upstream error to learned discharge.
The existing decomposition identifies error location, not the causal need for
a new merge-discharge model. Its necessity remains unmeasured after execution
repair; do not apply an unconditional capacity scalar or source-bias offset.
