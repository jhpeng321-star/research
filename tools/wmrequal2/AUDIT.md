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
claimed from the latter empty set. A CPU probe will quantify original-ramp
exposure and verify these paths; do not relabel the shared mismatch as measured
learned-model error. The SETUPAUDIT2 bridge-only repair does not by itself repair
the timeline callback. No affected-arm GPU dispatch is yet justified.
