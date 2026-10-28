# Entrant-control path audit

- `control/dreamer.py` / `policy14.py`: `_entry_slots` retains observable insertion
  CAV share; `imagine` already sets ALL-row padding active, BASE padding inactive.
  Individual/GROUP/FLIP interventions remain the declared t0 identities. Existing
  controller bridges build the ordinary observations and maintain per-slot command
  history. `entrant_control` records the explicit opt-in contract, default false;
  existing ALL continuity is preserved in both modes.
- METERVIS uses that Policy14Trainer imagination path, so its ALL entrants were
  already controlled. Its old duplicate-native discrepancy is a separate RNG issue.
- `train/inputs.prepare_inputs`: every padded timeline formerly native; queue CAV
  shares also deliberately removed. ON retains observable queue shares and wraps
  the original t0 timeline with same-arm entrant feedback. Direct/mixed commands
  require explicit hash-bound algorithm specs; they are never copied to births.
  Native/damping/PI unambiguous plans can be inferred from the algorithm channel.
- LRDIAG and E3 `e3_score.rollout_dj` both use `prepare_inputs`. Configurable
  `data.entrant_controllers` maps observable arm names to frozen checkpoint and
  exact plant adapter options. The full scene is available to observations, but
  rows never observe another arm. Callbacks persist across ticks/chunks. Entry CAV
  slots are virgin-only and never reused, preserving identity/history continuity.
- `loop14_checks/accept.free_rollout` / horizon_gate: the old harness prepared
  no entries: the attribution loader/dataset builder explicitly disabled them,
  and free_rollout also omitted the entry argument. ON restores both observable
  input construction and entry delivery; OFF preserves frozen evidence. This is
  an additional horizon-contract difference to disclose. Single-row gate rollouts
  receive identity-keyed lane/arrival/merge CRNs. CLI exposes fixed K/seed streams
  and post-load ramp adaptation overrides without dropping checkpoint tensors.
- `eval/predictor`: entries formerly used hold-speed callbacks and queue mode forced
  HDVs. ON requires an explicit same-arm feedback callback and retains observable
  queue CAV shares. No silent fallback for an unsupported arm.

K-sample CRNs: fixed draw seed, identity-keyed lane/arrival streams and persistent
merge keys; RNG scopes isolate rescoring from previous model loads/calls. Old
LRDIAG OFF rows are reusable only after full-horizon equality verification.

Pairs300 generic-random arms bind the recorded branch algorithm, episode parameters
and arm seed from the converted observable manifest. The source GenericRandomPolicy
is reused without importing/running a simulator; its reseed resets latents/segments,
and new tracks use predicted measurements. Existing calibrated lane/safety execution
is shared with the policy bridge. Unsupported direct arms fail explicitly.

Converted C4/pack loaders carry a separate `entry_feat_control`, derived only from
observable pre-t0 insertions and transmitted CAV identity. OFF continues to use the
unchanged legacy features; ON selects these features when queue shares are absent.
The controller bridge reports recorded t0 commands and executed lane intents on
the ordinary one-second communication clock, never discarded feedback actions.
Entrants keep ordinary bridge history. Both merge-owner masks are preserved.

`deterministic_crn` additionally keys imagination's no-queue candidate identities
across counterfactuals and offsets arrival/lane draws by K-sample index. OFF retains
the original streams. Evaluation accepts explicit callbacks or hash-bound specs;
unambiguous native arms resolve automatically, unsupported direct arms fail closed.
