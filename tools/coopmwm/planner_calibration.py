"""T1C: CPU-only calibration from existing, identity-checked verification receipts.

Code-level source contract (c4/track-base 13f184b5):
* sumo_verify.verify: predicted_saving[i] comes from saving_vs_policy, falling
  back to saving_vs_mappo; realised gain is J_policy_veh_s-J_candidates_veh_s[i].
  Read ALL graded candidates of accepted AND rejected decisions, never just
  chosen_plan/grade_veh_s (which would select positive outcomes).
* mppi_runtime.label_list8_search and mppi_search final scoring: saving_vs_policy
  is WM mean policy cost minus candidate cost. It is NOT a probability; p_best
  is also not P(candidate beats policy). Fit gain -> empirical P(gain_real > 0).
* verify_perseed.confirm_all8: confirmations_all8 carries first_seed and seed2
  receipt identities, optional extra receipt identities, and per_seed_v1.examples.
  Follow group receipt identities, not admitted-only examples or solo/extended
  outcomes. replay_bank.replicate changes post-t0 RNG, not episode.world.seed:
  split all replicas/candidates/updates/arms by their original world seed.

The fitted probability applies to the stored SUMO verification window. Existing
WM and SUMO horizons may differ (see viability_report); report this limitation,
preserve window provenance, and never shorten an online planner's look-ahead.
No SUMO runner, jobs, dispatch, remote access, policy observations or truth data
are needed. Only explicitly supplied files and linked group receipts are read.
"""
