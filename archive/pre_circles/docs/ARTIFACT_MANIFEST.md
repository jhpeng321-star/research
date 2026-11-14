# Experimental artifact manifest

`runs/` is intentionally excluded from Git. It currently contains historical
and active artifacts, so directory presence alone is not evidence of validity.

## Active IDM foundation

- `runs/eval/idm_analytical_pilot_*` — analytical controller pilots.
- `runs/eval/idm_transition_scan_v1` — transition-demand scan.
- `runs/eval/idm_phase5_*` — reward, observation, safety, and endpoint audits.
- `runs/eval/idm_residual_*` — residual controller and checkpoint evaluations.
- `runs/rllib/idm_phase4_*` and `runs/rllib/idm_residual_phase4_stage1_100k` —
  development/smoke training, not an accepted paper checkpoint.

## Active wall and escalation development

- `runs/eval/idm_wall_*` — IDM wall timing, feature, and multi-window pilots.
- `runs/probe/counterfactual_*` — paired wall/treatment/predictor development.
- Current row validity requires the treatment-dose and branch-identity contract
  declared by the matching current protocol. Older rows that lack it must not
  be pooled into new predictor training.

## Historical artifacts

- most non-IDM `runs/rllib/shepherd_*`, `wall_supervisor_*`, and `champ_*` runs;
- `runs/eval/fair_safe_*`, `shepherd_*`, `hybrid_meta_*`, and `noop_cache_*`;
- `runs/probe/metastable_feasibility/` and older autopsy/gate outputs.

These remain for provenance and prior-paper comparison. Their consolidated
findings are in `LEGACY_RESEARCH_RECORD.md` and `RESEARCH_OVERVIEW.md`.

## Storage policy

- Preserve run directories cited by a current protocol or manuscript table.
- New runs must include configuration, seeds, schema/contract version, source
  revision when available, and a completion summary.
- Smoke/development/locked-test roles must be explicit in the artifact metadata.
- Never reuse inspected development seeds as prospective paper-test seeds.
- Temporary logs and rendering intermediates belong under ignored `tmp/`, not
  at repository root.
