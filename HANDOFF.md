# S-RB-UNC — baseline READY; owner00:40 addendum IN PROGRESS

Fresh-session resume preserves clean baseline46c33fbe and all completed probes.
Next: fitted IDM/optional LC posterior + learned entrance-counter process +
identity-keyed action CRN. Frozen plan: `docs/paper_results/RBUNC_ADDENDUM_DESIGN.md`.
Remote terminal rechecked: old31children exited0; no old dispatch repeated.

Branch `c4/rb-unc`, base `f0db3c82`. Owner registration5683/5694 and WHY +
ENSPROBE read. Merge the branch; retain INT's own root HANDOFF on conflict.
Implementation commits: `1f918969`, `ebea8483`, `9739d304`, `73cb13be`.

## API / switch
- NEW `wms.control.hidden_state.plan_hidden_state_futures`:
  `(model, initial, history_kwargs, future_sampling='hidden_state', n_samples=8,
  seed=190, row_keys=world_keys, entrance_histories=histories, queues=queues)`.
  Returns K `HiddenFuture` objects: `.kwargs` for rollout, `.receipt` provenance.
- Default `future_sampling='off'` -> None, no encoding/RNG work.
- Caller pads first and supplies observed request-demand/backlog/CAV share.
  Create fresh plans/controllers per alternative or replay (entries are mutable).
  Equal row keys + equal starts/pools share drivers/demand/newcomers across arms.
- `sample_present=False` / `sample_arrivals=False` isolate the two latent sources.
- No shared wms/spec/factory/trainer changes. API example: `tools/rbunc/README.md`.

## Semantics / integration detail
- Present HDVs: calibrated supplied posterior samples, else empirical joint
  population spread around point estimates with pre-t0 motion likelihood.
  Existing encoder reliability mixes short histories toward population draws.
  Incumbent untrained uncertainty head ignored; embeddings stay fixed.
- Future requests: exact Poisson clocks from t0 at observed entrance demand.
  Newcomer settings drawn from motion-inferred joint population by birth ordinal.
- **SUMO bridge:** after MERGE configures insertion inside rollout,
  `PoissonEntries.insert` binds only its future-owned insertion `candidate` to
  its SAME weighted `BirthMix.candidate`. No legacy uniform-hash mismatch.
  Actual INT-runtime synthetic insertion/motion identity + replay smoke PASS.
- Known stochastic modules, train mode, runtime-noise/future-observation kwargs
  rejected. Physics remains deterministic within each conditional future.
- Old MOBIL outputs are NOT LC2013 settings; LC defaults stay defaults until
  motion-inferred LC uncertainty exists. Missing donor pools fail closed.

## Tests / complete FIT diagnostics
- Latest new-suite12CPUtestsPASS; earlier legacy-futures13PASS. Single thread.
- All32 native FIT969400..969431, K8/H120, authentic60s history COMPLETE.
  Reference deterministic IDM/MOBIL + frozen SUMOCORE encoder5b256941,
  NOT integrated WM-PHYS efficacy / main EVAL checkpointb8797242.
- Mean within-K team SD574.4 [506.6,644.9] vs real across-world SD4200.5veh-s.
  First-state drivers-only134.1, arrivals-only755.6, both696.7, fixed0.0veh-s;
  fixed futures have identical complete physical tape hashes.
- Queue conservation, finite dynamics, no missing padding/dropped requests,
  no global physics RNG consumption; all280v2 futures pass. Independent CRPS/
  variance/target/source-invariant checks PASS96metric-world cells.
- V1 retained; corrected raw-IDM likelihood to existing executed-plant bounds
  (max residual MSE3.18e10 ->42.51). No factual delay used to tune spread.
- Report `docs/paper_results/RBUNC.md`; final raw/summary/verification and own
  terminal receipts in `docs/paper_results/rbunc/`. All32 complete/31child rc0.

## Scope / remaining integration
- Reference spread is a source-plumbing diagnostic, not calibrated coverage or
  a WM-PHYS win. Full120/300/600s WM-PHYS action/distribution panel stays INT/EVAL.
- Request-clock spread checked120/300/600s; traffic-delay spread here only120s.
- Across-world SD includes observed-state heterogeneity: never tune conditional
  SD to equal it. Recorder `arrivals` are destination exits, not entrance demand.
- No SUMO/GPU/training, prohibited data, downloads, new venvs, deletions, push
  or peer-process actions. CPU nice19 owned workers; both versions preserved.
