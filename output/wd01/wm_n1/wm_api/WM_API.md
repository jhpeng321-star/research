# WM_API — stable WM scoring API for the planner pilot

**api_version: `1.0`** — the version returned in every `ScoreResult`.
Backward-incompatible schema changes bump the version and are documented
here.  Package: `output/wd01/wm_n1/wm_api/` (import root = the `wm_n1`
directory).  Zero SUMO; the scoring engine is the composed Idea A bundle
(``train.idea_a.build_idea_a_bundle`` modules) driven by the ``rollout2``
engine + the P5 correction head — no rollout2/train internals were
modified.

## 1. Interface

```python
from wm_api import WmScorer, StateSnapshot, state_from_physics_tick

scorer = WmScorer("ckpt/")          # sha-checked member-checkpoint dir
result = scorer.score(
    state,                          # StateSnapshot or its JSON mapping
    candidate_plans,                # non-empty list; plan 0 = reference
    horizon_s=300,                  # 300 s by registration (see below)
    members=None,                   # 0-based subset of loaded members
    n_samples=None,                 # rollout repetitions (default 1)
    seed=None)                      # default 0 (deterministic)
result.to_json()                    # JSON-serialisable dict
```

Deterministic CPU mode: 1-2 torch threads (`WmScorer(..., threads=2)`),
float64 engine math, and a per-(member, sample) rollout rng
`numpy.random.default_rng(seed + 1009*member0 + 7919*sample)`.  A fixed
seed reproduces the response bit-for-bit on the same machine.

### Refusals (all raise a `wm_api.errors.WmApiError` subclass)

| refusal | error | meaning |
| --- | --- | --- |
| unknown action | `UnknownActionError` | an action outside {native, damping, pi, wall} after alias canonicalisation (`pi-external-u`→`pi`, `wall-v2`→`wall`, ...) |
| missing checkpoint | `CheckpointNotFoundError` | directory or `member_*.pt` absent, or a SHA256SUMS-listed file missing |
| sha mismatch | `CheckpointShaMismatch` | a file hash != its `SHA256SUMS` line |
| K overflow | `KOverflowError` | vehicles + obstacles > the frozen `K = 128` slot pool — **reported, never silently dropped** |
| other horizon | `HorizonError` | `horizon_s != 300` without `allow_other_horizon=True` (then the outcome head reports its NEAREST registered tau, in `result.tau_s`) |
| bad state / plan | `StateSchemaError` / `PlanSchemaError` | schema violations (below) |

## 2. `state` — the per-vehicle t0 snapshot

| field | type | meaning |
| --- | --- | --- |
| `ids` | list[str] | vehicle ids (slot order) |
| `lane_phys` | list[int] | 0 main_left / 1 main_right / 2 accel; **-1 = excluded** from the model world (junction rows are never guessed) |
| `s_down_m` | list[float] | merge-relative downstream coordinate (0 at the merge start, negative upstream; `features.geometry` convention) |
| `v_mps`, `a_mps2` | list[float] | speed / last-step acceleration |
| `length_m` | list[float] | bumper-to-bumper length (5.0 m nominal) |
| `is_cav` | list[bool] | CAV flag (CAVs self-identify) |
| `t0_s` | float | snapshot time (s); control ticks fire on integer seconds |
| `entrance_counters` | float[2,30] *optional* | departure counts per entrance (0 mainline, 1 ramp), 10 s bins over the last 300 s, bin 0 most recent. **Present → the P4 arrival head runs in the loop** (seeded); absent → closed rollout |
| `obstacles` | list *optional* | held stopped vehicles `{id, lane_phys, s_down_m, length_m}` — pinned each step at `s`, `v = 0`, `a = 0` (the INC1 held-obstacle convention) |

Converter from the chunk-4 physics-log tick format (one parsed
`physics_steps.jsonl` row):

```python
from wm_api import state_from_physics_tick
from wm_api.state import counters_from_ticks

state = state_from_physics_tick(tick)          # {"time_s", "vehicles":
                                               #  {id: {lane_id, lane_position_m,
                                               #  speed_mps, acceleration_mps2,
                                               #  road_id}}, "departed_ids",
                                               #  "cav_ids"}
state.entrance_counters = counters_from_ticks(tick_history, t_now=540.0)
```

Only the frozen fields are read; junction/unknown-lane rows are EXCLUDED
and reported back in `state.excluded` (never silently dropped).  CAV flags
come from the tick's `cav_ids` list.

## 3. `candidate_plans` — the planned schedule (never an arm name)

Per ACTION_DESCRIPTOR_REQUIREMENT_20261001.md the action descriptor IS the
planned schedule: the per-CAV controller for each 30 s segment of the
scoring horizon, plus the wall hold / speed / cap.  "Damp 60 s then native"
and "damp the whole window" are different inputs.

```json
[{"actions": {}},
 {"actions": {"cav0": [{"time_s": 0.0, "action": "damping"},
                       {"time_s": 60.0, "action": "native"}],
              "cav1": "damping"}},
 {"actions": {},
  "wall": {"wall_id": 1, "speed_mps": 6.5, "hold_s": 120.0,
           "members": ["cav0", "cav1"], "onset_s": 0.0}}]
```

* A per-CAV value is one static action or a list of `{time_s, action}`
  events (times relative to `t0_s`).  Segment `k` — `[t0+30k, t0+30(k+1))`
  — runs the latest event at or before its start, default `native` (the
  `switch_schedule.json` convention).
* `wall` = wall-v2.2 GROUP membership + params: members run the wall law
  from `onset_s` for `hold_s` (HOLD phase from t0; assembly is not
  modelled) with the v2.2 merge release / hold cap applying inside the
  mover.  A wall member must not also carry its own schedule.  The frozen
  a-priori proposal gates (2-abreast + span ≤ 75 m, `D_min = 70 * v_wall`
  on the lead member, hold cap) are REPORTED per plan under
  `plans[i].wall_gates` — informational, never tuned.
* CAVs absent from `actions` stay native.  HDVs keep IDM and receive no
  commands (only CAV ids are plannable).

## 4. `ScoreResult`

Per plan (in candidate order):

```json
{
  "api_version": "1.0",
  "horizon_s": 300.0, "tau_s": 300.0,
  "members": [1, 2], "seed": 0, "n_samples": 1,
  "arrivals_enabled": false,
  "plans": [{
      "plan_index": 0, "descriptor": "cav0:nnnnnnnnnn;...",
      "is_reference": true,
      "metrics": {
        "delay_s_per_veh": {
          "quantiles": {"q05": ..., "q10": ..., "...": ..., "q95": ...},
          "mean": 12.3, "member_draws": [12.1, 12.5]},
        "ttc3_exposure_s": {...},
        "hard_brake_s": {...}},
      "ood_score": 0.87,
      "paired_diff_vs_plan0": {
        "delay_s_per_veh": {"quantiles": {"q05": ..., "q95": ...},
                            "mean": 0.0}},
      "wall_gates": {}}],
  "checkpoint": {"directory": "ckpt/", "sha256": {...},
                  "sha_verified": true, "members": [1, 2]},
  "meta": {"timing_s": ..., "validity": {...}, "n_collisions": 0, ...}
}
```

Metric semantics:

* `delay_s_per_veh` — mean delay, s/veh over `(t0, t0 + horizon]`
  (correction-head target `delay_s`, the registered y_delay_s label
  convention); `ttc3_exposure_s` — TTC < 3 s exposure seconds;
  `hard_brake_s` — hard-brake seconds.
* Quantile keys `q05..q95` are the frozen 19-level grid 0.05..0.95.  The
  plan-level curve is the unweighted mean of the member curves (the
  ensemble mixture convention of `WmN1Ensemble.mixture_quantiles`).
* A member "draw" is that member's median (q50) of the metric; `mean` is
  the unweighted mean of the member draws.
* `paired_diff_vs_plan0` — comonotone difference at the SAME levels
  (`q_plan_i - q_plan_0`; `mean = mean_i - mean_0`).  **Sign convention:
  cost difference = treatment − reference; negative = improvement.**
  Plan 0 is the reference (native by convention); its block is all zeros
  with `"is_reference": true`.
* `ood_score` — larger = less familiar (`corr.ood` conventions).  Default:
  `ensemble_disagreement` over the member CAV-motion summaries.  If the
  checkpoint ships `ood_model.json` (`{"mean": [...], "std": [...]}`, a
  diagonal-Gaussian density fitted OFFLINE on TRAIN summaries), the mean
  member summary is scored with `motion_density_score` instead
  (`meta.ood` says which).

## 5. Checkpoint directory

```
ckpt/
  member_1.pt .. member_k.pt    train.IdeaATrainer payloads
                                (format "wm_n1_idea_a_member/1")
  member_k.json                 sidecars (optional)
  ood_model.json                optional OOD density model
  SHA256SUMS                    optional; "<sha256>  <name>" lines
```

Loading rebuilds each member via its own embedded config and loads the
flat `hdv.* / arrival.* / corr.*` state dict (float64 for the rollout
engine).  Every member file's sha256 is computed and returned in
`result.checkpoint.sha256`; a `SHA256SUMS` file turns this into a
verification (any mismatch or missing listed file refuses).  The CIRCLES
damping/PI ONNX payload is sha-pinned inside the controller mirrors
(`ctrl.config.CIRCLES_MODEL_SHA256`) and re-checked on first use.

## 6. CLI

```
python -m wm_api score --ckpt ckpt/ --request req.json
python -m wm_api score --ckpt ckpt/ --request req.json --out resp.json
python -m wm_api score --example          # print an example request
```

Request JSON: `{"state": {...}, "candidate_plans": [...], "horizon_s": 300,
"members": null, "n_samples": 1, "seed": 0}`.  Response: the ScoreResult
JSON on stdout (or `--out`).  Refusals exit with code 2 and print
`{"error": {"type", "message", "api_version"}}`.  CPU-only, 2 torch
threads, BelowNormal process priority on Windows (best-effort).

## 7. Versioning

* `api_version "1.0"` is this document's contract: the `score` signature,
  the state/plan schemas, the metric/quantile conventions and the refusal
  set.  Additive optional request fields do not bump the version.
* The scoring internals (checkpoint format `wm_n1_idea_a_member/1`, the
  frozen `rollout2` engine, the P5 heads) may evolve independently; any
  change that alters returned values under a fixed seed bumps
  `api_version`.
