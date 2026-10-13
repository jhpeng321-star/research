# WMLANE — lane-change hazard model (A) and driver-setting updating (C)

Branch `c4/wmlane` (from `c4/int` 51cb52d). Both features are behind config switches whose
defaults are OFF; with them off the model, losses, flat config/receipt form and checkpoint
keys are bit-identical to `c4/int` (regression test `test_switch_off_is_bit_identical_to_pre_wmlane_code`
replays a C4-config unroll — loss terms, flip rate and the summed |grad| — against numbers
captured on the pre-WMLANE tree).

Motivation (WM-S-LANECHECK-20261012): with perfect lanes the physics-only 120 s error drops
53 -> 11 m; the residual/MOBIL lane path misses most overtakes and keep-right returns and merges
~8 s early. Second error source: per-car desired-speed drift (1 m/s v0 error = 120 m at 120 s).

## A. `model.lane_model = "hazard"` (default `"residual"`)

* `src/wms/model/lane_hazard.py`: `HazardLaneModel` (MLP 30 + embedding -> 64 -> 64 -> 2,
  final layer zero-initialised, deltas soft-capped at `lane_hazard_cap`, default 10) over
  interpretable features per HDV: for the left and right target lane — valid, front/rear gap,
  front/rear relative speed, speed advantage (target-lane leader speed vs current-lane leader,
  both capped at the car's v0; no leader = v0), physics prior logit and feasibility; ego — v,
  a, desired-speed deficit v0 - v, v/v0, current leader gap/dv, lane one-hot, time since the
  last change (lane age) + never-changed flag, distance to the merge end, in-merge-zone,
  position; plus the car's own learned driver embedding (encoder output).
* Output stays in the absolute-lane `[B,K,3]` format (lane logits = physics prior + hazard
  delta on the two adjacent lanes), so argmax decisions, lane loss, flip-rate guard and every
  diagnostic are unchanged. It REPLACES the gated residual lane delta (the residual net's lane
  outputs are unused when on). Hard feasibility: targets the physics masked (<= -8: unsafe
  gap, ramp entry, non-adjacent) stay masked; inside the last 25 m of the acceleration lane
  the physics forced-merge logit is kept (hazard can delay a merge, never strand a car).
  Zero-init => a fresh hazard model reproduces the physics prior exactly.
* Requires the structured `driver_params`/`hybrid` rollout (embedding); exclusive with the
  old `lane_head="history"` (the `HistoryLaneHead` stub uses relative logits and a different
  mask convention; its features were reused/extended instead of the module).
* Loss `loss.lane_event` (`src/wms/train/losses/lane_event.py`, registered in `build_loss`
  as an unroll term, default weight 0): teacher-forced — the hazard is evaluated on the
  RECORDED state of every HDV-step of the unroll (0.5 s) and scored against the recorded
  transition with focal (`gamma`, default 2) category-balanced CE. Categories stay /
  overtake 1->0 / return 0->1 / merge 2->1 / other; present event categories share a total
  weight equal to the stay weight (`w_c = n_stay / (C n_c)`), each clipped to
  `[1, max_weight=20]`; recorded changes into a physics-masked target are dropped (counted).
  Driver inputs = the same t0 encoding the rollout uses (leaf tensors when
  `driver_grad_across_chunks`), the physics prior is computed without gradient (memory).
  Lane age along the recorded sequence starts from the driver history. The existing
  flip-rate loss keeps calibrating the rollout change rate.
* Public: `WmSModel.lane_event_logits(state, driver_params=, driver_embedding=, lane_age=)`.

## C. `model.driver_update = "ema"` (default `"off"`)

* `src/wms/model/driver_update.py`: `EmaDriverUpdater`. When a car's TRUE state is observed:
  v0 <- v0 + r_v0 * w_free * (v0_hat - v0), with the IDM free-flow inversion
  `v0_hat = v / (1 - a/a_max - (s*/s)^2)^(1/4)`, trusted only in free flow (w_free from
  r, interaction term, speed); T <- T + r_T * w_follow * (T_hat - T), `T_hat = (gap - s0)/v`
  in steady following; embedding += r_e * tanh(W innovation) (W zero-init; innovation =
  observed - predicted v/a, weights, parameter innovations). Rates are learnable sigmoids
  initialised at `driver_update_rate` (0.3). Unobserved slots never change.
* Rollout: `WmSModel.rollout(..., observe_states=[B,T+1,K,7], observe_mask=[B,T+1,K])`
  (index i+1 = after step i). Logged prediction stays the model's own; the carried state of
  observed HDVs is reset to the observation and the update runs; the result carries
  `driver_parameters_updated` / `driver_embedding_updated`. Without observations (open-loop
  imagination) nothing changes (tested equal to the switch-off model). Deployment entry
  point for belief/RSU updates: `WmSModel.update_drivers(params, embedding, state, observed)`.
* Training mode: `model.driver_reveal_prob` (default 0) / `driver_reveal_max_s` (10 s):
  in training unrolls (backward only) each t0 HDV is revealed with that probability for a
  teacher-observed prefix of U{1..max_s/dt} steps; the update path (rates, embedding map,
  and the encoder through the updated settings) is trained by the later free-running loss.
  Updates are carried across truncated-BPTT chunks as detached offsets on the t0 encoding.
* Longer driver history: `data.history_s` already exists. The packs support any duration
  back to the run start (`prior_history` front-masks missing ticks); 120 s is valid —
  see the smoke section for coverage. Not switched on in `c4_wm_v2_lane.toml` (opt in with
  `--set data.history_s=120.0`); the encoder's GRU cost grows linearly with it.

## Config

`configs/c4_wm_v2_lane.toml` extends `configs/c4_wm.toml` with A and C on (lane_event
weight 1.0, reveal prob 0.3). New flat keys (`lane_model`, `lane_hazard_hidden`,
`lane_hazard_cap`, `driver_update`, `driver_update_rate`, `driver_reveal_prob`,
`driver_reveal_max_s`, `lane_event_weight`, `lane_event_gamma`, `lane_event_max_weight`)
are in `OMIT_WHEN_DEFAULT`, so existing receipts/checkpoint configs are unchanged; new
modules (`lane_hazard.*`, `driver_updater.*`) exist only when switched on and are created
after every existing module (key order / RNG stream of old models unchanged).

## Verification

* `tests/unit/test_wmlane.py` (14 tests): switch-off bit identity + flat forms, validation,
  hazard shapes / zero-init = physics prior, hazard never selects an infeasible change
  (extreme +-60 bias; mandatory merge kept), one-step logits == forward, event targets /
  categories / lane age, category weights + cap, loss sign / weighted-CE equality / focal /
  infeasible rows dropped, EMA update reduces a wrong v0 (25 -> true 33 m/s) monotonically,
  rollout observation updates v0 while open loop is a no-op, reveal mask = HDV prefix,
  end-to-end unroll with all switches on (finite, gradients reach hazard and update rates).
* CPU smokes (`tools/wmlane_smoke.py`, the c4_smoke two-update dry run with config +
  overrides and `--data-root E:/research-dense`): see results below.
