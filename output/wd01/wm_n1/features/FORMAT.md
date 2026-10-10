# WM-N1 featurizer — FORMAT

Frozen tensor layout for the WM-N1 memmap dataset (design
`output/wd01/wm_n1/WM_N1_DESIGN.md` sections 1, 3, 4; registration
`REGISTRATION_WM_N1.md` section 3).  Implemented by the package in this
directory; **zero SUMO/TraCI**, pure offline reconstruction over frozen
physics logs.

Every shard is one run (`<chunk>/<world>/<arm>`).  Arrays are `.npy`; the big
ones are written with `numpy.lib.format.open_memmap`.  All times are seconds;
the physics tick is `DT = 0.5 s`.

## 0. Provenance and honesty

* Read only from `<arm>/run/physics_steps.jsonl` the fields `lane_id`,
  `lane_position_m`, `speed_mps`, `acceleration_mps2`, `road_id`, and the tick
  lists `departed_ids` / `cav_ids`.
* **Excluded inputs** (never featurised, tested): `type_id` (driver/vType),
  `speed_mode`, true penetration, `pre_fix`, world strings, seed, `t0`.  The
  one exception is `is_cav` (CAVs self-identify to the RSU).
* CAV actions come ONLY from `output/wd01/partB_v0data/actions_v04.py`
  (`kind == "command"` rows for the token fallback; the PLANNED action from
  `switch_schedule.json` is authoritative per the owner-confirmed **v0.4a**
  rule, so a PI-external-U CAV is `pi`, not the `damping` writer owner).
  Wall tokens come from `wall_tokens_v04.py`.
* The per-tick dispatcher action log and the realised-insertion log are
  **never opened** (static + functional open-guard tests).

## 1. `s_down`, `lane_phys` and junction conventions

`lane_phys` (WM-30 / `partA_v2g/adapter.py`):

| lane_phys | name | SUMO edges (`edge_lane`) | segment starts (m) |
| --- | --- | --- | --- |
| 0 | main_left | `1191554636_1`, `1487606179_2`, `1487606178_1` | 0, 2645.67, 2721.11 |
| 1 | main_right | `1191554636_0`, `1487606179_1`, `1487606178_0` | 0, 2644.22, 2719.66 |
| 2 | accel | `11553048#1_0`, `1487606179_0` | 0, 378.66 |

Physical-lane coordinate `s = segment_start + lane_position_m`.  The stored
feature is the Part-A downstream coordinate

```
s_down = s - lane_offsets[lane_phys],  lane_offsets = (2645.67, 2644.22, 378.66)
```

so `s_down = 0` at the merge-edge start and is negative upstream.  Junction
internal lanes (`:`-prefixed) get `lane_phys = -1` (lane one-hot all zero),
`junction = 1`, and the vehicle's **last physical `s_down`** carried forward
(a documented approximation; `pdl` uses the junction boundary `s`).  The
mapping table is unit-tested (`tests/test_geometry.py`).

Entrances: mainline entry edge `1191554636` -> counter entrance 0; ramp entry
edge `11553048#1` -> counter entrance 1.

## 2. Vehicle-step tensor — `veh_step.npy`

dtype **float16**, shape **[N_ticks, 128, 16]**; `veh_mask.npy` uint8
`[N_ticks, 128]` (1 = slot occupied).  `N_ticks` covers the run from
`t_origin` (`0` for `base`/`native_base`, else `t0`) to the last tick.

| idx | feature | meaning |
| --- | --- | --- |
| 0 | `s_down_m` | downstream coordinate (§1) |
| 1 | `lane_main_left` | lane_phys 0 one-hot |
| 2 | `lane_main_right` | lane_phys 1 one-hot |
| 3 | `lane_accel` | lane_phys 2 one-hot |
| 4 | `v_mps` | speed |
| 5 | `a_mps2` | acceleration |
| 6 | `nominal_length_m` | 5.0 m (`params_nominal.json`) |
| 7 | `is_cav` | 1 for a designated CAV (from `cav_ids`) |
| 8 | `action_native` | current CAV action one-hot (`native`) |
| 9 | `action_damping` | (`damping`) |
| 10 | `action_pi` | (`pi`) |
| 11 | `action_wall` | (`wall`; set for active wall members) |
| 12 | `action_unknown` | (`unknown`; no evidence — never a silent native) |
| 13 | `wall_member` | 1 while locked by an active wall (requested/assembling/holding) |
| 14 | `junction` | 1 for a junction internal lane row |
| 15 | `present` | 1 for an occupied slot (redundant with `veh_mask`) |

**Slots.**  A persistent pool of 128 concurrent slots with reuse
(`slot_index.npy` int32 `[N_ticks, 128]`, value = index into `slot_vid` or
`-1`).  A continuing vehicle keeps its slot; a slot freed by a leaving
vehicle is reused.  `slot_vid.json` is the run's vehicle-id table
(first-appearance order).  More than 128 *concurrent* vehicles drops the
excess (counted in `meta.n_dropped_slot_steps`).

## 3. Plan tensor — `plan.npy`

dtype **float16**, shape **[N_dec=20, 128, 19, 8]**; `decision_t.npy` float64
`[20]` = `t0 + 30*k`, k = 0..19.  Segment `s` covers `[t + 30s, t + 30s + 30)`
for 19 segments (570 s horizon).  The 128 slots are the decision tick's
slots; only designated CAVs carry a token.

| plan idx | feature |
| --- | --- |
| 0-4 | planned action one-hot `{native, damping, pi, wall, unknown}` |
| 5 | `wall_id_index` (index into `meta.wall_ids`, -1 = none) |
| 6 | `wall_speed_mps / 40` |
| 7 | `planned_hold_remaining_s / 600` |

The decision tick is `round((t_dec - t_origin)/DT)` (nearest present tick if
that grid tick is empty).

## 4. Wall tokens — `wall.npy`

dtype **float16**, shape **[N_ticks, 4, 11]**; `wall_mask.npy` uint8
`[N_ticks, 4]`.  Fields follow `wall_tokens_v04.WALL_TOKEN_FIELDS`, events
`<= t` only:

`0-4` phase one-hot `{requested, assembling, holding, released, aborted}`,
`5` speed/40, `6` time in phase/600, `7` planned hold remaining/600,
`8` n_members/4, `9` members mean s_down/1000, `10` members lane span/2.

## 5. Counter history — `counter.npy`

dtype **float16**, shape **[N_ticks, 2, 30]**.  Entrance 0 mainline, 1 ramp;
bin `b` covers `[t - 300 + 10b, t - 300 + 10(b+1))`.  Counts come only from
`departed_ids` with departure time `<= t`.  For a branch arm (physics starts
at `t0`) the shared pre-restore departure history is merged from the base
arm's log (`--base-history`, default on in the CLI).

## 6. Step targets

* `accel_next.npy` float16 `[N_ticks, 128]` — next-tick acceleration of the
  **same vehicle** in the slot (`target_mask.npy` uint8 `[N_ticks, 128]` = 1
  iff that same vehicle is present next tick).
* `lane_next.npy` int8 `[N_ticks, 128]` — class `{0 stay, 1 left, 2 right,
  -1 none}` of the next-tick lane transition; `1->0` left, `0->1` right, from
  lane 2 (accel) is the merge (right).
* `arrivals.npy` int16 `[N_ticks, 2]` — departures per entrance per tick.

## 7. Outcome targets — `outcomes.parquet` (reused read-only)

Rows are copied verbatim from the matching v0.4a label source:

```
E:/research-dense/output/wd01_partB_v0data_v04a/worlds/<chunk>/<world>__<arm>_labels.parquet
```

**Key join:** `(world, source, t, tau)`, with `source == arm` and `t` in
`decision_t`, `tau` in `{30, 60, 120, 300, 570}`.  The label window
convention (base `t0` row on the branch interval) is preserved verbatim; no
target is recomputed.  `--outcomes-root` overrides the root (tests).

## 8. Split guard and pair index

* The split of a world is read from its chunk `WORLD_PLAN.json` **before any
  run file is opened**.  HOLDOUT worlds are refused, as are worlds in
  `protocols/holdout_exclusions.json` (via `tools/holdout_guard.py`).
  TRAIN and CALIBRATION are allowed.
* chunk-5 (sealed TEST) needs `--final-test` AND `--registration-sha <the
  LF-normalised sha256 of REGISTRATION_CHUNK5_TEST.md, i.e.
  f8c7d760…4b31e>` (the CRLF-safe gate constant in `splits.py`).
* `INDEX.json` groups shards by `pairs[<world>@t0=<t0>]` = sibling arms of the
  same world at the same decision time.

## 9. Writer

Per-run `.npy` shards + `INDEX.json` + `SHA256SUMS`.  `DONE.json` records the
input physics sha256 and every output sha256; a completed shard is skipped
(resumable by sha), and a non-empty shard directory without a valid
`DONE.json` is never overwritten (`--force` rebuilds it).  `--workers N`
featurizes runs in parallel; `estimate` prints expected bytes without opening
a run; `verify` re-checks every shard's shas against the index.

## 10. Micro-smoke evidence (chunk-4 TRAIN, ≤2 runs, 1 process, BelowNormal)

Command (cwd `output/wd01/wm_n1`, 1 worker):

```
python -m features featurize --chunk chunk4 --worlds w8623m2600p02t540 \
    --arms base,native --data-root output/wd01/wm_n1/features/data --workers 1
```

stdout:

```
featurize: runs=2 written=2 skipped=0 wall=24.1s
  [written] chunk4/w8623m2600p02t540__base n_ticks=2281 bytes=13300267 (13.300 MB) pair_key=w8623m2600p02t540@t0=540
  [written] chunk4/w8623m2600p02t540__native n_ticks=1201 bytes=7388218 (7.388 MB) pair_key=w8623m2600p02t540@t0=540
```

`python -m features verify` -> `{"ok": true, "n_runs": 2, "errors": []}`;
`INDEX.json` pairs group both arms under
`w8623m2600p02t540@t0=540`.

**Extrapolation.**  Measured: 13.300 MB (base, 1140 s) and 7.388 MB (branch,
600 s); 24.1 s wall for the two runs (≈12 s/run on this laptop, one worker,
including hashing the physics).  The `estimate` mode reports 5.337 GB for
chunk-4 TRAIN+CAL (648 runs); the design's full chunks 1-4 + J3/J4 corpus
extrapolates to roughly 20-25 GB uncompressed, i.e. several hours at 4
workers (the separate registered featurization allocation).

## 11. Revision 2 — v3 format: K_max is a format parameter, v3 = 192

**Status: registered 2026-10-01 (CODE PREP, branch `zcode/kfix192`),
amending sections 2, 3, 6 and the module docstrings.**  Implements option
"K = 192" of `REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md` section 4.
Supersedes the single fixed `K_max = 128` of sections 2-6 for NEW shards;
the v2 text above remains the frozen description of every existing v2/v2c
pack.

* **Rule.**  The slot width `K_max` follows the *format*, not a global
  constant: `v2 = 128` (frozen, byte-reproducible) and `v3 = 192`.  The
  slot-pool rule itself is unchanged (persistent pool, reuse, drop the
  excess beyond `K_max` concurrent and count it in
  `meta.n_dropped_slot_steps`); only the pool size follows the format.
  Registered pairs live in `features/constants.py`
  (`FORMAT_K_MAX = {v2: 128, v3: 192}`, `resolve_k_max`); `model/constants.py`
  mirrors them (`model.tests.test_contract` pins both).
* **v2 is unchanged.**  The default featurization path (`build_shard`
  without `format`/`k_max`, `features featurize` / `corpus featurize-all`
  without `--format`) still produces byte-identical v2 shards: `K = 128`
  shapes, `meta.json` schema `wm_n1_features_format/1` and no
  `format`/`k_max` keys.  A pinned synthetic fixture
  (`features/tests/test_format_k192.py::test_v2_fixture_is_still_byte_identical`)
  freezes the pre-change output sha256 of every v2 array.
* **v3 shards self-describe.**  A shard featurized with
  `--format v3` (`build_shard(..., format="v3")`, K = 192) writes
  `meta.json` with schema `wm_n1_features_format_v3/1` plus explicit
  `"format": "v3"` and `"k_max": 192` keys; all other tensors, dtypes and
  conventions are exactly sections 1-8 with 192 in place of 128.
* **Packs.**  `corpus.pack` derives `k_max` from the shards (raising on any
  mixed-K tree), records `"format": "v3"` in both `manifest.json` and
  `INDEX.json` when the pack K is a non-default registered format, and
  `corpus.verify` re-checks the K contract (manifest/index agreement, slot
  dims of every slot-axis array, format↔K consistency) and reports
  `k_max`/`format`.  v2 packs verify exactly as before.
* **Read side.**  `train.dataset.MemmapCorpus` resolves and validates the
  pack `k_max` (manifest vs packed slot dims, declared `format`, and the
  optional `k_max=` argument): a K mismatch between a pack and the
  model/config (e.g. a v3 pack opened with `k_max=128`) is refused with
  `CorpusRefusal`.  `IdeaATrainConfig.k_max` (amendment 5 parameterisation)
  threads the model-side K; `None` takes the pack's K.
* **Everything downstream is K-carried, not hardcoded.**  The model net, HDV
  residual net, rollout2 world state, adapter predictor and Idea A channels
  take K from the tensors; the timing gates size scenes via
  `n_vehicles` (`make_config(format="v3")` selects 192), and
  `bundle.envcheck --format v3` reports the K=192 saved-activation memory
  estimate (`KMAX_OPTIONS.md` section 2 fit).
* **Cost anchor (KMAX_OPTIONS.md):**  the full-corpus v3 pack is ~52.05 GB
  (vs 35.07 GB at K=128); the timing gate must be re-run on pack_v3 before
  member-1 training (amendment 5 section 4).
