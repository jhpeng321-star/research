# Data formats

This document describes the data that flows from SUMO into WM-S training and C4 imagination: raw
episode recordings, paired branches, the converted `c4_episodes` sources, training windows and
batches, the legacy packs and evaluation states, the data registry, and the seed blocks.
**No data is shipped with the repository**; datasets are addressed by logical name and resolved
to local copies (section 6). Paths are relative to the repository root; where a design note and
the code disagree, the code is authoritative.

## 1. Shared conventions

- **Model frame** (`traffic_shepherd/c4/geometry.py`, the same tables the packs were built
  with): lane 0 = left mainline, 1 = right mainline, 2 = ramp/acceleration lane, -1 = no
  modelled lane (absent, or on a SUMO junction-internal lane, where `s` is carried from the last
  modelled position). `s` is merge-relative: merge zone `[0, 75.44]` m, upstream negative.
- **Slots.** A persistent slot pool of `K` columns (default 192); a vehicle keeps its slot while on
  the network, and slots are reused after it leaves. `target_mask` marks that the same vehicle is
  still in the slot at the next tick.
- **Timing.** Physics/recording every 0.5 s; one control decision per second. Control row `c` is
  decided on state index `2c` and holds for states `2c+1, 2c+2`.
- **Kinematics vector** `veh_kin [.., 7]` = (`s`, lane one-hot x3, `v`, `a`, length); length is the
  nominal 5 m in packs and converted sources.
- **Controller ids** (`src/wms/data/action_timeline.py`): native 0, damping 1, pi-external-u 2,
  direct 3.
- **World ids**: `w<seed>m<demand veh/h>p<penetration %, 2 digits>t<warmup s>`, e.g.
  `w4107m2700p20t300` (`src/wms/eval/states.py::world_meta` parses them).

## 2. SUMO episode recordings (`traffic_shepherd/c4/recorder.py`)

`python -m traffic_shepherd.c4.collect` writes one write-once directory per episode:

```
manifest.json                 status (running | complete | failed), identity, config; schema c4_sumo_episode/1
summary.json                  SUMO outcomes and wall-clock speed (complete episodes only)
episode.npz                   full-episode slot arrays (below)
run/physics_steps.jsonl.gz    0.5 s rows, schema wm41_logging/1 (omitted with --no-physics-log)
run/action_timeline.jsonl.gz  1 s per-CAV issued commands, schema traffic_shepherd_action_timeline/1
scenario/                     requests.rou.xml + scenario.sumocfg
```

`episode.npz` (`T` states including t = 0, `C` control ticks):

| array | shape / dtype | content |
|---|---|---|
| `time_s` | `[T]` | state times |
| `s_m`, `v_mps`, `a_mps2`, `length_true_m` | `[T,K]` f32 | model-frame kinematics; `a_mps2` is the realised (SUMO) acceleration |
| `lane`, `lane_carry` | `[T,K]` i8 | lane (-1 absent/junction), carried lane |
| `present`, `is_cav`, `junction` | `[T,K]` bool | masks |
| `vid_index` | `[T,K]` i32 | index into `vehicle_ids [V]`; `cav_vehicle_ids` lists the CAVs |
| `control_time_s` | `[C]` | decision times |
| `controller` | `[C,K]` i8 | controller id per CAV slot |
| `issued_acceleration_mps2`, `issued_target_speed_mps` | `[C,K]` | command after the environment's speed-bound clip |
| `requested_acceleration_mps2` | `[C,K]` | raw policy output (NaN where none) |
| `lane_command_target`, `lane_request_target` | `[C,K]` i8 | issued / requested target lane (-1 none) |
| `meta_json` | | seed, demand, penetration, warmup, horizon, controller, slot count, CAV speed mode, `dt_s`, `control_dt_s` |

Requested, issued and realised accelerations are therefore all recoverable and generally differ.

**Collector layout.** `<out>/episodes/<policy_tag>/s<seed>_m<demand>_p<penetration x 1000, 4
digits>/`, plus `<out>/INDEX.json` (all jobs) and `<out>/COLLECT_SUMMARY.json` (measured SUMO
speed). Collection is resumable: complete episodes are skipped, an incomplete directory is kept as
`<job>.incomplete-<n>` and the job re-runs. `--case-plan product` runs every seed x demand x
penetration; `weighted-cycle` runs one world per seed, cycling penetrations by `--pen-weights`.
`--population textbook` draws continuous per-vehicle IDM/LC2013 HDV parameters
(`traffic_shepherd/c4/population.py`); `legacy3` keeps three discrete driver types.

## 3. Paired branches

**Random-action branches** (`traffic_shepherd/c4/genrand.py`, `collect --policy generic_random
--branch-cadence <s>`). A base episode runs uninterrupted; at each decision time the full SUMO
state (`saveState` with full precision and RNG state) and a deep copy of the environment's Python
state are saved. From each saved state, several arms run for `--branch-horizon` seconds with the
same SUMO RNG state: re-seeded random-command arms and an all-native reference arm. Each arm is
written as an ordinary `c4_sumo_episode/1` directory
`<out>/branches/<policy_tag>/<job>/t<TTTT>_{native|seed<j>}/` containing the base pre-history
plus the branch, with `manifest.branch = {pair_key, base_world, branch_t_s, arm, kind, ...}` and
`pair_key = <world>@t<branch time>`. `BRANCHES_COMPLETE.json` marks a finished job.

**Cooperation measurement** (`traffic_shepherd/c4/branching.py`). The same save/restore mechanism
branches any controller into `B` (all native, plus a repeat for determinism), `ALL`, `ONLY:i`,
`GROUP` and `FLIP:i` arms and writes `branches.jsonl`, `points.jsonl` (per decision point: the group,
per-arm metrics and the improvements `D_X`, positive = better than `B`, with `C_G`, `M_i`, `I_i`),
`BRANCHING_SUMMARY.json`, `MANIFEST.json` and the saved states. This is an evaluation tool; its
outputs are not training data.

## 4. Converted sources (`src/wms/data/c4_episodes.py`)

```
python -m wms.data.c4_episodes --episodes <collect dir | episode dir> [...] --out <new dir> \
    [--val-frac 0.2] [--split-seed 0] [--allow-designed]
```

The converter takes every complete episode under the roots and writes (write-once)
`<out>/C4_EPISODES_MANIFEST.json` (schema `wms_c4_episode_source/1`, split `TRAIN`, slot count,
controller vocabulary, split rule, and one entry per episode with world, arm, seed, demand,
penetration, warmup, counts, source sha256, optional `branch`/`pair_key`/`genrand`, and
`split_role`) and one memory-mappable directory `<out>/episodes/<world>__<arm>/` of `.npy` arrays:

| array | shape / dtype | content |
|---|---|---|
| `veh_kin` | `[T,K,7]` f32 | kinematics (junction/absent rows have a zero lane one-hot) |
| `veh_mask` | `[T,K]` u1 | slot occupied (junction transits included) |
| `target_mask` | `[T,K]` u1 | same vehicle in the slot at the next tick |
| `counter` | `[T,2,30]` f32 | entry counts in 10 s bins over the last 300 s (bin 0 newest; entrance 0 mainline, 1 ramp) |
| `vid`, `is_cav` | `[T,K]` | episode-local vehicle index (-1 empty), CAV mask |
| `time_s`, `control_time_s` | `[T]`, `[C]` | times |
| `controller`, `issued_acceleration_mps2`, `issued_target_speed_mps` | `[C,K]` | per-CAV action timeline |
| optional | `[C,K]` | `requested_acceleration_mps2`, `lane_command_target`, `lane_request_target` (stored, not read by the loader) |

Rules: the train/validation split is by **world** (`ceil(frac x n_worlds)` worlds with the smallest
`sha256(seed:world)` are validation), so one demand/driver draw never straddles the split;
episodes whose timeline contains damping or PI are refused unless `--allow-designed`; all episodes
must share one slot count. Branch episodes keep the base world id and carry a unique arm tag.

**Use in training.** A WM config lists sources under `[[data.c4_sources]]` (`name`, logical `path`,
`mix_ratio`, `window_stride_s`, `start_s`, `use_val`, `paired`); `data.c4_base` names the source
that replaces the legacy pack data, and its `mix_ratio` must equal one minus the others. A world
that is validation in any source is validation in all of them (`src/wms/data/c4_pairs.py`).

## 5. Windows and batches

Windows are cut at load time (`C4WindowLoader`): `t0` every `window_stride_s` seconds from the
episode's warmup end (or `start_s`), horizon = the longest curriculum unroll. A record holds
`states [T+1,K,7]`, `present` (alive since t0: the cumulative product of `target_mask`, so
mid-window entrants are never added and reused slots never revive), `loss_valid` (junction steps
excluded), `cav_teacher` (recorded realised CAV motion), short `hist` and long `driver_hist`
built only from ticks `<= t0`, optional `entry_feat` (observed pre-t0 entry history), and
`action_timeline` (`controller`, `acceleration_mps2`, `target_speed_mps`, `control_dt_s`) bound to
the t0 slot columns. `src/wms/data/collate.py` stacks records into batches; optional keys appear
only when every record has them. Paired batches (`loss.paired_diff`) group one window per arm of a
`pair_key` at the branch time and add `pair_group` / `pair_ref`; arms whose t0 slots or
kinematics disagree are dropped.

**Legacy packs** (`src/wms/eval/states.py::Pack`, `src/wms/data/corpus.py`): memory-mapped corpora
with `manifest.json`, `INDEX.json`, `veh_kin [N,K,7]`, `veh_mask`, `veh_step [N,K,16]`
(column 7 = `is_cav`) and `target_mask`, plus identity sidecars and window lists
(`wm_s_lpo_windows/1`). Older configs train on them; the current C4 configs use converted sources
only. **Evaluation states** are JSON records of an initial frame and the recorded future frames per
arm, where a frame is `{"time_s", "vehicles": [{"id", "s", "lane", "v", "present", "is_cav"}]}`
(`src/wms/eval/normalize.py`); `traffic_shepherd/c4/wallregen_states.py` writes unseen-wall sets in
this format.

## 6. Data registry

`src/wms/data/registry.py` is the only code that maps names to paths. A name is
`<dataset>[/<relative path>]`, resolved in this order:

1. environment variable `WMS_DATA_<DATASET>` (upper case, non-alphanumerics -> `_`; e.g.
   `genrand_plain` -> `WMS_DATA_GENRAND_PLAIN`, `v6_policy` -> `WMS_DATA_V6_POLICY`);
2. `configs/data_registry.toml` (`[datasets.<name>] path = "..."`, relative to the repo root);
3. `tools/data_registry.json` (`local_path` below `data_cache/`, which is git-ignored).

Absolute paths are taken as resolved. Every result passes the restricted-token guard
(`src/wms/core/guard.py`), which refuses names of sealed evaluation splits; trainers fail closed
when a declared input is missing. `tools/data_registry.json` (schema in
`tools/data_registry.schema.json`) records, per dataset, its remote location, size, format and an
md5 manifest; `python tools/fetch_data.py list | status [name] | verify [name] | get <name>`
(also `python -m wms fetch ...`) copies a dataset with `rclone` into `data_cache/` and verifies
it. The remote locations are private to the original project, so external users should point
`WMS_DATA_*` variables or `configs/data_registry.toml` at their own copies, e.g. converted sources
produced by the commands above.

## 7. Seeds (`traffic_shepherd/c4/seeds.py`)

Every SUMO case passes `check_case(seed, demand)` before any simulation. Allowed blocks, each with
one role:

| block | role |
|---|---|
| 4107-4148 | exposed controller-development seeds |
| 961000-961999 | undirected random-action WM training data |
| 962000-962199 | regenerated unseen-wall evaluation states (never training) |
| 963000-963999 / 964000-964199 | textbook-population training / evaluation |
| 965000-965999 | long paired-branch training data |
| 966000-969999 | Dreamer-loop training collections |

Sealed acceptance, test and validation blocks are refused explicitly, and module-level assertions
check that training and evaluation blocks are disjoint. The loop orchestrator additionally limits
itself to its own training range and refuses evaluation seeds. New experiments need a new
disjoint block registered in `traffic_shepherd/seeds.py` and `traffic_shepherd/c4/seeds.py`.
