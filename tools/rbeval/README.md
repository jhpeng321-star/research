# S-RB-MERGE FIT diagnostics

Run with existing Python + numpy/torch; one CPU thread. No SUMO, training,
driver-truth files, assignment files, or directories outside the exact FIT range.

```
python tools/rbeval/validate_merge.py --root /home/ubuntu/wmdata20261031/views --seeds 969400 969401 --horizon 300 --output receipts/rb_merge_fit.json
```

Reads `world_<FIT seed>.json` then only episode metadata and observable
`time_s.npy`, `veh_kin.npy`, `veh_mask.npy`, `vid.npy`. Refuses prohibited paths,
including symlink destinations, and any seed outside 969400–969431.

- **Recorded-state mechanics:** two-sided merge acceptance after causal
  constant-acceleration motion, versus next recorded lane. Reports confusion
  counts and first accepted times. Entrance admission recall uses the source-IDM
  check at observed births; first visibility only (junction reappearance excluded).
- **Autonomous cohort:** all t0 vehicles run native population physics with
  SUMO merge, physics MOBIL mainline; exits are autonomous. No future positions,
  lanes, presences or service are fed back. Reports merge time/position errors,
  censored/missed/extra event slots, normalized ramp speed-loss vehicle-time and
  per-vehicle delay quantiles. Omits entrants; does not certify fleet/source delay.
- **Full population:** optional `--prediction-dir DIR` consumes integrator output
  `<seed>_<arm>.npz`, keys `s,v,lane,mask` [time,slot], `vid` [time,slot] (or fixed
  [slot]), initial identities exactly matching the view, `time_s` exact recorded
  clock including t0. Slots may recycle/reorder; scoring aligns persistent IDs.
  This is the contract for the overnight fully integrated WM-PHYS candidate.

`--stride 1` is the complete one-step diagnostic; larger strides are explicitly
subsampled and MUST NOT be interpreted as complete insertion/event counts.
`--inspect` reports only metadata keys and observable array shapes.

The historical 562.4 real / 1265.4 WM2 ramp delay was a different 48-world
300s fleet contract. No cohort-only score is presented as a matched improvement.

# S-RB-EVAL: distributional traffic evaluation

The frozen contract is `docs/paper_results/RBEVAL_DESIGN.md`, registered centrally
before scoring. This is a 20-world comparative overnight pilot; sparse cells
cannot qualify a world model under the powered S-WMBREAKDOWN gate.

## Existing WM2 dispatch: reuse it

- Host `gpuR`, root `/root/rbeval20261031`, shards `wm2_shard0` through `wm2_shard3`.
- Full dispatch receipt `DISPATCH_1793460699582341507.json`, original PIDs
  706825–706828. Each worker is capped at 5 GiB, one CPU thread, nice19.
- The first five-arm task was independently verified and is skipped on resume.
- `python -B -m tools.rbeval.ops status` checks load, GPU and owned workers.
- `python -B -m tools.rbeval.ops fetch` copies immutable outputs and receipts to
  ignored `tools/rbeval/evidence/`. Neither command launches inference.
- Never dispatch another copy of the panel while those workers are alive.

CPU-only finalizer is now armed (PID708801, `FINALIZER_708801.json`): every120s
it checks only the owned panel/worker receipts, verifies all160 completion hashes,
then runs the frozen scorer once on a free CPU. It launches no replacement
inference. The durable result is `/root/rbeval20261031/FINALIZED.json`, with the
report path and every report hash. `ops fetch-reports` retrieves finalizer
receipts/logs and reports. Its eight-hour deadline ends around07:44 host time;
an incomplete/stopped worker leaves a blocked receipt rather than a false result.
`stage-scoring` and `finalize` are explicit deployment commands; do not repeat
`finalize` while its owned receipt is live. Score sources are isolated from and
do not modify the identity-bound inference runtime.

## Score a completed panel (Windows or Linux)

```powershell
python -B -m tools.rbeval.score --inputs tools/rbeval/evidence/wm2_shard0 tools/rbeval/evidence/wm2_shard1 tools/rbeval/evidence/wm2_shard2 tools/rbeval/evidence/wm2_shard3 --out tools/rbeval/reports/wm2_complete
```

The output parent must already exist; output directory must be NEW. For a
progress report add `--allow-partial`: only complete eight-future worlds are
scored; an incomplete K ensemble is never silently treated as K8. Input source
and task identities, exact BIC decisions, delay decomposition, initial-state
hashes, prehistory and support are checked before scoring. Duplicate copies or
mixed model contracts fail rather than being pooled. No input files are changed.

Outputs:
- `audit.json`: identities, file hashes, complete worlds and missing tasks.
- `distribution_world_bins.csv` / `distribution_worlds.csv`: empirical/fair-K
  delay/speed CRPS; pooled and cell×arm/time climatology; support, bias, range/PIT.
- `distribution_summary.csv`: split-specific arm/cell summaries and simultaneous
  whole-world max-stat intervals. Time bins collapse inside world before CIs.
- `rates.csv`: per-cell/arm/horizon/region rates, Wilson and simultaneous Wilson
  intervals, separate pre-existing mass, paired differences including K8 MC.
- `events_worlds.csv`, `timing_worlds.csv`, `timing_summary.csv`,
  `onset_cdf_bands.csv`: onset/censoring distributions, Brier skill and real bands.
- `expected_shifts*.csv`, `expected_order.csv`: controller−native/CTRL0 expected
  delay and upstream-event shifts, resolved signs and group arm-order evidence.
- `calibration.csv`: randomized ranks/range, simultaneous exchangeability bands
  from whole-world bounded Hoeffding averages (arbitrary within-world dependence).
- `energy_diagnostic.csv`: joint speed-field energy and supported dimensions.
- `gate.json`: qualification reasons, explicitly diagnostic/pilot-only. Missing
  cells, support, precision or uncertainty prevent qualification.

Climatology uses only 969400–969431 and excludes the entire target world.
Reserved8 labels never enter any baseline. Bootstrap: 10,000, seed20261031,
demand×penetration×split-stratified worlds, joint max-stat families. Constant or
singleton cells have no demonstrable sampling uncertainty. K8 future resamples
are paired across arms for the rate difference. Chunked calculations retain
one joint maximum across the full family and bound working memory.

## WM-PHYS integration

`run_wm.py` runs in the existing Linux runtime and uses the same frozen
motion/controller/start helpers. Supply a NEW output directory, model name,
checkpoint, config, and an integration runtime containing the physics changes:

```bash
nice -n 19 /root/miniconda3/bin/python -B -u tools/rbeval/run_wm.py --out /root/rbeval20261031/phys_A_shard0 --runtime-root /path/to/integrated/runtime --wm /path/to/wm_s.pt --wm-config /path/to/wm_phys.toml --model-name wm_phys_A --shard 0 --shards 4 --memory-gib 5
```

`--switches file.json` sets explicit **existing model attributes** after loading;
config-based construction is required for new modules. Every selected runtime
Python source, model, controller, config and observable start is identity-bound.
Keep the same absolute runtime paths when resuming. The CLI preserves K8, five
arms, t0=300 and 600 s open loop. Per-process exclusive locks and completed-task
checks protect resumes. `--max-tasks 1` is an engineering canary, not a new panel.

INT's `configs/wm_phys.toml` is a **rebuild recipe**, not a normal training config.
Use its checkpoint-preserving constructor through a small INT-owned entrypoint:

```python
from tools.rbeval.run_wm import main

def load_candidate(checkpoints, configs, device):
    # Strictly load WM2, then use INT's tested rebuild_with_motion_encoder.
    # Apply recipe/variant, freeze with eval().requires_grad_(False).
    # Return ([candidate], JSON_serializable_construction_receipt).
    ...

main(model_loader=load_candidate,
     identity_sources=[__file__, '/path/to/tools/rbint/rebuild.py', '/path/to/recipe.toml'])
```

The same CLI arguments are parsed by `main`; `--model-name` must identify the
candidate, never `wm2`. The loader function's file is automatically hashed, and
every extra recipe/helper must be supplied in `identity_sources`. Normal config
loading remains the default. Stage this updated runner into a NEW code directory
for candidates: the live WM2 runner/source hashes under `code/` must stay intact.
The construction receipt must be deterministic across shards and include the
selected variant/settings; scoring compares it as well as the loader source.

The runner currently uses the registered arrival process and learned entrant
driver mix; it rejects `physics.entry_learned`. A candidate that changes entry
sampling needs a separately frozen adapter/identity before inference. Candidate
comparisons must disclose that sampling contract, not silently replace it.

## S-SUMOCORE / external observable adapter

```powershell
python -B -m tools.rbeval.import_external --manifest observable_manifest.json --out new_normalized_output
python -B -m tools.rbeval.score --inputs new_normalized_output --out new_report
```

The adapter only reads already produced observables. It runs no simulator.
Manifest fields: `model`, `model_sha256`, `config_sha256`, `sources` (path→SHA256),
and `tasks`. Source paths are relative to the manifest or absolute. Each task
has `seed`, `future` (0…7), and five `episodes`. Each episode has `arm`, the
registered observable `initial_sha256`, `clocks`, and either:

1. `bins`: `upstream`/`merge`/`ramp`, each containing 60 post-t0 `speed` and
   `post_counts` values. Empty support must be zero with null speed; or
2. `trajectory_npz`, `trajectory_sha256`: flat equal-time samples or [time,vehicle]
   arrays with allowlisted `t,s,v,lane,present` (present optional for flat rows).
   Times are absolute seconds; lanes 0/1 mainline, 2 ramp; other lanes excluded.
   Variable-time sampling must be resampled to equal-time observations upstream.

`clocks` has keys `120`, `300`, `600`; each contains exactly `team_delay_vs`,
`road_delay_vs`, `source_wait_vs` on the registered reported-boundary convention.
Team must equal road+source. Pre-t0 speed bins and exact BIC flags are attached
from the factual receipt, never supplied as predicted future states. Future
source trajectories must not be included in model inputs. Import manifest and
adapter hashes become part of the immutable identity.

## Verification

```powershell
python -B -m unittest tools.rbeval.test_core tools.rbeval.test_score tools.rbeval.test_finalize -v
```

Tests independently verify the registered 1,800 real BIC decisions, CRPS by CDF
integration, births/exits/invalid lanes/endpoint bins, sparse-cell precision,
world-cluster and K uncertainty, chunked family equivalence, leakage prevention,
randomized ties, duplicate rejection, and external-import-to-gate plumbing.
Synthetic test outputs are distinctly named and never scientific predictions.

The GLM targets helper has completed: `E:/agent-work/rb-shared/targets/DONE.md`.
Its development-only bins and BIC flags reuse the same verified observable
receipt. No extraction work is duplicated by this harness.
## Committed diagnostic receipts

`docs/rb_merge/fit_native32_post_helper_summary.json` aggregates all 32 native
FIT worlds for source version 2977be6d. Its runtime source hashes were checked
against this checkout. Both task-owned cpuQ jobs have exited; do not relaunch
the retained dispatch scripts to reproduce already completed receipts.

Re-aggregate without running another rollout:
```
python -m tools.rbeval.summarize_merge docs/rb_merge/post_helper_fit_300.json --version 2977be6d-source-corrected-population-MOBIL-mainline --dispatch docs/rb_merge/post_helper_fit_dispatch.json --output docs/rb_merge/fit_native32_post_helper_summary.json
```

Earlier source-version receipts and the adverse first FIT32 screen are retained
separately. Different versions are not pooled. The ramp-lane speed-loss
diagnostic uses reference speed 22.22m/s and excludes a ramp vehicle after its
merge; use full ramp-origin/group clocks for the owner's fleet-delay target.
## Entrant-inclusive LC2013 continuation

`full_fleet.py --runtime <immutable integrated root> --root <observable views>
--targets <shared target CSV directory> --output <new result directory>
--seeds 969400 ... 969431 --horizon 300` runs population-default physics with
LC2013 mainline, SUMO merge/insertion and the existing native CAV executor.
Uses identity-hashed observable `start.pt` pre-t0 insertions and entrance backlog
to infer demand (the existing WMCALIB source contract), preserving `conserve_queue` /
`SourceLedger`. The recorded per-world ramp-entrance-group clock is compared with
ramp-origin road delay **plus source waiting**, including travel after merging.
Cold LC memory and the actual CF selector are explicit in every receipt. No future
observation supplies a prediction or driver setting. Exact future identity timing
is meaningful only for the t0 cohort; synthetic entrants are scored distributionally.

`remote_full_fleet.py status|stage|launch|poll|collect --version v4` owns one
CPU-only dispatch on cpuQ: INT `b0f66943` plus MERGE `0090c168` kernel/two
lookahead hooks, `--cf-model sumo_idm --warm-lc`. Stage/launch/result paths refuse
replacement. Earlier v3 (INT `ff8a637f`, legacy CF, cold LC) is retained separately.
Default version is now `v5`: same INT base plus MERGE `3df73a27` network/insertion
rules and `--trace-merge`, retaining the same arrival stream and population.
Use an explicit `--version v3`/`v4` to inspect or collect those completed runs.
Current default `v6` adds source-EPS5870b556 and the opt-in physics-native CAV
callback d7be510a (`--native-model sumo`). Prior v5 retains its frozen head path.
The validator's direct CLI default stays `--native-model head`; selecting sumo
requires `--cf-model sumo_idm`. Every row reports its actual native model.
`python -m tools.rbeval.summarize_full_fleet --panel <collected directory>
--targets <shared targets> --output <new score.json>` checks complete FIT32,
collection hashes and conservation before reporting per-world and marginal clocks.
