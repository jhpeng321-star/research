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
python -B -m unittest tools.rbeval.test_core tools.rbeval.test_score -v
```

Tests independently verify the registered 1,800 real BIC decisions, CRPS by CDF
integration, births/exits/invalid lanes/endpoint bins, sparse-cell precision,
world-cluster and K uncertainty, chunked family equivalence, leakage prevention,
randomized ties, duplicate rejection, and external-import-to-gate plumbing.
Synthetic test outputs are distinctly named and never scientific predictions.

The GLM targets helper has completed: `E:/agent-work/rb-shared/targets/DONE.md`.
Its development-only bins and BIC flags reuse the same verified observable
receipt. No extraction work is duplicated by this harness.
