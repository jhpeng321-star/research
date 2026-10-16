# WM-S / C4: a physics-informed traffic world model and learned multi-CAV control

Research code for regulating mixed traffic at a highway on-ramp merge with a small share of
connected automated vehicles (CAVs) among human-driven vehicles (HDVs), simulated in SUMO.

The repository has two parts.

- **WM-S**, a per-vehicle, per-step (0.5 s) world model of mixed CAV/HDV traffic. Each HDV moves
  by an **IDM car-following + MOBIL lane-change physics core** whose **per-driver settings**
  (desired speed, time headway, gaps, politeness, ...) are inferred from each vehicle's observed
  motion and learned from prediction error only (no driver-type labels), plus a **gated, bounded
  learned residual**. CAVs are exogenous inputs: the model is told what they do. As a result, a
  controller that never appeared in the training data can be evaluated zero-shot.
- **C4**, a shared per-CAV policy that outputs absolute acceleration and lane decisions directly
  (no designed controller underneath). It is trained **Dreamer-style inside the world model**:
  gradients flow through imagined WM-S rollouts. A **SUMO data loop** periodically collects real
  episodes with the current policy, refreshes the world model on them and resumes the policy on
  the refreshed model. A MAPPO baseline with the same actor is trained directly in SUMO for
  comparison. CIRCLES damping, PI and a coordinated "moving wall" serve as frozen baselines.

Design documents:

| document | content |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | WM-S model step, physics core, training, evaluation, the SUMO layer, the config system |
| [docs/C4_DESIGN.md](docs/C4_DESIGN.md) | C4 observation/action contract, reward and safety, Dreamer trainer, MAPPO baseline, deployment |
| [docs/DREAMER_LOOP.md](docs/DREAMER_LOOP.md) | the collect -> convert -> refresh WM -> resume policy loop and its orchestrator |
| [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) | SUMO episode recordings, converted WM episodes, packs/windows, the data registry, seeds |

## Repository map

```
src/wms/                 world model and C4 control (Python package `wms`)
  physics/               differentiable IDM/MOBIL core, merge-zone rules
  model/                 WmSModel, driver encoder, gated residual, lane models, entries, checkpoints
  data/                  data registry, C4 episode conversion, packs, windows, datasets
  train/                 trainer, unroll/curriculum, losses, selection, preflight
  eval/                  evaluation states, predictor, metrics, scoring, comparison
  control/               C4 policy, Dreamer trainer (dreamer.py, train_cli.py), reward, safety,
                         RSU field/belief, deployment; mappo/ = MAPPO baseline
  cli/                   `python -m wms train|preflight|eval|score|fetch`
  legacy/                older baselines/readers kept to score earlier checkpoints
traffic_shepherd/        SUMO layer (TraCI)
  c4/                    C4 environment, sensing, HDV population, collection, saved-state
                         branching, random-action data generation, seed guard, unseen-wall regeneration
  *.py                   scenario/config, CIRCLES damping, PI session, wall group, command ownership
baselines/circles/       CIRCLES damping controller (ONNX weights + adapter; own LICENSE)
output/controller_verification_v{2,4,7}/   PI controller laws imported by the PI baseline
configs/                 TOML presets with `extends` chains (WM: c4_wm*.toml, policy: c4_policy*.toml,
                         MAPPO: c4_mappo.toml, scoring: eval_unseen.toml, ...);
                         examples/ = Dreamer-loop config template
scripts/c4/              launchers (WM, policy, MAPPO, evaluation campaign, Dreamer-loop orchestrator)
scripts/boxops/, box/    remote CPU/GPU host setup and job wrappers
scripts/genrand/         random-action data collection launcher
tools/                   paired SUMO comparison, unseen-wall scoring, lane-model calibration and
                         pretraining, data fetch/guard, pre-push safety check
tests/unit/              unit tests (CPU; most need no SUMO)
assets/research/         SUMO network of the merge scenario
scenarios/idm/           driver route template
docs/                    design documents
```

## Installation

Requirements: Python 3.10+, PyTorch, and SUMO **1.27.1** (TraCI interface).

```bash
python -m venv .venv && source .venv/bin/activate     # or a conda env with Python 3.10
pip install torch                                     # pick the build for your CUDA/CPU platform
pip install -r requirements.txt                       # numpy, pandas, scipy, onnxruntime, traci, sumolib
pip install -r requirements-dev.txt                   # pytest, ruff (optional)
pip install eclipse-sumo==1.27.1                      # or install SUMO 1.27.1 from sumo.dlr.de
export SUMO_HOME=$(python -c "import sumo, os; print(os.path.dirname(sumo.__file__))")
```

Run everything from the repository root with `src/` and the root on the path (or `pip install -e .`
for the `wms` package):

```bash
export PYTHONPATH=src:.
```

### Data

No data or checkpoints are part of this repository. Configs refer to datasets and checkpoints by
**logical name**; `src/wms/data/registry.py` resolves a name from, in order, the environment
variable `WMS_DATA_<NAME>` (for example `WMS_DATA_V5_NATIVE=/data/v5_native`), then
`configs/data_registry.toml`, then `tools/data_registry.json` (paths below `data_cache/`). All
training data can be regenerated in SUMO with the commands below; see
[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

Seeds are guarded: `traffic_shepherd/c4/seeds.py` only accepts the registered development,
training and evaluation blocks (for example 961000-961999 random-action training data,
962000-962199 unseen-wall evaluation, 964000-964199 evaluation, 966000-969999 Dreamer loop).

## Quickstart

All commands are run from the repository root; output directories must be new.

### 1. Generate world-model training data in SUMO

Undirected random direct-output CAV actions (no designed controllers in the WM training data),
plain episodes plus paired branches, converted to the WM episode format:

```bash
python -m traffic_shepherd.c4.genrand_driver --out runs/genrand \
    --plain-seeds 961000-961099 --branch-seeds 961480-961489 --workers 8
```

Native (uncontrolled) episodes and any other controller can be collected with
`python -m traffic_shepherd.c4.collect --policy native|random|generic_random|<checkpoint.pt> ...`.

### 2. Train the world model

```bash
export WMS_DATA_GENRAND_PLAIN=runs/genrand/c4_episodes      # point each logical name at your data
python -m wms preflight --config configs/c4_wm_v6_full.toml                 # config + data check
python -m wms train --config configs/c4_wm_v6_full.toml --mode run --out runs/wm_v6
# overrides: --set runtime.device=cpu --set curriculum.stage_steps=[4,4,3]
```

The run directory holds `wm_s_best.pt` and `physics.json` (the physics switches the checkpoint must
be rolled out with); both are needed downstream. `scripts/c4/run_wm.sh` wraps preflight, smoke and
detached runs.

### 3. Train the C4 policy inside the world model

```bash
python -m wms.control.train_cli --config configs/c4_policy_v6_full.toml \
    --wm-ckpt runs/wm_v6/wm_s_best.pt --device cuda --output-dir runs/policy_v6
# 2-update CPU smoke test: add --smoke
```

Several `--wm-ckpt` arguments form an ensemble. The trainer reads only files inside the checkout
or inside the directories listed in `C4_ALLOWED_ROOTS`. Outputs: `policy_best.pt`,
`policy_latest.pt`, `updates.jsonl`, `validation.jsonl`, `RUN_SUMMARY.json`.

MAPPO baseline (directly in SUMO, one SUMO process per worker):

```bash
python -m wms.control.mappo.train --config configs/c4_mappo.toml --workers 8 --out runs/mappo
```

### 4. Run the Dreamer data loop

Copy `configs/examples/dreamer_loop.example.json`, fill in your hosts and paths, then:

```bash
python scripts/c4/dreamer_loop.py dry-run --config my_loop.json \
    --init-wm runs/wm_v6/wm_s_best.pt --init-wm-val-log runs/wm_v6/val_log.jsonl
python scripts/c4/dreamer_loop.py run     --config my_loop.json \
    --init-wm runs/wm_v6/wm_s_best.pt --init-wm-val-log runs/wm_v6/val_log.jsonl
python scripts/c4/dreamer_loop.py status  --config my_loop.json --live
python scripts/c4/dreamer_loop.py stop    --config my_loop.json
```

### 5. Evaluate controllers in SUMO

Paired comparison (same seed, demand and penetration for every controller; bootstrap CIs of the
paired differences against `native`):

```bash
python tools/c4_compare.py run --controllers native damping pi c4=runs/policy_v6/policy_best.pt \
    --penetrations 0.05,0.2 --demands 2400,2700 --seeds 964000-964009 \
    --warmup 300 --horizon 600 --workers 6 --out runs/compare
python tools/c4_compare.py summarize --out runs/compare        # COMPARE.md + compare.json
```

`scripts/c4/eval_campaign.py` and `scripts/c4/run_eval.sh` run larger resumable campaigns.

### 6. Score world models on unseen walls (zero-shot controller prediction)

The coordinated wall controller never appears in the WM training data. Regenerate wall/native
branch pairs in SUMO, convert them to evaluation states and score checkpoints (HDVs predicted,
CAVs teacher-forced from the recording):

```bash
python -m traffic_shepherd.c4.wallregen plan --out runs/wallregen/plan.json \
    --seeds 962000-962035 --demands 1600,1800
python -m traffic_shepherd.c4.wallregen run --plan runs/wallregen/plan.json \
    --out runs/wallregen/sim --workers 8
python -m traffic_shepherd.c4.wallregen_states --run runs/wallregen/sim --variant h60 \
    --out runs/wallregen/states_h60
python tools/wallregen_score.py --set-dir runs/wallregen/states_h60 \
    --ckpt v6=runs/wm_v6/wm_s_best.pt --out runs/wallregen_score --benefit
```

`python -m wms score --config configs/eval_unseen.toml` and
`python -m wms eval --config configs/eval_unseen.toml --checkpoint <ckpt>` score registered
evaluation sets against a previous-best checkpoint.

## Tests and checks

```bash
python -m pytest tests/unit -q -p no:cacheprovider
python -m ruff check src/wms traffic_shepherd tests
python tools/prepush_check.py            # size / secret / forbidden-path gate on the tracked files
```

Some tests need SUMO, generated data or private evaluation sets and skip themselves when those
are absent.

## License

No license has been chosen yet. `baselines/circles/` carries its own license.
