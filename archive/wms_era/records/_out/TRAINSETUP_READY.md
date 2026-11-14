# WM-S-TRAINSETUP-20261009 — prepared, CPU verified

**Engineering setup is complete on `wms/trainsetup`. No GPU, SUMO, remote access
or real-data fit was launched. Real-pack census, external split lists and the
GPU smoke/hours gates must pass before the prepared full run starts.**

## Data and training contract

- Original recipe is the supplied C1 structured receipt and `_in/run3_chain.sh`:
  base pack_v3, pack_dspeed (`policy=trajectory`, mapped to direct), pack_damping,
  with their existing TRAIN identity/outcome/overlay sidecars. Recorded constant
  controllers become timelines; direct commands come from the packed command
  channels. Heterogeneous legacy t0 labels bind by slot; unsupported/wall labels
  refuse rather than silently becoming native.
- TRAIN switching root defaults on gpuB to `/root/autodl-tmp/swpack`. The supplied
  TRAIN index has 3,240 items / 180 runs, K=192, 120 one-second controls per item.
  `controller_timeline.npy`, `issued_acceleration_mps2.npy` and
  `issued_target_speed_mps.npy` are mmap-read from the pack. Manifest **original
  position + world + arm + tick** must match the index. Columns are the stable
  t0 identities; departures cannot become a switch merely because a slot is reused.
- Default sampling is **50% original / 50% switching**, not concatenation-count
  proportional. `SWITCH_MIX_RATIO` and `EVENT_OVERSAMPLE` configure the chain.
  Inside switching, a switch/direct event within the current horizon receives
  weight 2 by default; source shares remain normalized at every stage.
- **Leave PI AND wall out:** `WINDOWS_TRAIN`, `WINDOWS_VAL`, `EXCLUDE` are required
  externally produced original lists; no stale noPI default is used. Runtime
  `--leave-pi-wall-out` checks every CAV at every tick, including entrants and
  wall membership, across the max 120 s window and its endpoint. The switching
  compact timeline also excludes any PI segment. Both monitoring sources use
  the same restrictions. Original start shifting is disabled: shifting after
  filtering could introduce held controllers. Switching start shifting is always
  refused because its timeline/identity columns are tied to t0.
- **Zero-shot exclusion is fail closed:** 30 held worlds are refused in selected
  original windows and the entire switching TRAIN index; direct family 3 is
  deterministically recomputed from world seed + vehicle ID. Any item containing
  a held-family direct segment is dropped (`refuse` is optional; **no `allow`**).
  Unknown direct identities are treated as held; mismatched identity worlds
  refuse. Identity hashes, vocabulary, array shapes, finite commands and joins
  are checked. More than 85% family drops refuses rather than using a sliver.
  Real retained counts after these exclusions remain unknown until box preflight.
- Official switching VAL/TEST packs are not training inputs. Monitoring uses
  a world-disjoint split of the TRAIN list (10%, seed 0), or an explicitly
  TRAIN-scoped `SW_WINDOWS_VAL`; overlapping train/monitor worlds refuse.

## CAV driving and optional hooks

`--cav-drive teacher` / `CAV_DRIVE=teacher` is the default. **Every original and
switching CAV follows its recorded trajectory**, including direct segments;
direct commands do not override the teacher. Projection does not move the CAV
teacher. This is the mode used in the CPU CLI check and proposed main run.

`CAV_DRIVE=closed_loop` invokes `TimelineControllers`: native model IDM,
published CIRCLES damping and PI code on predicted states, plus recorded direct
commands, at 1 s control / 0.5 s model physics. Damping instances persist, PI
resets on re-entry and invalidates on exit/departure, and commands are held over
both physics halves and BPTT chunks. One dispatcher call occurs per physics tick,
even with both base and entrant CAVs. Missing PI sources fail closed if PI is
actually requested. These callbacks use the approximate model plant, not SUMO;
only teacher mode reproduces measured generation CAV trajectories exactly.

`FLAGS=''` is the optional hybrid rollout hook (e.g. `--rollout-mode ...` after
integration) and `BRAKING_FLAGS=''` is the braking-fix hook. Both default empty;
all supplied options must exist in merged CLI help. Extras cannot override the
data/split/budget/teacher contract. Driver ground-truth sidecars are diagnostic
only and never targets in the training loss.

## Chain, staging, disk and time

`_out/train_main_chain.sh` has LF bytes and defaults to **preflight**. `stage`
copies only six named TRAIN inputs to a **new** local destination using local
rsync; `run` is an explicit future GPU launch. Existing staging/run outputs
refuse, partial outputs/failure logs are retained, and there is no automatic retry.

Four stages: **10/30/60/120 s**. Full steps: **2500/2500/1875/1250** = 8125.
Smoke: six steps per stage = 24, using the same active position, speed, lane,
traffic, stability and conservation terms. The chain checks finite terms,
dominance/growth and per-stage gradient-norm shares. Shares can genuinely be zero
for a constraint with no violations; they are not evidence of a nonzero penalty.
Traffic ramp is explicitly 0..1 s for both smoke and full so the short-stage
objective is exercised. No 300 s stage is included.

Smoke timings project the fixed full-step budget without shrinking it; projection
over `MAX_HOURS=10` refuses. **GPU hours are not measured here**; the supplied C1
step counts are reused, not CPU timing extrapolated to a GPU. Projection covers
optimizer-step time; staging, validation and checkpoint overhead are additional.

The chain checks CRLF, required sidecars, at least **10 GiB free** before and after
staging, and before training. Exact logical source bytes drive the staging space
check. TRAIN switching has a planning estimate of **about 7.7 GiB**, derived from
the copied builder receipt's 60-run projection (2.556 GiB) scaled to 180 runs;
budget **at least 17.7 GiB free** before transfer, plus code/checkpoint overhead.
Original pack sizes were not supplied and were not measured; measure them before
any gpuB staging. The low-transfer route is switching TRAIN -> gpuA, where originals
already live. Full official VAL/TEST packs and raw runs are unnecessary.

CPU idle is checked immediately before timing (busy/load <=1 core), and smoke
samples are invalidated if busy cores exceed OMP+1. Per-process RSS is sampled
and training is terminated at 6,000,000,000 bytes. `NUM_WORKERS=0` avoids lazy
closure pickling and worker memory replication; all packs/timelines are mmap-backed.

### Coordinator staging and launch commands (not executed here)

On gpuA, copy only needed switching inputs from gpuB to a fresh incoming path:

```bash
# Use configured host aliases. Apply source symlink/allowlist checks before transfer.
ssh gpuB 'du -sb /root/autodl-tmp/swpack/{pack_switching_train,identities_switching_train,driver_params_switching_train}'
ssh gpuA 'df -BG /root/autodl-tmp'
ssh gpuA 'test ! -e /root/autodl-tmp/sw_train_incoming && mkdir /root/autodl-tmp/sw_train_incoming'
for p in pack_switching_train identities_switching_train driver_params_switching_train ACTION_TIMELINE_INDEX_train.json windows_train_hetdata.json ZERO_SHOT_HELDOUT.json; do
  ssh gpuA "rsync -rt --itemize-changes gpuB:/root/autodl-tmp/swpack/$p /root/autodl-tmp/sw_train_incoming/"
done
# After transfer, run the local measured-size/disk-guarded staging step on gpuA:
export BOX=gpuA SW_STAGE_SRC=/root/autodl-tmp/sw_train_incoming
export SW_ROOT=/root/autodl-tmp/sw_train_main
export WINDOWS_TRAIN=/path/from/split-agent/windows_train_noPIwall.json
export WINDOWS_VAL=/path/from/split-agent/windows_val_noPIwall.json
export EXCLUDE=/path/from/split-agent/pi_wall_excluded_runs.json
bash _out/train_main_chain.sh stage

# Actual real-data loader/guard census; only TRAIN pack and exclusion metadata:
export PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python
export PYTHONPATH="$PWD/output/wd01/wm_n1"
"$PY" -m wms.mixdata preflight --config _out/train_main_data.json --sw-root "$SW_ROOT" --full-scan
export W=/root/autodl-tmp/wms_train_main_20261009/work
bash _out/train_main_chain.sh preflight
bash _out/train_main_chain.sh run  # future explicit GPU smoke -> hours gate -> full
```

gpuB alternative: original packs must first be staged under paths supplied to
`PACK_V3/PACK_DSPEED/PACK_DAMPING`, with TRAIN identity, outcome and overlay
sidecars. Copy pack bytes unchanged; do not reconstruct mixed frozen metadata
or stage whole run roots. Exact original byte size and destination headroom must
be checked before transferring each named directory. Then:

```bash
export BOX=gpuB ORIGINAL_ROOT=/root/autodl-tmp/original_train_inputs
export SW_ROOT=/root/autodl-tmp/swpack
export PACK_V3="$ORIGINAL_ROOT/pack_v3" PACK_DSPEED="$ORIGINAL_ROOT/pack_dspeed" PACK_DAMPING="$ORIGINAL_ROOT/pack_damping"
export IDENTITIES_V3="$ORIGINAL_ROOT/identities" IDENTITIES_DSPEED="$ORIGINAL_ROOT/identities_dspeed" IDENTITIES_DAMPING="$ORIGINAL_ROOT/identities_damping"
export OVERLAY_DSPEED="$ORIGINAL_ROOT/labels_overlay_dspeed" OVERLAY_DAMPING="$ORIGINAL_ROOT/labels_overlay_damping"
export TRAIN_OUTCOMES="$ORIGINAL_ROOT/labels_cov_20261004T072826Z"
export WINDOWS_TRAIN=/path/from/split-agent/windows_train_noPIwall.json WINDOWS_VAL=/path/from/split-agent/windows_val_noPIwall.json EXCLUDE=/path/from/split-agent/pi_wall_excluded_runs.json
export W=/root/autodl-tmp/wms_train_main_gpuB_20261009/work
bash _out/train_main_chain.sh preflight
bash _out/train_main_chain.sh run
```

For a real local CPU slice, the coordinator should copy the admitted TRAIN pack
slice and matching sidecars **into this workspace**, then use the exact argv in
`trainsetup_cpu_20261009_02/CPU_RECEIPT.json`, replacing only the pack/sidecar/list
and new output paths. The external `ws-cl/_ext/pack_v3` junction was not opened.

## Verification and exact results

All Python checks used `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`
directly. From workspace root:

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest -q output/wd01/wm_n1/wms/tests --basetemp="_out/pytest_trainsetup_full01"
# 276 passed, 2 skipped (unavailable TRAIN parity fixtures), 1 warning; 80.24 s.

& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest -q output/wd01/wm_n1/wms/tests/test_mixdata.py output/wd01/wm_n1/wms/tests/test_trainsetup_contracts.py output/wd01/wm_n1/wms/tests/test_wms.py output/wd01/wm_n1/wms/tests/test_wms_lpo.py --basetemp="_out/pytest_trainsetup_final01"
# After final adapter changes: 128 passed, 1 warning; 27.85 s.

& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest -q output/wd01/wm_n1/wms/tests/test_mixdata.py output/wd01/wm_n1/wms/tests/test_trainsetup_contracts.py --basetemp="_out/pytest_trainsetup_accept01"
# Added original direct-command binding, explicit monitoring-overlap and packed
# wall/timeline-PI integration regressions: 99 passed, 1 warning; 12.98 s.

& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -B "_out/test_train_main_chain.py"
# 16 tests passed, including failed-output preservation and CPU/RSS refusal.
& "C:/Program Files/Git/bin/bash.exe" -n "_out/train_main_chain.sh"
# Passed; embedded Python syntax also checked by chain tests.

& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m ruff check output/wd01/wm_n1/wms _out/trainsetup_cpu.py _out/test_train_main_chain.py
# All checks passed. git diff --check also passed.

& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" "_out/trainsetup_cpu.py" --out "_out/trainsetup_cpu_20261009_02"
# Passed real wms.cli train argv, 1/1/1/1 steps; 12 records, 50/50 sources,
# all six terms and four gradient-share records; sampled child peak RSS
# 439,238,656 bytes, wall time 101.437 s. CPU timing is NOT a GPU estimate.
```

The fixture uses real copied TRAIN identity names/IDs and held-world metadata,
but synthetic 8-slot kinematics and action commands. Hash-checked copied TRAIN
metadata has 3240 x 192 identities, matching index count, and 30 held worlds.
Actual full switching timeline arrays were not locally available. CLI receipt
and logs are in `_out/trainsetup_cpu_20261009_02/`; input SHA256s and exact argv
are in `CPU_RECEIPT.json`. Reproduction requires a **new** `--out` path.

`wms.mixdata check-receipt` passed on that TRAIN receipt. Running
`wms.gate_stages ... --stages 10,30,60,120 --require-terms
position,speed,lane,traffic,stability,conservation --ramp-end-s 1` on the synthetic
one-step fixture **failed**: traffic grew 215x from 10 to 30 s (>10x limit).
All numeric terms were finite. The gate was not weakened or bypassed; this is
a CPU plumbing fixture, not evidence that the future real-data smoke passes.

Earlier failures are preserved: initial contract tests identified the callback
double-call and hidden nonfinite timing problems (fixed and independently retested);
first full-width CLI check was interrupted after one finite step by the harness
timeout (`_out/trainsetup_cpu_20261009_01/INTERRUPTED.json`). The NumPy mmap
non-writable-tensor warning originates in the frozen IdeaA loader; no pack
bytes are written by training.

Task record: `_out/TRAINSETUP_REGISTRATION.json`; verification record:
`_out/TRAINSETUP_VERIFICATION.json`. Master ledger has the stable
protocol/engineering-only entry. ROUTE, README, INDEX, open questions and history
are updated; no scientific efficacy conclusion is changed. Historical source
verification requires the coordinator's full workspace; no old outcomes were
opened to validate this engineering task. READY status does not waive the
real-data guard census, provided leave-PI-and-wall-out lists or smoke gate.

Global ledger consistency fails in this partial checkout because historical
registered artifact paths are absent (including externally recorded paths).
It reports **1560 hard errors**, identical to the pre-task ledger; introduced
errors are **0**. Ledger tests: **4 failed, 18 passed, 5 skipped**, all failures
from unavailable historical paths/source coverage. `summary --check` passes.
These frozen entries were not altered to silence failures. The task's entry and
local evidence paths introduce no additional consistency errors. `summary --write`
regenerates the status from the ledger; coordinator must run `validate`,
`verify-sources` and the ledger tests in the full merged checkout. Scoped copied
TRAIN source hash verification passed here; full historical `verify-sources` was
not run because it would open unrelated registered outcomes outside task scope.
