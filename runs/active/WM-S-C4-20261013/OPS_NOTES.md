# WM-S-C4 ops notes (scripts/c4) — draft 2026-10-12

Nothing here starts a machine. The owner powers boxes on; every script below is run by a person or by the coordinator
after the owner says so. MAPPO is deferred (stub only). All shell files are LF (`.gitattributes` has `*.sh text eol=lf`;
`pack_bundle.py` normalises again and the scripts refuse to run with CR bytes).

## 0. Files

| File | Purpose |
|---|---|
| `scripts/c4/bundle_manifest.json` | declarative bundle content (code dirs, data groups, checkpoints, policy filter) |
| `scripts/c4/pack_bundle.py` | builds the bundle (laptop). `--dry-run [--list-files]` prints sizes without writing |
| `scripts/c4/unpack_bundle.py` | verifies + unpacks on the box, rebuilds the sparse pack arrays |
| `scripts/c4/selftest_bundle.py` | synthetic round-trip test of pack/unpack (no real data) |
| `scripts/c4/box_setup.sh` | env report, installs pins, SUMO 1.27.1, unpack, sha verify, unit-test smoke |
| `scripts/c4/c4_env.sh`, `c4_jobwrap.sh` | shared helpers: fresh run dirs, nohup/setsid, pid, STATE, monitor.csv, RUN_SUMMARY_C4.json |
| `run_wm.sh`, `run_policy.sh`, `run_loop.sh`, `run_eval.sh` | stages a, b, c, e; `run_mappo.sh` = deferred stub |
| `scripts/c4/make_eval_seeds.py`, `eval_campaign.py` | evaluation seeds and the paired campaign driver |
| `scripts/c4/fetch_results.py` | pull logs / checkpoints back with sha256 verification |
| `runs/active/WM-S-C4-20261013/eval_seeds.json` | the chosen development seeds (sha256 inside) |

## 1. Build the bundle (laptop)

```
python scripts/c4/selftest_bundle.py                       # 2 s, must print "selftest OK"
python scripts/c4/pack_bundle.py --dry-run                 # sizes, counts, what the filter dropped
python scripts/c4/pack_bundle.py --out E:/c4_bundle_20261013     # fresh dir; ~17 GB; refuses a non-empty dir
```
Dry run on 2026-10-12: code 3.3 MB, checkpoints 8.4 MB, lists 0.2 MB, lpwo lists 0.4 MB, sidecars 89.5 MB,
pack_v3 3.3 GB, pack_dspeed 9.6 GB, pack_damping 0.6 GB, pack_switching_train 3.5 GB: **17.1 GB**. Real size on the
box after unpacking is the same, but the sparse arrays have an apparent size of about 75 GB (`du -sh --apparent-size`).
If the box filesystem does not support holes the unpack needs the full ~75 GB (the box_setup disk check uses 60 GB).
Role-specific transfer: the CPU box needs only `c4_code`, `c4_ckpt`, `c4_data_lists`, `c4_data_lpwo_v3`,
`c4_data_sidecars` (~100 MB; `box_setup.sh --role cpu`); the big packs go to the GPU box only.

Hard filter: any path with holdout, heldout, cal, calibration, cal_split*, sealed, chunk5, j4b, inc1 is dropped (reported in
`BUNDLE_MANIFEST.json`) and every tar is re-opened and re-checked before the build is declared good. Currently dropped:
`traffic_shepherd/planner/holdout.py` and `tests/test_holdout_exclusions.py` (not imported by active code).
Code-only allowlist: `src/wms/physics/merge_cal.py` (merge-zone physics, imported by `wms.physics`; no calibration data).
Consequence for configs: the C4 WM config must **not** reference the datasets `zero_shot_heldout` (its file is filtered out);
use `heldout_family_policy`-free settings or the derived `c4_lists/*` instead. The manifest JSON copies inside the packs
still list non-TRAIN runs by name (indices must stay valid); their array rows are absent (zeros).

Policy filter: default `--policies native,trajectory` (no designed CIRCLES damping windows, no switching arm). To ship
damping windows for evaluation-only use: `--policies native,trajectory,damping`.

## 2. Transfer (never in a retry loop; ssh drops happen)

```
scp -r E:/c4_bundle_20261013 gpuA:/root/autodl-tmp/c4/bundle        # or rsync -a --partial --progress ...
```
After a drop, rerun the **same command once** with `rsync --partial` or per-file scp; verification is the `SHA256SUMS`
check inside `box_setup.sh`, not the transfer tool. Outputs must live under `/root/autodl-tmp/` (system disk is small).

## 3. Box setup (once per box)

```
ssh gpuA 'cd /root/autodl-tmp/c4/bundle && bash box_setup.sh --role gpu --detach'
ssh gpuA 'tail -f /root/autodl-tmp/c4/setup.log'      # then: cat /root/autodl-tmp/c4/SETUP_REPORT.txt
```
CPU box: `bash box_setup.sh --role cpu --detach` (installs SUMO via pip if missing). It checks python >= 3.10, torch + CUDA
(gpu role), **pandas==2.3.3** (pandas 3 breaks the outcome-sidecar dtype check; installed if different), pyarrow,
onnxruntime, SUMO **1.27.1** (pip `eclipse-sumo/traci/libsumo==1.27.1`; fallbacks printed if no wheel), nproc **and** the
cgroup CPU quota (containers show the host nproc; scripts use the quota as `C4_CORES`), RAM, disk, GPU. It unpacks into
`$C4_HOME/rd` (default `/root/autodl-tmp/c4/rd`), writes `$C4_HOME/env.sh` and `UNPACK_OK`, runs `pytest tests/unit`,
`python -m traffic_shepherd inspect` and (gpu role, if `configs/c4_wm.toml` exists) `wms preflight`.
On the Windows rented box (alias `rented`, if re-rented): the `.sh` wrappers need Git Bash/WSL; `eval_campaign.py`,
`make_eval_seeds.py`, `fetch_results.py` and `unpack_bundle.py` are plain Python and run natively
(`python scripts/c4/eval_campaign.py run --out E:\c4\eval --seeds ... --workers 44`).

## 4. Stage a — world model (gpuA, ~5 h)

```
cd /root/autodl-tmp/c4/rd/scripts/c4
bash run_wm.sh preflight                    # foreground, no GPU
bash run_wm.sh smoke                        # real pack, curriculum.stage_steps=[5,5,3,3]; sets the smoke-ok marker
bash run_wm.sh run                          # ONE detached run (refuses without a passing smoke for this config sha)
tail -f /root/autodl-tmp/c4/runs/wm_main_*/launch.log ; cat .../STATE ; cat .../RUN_SUMMARY_C4.json
```
Check from the smoke (pitfall: gates must use the real data path): finite loss in all 4 stages, `init_from` loaded N/M,
s/step per stage, CUDA peak, and that **every loss term is active** in `train_log.jsonl`. R5b reference: ~8 s/step at 60 s
(batch 32), ~16 s/step at 120 s, GPU 4.5 GB at 60 s; 940 + 625 steps ~5-5.5 h. `monitor.csv` (60 s samples) records GPU
memory, RAM, load and free disk; the summary stores the peaks. A killed or failed run keeps its directory;
`RESTART=1 [RESTART_CKPT=.../wm_s_best.pt] bash run_wm.sh run` creates `<RUN_ID>_rN` warm-started from the checkpoint
(a new allocation: record it). Stop a run by PID of the wrapper only (`kill -TERM $(cat runs/<id>/pid)`).

## 5. Stage b — policy in imagination (gpuA)

```
bash run_policy.sh check
WM_CKPT=/root/autodl-tmp/c4/runs/wm_main_*/out/wm_s_best.pt bash run_policy.sh run
```
Assumes `python -m wms.control.train_cli --config configs/c4_policy.toml --out DIR --set k=v`; the WM checkpoint key is
`wm.checkpoint` (override with `POLICY_WM_KEY`), policy init key `policy.init_from` (`POLICY_INIT_KEY`).

## 6. Stage c — Dreamer loop

```
bash run_loop.sh plan
WM_CKPT_0=... POLICY_CKPT_0=... COLLECT_SEEDS_FILE=<registered, audited collection seeds> ITERS=3 bash run_loop.sh run
```
Single box with GPU + many cores: all four stages run in order inside one detached driver. Two boxes: on the CPU box
`STAGES=collect,convert`, copy `iter_K/convert/out` and the checkpoints to the GPU box (rsync, then verify with
`sha256sum`), there `STAGES=wm,policy START_ITER=K ITERS=1 LOOP_ID=<same>`, copy the new checkpoints back, repeat.
Resume after a stop: `RESUME_LOOP=1 LOOP_ID=<id> bash run_loop.sh run` (finished stages are skipped; a failed stage needs
`RESTART_STAGE=1`, which renames the failed directory). A stage failure stops the loop; nothing is retried.

## 7. Stage e — evaluation (CPU box)

```
python scripts/c4/make_eval_seeds.py              # already done; file is in the bundle
bash run_eval.sh check                            # SUMO 1.27.1 + traffic_shepherd inspect
POLICY_CKPT=... WM_CKPT=... bash run_eval.sh plan
POLICY_CKPT=... WM_CKPT=... [BELIEF_CKPT=...] [INCLUDE_WALL=1] WORKERS=44 bash run_eval.sh run
RUN_ID=<eval id> bash run_eval.sh status ; RUN_ID=<id> RESUME=1 bash run_eval.sh run     # continue after a stop
```
Layout: `out/<controller>/m<demand>/p<pen>/s<seed>` (+ `.log`, `.DONE` / `.FAILED`). Native/damping/PI call the existing
`python -m traffic_shepherd evaluate` (new output dir per job, role gate `require_development_case`), or the C4
runner with `C4_CLASSIC_HARNESS=deploy` (recommended for paired comparisons: same harness/summary). Controllers
`wall`, `c4_policy`, `c4_policy_rsu_check_{oracle,belief}` call `python -m wms.control.deploy --controller X
--seed --loading --penetration --role --output [--policy-ckpt --wm-ckpt --rsu-state oracle|belief] $C4_DEPLOY_ARGS`
(implemented; DEPLOY_NOTES.md at the repo root). The belief variant runs now (RSU BeliefTracker, no checkpoint;
`BELIEF_CKPT` is optional and ignored). Failed jobs are not retried; `RERUN_FAILED=1` renames them aside and reruns once.

## 8. Fetch results (laptop)

```
python scripts/c4/fetch_results.py --host gpuA --run wm_main_<ts> --list-only
python scripts/c4/fetch_results.py --host gpuA --run wm_main_<ts> --include-large "out/wm_s_best.pt"
```
Small files (<= 5 MB, no `.pt/.npy/.bin/.tar`) -> `runs/active/WM-S-C4-20261013/fetched/<host>/<run>/`; named large
files -> `tmp/c4_fetch/<host>/<run>/` (git-ignored). All sha256-verified on the box and locally; existing files are never
overwritten. After a verified fetch the owner stops the instance (box off when not training).

## 9. Machine sizing

| Stage | Hardware | Cores | RAM | Disk | Duration (expected) |
|---|---|---|---|---|---|
| bundle build (laptop) | any | 1 | < 2 GB (range reader streams 16 MB) | 17 GB free | 10-20 min |
| box_setup (either) | any | 1 | 2 GB | 17 GB real (+ sparse), 60 GB free advised | 10-25 min (sha of 17 GB + unpack) |
| a. WM | 1 GPU >= 24 GB (gpuA 48 GB; peak seen 4.5 GB at 60 s, 120 s stage not yet measured beyond 1.5 GB at batch 8) | 4-8 (OMP 4) | 32 GB (mmap packs) | 60 GB | ~5 h (6-8 h ceiling) |
| b. policy | 1 GPU | 4-8 | 32 GB | 10 GB | **TBD** (policy worker smoke) |
| c. collect | CPU, no GPU needed unless the policy runs on GPU (CPU torch is enough for a small actor) | workers = usable cores - 2 | ~1.5-2 GB per SUMO worker (estimate, **TBD**) | ~60 MB per 1200 s feature run -> ~120 MB per 2400 s episode; 360 episodes ~45 GB | episodes x min/episode / workers |
| c. wm/policy FT | gpuA | 4-8 | 32 GB | 10 GB | 2 h + 2 h per iteration (proposal) |
| e. evaluation | CPU (RSU check variants run a WM rollout per decision: CPU torch is slow, use a GPU or lower the check rate) | usable cores - 2 | 2 GB per worker | ~50-100 MB per episode -> up to ~115 GB for 1152 episodes **[estimate, TBD]** | table below |

SUMO speed (min per 2400 s episode) is **TBD from the sumo worker**; the table gives the wall time of the full
1152-episode evaluation (6 controllers incl. the belief variant x 8 pen x 3 demands x 8 seeds) and of one collection
iteration (120 episodes) for assumed speeds:

| min/episode | 16 workers | 46 workers (48-core box) | 94 workers (96-core box) |
|---|---|---|---|
| 3 | eval 3.6 h / collect-iter 0.4 h | 1.3 h / 0.13 h | 0.6 h / 0.06 h |
| 6 | 7.2 h / 0.75 h | 2.5 h / 0.26 h | 1.2 h / 0.13 h |
| 10 | 12 h / 1.25 h | 4.2 h / 0.43 h | 2.0 h / 0.21 h |

(Formulas use 1152 episodes; with the belief variant still pending 960 are runnable, the optional wall law adds 192.)
The laptop alone (22 usable cores) would need 3.6 h to 12 h of wall time per 1152 episodes at 16 workers: use a CPU box
for the evaluation and for collection. The earlier rented 48-vCPU Windows box was terminated 2026-09-30; a new rental
means ~45 min setup and a parity check (AMD vs laptop Intel) before trusting numbers.

## 10. Interface assumptions to reconcile before the first launch

| Assumed | Owner of the real thing |
|---|---|
| `configs/c4_wm.toml` (init from `ckpt_r5b` via `WMS_DATA_CKPT_R5B`; data lists `c4_lists/*` or equivalent) | WM worker |
| `python -m wms.control.train_cli --config --out --set` | policy worker |
| `python -m traffic_shepherd.c4.collect --policy-ckpt --penetrations --demands --seeds-file --iteration --workers --out` (writes `episodes/`) | SUMO worker |
| `python -m traffic_shepherd.c4.to_pack --episodes --out --iteration` (writes `pack/`, `identities/`) | SUMO/data worker |
| ~~`traffic_shepherd.c4.evaluate`~~ RESOLVED 2026-10-12: `python -m wms.control.deploy --controller {native,damping,pi,wall,policy\|c4_policy,policy_check\|c4_policy_rsu_check} --state\|--rsu-state oracle\|belief ...` (campaign argument names accepted; outputs `episode/` (collector format), `decisions.jsonl`, `segments.jsonl`, `DEPLOY_SUMMARY.json`); `eval_campaign.py` default `--c4-module wms.control.deploy`, `--classic-harness`, `--deploy-args` | deploy worker (c4/deploy) |
| **OPEN (blocker): seeds.** `eval_seeds.json` lists WM28_FRESH_VALIDATION_SEEDS (92xxxx); the C4 env gate `traffic_shepherd.c4.seeds.check_case` admits only 4107-4148 and explicitly blocks WM28 fresh-validation seeds, so every C4-runner job fails at reset (`SeedRefusal`, verified) while `traffic_shepherd evaluate` would accept them (unpaired). Owner decision needed: a registered, exposure-audited block admitted by both gates | owner / integrator |
| **OPEN: sizing.** `policy_check` with defaults (5 candidates x 4 driver samples, 20 s imagination, 5 s period, 1 R5b member) measured ~12.7 s/decision with 4 torch threads (laptop; the campaign sets OMP_NUM_THREADS=1, so pass `--threads N` in `C4_DEPLOY_ARGS` and size workers accordingly); 1 thread ~21 s/decision -> 120 decisions ~ 25 min (4 threads) to ~45 min (1 thread) per 600 s episode (+ SUMO); `MIN_PER_EP=6` underestimates these jobs (set `--minutes-per-episode` accordingly or use a cheaper `C4_DEPLOY_ARGS`) | integrator |
| policy checkpoint named `policy_best.pt` / `policy.pt`, WM `wm_s_best.pt` / `wm_s.pt` (`*_CKPT_NAMES` env) | policy / WM workers |

## 11. Pitfall checklist (every launch)

1. LF endings: scripts refuse CRLF; do not hand-edit them on Windows without LF.
2. ssh drops: launch once, verify with `cat runs/<id>/STATE` and `ps -eo pid,lstart,args | grep -E 'wms|traffic_shepherd'`; never loop a launch.
3. Sidecar/identity directories exist: `box_setup.sh` + `wms preflight` resolve every dataset and fail closed.
4. pandas pinned to 2.3.3, pyarrow present.
5. Disk: outputs under `/root/autodl-tmp`; `run_*` check free space; SUMO jobs fill disks and starve training CPU (~4x slower):
   do not run collection or evaluation on the GPU box while it trains.
6. One GPU, one main job: `c4_require_gpu_idle` refuses to start next to a GPU user.
7. Gates must touch the real data path: the box smoke (not only the unit tests) has to pass first.
8. Kill only your own PIDs (wrapper pid in `runs/<id>/pid`, process group via `kill -TERM -- -<pgid>`); never by image name.
9. The box is switched off by the owner after a verified fetch.
