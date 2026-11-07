# Option A deploy: early R start + R's TRAIN wave on cpuV

Operator procedure for the switch in DESIGN_OPTIONA.md section 5. Nothing here runs SUMO except the parity gate
(step 5, ~2,400 sim-s, registered first). Hosts: cpuZ = 172.17.16.11 (run owner), cpuV = 172.17.0.15 (replica).
Run root (both hosts, identical path): `/home/ubuntu/coopmwm20261107_clean5bf8p` (below: `$RUN`).
Frozen source (both hosts, identical path): `/home/ubuntu/clean5b_src_<sha>` (below: `$SRC`).

## 0. Env vars (set ONLY in the restarted B and R owners; unset = byte-identical to resume3)

| var | owner | value |
|---|---|---|
| `COOPMWM_EARLY_R` | B and R | `1` (R waits for `replay_bank/B/PRACTICE_PLAN_B_<u>.json`; B's coordinator runs the guard; atomic TTC cache) |
| `COOPMWM_REMOTE_R` | R only | `172.17.0.15` |
| `COOPMWM_REMOTE_R_KEY` | R only | `/home/ubuntu/.ssh/id_ed25519_optiona_cpuv` |
| `COOPMWM_REMOTE_R_KNOWN_HOSTS` | R only | `/home/ubuntu/.ssh/known_hosts_optiona_cpuv` |
| `COOPMWM_REMOTE_R_USER` | optional | default `ubuntu` |
| `COOPMWM_REMOTE_R_ENVSH` | optional | default `/home/ubuntu/cpus_env/env.sh` (sha compared on both hosts every wave) |
| `COOPMWM_REMOTE_R_JOB_TIMEOUT_S` | optional | default 14400 (per remote episode; exceeding it = RemoteWaveLost) |

Everything else (CONFIG.json, `COOPMWM_DISK_MIN_GB`, the run-chain `COOPMWM_*` bindings such as
`COOPMWM_FROZEN_TRUNK`, `COOPMWM_MAP_SPEC`, `COOPMWM_G5_RUNTIME`, `COOPMWM_LC_RUNTIME`, `COOPMWM_RESUME_FAILED_EPISODES`)
is unchanged. The remote worker receives the R child's `COOPMWM_*`, `C4NM_*`, `OMP_*`, `MKL_*`, `OPENBLAS_*`,
`NUMEXPR_*`, `KMP_*`, `TORCH_*`, `PYTHONHASHSEED`, `SUMO_HOME`, `PYTHONPATH` on top of cpuV's sourced env.sh
(`COOPMWM_REMOTE_R*` are not forwarded). `LD_PRELOAD` must come out identical from env.sh on both hosts (checked).

## 1. Source tree on cpuV (identical path, manifest check)

```bash
# on cpuZ, from the new frozen tree $SRC (built and manifest-sealed as usual for the new head)
rsync -a --delete -e "ssh -i ~/.ssh/id_ed25519_optiona_cpuv -o IdentitiesOnly=yes -o BatchMode=yes \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts_optiona_cpuv" \
  --exclude=__pycache__ $SRC/ ubuntu@172.17.0.15:$SRC/
ssh -i ~/.ssh/id_ed25519_optiona_cpuv -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=~/.ssh/known_hosts_optiona_cpuv ubuntu@172.17.0.15 \
  "cd $SRC && python3 -c 'import hashlib,json;m=json.load(open(\"SOURCE_MANIFEST.json\"));\
bad=[n for n,s in m[\"files\"].items() if hashlib.sha256(open(n,\"rb\").read()).hexdigest()!=s];print(\"BAD\",bad)'; \
  sha256sum SOURCE_MANIFEST.json"
```
`BAD []` and the SOURCE_MANIFEST.json sha must equal cpuZ's. RemotePool re-checks this (plus CONFIG, env.sh,
Python/numpy/torch versions, LD_PRELOAD and every input below) at the start of EVERY R wave, before any reserve.
The tree must contain `tools/coopmwm/remote_worker.py` and `remote_wave.py` (they are manifest files of the new head).

## 2. Replica run root on cpuV (identical path `$RUN`)

Derived from the worker code path (`remote_worker.op_run` -> `accel_plant.episode` -> `bind_runtime`,
`track2_backend.episode_components`, `g4_runtime.BoundaryCapture`, `fair_vehicle_log.VehicleLog`,
`warm_health.assert_running/live_safety`). The job dict itself carries the policy weights, the track2 config +
worker state, the safety reference and any practice snapshot inline, so nothing from replay_bank, g4/, checkpoints
or A/B results is needed on cpuV.

| path on cpuV | content | why |
|---|---|---|
| `$RUN/CONFIG.json` | byte copy of cpuZ's (sha compared every wave) | replica identity |
| `$RUN/CHARGES.jsonl/` | an EMPTY DIRECTORY | any ledger write on cpuV fails loudly; every remote op refuses otherwise |
| `$RUN/jobs/` | empty directory | the worker writes `jobs/<id>/` (CLAIM, observable tape, g4_captures, vehicle log, RESULT) |
| `$RUN/remote_r/` | empty directory | staging (JOB.pkl + sha, PID, DONE/ERROR, worker.log); never landed |
| no `STOP.json`, `FAULT.json`, `PAUSE_*.json`, `WARM_*.json`, release/rule/diagnosis files | | these are MIRRORED from cpuZ on every poll (health guards read them each tick) |
| `job['runtime']` (frozen runtime dir) and `job['checkpoint']` | byte copies at their identical absolute paths | `bind_runtime` / leverprobe setup; digests compared every wave |
| `COOPMWM_FROZEN_TRUNK` file (path before `\|`) | byte copy, identical path | `frozen_trunk.setup` |
| `/home/ubuntu/coopmwm20261106_clean4/inputs/map/wz3to2/` (dir of `COOPMWM_MAP_SPEC`) | whole directory, identical path | `map_runtime.install` (tree digest compared every wave); copy the whole `inputs/map/` tree so SUMO-side files referenced by the spec are present too |
| every absolute `PYTHONPATH` entry of the R owner (other than `$SRC`) | identical path and bytes | forwarded to the worker; tree digest compared |
| `/home/ubuntu/cpus_env/env.sh` | identical (md5 aa316e06) | sha compared every wave |

Do NOT copy replay_bank/, g4/, sealed/ or any holdout/heldout/cal_split/chunk5/j4b/inc1 path, A/B results or
checkpoints. Setup (cpuV, as ubuntu):
```bash
mkdir -p $RUN/jobs $RUN/remote_r $RUN/CHARGES.jsonl
# from cpuZ: CONFIG.json, frozen runtime/checkpoint/trunk, map inputs (rsync -a over the dedicated key, as in 1)
```
The parity gate's audit hook lists every file the remote worker opened outside its job dir
(`OPTIONA_PARITY_<tag>.json: opened_files / opened_mismatched`); anything missing or different there must be fixed
before the switch.

## 3. Disk and window
cpuV keeps >= 12 GB free (remote admission waits, then RemoteWaveLost after 1 h). Landed job copies are deleted on
cpuV only after `LEARNING_MONITOR_u<u>.json` exists and the cpuZ copies re-verify against their landing manifest
(captures removed by disk_prune count only with the identical sha in `pruned/PRUNE_u<u>*.json`). The R child does
this at the start of update u+1; manually: `python -B -m tools.coopmwm.remote_wave cleanup --config $RUN/CONFIG.json --update <u>`.
Use cpuV only while no sealed eval wave runs there; all R waves end at u12, before the u12 sealed eval.

## 4. Linux qualification (cpuZ, new tree)
Run the usual coopmwm CPU suite plus `tools/coopmwm/test_optiona_cpu.py`.

## 5. Parity gate (after the REGISTRATION entry; ~2,400 sim-s charged to the run's ledger)
Same shell env as the R owner (env.sh + run-chain vars + the REMOTE vars). K = the last update R COMPLETED.
Seed: a fresh TRAIN-type seed from an allocated unused range (never 964000-964199, 962xxx, 977xxx, 9906xx,
989600-989699; the tool refuses those, any registered world and any existing job dir).
```bash
cd $SRC && python -B ops/optiona_parity.py --config $RUN/CONFIG.json --update K --tag p1 \
  --world-json '{"seed": S, "cell": C, ...full TRAIN world dict...}' --dry-run     # prints the job id, zero charge
cd $SRC && python -B ops/optiona_parity.py --config $RUN/CONFIG.json --update K --tag p1 --world-json '...'
```
PASS = identical RESULT pointer/payload, g4_captures, decompressed observable tape (and every other job file except
CLAIM.json) and no opened-file mismatch. FAIL: stop, do not switch, investigate thread/MKL determinism.

## 6. Owner restart at an update boundary (by PID, own processes only)
After `LEARNING_MONITOR_u<k>.json` exists and before either B or R child of u<k+1> has reserved anything
(or after both have finished and are waiting): for each of B and R
1. read `$RUN/owners/<arm>.json` and `$RUN/owners/children/<arm>_u<k+1>.json` (pid + start ticks); verify the
   ticks against `/proc/<pid>/stat` field 22;
2. `kill -TERM <owner pid>` first, then `kill -TERM <child pid>` (never SIGINT; SIGTERM leaves the update unwritten,
   an attempted update is never redone, so stop only children that have not reserved: check CHARGES.jsonl for
   `_u<k+1>_<arm>_` rows; if they exist, use the operator resume mechanism instead);
3. confirm both PIDs are gone; remove nothing (FAULT/STOP files written by the dying child must be diagnosed and
   released by the registered procedure, never deleted silently);
4. restart from `$SRC` with the usual chain command and env, adding the vars of section 0
   (B: `COOPMWM_EARLY_R=1`; R: `COOPMWM_EARLY_R=1` + the three `COOPMWM_REMOTE_R*`).
Then N6 rechecks plant identity on the new head and IMITREPLAY restages the u12 source.

## 7. Failure semantics (what the operator sees)
- Transport error, timeout, sha mismatch, dead/missing remote worker, replica identity mismatch, remote setup error:
  `RemoteWaveLost` -> FAULT.json + STOP.json (launch_training.main). No replacement, no retry, no extra charge.
  Receipts: `$RUN/remote_r/REMOTE_WAVE_LOST_<ns>.json`; remote workers still running are SIGTERMed by recorded PID
  (best effort). Their charged jobs have no cpuZ CLAIM, so a resume fails closed ("attempted job cannot be retried")
  until the operator decides.
- A genuine episode exception on cpuV is re-raised with its type/args: the unchanged episode-local rule applies
  (EPISODE_FAILED receipt on cpuZ + charged same-cell replacement, also run on cpuV).
- B's realised practice/ordinary counts differ from the plan early R used: ValueError in B's coordinator BEFORE any
  G4 ledger write (R's wave is already spent; disclosed in the design).
