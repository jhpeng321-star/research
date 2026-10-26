# S-RESTART — round16 native-residual restart

## Current status

**SCIENTIFIC WAIT_OWNER after a successful new-cycle0 launch.** Full MINI_PASS/mechanics validation passed, the official detached main driver launched on gpuA at **2026-10-27 00:58:06 UTC+8** (2026-10-26T16:58:06Z), and completed30policy updates. Update10 and the final SHA-bound native-relative gate passed. The registered fresh four-world SUMO guard was **INCONCLUSIVE**, so the driver stopped at01:21:33 UTC+8 without releasing the other9worlds. No scientific failure was retried or threshold relaxed.

- Runtime source: **daac879c59b9570bf7d10ec5b840aa028305cfa4**, branch `c4/restart`; no push.
- gpuA release: `/root/autodl-tmp/c4/restart20261026`; arm: `.../arm`; fresh main root: `.../arm/loop`.
- Main driver **56713/start841410018** and automatic launcher/mirror **47015/start841200902** exited after the scientific stop/backup path. Mini coordinator37473 and trainer37793 also exited; no live task-owned detached jobs remain.
- WM mini steps1/2 are finite, **264.155/284.389s**; step2 loss289.6131309, gradnorm30.3725, peakallocated1954.78MiB. Original16-window, multi-horizon validation/export completed, exit0, candidateSHA10a903be9da03d4fa0a7dcb8e92325ddcb9ea106e0a6b01cd257b38c783ec70a. No validation was skipped. This mini candidate was never adopted; the main retained R16.

### Main first-gate verdict

Selected policy **1b9949cc0eadb32db3f56b08684fa512fc2e70a26a0b7a360edb5599c6226ab5**, selected update30. Update10 improved16/16 starts, mean paired progress delta−0.03383159637451172veh-s; final native-relative mean delta **−0.09655570983886719veh-s**, all16 noninferior. These imagination gates do not establish real-SUMO efficacy.

Fresh seeds **968144–968147**, two300s paired starts/world, produced policy−native J deltas **[−2729.874502, +665.167363, −1434.149218, −1488.871471] veh-s/world**. Mean **−1246.931957**, registered whole-world95% bootstrap CI **[−2405.943181, +126.657654]** in the same units. Upper bound>0 at the fixed margin0 means **INCONCLUSIVE**, despite3/4 negative point deltas. This was the single registered look. Main state `stopped="native smoke inconclusive"`; WAIT_OWNER receipt preserved.

Only4worlds were dispatched/charged (**11,364sim-s**). Seeds968148–968156 remain allocated in the durable journal but **were never dispatched or charged**. Main conversion/WM-FT did not start. Total charged **102,607/210,000**, remaining107,393; every failed reservation remains counted. Do not resume or publish this as a real-efficacy PASS without an explicit owner scientific decision.

## Authority, integration and resources

The owner's renewed continuation3 explicitly permits **up to3 recorded fixes/retries per non-scientific setup/staging/SSH/fixture error**, superseding earlier continuation-wide second-failure stops. Scientific native identity/native-relative veto/smoke behavior/update10/native-gate failures are not retried. E: below300MB pauses/reports. Shared registration amendments and exact pre-dispatch bindings were appended before execution; prior failures/stop receipts remain historical.

Reused the existing sparse `E:/agent-work/c4-restart`, with no new worktree or laptop model/data/archive copies. E: was checked before disk-heavy steps and remained above300MB (latest checks about14GB free as the owner's cleanup progressed). Heavy tests, basetemp, models, data, checkpoints and simulation outputs stayed on boxes. No peer process, round17 task or pairs300b move was modified; agent reads did not open restricted/truth/type assignment or evaluation-seed files. No process-name discovery/signaling, reset, clean, stash or push.

Merged nativeinit93426d22 (`c248a86d`), resumeprepab876da4 (`5cc4a5c1`), longcritic7a210420 (`05728f0c`), preserving the first-four-world barrier, historical-ledger generation offset and8,000combined long-attempt cap. Added the requested SHA/panel/WM/config veto **before `loop14.run_policy` adoption**, as well as collection checks; collectors now receive the gate receipt beside policy.pt. Native lanes remain native; config policy_ramp_merge=true aligns learned-lane ownership, while residual mode abstains on lanes/native WM-law ownership.

Production Dreamer and confident-gate native calls pass explicit mode/head settings. Timeline replay uses the model's native acceleration path. No config-object-to-HOLD caller was found on these active paths, so R17ENTRY's optional config dispatch was not activated. R16 weights/learned drivers remain unchanged; nonzero residual WM-twin versus SUMO junction/stop/lane-change finalization parity is not established.

gpuA remains the owner-selected main host: matched synthetic timing policyR/A7.836/8.616s, WMR/A60.774/57.771s shows no decisive overall winner;48GB headroom favors A. Main policy/WM CUDA allocator caps **0.20 each**, numerical threads **2+4**, combined memory/free-space guards; no peer settings changed. Actual policy mini updates were about29–32s, with longcritic ON. Main has30measured updates/cycle,40WM steps, timeout **and** trainer max-hours **4.5h**, safe SPEED2 flags ON and immutable-WM policy overlap ON. Numerical shortcuts remain OFF.

Imagination starts at10s with registered10/30/60s current-WM gates and a **60s ceiling**. LONGWM loss/new uncertainty interfaces stay OFF. LONGCRITIC real native anchor is paired B300s, never native600s; only the SHA-bound60s reliability weight0.386339 is currently active. The owner-approved measured real-share1.0 is a pinned **off-policy real critic bootstrap** for the fresh residual actor, with source/partition checks; rejected actor/critic tensors are never loaded. Subsequent matched-policy phases run the fixed-grid selection. Development labels stay excluded. Training-only30–60s group noise multiplies the residual latent, so the zero head remains exactlyzero for all draws; deterministic deployment is unchanged.

## Gates, smoke and accounting

Accepted base: R16 FINAL **5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2**, configSHA **d218190193dcab5881b307c8dab7165b16bd3f6d618baae7d909268607e78bf1**. Rejected predecessor **fa2427772a49e9462c0be3172e402c515616a15e2930a263e1eafeacb23a0d5d** is preserved and blocked for absent native acceptance, never adopted.

Real10-update mini checkpoint **4a61fa8be1c8681f097ea2fa8b577b28dd5c86c974fd6941cbec932253215df3**: update0 exact native identity16/16, update10 signal/native gates16/16improved, mean native-relative delta **−0.033831119537353516veh-s**. Its final-report SubsetProvider error was corrected without changing training/selection; two failed serialization attempts and their gates/checkpoints remain. The accepted tiny policy is reused only for the executor setup retry; main cycle0 still starts fresh and measures30updates.

Registered S-C0DIAG-SMOKE-1: seeds968142/968143,1900vph,5%/40%,native warmup300/start360,paired300s continuations,extrasOFF. First dispatch exposed an executor contract bug: NaN absence sentinels on unobserved junction CAVs were rejected as residual actions. Recorded native observations reproduced1/2junction sentinels with **zero nonfinite acting commands** and finite parameters; the actual failing POLICY tick was not preserved. Corrected typed abstention is **zero outside obs.cav_mask**, with genuine nonfinite acting commands still refused. Full regression suite **404PASS18.31s** includes this distinction and no-double-charge retry contracts.

The failed attempt's full1,922charge is retained. Native bases reached361s each (>=722actual sim-s); partial failed POLICY simulation is unmeasured, so failed aggregate simulated_seconds=0 is not treated as actual zero. An explicit pre-registered operational retry (`mini_retry1`) retained the same seeds/checkpoint/WM and used fresh output directories, another1,922reservation charged before dispatch. **Pre-main total91,243**, including87,399historical and3,844smoke-attempt charges; after the main first4worlds the final total is102,607. No refund. Main excludes968100–968102 and968142/968143, continuing its registered968100–968227block. Main primary36,933/cycle includes7,800long SINGLE/GROUP sim-s; combined long-attempt cap8,000 remains.

Corrected smoke completed4/4branches,1,922scheduled sim-s,0incidents/drops/policy lane commands. **8,917 CAV decisions** on real native observations replayed exact zero-head native abstention, with0additional SUMO. This is command-level initialization replay, not a separately simulated zero-head branch; real POLICY arms use trainedu10:

- Seed968142/5%: native→policy CAV speed26.48654→23.24920m/s; lane changes15→13; ramp merges3→3; arrivals201→187; total delay4076.034→5652.066veh-s, delta **+1576.032**.
- Seed968143/40%: speed26.39285→26.68875; lane changes53→57; merges12→11; arrivals167→170; delay2955.224→2896.306, delta **−58.918**.

These are **mixed mechanics observations, not efficacy certification or native-identical trained trajectories**. No acceptance threshold is fitted from these two worlds. Main dispatches/charges only its first4fresh worlds, two300s starts each; one fixed5000-draw/seed20261026/world-bootstrap95% look at policy−native J, margin0. Harmful or inconclusive stops scientifically before releasing the other9worlds. No scientific failure is retried.

## Binding and evidence

Exact corrected CONFIG SHA **201aadcc7024dd72fba570d39414a5332c84c6f44324633372fd1fd829f53184**, REGISTRATION SHA **f33c177703afd50fce40fd24ae0ec8b479a9a76946660e6192d0f4e731979952**. All3hosts verified the same95-file overlay SHA **ed5af55f87b0ec025f934f08d4b265bb8971fc2b244b44560007d4ba3ffbc032**. Corrected smoke-binding SHA **94ead76ecab1bb862dde1b588879449f1a737affa814c938cfdcd6f0f46da66a** was appended/acknowledged before dispatch; original827b2ad... bindings/ack remain archived.

Box staging verified1,072baseline files,16core snapshot files and2paired300s return exports. Exact q views:17,282verified files,13,440links to identical existing TRAIN arrays,3,842missing files(1.128GB) copied gpuR→gpuA; manifestSHAca91b4358c96f061bc357a887ecb924d6d0e8e55f12b606c33c4b79e257fd70c. Both CPU runtime/fixture readiness checks pass (SUMO1.27.1, published ONNX SHA exact); absent output modules were restored by linking existing approved fixtures, no simulation. Original directories/state/ledger remain unchanged.

Important gpuA evidence under the release:
- `integrated_tests.log` (404PASS); previous failed/passing logs retained.
- `arm/PREARM.json`, `SMOKE_BINDINGS.json`, `SMOKE_MASK_DIAGNOSIS.json`, `SMOKE_RETRY_PREPARED.json`, `SMOKE_OBSERVATIONS.json`, completed `SMOKE_RESULT.json` / `MINI_PASS.json`.
- `arm/mini/loop` retains failed setup/serialization history; `arm/mini_retry1/loop` holds successful world artifacts, four converted episodes, paired real/native returns and WM mini outputs.
- `arm/AUTO_PID.json`, `AUTO_EVENTS.jsonl`, `auto_restart.log`, `PID.json`, final `FIRST_GATES.json`, `WAIT_OWNER.json`, `MIRROR_RECEIPT.json`, `RESTART_FINAL.json`.
- `OPERATIONS_C3.jsonl` records repairs. Read-only profiling of exact owned WM37793 was denied by container permissions; no privilege/process configuration was changed. Its step timing/CPU activity and finite logs remain observable.

The watcher and no-SUMO artifact validator lived **outside the hash-bound runtime**, with their own script SHAs, so monitoring did not invalidate MINI_PASS. After the scientific stop, this arm/accepted base was mirrored to cpuP **`/home/ubuntu/c4/main14/restart20261026/mirror`**, with path remapping and matching current/portable state SHA **95202320d9948ae6bf18db2864f0c4ae9c197a65acc8a7bbea89f272f9beb3bf**, policySHA1b9949cc... and baseSHA5b256941.... Full original training corpora remain on gpuR. The watcher exited after verifying the backup; no automatic scientific retry is armed. Cooperative stop paths remain `arm/STOP_ARM` / `arm/loop/STOP_DRIVER`; no peer signals.

Small local handoff receipt: `tools/restart/RESTART_FINAL.json`. `python -B tools/restart/verify_final.py` verifies the scientific stop/units-independent guard status, charge conservation, nine undispatched worlds, mirror hashes and operator syntax/source without loading models or running simulation. Full numerical/simulation artifacts remain remote.
