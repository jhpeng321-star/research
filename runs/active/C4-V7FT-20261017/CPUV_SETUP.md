# cpuV setup — 2026-11-05 (S0 PASS; integration pending)

## Host and copy authority
- Owner: S-LAUNCH, new clean rerun on cpuV after cpuS loss; “set it up”.
- cpuV: `101.35.160.72` / `172.17.0.15`, hostname `VM-0-15-ubuntu`, 48 vCPU,
  AMD EPYC 9754 (owner inventory), 92 GB RAM, 185 GiB initially free.
- Source cpuQ `172.17.0.3`; direct internal-network SSH verified.
- Dedicated cpuQ key `~/.ssh/id_ed25519_cpuv_copy_20261105`, public comment
  `cpuQ-to-cpuV-copy-20261105`. Exactly one new authorized_keys line on cpuV,
  `restrict,from="172.17.0.3"`; prior file backed up. Private key stays on cpuQ.
- cpuV host key pinned in cpuQ `known_hosts_cpuv_copy_20261105`.
- Copy coordinators ran nice19/ionice-idle. Initial own PID416635/start82532462
  blocked while inventorying a dead cpuS FUSE mount and was identity-checked and
  sent SIGTERM. Final traversal guards lexical mount names before stat and never
  follows symlinks; own final PID437813 exited after copy. Existing cpuQ processes
  were neither signalled nor adopted. Prior attempt receipts/logs are retained.

## Environment — SHA verified
- Same path `/home/ubuntu/miniconda3`; 32,130 manifest entries, 2,142,044,456 bytes,
  zero mismatches in independent cpuV `ENV_VERIFY.json`. Python caches excluded.
- Python **3.10.16**, torch **2.12.1+cpu**, SUMO **1.27.1**.
- Current pip-freeze SHA256, equal on cpuQ and cpuV:
  `7c1858dba9279feb798ab7d790db7dda44ac987580a1726c1e75613efcfe8a8e`.
  The historical Nov4 report's short hash is not substituted for this fresh comparison.
- System dependencies installed: `libgl1=1.4.0-1`, `libxrender1=1:0.9.10-1build4`;
  GCC11 already installed. SUMO version executable check succeeds.
- Shim `/home/ubuntu/cpus_env/libmkl_intel_shim.so`, compiled from
  `int mkl_serv_intel_cpu_true(void){return 1;}` with GCC `-shared -fPIC -O2`.
  Binary SHA **505bcbd243e89cd49b451ae5fe6e7d3e0d5a8a62f43a36943a7809e601a71f79**,
  exactly the documented cpuS binary. Mandatory `LD_PRELOAD` retained in subprocesses.
- `/home/ubuntu/cpus_env/env.sh`: SUMO_HOME, conda PATH, source/runtime PYTHONPATH,
  empty CUDA visibility, OMP/MKL/OpenBLAS threads1 and LD_PRELOAD.
- Zero-SUMO seeded torch probes equal to cpuQ:
  matmul `284abf5b78efa2b6ea2606f6878c4f368e2f9b7cdf66e7d5a79a0ce3aa2de637`;
  tanh `349db30ca0a30d1538b4ba238c742290884d8bd77af1176b85d60d84da116cd5`;
  exp `89a65f041a3a1a5c613f391fbbcbe0f6da4c7a1dd526452649e7213fd63d240e`.

## Data receipts
- Six rescued legacy confirmation trees: **10,108 files / 4,396,534,952 bytes**,
  full source SHA inventory and independent cpuV verification, **zero mismatches**.
  Source `/home/ubuntu/rescue_confirm/home/ubuntu/coopmwm20261102_warm1/replay_bank/`;
  destination `/home/ubuntu/coopmwm20261102_warm1/replay_bank/{B,R}/confirm_{replication,extended,solo}`.
  Restored as ordinary original-path directories, not cpuS mounts or rescue-path aliases.
- Main rerun research/source bank superset: **133,119 entries / 28,320,472,526 bytes**,
  full independent cpuV SHA verification, **zero mismatches** in `RESEARCH3_VERIFY.json`.
  It includes warm1, LS source, original solo1755_v2/v3 link targets, coop3 snapshots,
  MAPPO u0, frozen sources, e3final/runtime and transitive source-code roots.
- The first verification found exactly the two known live administrative files
  `LS_COLLECT_STATUS.json` and `SUPERVISOR_REPLAYDRAIN0735.log` changing on cpuQ.
  Their current stable bytes were frozen into immutable copy-control snapshots,
  SHA-copied, and the complete inventory reverified. Earlier cpuV versions and the
  failed verification are retained. Neither file is a training input; no cpuQ writer stopped.
- A static Python implementation module required by the frozen runtime source manifest
  was recovered as an opaque source-code artifact (40,205 bytes, SHA verified).
  No hidden vehicle/driver-value or truth dataset was copied or parsed. Exact backend
  setup then verified every required runtime source hash and the MAPPO checkpoint.
- Combined preservation inventory: `/home/ubuntu/STAGING_MANIFEST_S0.json`,
  **175,358 entries / 34,859,092,139 bytes**, SHA256
  `ac87a129e7bcfbab622c63cfdbb1c383894b5f6d751c23b1f6d558a66aabd289`.
  This SHA-verified S0 superset is separate from the final accepted T2C runtime binding.
- Read-only real-bank audit: B143 decisions /75 snapshots resolved /68 previously
  documented unresolved LS records; R125/125 resolved. All **B7/R3** certified
  cooperative contexts resolve, **zero required cooperation snapshot inputs missing**,
  **zero data-missing bank examples**. Imported snapshot envelopes, snapshot payloads
  and source episode RESULT(.gz) hashes are covered by the combined inventory.
  Details `BANK_DEPENDENCY_AUDIT.json`; the known68 gaps are disclosed, not repaired
  by changing selection or evidence.
- Receipts on both hosts under `/home/ubuntu/cpuv_setup_20261105/`:
  `ENV_MANIFEST.json`, `ENV_VERIFY.json`, `RESCUE_CONFIRM_MANIFEST.json`,
  `RESCUE_CONFIRM_VERIFY.json`, `ENV_QUALIFICATION.json` (cpuV), and copy log (cpuQ).
- Prohibited-name and truth files excluded before content reads/copy; symlinks preserved
  only for admitted destinations. Dead cpuS confirm mounts skipped at source.

## SUMO parity — PASS
CPUV-PARITY-1 preregistered in REGISTRATION.md at system clock00:52+08.
P3 510 sim-s, then only if PASS P4 450 sim-s; hard cap960, no retries.
Same cpuQ reference identities as CPUS-PARITY-2. The960 is reserved inside the
3,500,000 owner research ceiling and must carry into clean1 cumulative accounting.
Execution on cpuV completed at unix1793812141 (~01:09+08), with no retries:
- P3: **510 charged/actual sim-s**, 210 steps; uncompressed RESULT SHA
  `90d82a9b398d60c2fa532da6c369033d0699ab2ce3045f826a949ac903f16cf2`,
  run-branch digest `57a9bc83d7d18ce8be270a8b464b5fda1133d379210c14943844475cd833c2c7`.
- P4: **450 charged/actual sim-s**, 150 steps; uncompressed RESULT SHA
  `ca85ea6d0682d734309f90f9eb30ab1a4fc9604fd6a7eecbfcf81a70de118d2b`,
  run-branch digest `d85a91e8ceadd0940a3ccce2f5913bafd4b1b661c25882985e7b7e231d896fb6`.
- Both full branch receipts, every step, all prefixes, continuation digests, final
  fingerprints, totals and complete decoded result objects equal cpuQ. Both have
  zero collisions and zero teleports. Wall time13.6s and11.3s respectively.
- Receipts `/home/ubuntu/cpuv_parity_20261105/{P3_COMPARE,P4_COMPARE,PARITY_PASS}.json`;
  exclusive execution claim and independent local engineering CHARGES.jsonl retained.
- Exactly **960 sim-s total**; clean1 cumulative accounting must include this spend,
  leaving **3,499,040 research +20,000 smoke** within the current approval.

## Integration status
Waiting for `E:/agent-work/c4-rerun-tracks/T2C_ACCEPTED.json`.
Current pending T2C budget quote exceeds the approved ceiling and its production
adapter is absent; see `E:/agent-work/_sol/S_LAUNCH_BLOCKERS_20261105.md`.

S0 qualification receipt `/home/ubuntu/cpuv_setup_20261105/S0_READY.json`;
compact local copies `E:/agent-work/_sol/cpuv_receipts/`. At qualification, cpuV had
162,012,954,624 bytes free. No clean1 training, end-to-end smoke or final safety gate
has been dispatched; parity does not substitute for the final-runtime safety gate.

## Fresh-session resume — 2026-11-05 01:38 +08
Read-only remote verification retained S0; no parity/copy job was repeated. Remote
`S0_READY.json` SHA is `9781e6b4372e833b603de42cee68bd41e6ff61d91f15df7db553ea5aa5505420`;
`PARITY_PASS.json` SHA is `924be730fd0453dfda0b65598cbac5ef080db356a75ce35a9241ab25be39ab74`.
The staging inventory, mandatory shim/env and current pip freeze still match the
recorded hashes above. No research process matched the launch/copy/parity modules;
prior own copy PIDs are absent. Clean1 root and launch source have not been created.
Free disk is 162,010,009,600 bytes. This verification spent **zero new sim-s**.
T2C fix worker PID25084 is active locally; acceptance remains absent. Existing
uncommitted T2C fixes are preserved. See `S_LAUNCH_RESUME_20261105.json` for the
durable resume checkpoint and remaining prerequisites.

The later owner amendment at REGISTRATION.md 01:39 ("approve the cut") supersedes
the original research allowance: **5,000,000 sim-s including smoke**. Parity960
still counts inside that total. With the launch instruction's20,000 smoke cap
reserved, remaining research is4,979,040. The active T2C worker owns the revised
quote and production backend; acceptance remains the explicit integration gate.
