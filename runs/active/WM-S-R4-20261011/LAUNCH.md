# WM-S-R4-20261011 — gpuA launch record

## Staging and data

- Box: `gpuA` (`/root/autodl-tmp`, 14 GB free before staging); device: RTX 4090 D,
  49,140 MiB total GPU memory.
- Code commit requested and staged: `d4de8f75baad14f920d8dcf90ce663065b7b1be4`.
  Archive: `/root/autodl-tmp/wms_train_r4/code/code_d4de8f7.tar.gz`, SHA256
  `76eb871d3736d9b8222d38367d2ddd98a271205b2db9b9454bb62453e0a8ea78`.
  The selective `git archive` contains the R4 package/config files only, excludes
  restricted-name files, and normalizes every included `.sh` to LF. Remote hash
  matched. Extracted root: `/root/autodl-tmp/wms_train_r4/code/`.
- Registration discrepancy: `REGISTRATION.md` line 4 says code `711b078`; this
  launch followed the task-directed `d4de8f7` commit and configuration.
- Reused R3 data locations; no datasets were uploaded. All three pack SHA256SUMS
  manifest hashes matched their local `CACHE_RECEIPT.json`, and all 18 files in
  each pack verified (54/54). Required sidecar, identity, overlay and configured
  LPWO list file SHA256 values matched local cache receipts. R4 resolved to the
  expected 911 train / 151 validation windows; `pack_v3` resolved as R3's `pack`.
- Box environment used for preflight, smoke and full run:
  `WMS_DATA_PACK_V3=/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack`,
  `WMS_DATA_PACK_DSPEED=/root/autodl-tmp/wmn1_a9_data/pack_dspeed`,
  `WMS_DATA_PACK_DAMPING=/root/autodl-tmp/wmn1_a10_data/pack_damping`,
  `WMS_DATA_SIDECARS_PACKV3_V2=/root/autodl-tmp/wms_r3/sidecars_packv3_v2`,
  `WMS_DATA_IDENTITIES_DSPEED=/root/autodl-tmp/wms_run2_20261006/work/sidecars/identities_dspeed`,
  `WMS_DATA_IDENTITIES_DAMPING=/root/autodl-tmp/wmn1_a10_data/identities_damping`,
  `WMS_DATA_OVERLAY_DSPEED=/root/autodl-tmp/wmn1_a9_data/labels_overlay_dspeed`,
  `WMS_DATA_OVERLAY_DAMPING=/root/autodl-tmp/wmn1_a10_data/labels_overlay_damping`,
  `WMS_DATA_LPWO_V3=/root/autodl-tmp/wms_r3/lpwo_v3`.

## Preflight and smoke

- Remote `python -m wms preflight --config configs/r4.toml`: passed, no failures;
  resolved 10/30/60/120 s curriculum, 2500/2500/1875/1250 registered steps.
- GPU smoke command used `--set curriculum.stage_steps=[20,20,10,10]`, output
  `/root/autodl-tmp/wms_train_r4/smoke`. Passed all 60 steps with finite loss;
  `gate_g_by_k` was logged and monotone, final and best checkpoints were written.
  The checkpoint contains `gate_g_max_logit=-0.83019656` and
  `gate_tau_raw=49.97061539` (initialized for `g_max=0.30`, `tau=50`). Peak GPU
  allocation was 21,941 MiB.

## Full run

- Exactly one detached launch; no relaunch:
  `/root/autodl-tmp/wms_train_r4/r4_main_a1`
- Start time: `2026-10-10T18:12:50Z` (gpuA UTC clock).
- PID: `2927`.
- Launcher log: `/root/autodl-tmp/wms_train_r4/launch_r4_main_a1.log`.
- Expected duration: approximately 9–10 hours of training; registered cap is
  10 hours for training / 12 hours overall including smoke.
- Startup check after about 6 minutes: process remained active; `train_log.jsonl`
  grew to 421 KiB and reached step 423 with finite loss and logged `g(k)`;
  all registered train/validation windows were reported. GPU memory use at the
  check was 11,790 MiB.

### Monitor

```powershell
ssh gpuA "ps -p 2927 -o pid,stat,etime,cmd; tail -n 12 /root/autodl-tmp/wms_train_r4/launch_r4_main_a1.log; tail -n 2 /root/autodl-tmp/wms_train_r4/r4_main_a1/train_log.jsonl; nvidia-smi"
```

### Fetch results after completion

Streams the output and launcher log into the permitted local temporary tarball
location without creating a duplicate archive on gpuA:

```powershell
python -c "import subprocess; f=open('tmp/r4_box/r4_main_a1_results.tar.gz','wb'); p=subprocess.run(['ssh','gpuA','tar','-C','/root/autodl-tmp/wms_train_r4','-czf','-','r4_main_a1','launch_r4_main_a1.log'],stdout=f); f.close(); raise SystemExit(p.returncode)"
```

## Handling note

A blanket local cache integrity pass unintentionally hashed one non-configured,
holdout-named LPWO metadata file. It was not included in the staged archive,
transferred to gpuA, selected by R4, or inspected as dataset contents. The pass
was stopped for that file thereafter; only the three R4-configured LPWO files
were remotely verified.
