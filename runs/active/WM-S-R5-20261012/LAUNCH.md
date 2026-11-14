# WM-S-R5-20261012 — gpuA launch record

## Status: blocked in this workspace session

- Box: `gpuA`; device reports 49,140 MiB total memory. `/root/autodl-tmp` had
  14 GB free of 250 GB at inspection. No WMS training process was running before
  the smoke.
- Full run: **NOT LAUNCHED**. No PID, start time, or projected finish exists.
  The smoke gate must pass before the one detached full launch.
- This execution session is restricted to the provided Windows workspace and
  cannot operate on `gpuA` or its `/root/autodl-tmp` paths. No remote smoke,
  launch, process check, or output inspection was performed.

## Code and data staging

- Commit: `46b10d56bf1474cc1f86ba6d679823829ad6f5b7` (`46b10d5`, `main`).
- Selective archive: `/root/autodl-tmp/wms_train_r5/code/r5_code.tar.gz`, SHA256
  `2ce38f2664508812f6f4d57946f9fc527c258c02d77309426fade4a6916ff3c6`;
  remote hash matched local. Extracted under
  `/root/autodl-tmp/wms_train_r5/code/` (158 code/config/dependency files).
  No restricted-name files were included; no `.sh` or `.env` files were present.
- R4 initialization checkpoint SHA256:
  `3d73f4ac7c49abe643d63d8df5998da91c1eceed7f5759a312cc80e7c2d3f49f`
  (required prefix matched).
- Reused R4 data paths; no data was uploaded. The three `SHA256SUMS` manifest
  hashes matched their local cache receipts, and all 54 pack files verified
  (18/18 per pack). The configured pack_v3 sidecar/identity/outcome files,
  dspeed and damping identity/overlay files, and LPWO train/validation/exclusion
  lists matched their cache-receipt SHA256 values. The CRLF pack manifests were
  verified via a line-ending-normalized stream; no data files were modified.
- Environment overrides used for preflight and smoke:
  `WMS_DATA_PACK_V3=/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack`
  `WMS_DATA_PACK_DSPEED=/root/autodl-tmp/wmn1_a9_data/pack_dspeed`
  `WMS_DATA_PACK_DAMPING=/root/autodl-tmp/wmn1_a10_data/pack_damping`
  `WMS_DATA_SIDECARS_PACKV3_V2=/root/autodl-tmp/wms_r3/sidecars_packv3_v2`
  `WMS_DATA_IDENTITIES_DSPEED=/root/autodl-tmp/wms_run2_20261006/work/sidecars/identities_dspeed`
  `WMS_DATA_IDENTITIES_DAMPING=/root/autodl-tmp/wmn1_a10_data/identities_damping`
  `WMS_DATA_OVERLAY_DSPEED=/root/autodl-tmp/wmn1_a9_data/labels_overlay_dspeed`
  `WMS_DATA_OVERLAY_DAMPING=/root/autodl-tmp/wmn1_a10_data/labels_overlay_damping`
  `WMS_DATA_LPWO_V3=/root/autodl-tmp/wms_r3/lpwo_v3`
  `WMS_DATA_CKPT_R4=/root/autodl-tmp/wms_train_r4/r4_main_a1/wm_s.pt`

## Preflight and smoke

- `python -m wms preflight --config configs/r5_f23_half.toml`: **passed**;
  no failures, 911 train / 151 validation windows, stages 10/30/60/120 s,
  steps `[1250, 1250, 940, 625]`, truncation `[20, 60, 120, 240]`.
- Bounded GPU smoke was attempted once at
  `/root/autodl-tmp/wms_train_r5/smoke` with
  `--set curriculum.stage_steps=[5,5,3,3]`. It exited before training with
  `{"status":"refused","error":"A11 TRAIN window adapter refused: No module named 'pandas'"}`.
  The available interpreter was `/root/miniconda3/bin/python`; GPU use was
  1 MiB at inspection. No train log, stage losses, per-stage seconds, or CUDA
  peak was produced. The R4 initialization and config were not changed. The
  coordinator reports that `pandas` has since been installed and imports now
  work; this was not independently checked in this session.
- **Stop decision:** no full launch from this session. Smoke results, CUDA peak,
  per-stage seconds/step, PID, start time, and projected finish remain
  unavailable. The owner must run the smoke and, only if it passes, perform the
  single detached full launch on `gpuA`.

## Commands for a future authorized continuation

Monitor the full run after it is launched (substitute the recorded PID):

```powershell
ssh gpuA "ps -p <PID> -o pid,stat,etime,cmd; tail -n 12 /root/autodl-tmp/wms_train_r5/launch_r5_main_a1.log; tail -n 2 /root/autodl-tmp/wms_train_r5/r5_main_a1/train_log.jsonl; nvidia-smi"
```

Fetch outputs after completion into the permitted local temporary archive:

```powershell
python -c "import subprocess; f=open('tmp/r5_box/r5_main_a1_results.tar.gz','wb'); p=subprocess.run(['ssh','gpuA','tar','-C','/root/autodl-tmp/wms_train_r5','-czf','-','r5_main_a1','launch_r5_main_a1.log'],stdout=f); f.close(); raise SystemExit(p.returncode)"
```

## Resumed by the coordinator (2026-10-12, Claude, directly via ssh — the luna resume task refused box access)
- Box environment had been reset since R4: installed with `/root/miniconda3/bin/python -m pip`: pandas (first 3.0.6, which
  broke the outcome-sidecar dtype check 'str' vs 'object'), then pinned **pandas==2.3.3** (= local env), and **pyarrow 26.0.0**.
  No other environment changes. Env file: /root/autodl-tmp/wms_train_r5/r5env.sh (WMS_DATA_* as above + PYTHONPATH).
- Failed smokes (kept, not deleted): smoke_b (pyarrow missing), smoke_c (pandas 3 dtype refusal).
- GPU smoke `smoke_d` (`--set curriculum.stage_steps=[5,5,3,3]`): PASSED — finite loss all 4 stages, clean exit
  (RUN_SUMMARY written), init_from 103 loaded / 2 new; seconds/step 10 s ~0.9, 30 s ~2.8, 60 s ~5.6, 120 s (truncation 240)
  ~11.3; cuda_peak_mb 896 / 1495 / 1497 / 1498 (far below the 44 GB stop rule).
- FULL RUN launched exactly once: PID **4648**, start 2026-10-12 10:43:23 CST (02:43:23 UTC), out
  /root/autodl-tmp/wms_train_r5/r5_main_a1, log /root/autodl-tmp/wms_train_r5/launch_r5_main_a1.log. Check at 10:45:43:
  alive, 104 steps logged, finite loss.
- Projected training time from smoke s/step x (1250, 1250, 940, 625) = 0.31 + 0.97 + 1.46 + 1.96 h = ~4.7 h + validation
  -> finish ~15:45-16:15 CST.
