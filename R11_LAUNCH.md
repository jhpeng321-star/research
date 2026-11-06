# R11_LAUNCH - round 11 launch record (S-R11LAUNCH), 2026-10-24

## Registration
Commit 4d8de32b (branch c4/v10r11; includes review changes C1 head-saturation gate rule, C2 launch choice, C4 RESULT_5B resolved).
Launch choice (C2): R11_HEAD_INIT=zero (gate passes weight 9.067, lr scale x1000), R11_INIT default = round-10b stage300 (else best). No other config change.
Run dir: /root/autodl-tmp/c4/v5/runs/c4_wm_v10_round11 (smoke: runs/smoke_r11, log smoke_r11.log, verdict smoke_r11.verdict). Gate log: gate_r11.out.

## Ship (02:10 CST)
Repo dir /root/autodl-tmp/c4/v5/repo_r11_4d8de32b = cp -a of repo_r10b_6a6f2afd (6a6f2afd) + the 38 files changed 6a6f2afd..4d8de32b as LF git blobs (delta tar in ship_r11_4d8de32b/, delta md5 verified 38/38).
Verification: all 7779 tracked files at 4d8de32b checked on gpuR by git blob hash: 547 exact, 7227 identical after CR removal (round-10 base copy has CRLF), 5 absent (R8_A/B/C.md, tests/test_merge_core_v2_onpolicy.py, tests/test_merge_core_v2_seqtf.py; not used by the gate), 0 differ. Gate .sh md5 a3bafa721bd510c7b7a819aa5ecb83a3 matches local blob.
Disk after copy: 467 MB free.

## Waiter
Round 10b had 90/90 steps logged at 02:07 but was still in final validation (no RUN_SUMMARY.json, PID 400794 alive). Waiter ships as round11_wait.sh, PID 423697 (gate_r11.pid): waits for RUN_SUMMARY.json + PID 400794 gone + wm_s_stage300.pt, then exec's the gate.

## Run (gate started by the waiter at 02:34:10 CST, after RUN_SUMMARY.json of round 10b appeared and PID 400794 exited)
Gate: `R11_HEAD_INIT=zero bash scripts/round11_gate.sh repo_r11_4d8de32b`, init = runs/c4_wm_v10_round10b/wm_s_stage300.pt (default), weight 9.067, lr x1000, zero head (INIT_FROM.json: follower_courtesy_init "zero", expanded []).
Smoke (runs/smoke_r11; verdict "smoke: ok"; smoke_r11.gate.json errors []): 4 optimizer steps, cuda_peak_mb 3308.5, courtesy rows 4298 (last loss 0.689), contrast controls 72111, partnered controls excluded 4015, cut-in events 88 (454 event rows), TF zero-mass events 0, head_range_bounds [-9, 3] passed (C1 rule), all 3 checkpoints reload strictly, head moved from zero init, nothing changed outside the trainable set. Smoke took about 51 min (02:34 - 03:25).
Full run: runs/c4_wm_v10_round11, PID 423697 (same PID as the waiter, exec chain), log gate_r11.out, started about 03:25 CST. validation cap 16 windows. GPU 4.3 GB total used at step 20.

## Verification at step 20 (04:34 CST, read-only)
- Alive, 20 rows in train_log.jsonl, all values finite except merge_pivot_on_diagnostic_nll = inf at step 2 (diagnostic only, also present at round 10b step 2).
- Loss 159-382. Courtesy loss finite every step (0.45-0.94, last 0.774), courtesy rows 472+ per step, contrast controls 12172+ per step, cut-in events 10+ per step, TF zero-mass events 0, merge_pivot_on_objective_disabled 1 in all steps.
- Head (C1 bounds -9/+3, +-12): per-step min -1.30, max +1.49, mean between -0.42 and +0.36. Step 1 head exactly 0 (zero init no-op).
- wm_s_latest.pt (step 20) vs round-10b stage300: 151 old tensors bitwise equal (all 8 merge_pivot tensors included), 12 changed = gate_g_max_logit, gate_tau_raw, temporal.{0,2}.{weight,bias}, follower_courtesy.net.{0,2,4}.{weight,bias}; no unexpected change; all finite (script gpuR:/root/autodl-tmp/c4/v5/r11_cmp.py).
- Step time (batch 4, stage 120): median 228 s, mean 200 s (range ~93-240 s; alternates short/long), GPU shared with the reviewer-check job at times. Disk 3.3 GB free.
