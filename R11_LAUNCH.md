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
