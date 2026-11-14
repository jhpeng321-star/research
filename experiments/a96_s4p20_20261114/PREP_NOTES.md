# A96 S4-p20 prep notes (2026-11-14 ~02:50 UTC+8): the S4 WM plan search at D2800_p20

Status: PREPARED and INSTALLED on the A800 (/root/autodl-tmp/a96_s4p20). NOT LAUNCHED. Zero SUMO, zero GPU, zero
training, zero WM rollouts. Nothing written into /root/autodl-tmp/a85 or /root/autodl-tmp/a75p20 (read/cp only); no
process signalled; REGISTRATION and ledger not edited.

## 1. What changes vs S4 (a85, D2800_p60)
Only the cell: `--cell D2800_p20`, `--states /root/autodl-tmp/a75p20/p20/regime_STATES_p20.json` (sha 7860a5d1, the
registered A73/S3 p20 file, read in place; the launcher refuses on a sha mismatch and records STATES_SHA256, exactly as
run_search_s3b.sh does), plus paths/labels/out dir. The cfg holds no cell key, so the cfg is BYTE-IDENTICAL to S4
(A96S4P20_SEARCH_CFG.json = A85S4_SEARCH_CFG.json, sha 0a80c0f4...): seed 20261185, exclude_lane_sets [[1,2]], n_init
216, 4 generations x (60 children + 36 immigrants), 6 elites, finalists 18, K_final 6, novelty 40, grids, hold_only,
harm_cal. Code a85ship/ is a byte copy of a85/a85ship (a85s4 7ca82976, a75ss cae54dc1, a75_wmforms 3af8f45d,
wm_fix_crn 62e8e38f); bundle a75_search_box.tgz 9eb32080 (cp from a85/), wm_s.pt 6da9d333, H/look-ahead, --max-rows 64,
WZPROP_SPLIT_ACTIONS 32, --workers 14, MPS client env, unchanged TOP writer (top 12). Full diff: DIFF_vs_S4.txt.

## 2. Items in the S4 cfg tied to 60 % (none re-derived, none invented)
1. `exclude_lane_sets [[1,2]]`: motivated by S2b p60 outcomes (43/63 finalists used {1,2}). It is a structural rule
   (the definition of S4), kept as is. Whether {1,2} also dominates at p20 was not checked (would need S3b outcomes).
2. `harm_cal`: fitted on p60 SUMO outcomes (REGISTRATION l.9448 "fitted at p60"). No p20 fit exists. Kept unchanged,
   the same practice and caveat as A73 (l.11579) and S3b (A79 l.11738). Faithful copy = same values; flag the caveat.
3. Hand-library seeds, the 11 removed {1,2} seeds, comparators (wall_g_solo2, solo2), novelty set: hand-written, not
   outcome-derived; identical.
4. State counts: n_screen_states 18 / n_final_states 30 meant "all 18 states" at p60. The p20 file has 30 valid states,
   so screening uses a seed-drawn 18 of 30 and finals use all 30 (as in S3/S3b). Inherent to the cell, not a cfg change.
5. Seed: kept 20261185 as instructed (S2b/S3b convention gave each cell its own seed: 20261176/20261177).
6. States carry the A78 HOLD-OUT OVERLAP FLAG (worlds 989200-202, 989350-352 are WM-D reserved list; not in forbidden
   blocks); same disclosure as S3/S3b.

## 3. Checks (same as S4's, no more)
- Laptop: S4's tests/test_a85s4.py unchanged (code is byte-identical): 10 passed.
- Box setup_s4p20.sh (CPU only, nice): bundle sha + MANIFEST, wm_s.pt pin, MANIFEST_A96, ship == bundle and == a85,
  cfg == S4 cfg, then check_s4p20_gen0.py (copy of check_s4_gen0.py with p20 states and the S3b p20 reference, keys
  only): `GEN0_OK cell D2800_p20 states 30 s3b_keys_reproduced 229 s4p20_pop 233 s4p20_admitted 221 rejected_seeds 11
  rejected_random 35`, SETUP_OK. No GPU smoke (S4 did none).

## 4. Hashes (sha256)
| file | sha256 |
|---|---|
| a96_s4p20_pkg.tar | a8471c9aa73cc3c474f4df34b329208a438a0e63bd8d7d162cef56c60d00d636 |
| cfg/A96S4P20_SEARCH_CFG.json (= S4 cfg) | 0a80c0f4c6a8858faf4430c8e55be04a9abc77779337cd91921e1df873b82f5e |
| run_search_s4p20.sh | 1c022cc29018a80545078c510e78b4d9085b44e179b6ff63005447c7c09111ab |
| queue_s4p20.sh | 20eb90459559e92f84b5678b3e2a4219eff64e40c381ce16c846d39001cc5e20 |
| setup_s4p20.sh | e3597fce2437fba02c67c671c55012b1fc444c581693ae686668f419c8de58b3 |
| check_s4p20_gen0.py | fcd4c0f46b7f685fa7d2db1af715336c5cad516ccf98d73d9d68e9f85c8a4698 |
| a85ship a85s4 / a75ss / a75_wmforms / wm_fix_crn | 7ca82976... / cae54dc1... / 3af8f45d... / 62e8e38f... (= S4) |
| a75_search_box.tgz | 9eb32080ea789219202f61f1819944870bb28c428f4d417762ca310631a63277 |
| regime_STATES_p20.json (read in place) | 7860a5d194fa95bec102b8f113c4565606c09a5cde523a42a996ae20e67d4f8a |
| wm_s.pt | 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 |
| setup_s4p20.out (box) | 7093a33bb3118b7efd84b67f4e5f111d592b5052d31555f541fc421c4b391270 |
| DIFF_vs_S4.txt | 5e27a64aba685ec5c016b8f7e7ffce30dc9d5672d77837beaab2924095797bf0 |
All .sh/.py: 0 CR bytes (`file`: ASCII bash).

## 5. Queue / launch (NOT started)
queue_s4p20.sh: lock dir queue_s4p20.lock (pid inside), refuses if out_s4p20(.meta) exist, waits while any
`python -B -m (a75ss|a85s4) --states` runs (none now), same MPS env, logs MPS/GPU, runs once, writes `S4p20 rc N` and
QUEUE_DONE. Note: S4's queue logged "no MPS control daemon" because `pgrep -x nvidia-cuda-mps-control` cannot match
the 15-char comm; daemon pid 5702 was (and is) alive, so S4 most likely ran under MPS. The p20 queue uses pgrep -f
(verified match); log line only, env identical.
```bash
timeout 30 ssh -o BatchMode=yes -p 37084 root@connect.bjb1.seetacloud.com 'setsid -f nohup bash /root/autodl-tmp/a96_s4p20/queue_s4p20.sh > /root/autodl-tmp/a96_s4p20/queue_s4p20.nohup 2>&1 < /dev/null' < /dev/null
```
Log /root/autodl-tmp/a96_s4p20/chain_s4p20.log; done marker `QUEUE_DONE` (+ out_s4p20.meta/RC). Outputs in
/root/autodl-tmp/a96_s4p20/out_s4p20: STRATEGY_SEARCH_D2800_p20.json, TOP_OFFER_CONDITIONS_D2800_p20.json,
LOG_D2800_p20.jsonl, ARMS.json, crn_log/; sidecars out_s4p20.meta (T_START/T_END, CFG/STATES/SHIP_SHA256, SHA256SUMS,
run.log). Withdraw before start: kill the pid in queue_s4p20.lock/pid only.

## 6. Runtime
S4 p60: 13,930 s (3 h 52 min), 599 evaluated, 23,914 rollouts. S3b p20 (same a75ss path, same sizes, 14 workers):
5,159 s (1 h 26 min), 613 evaluated, 10,458 rollouts (S2b p60: 29,518), i.e. fewer CAVs -> fewer plan activations ->
fewer rollouts, despite finals on 30 states. Expect S4-p20 about 1.3-2 h (S4/S2b rollout ratio 0.81 applied to S3b
gives ~1.2 h; allow for the filter shifting activation). Headroom: the only exposed knob is --workers (14); 16 was
measured to exceed 80 GB (5.1 GB/worker, A79), and wm_fix_crn is designed so rows do not depend on worker/chunk but
byte-equality across worker counts was never tested. Recommendation: keep 14 (no safe speed-up). GPU idle (46 MiB),
50 G disk free; install 19 MB.

## 7. Copy back (after `S4p20 rc 0`)
`timeout 300 ssh ... 'tar -C /root/autodl-tmp/a96_s4p20 -cf - out_s4p20 out_s4p20.meta chain_s4p20.log' < /dev/null > .../a96_s4p20_out.tar`,
then `cd out_s4p20 && sha256sum -c ../out_s4p20.meta/SHA256SUMS`. Registration of A96 still to be written by the coordinator.
