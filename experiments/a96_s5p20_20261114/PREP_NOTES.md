# A96 S5-p20 prep notes (2026-11-14): S5 closing-lane niche-spread WM search at D2800_p20, gpuR

Status: PREPARED and INSTALLED on gpuR at /root/autodl-tmp/a96_s5p20 (SETUP_OK). **NOT LAUNCHED.** Zero SUMO, zero
training, zero WM rollouts, zero GPU. /root/autodl-tmp/a86 was read only (sha/cmp/T_START/T_END/chain log). On the
A800, the only actions were sha256sum and a scp read of a75p20/p20/regime_STATES_p20.json; a96_s4p20 was not touched.
No process was signalled. REGISTRATION and the ledger were not edited.

## 1. What changed vs S5 (A86, 60 %): only the cell
- **Code:** ship/a75ss.py, a75_wmforms.py, wm_fix_crn.py, a85s4.py and a86s5.py are byte-identical to S5's. The
  a86s5 entry point is cell-parametric: `--cell D2800_p20` gives STRATEGY_SEARCH_D2800_p20.json, LOG_D2800_p20.jsonl,
  TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED_D2800_p20.json, **TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p20.json**
  (the SUMO list), NICHE_TOP_D2800_p20.json and ARMS.json.
- **Search cfg:** byte-identical. A96S5P20_SEARCH_CFG.json == A86S5_SEARCH_CFG.json, sha 3adc2663 (checked with cmp on
  gpuR). Identical items: seed 20261187; exclude_lane_sets [[1],[2],[1,2]]; niching (parents, finalists, top_cap 2);
  n_init 72, generations 3, children 20, immigrants 12, n_elite, finalists 18, K_final 6, n_screen_states 18,
  n_final_states 30; grids, forms and novelty_weight 40; harm_cal. H 300, K_screen 4 and the scorer are code defaults
  and unchanged. Top-12 rule: --top-bank default 12, cap 2 per niche.
- **Launcher:** same `--max-rows 64 --workers 3`, no MPS, WZPROP_SPLIT_ACTIONS 32, same WM ckpt path. Changed:
  `--states /root/autodl-tmp/a96_s5p20/p20/regime_STATES_p20.json` (sha checked before the start), `--cell D2800_p20`,
  install paths, labels, and the log, lock and output names. Full diff: DIFF_vs_S5.txt (e5a58e12).
- **WM checkpoint:** wm_s.pt 6da9d333 on gpuR in both a96_s5p20/a75 and a86/a75 (the same sha S5 used).
  Bundle a75_search_box.tgz 9eb32080. The unpacked bundle tree is identical to a86/a75 (`diff -r`).
- **States:** D2800_p20, 30 valid states. regime_STATES_p20.json 7860a5d1 is the registered S3/S3b file.
  - Copy path: A800 /root/autodl-tmp/a75p20/p20 -> laptop inputs/ -> pkg tar -> gpuR, sha verified at each hop. It
    also equals experiments/a75_holdforms/search_p20/p20.
  - gpuR had no p20 states before this.
  - **Cell-inherent difference:** p60 has only 18 states, so S5 screened and finalled on all 18. At p20, screening
    draws 18 of 30 states and the finals use all 30 states (n_final_states 30). S3b did the same.

## 2. Items in the 60 % config that come from earlier outcomes
- **harm_cal:** per-group a/b/margin, n 16. It was fitted on SUMO U0 outcomes at p60 (HARM_CAL_D3200_p60; see
  REGISTRATION lines 9448 and 11579). No p20 fit exists, and I did not invent one. It is kept unchanged, which matches
  the registered precedent (A73 and S3/S3b at p20: "fitted at p60; applied as is at p20 - caveat"). It is part of the
  scorer, so keeping it is what an exact copy requires. **Caveat to disclose:** the calibration was fitted at p60.
- **No other 60 %-derived values:**
  - The incumbent and comparators (wall_g_solo2, solo2) are fixed hand plans in strategy_search.
  - The exclusion list and niche definition are structural.
  - The novelty reference is the hand library.
  - The S5 config contains no reference score, no exclusion fitted on S5 results and no incumbent score.
  - f_vs_base is computed per run against its own base.

## 3. Checks (exactly S5's: CPU pytest + box setup check; no smoke, as S5)
- **Laptop tests:** `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I -B -m pytest -q -p no:cacheprovider tests`
  gives **20 passed** (tests/test_a86s5.py 537c2a65 unchanged, on the identical code).
- **gpuR setup_s5p20.sh** (CPU only, nice 19) gives `GEN0_OK s2b_keys_reproduced 229 s4_admitted 221 s5p20_pop 72
  s5p20_admitted 72 rejected_seeds 28 rejected_random 12 niches 15 gen0_key_sha bf9583a4f08768a5`, then IMPORT_OK and
  SETUP_OK. The MANIFEST_A96, bundle MANIFEST, wm_s.pt and ship==bundle checks all pass.
  - check_s5p20_gen0.py differs from S5's check in one way: part 3 uses the p20 states.
  - The p20 generation-0 key sha bf9583a4f08768a5b772ed59a146c1ce92807e6100acf15b764579a50ee4fb2e was computed on
    the laptop and reproduced on gpuR.
  - p20 generation-0 niches: gap:L01 3, gap:L012 13, other:L012 1, pulse:L01 4, pulse:L012 13, pulse:L02 3,
    rolling:L01 2, rolling:L012 4, rolling:L02 3, taper:L01 3, taper:L012 3, taper:L02 4, zone:L01 2, zone:L012 4,
    zone:L02 10.
- **Stub rollout count** (tools_local/stub_rollout_count.py 2ec5cf69, with STATES_F/CELL env added):
  - p60 reproduces S5's 2,004 generation-0 rollouts.
  - p20: generation 0 has 852 rollouts. The stub total is 3,208 (generations [852, 1228, 1660, 2116] cumulative,
    finals 1,092). See tests/STUB_ROLLOUT_COUNT_p20.txt.

## 4. Hashes (sha256)
| file | sha256 |
|---|---|
| a96_s5p20_pkg.tar (laptop == gpuR) | 71b301a26b73abcf3e3a8cb83086f038a351430d3eef7e7b8d5afe4a4a8650ff |
| cfg/A96S5P20_SEARCH_CFG.json (== A86S5) | 3adc266397fad96a11430e4251f6155eaf4a8ad90b47ffe7b0ba50b1e9b9c215 |
| p20/regime_STATES_p20.json | 7860a5d194fa95bec102b8f113c4565606c09a5cde523a42a996ae20e67d4f8a |
| run_search_s5p20.sh | 108dceedda2daed95789b80c0a033cbe61c66817608fab79fa3059c89309fc86 |
| start_s5p20.sh (queue/launch) | 968f8d5a61043c3e226bc57a6cc62370c5246f58720b56b53cf5fecb1ddfc26d |
| setup_s5p20.sh | bd4bbcdd547a1233e4b27c09cfeaf93be7233d3fe8b318d2477aea55c402e0f9 |
| check_s5p20_gen0.py | a7a2ee799995b309832cfa004306dcebf7909298a714bbd5771c788f373dd880 |
| ship a86s5 / a85s4 / a75ss / a75_wmforms / wm_fix_crn | 71dda3e8 / 7ca82976 / cae54dc1 / 3af8f45d / 62e8e38f (full in pkg/MANIFEST_A96.sha256) |
| wm_s.pt (gpuR a96 == a86) | 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 |
| a75_search_box.tgz | 9eb32080ea789219202f61f1819944870bb28c428f4d417762ca310631a63277 |
| DIFF_vs_S5.txt | e5a58e12e8b23e1a04289a71b6379c5f462a97504d88a520a1c007b6965bcedc |

All .sh files are LF-only (0 CR, checked on gpuR).

## 5. Launch (coordinator only, after registration; NOT done)
```bash
timeout 30 ssh -o BatchMode=yes gpuR 'setsid -f nohup bash /root/autodl-tmp/a96_s5p20/start_s5p20.sh > /root/autodl-tmp/a96_s5p20/start_s5p20.nohup 2>&1 < /dev/null' < /dev/null
```
- **start_s5p20.sh refuses to start** if SETUP_OK is missing, the lock start_s5p20.lock exists, out_s5p20 or its
  .meta exists, or any compute process is on the GPU. The gpuR GPU was idle at prep (1 MiB).
- **Log and done marker:** /root/autodl-tmp/a96_s5p20/chain_s5p20.log gets `S5P20 start <date>`, then
  `S5P20 rc <N> <date>`, then `START_DONE`.
- **Run sidecars:** out_s5p20.meta/{run.log,RC,T_START,T_END,CFG_SHA256,STATES_SHA256,SHIP_SHA256,SHA256SUMS}.
- **Outputs:** /root/autodl-tmp/a96_s5p20/out_s5p20/:
  - TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p20.json (the SUMO list)
  - TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED_D2800_p20.json
  - NICHE_TOP_D2800_p20.json
  - STRATEGY_SEARCH_D2800_p20.json
  - LOG_D2800_p20.jsonl
  - ARMS.json
  - crn_log/
- **Withdraw:** stop by the pid in start_s5p20.lock/pid and its children only, never by name.
- **Copy back:** `tar -C /root/autodl-tmp/a96_s5p20 -cf - out_s5p20 out_s5p20.meta chain_s5p20.log`, then on the
  laptop run `sha256sum -c ../out_s5p20.meta/SHA256SUMS`.

## 6. Runtime estimate
- **S5 at 60 % on gpuR:** T_START 1794580688 to T_END 1794585214, which is **4,526 s (75.4 min)** for 5,792 rollouts.
  That is 1.28 rollouts/s including startup, with 3 workers and no out-of-memory.
- **p20 rollout estimate:** about 3,300, with an upper bound of about 4,500. The stub total is 3,208; correcting with
  the S5 real/stub ratios (generations 0.96, finals 1.17) gives about 3,300. The upper bound allows the 30-state
  finals to activate more.
- **Rate:** S3b ran p20 about 18 % faster per rollout than S2b did p60 on the A800.
- **Estimate: about 35-60 min (central about 45 min); worst about 75 min.**
- **Out-of-memory risk** should be at or below S5's, because fewer actions per state activate at p20. This has not
  been measured.

## 7. Doubts
1. harm_cal was fitted at p60 and is applied as is (disclose; same as A73/S3b).
2. Same seed as S5 (as instructed). S3/S3b used seed+1 for p20 (20261177 vs 20261176); here the seed is unchanged.
   Generation 0 still differs from S5 because the states differ (key sha bf9583a4 vs 275c80a9).
3. The finals use 30 states, not S5's 18. This is a property of the cell, not a config change, but scores are not
   comparable in scale to S5.
4. The same S5 limits carry over: novelty is likely inactive, the real WM path was not exercised before launch, and
   tests use the StubWM only.
