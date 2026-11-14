# A85 S4 prep notes (2026-11-13 ~21:25 UTC+8): WM plan search that may not use the two-through-lane form

Status: PREPARED and INSTALLED on the A800. NOT LAUNCHED. This task ran zero SUMO, zero GPU, zero training and zero
WM rollouts. Nothing was written into /root/autodl-tmp/a75 or a75p20, no process was signalled, and the
REGISTRATION and ledger were not edited.

## 1. What S4 is (only changes vs S2b)
1. **Lane-set filter**, applied at generation time. A candidate whose role-lane set `{r['lane'] for r in roles}`
   equals {1, 2} is rejected and replaced by a fresh draw from the same `random.Random(seed)` stream. The lane field
   is the SUMO lane index, and lane 0 is the closing lane. `SS.field_speed` reads the closing-lane field from lane 0
   (the module docstring says the same). The hand-written duo12 plans `wall_g_duo12`, `wall_g_duo12_gate0` and
   `duo12_v18..v21` all have lane set {1, 2}; this is asserted in the tests.
   - Initial random plans and immigrants: `SS.random_strategy` is drawn again until the plan is accepted.
   - Mutation children: both the parent choice and the mutation are drawn again (`rng.choice(parents)` + `SS.mutate`).
     A mutation that returns None is NOT replaced; run() drops it later, exactly as in S2b.
   - Hand-library seeds with lane set {1, 2} are left out of the initial population. For S4 that is 11 seeds:
     wall_g_duo12, wall_g_duo12_gate0, wall_g_alt21, wall_g_alt21_gate0, wall_g_relay21, wall_g_relay12,
     wall_g_duo12_stagger40, g2_gap1, g2_gap1_gate0, duo12_v18 and duo12_v20. duo12_v19 and duo12_v21 were already
     removed in S2b by the search grid. These are unchanged: the comparators (wall_g_solo2, solo2; lane set {2}, added
     to the finals as before), the novelty reference set (all seeds, including the removed ones) and extra_seeds
     (empty).
2. **Seed** 20261185 (S2b: 20261176).
3. **Output** /root/autodl-tmp/a85/out_s4 (sidecars in out_s4.meta).

Everything else is the same as S2b: cell D2800_p60, the same 18 states (n_screen_states 18 and n_final_states 30 both
use all 18 valid states; the new seed only changes their screening order), novelty_weight 40, the grids, n_init 216,
4 generations, 6 elites, 60 children, 36 immigrants, min_roles 2, hold_only, WM wm_s.pt 6da9d333, 14 workers, MPS
client env, WZPROP_SPLIT_ACTIONS=32 and the unchanged TOP writer (top 12, a75ss.main).

## 2. Implementation
- New wrapper `a85ship/a85s4.py`. It copies `a75ss.search_cell` and marks every changed line `# A85-S4`; generation
  0 is factored into `initial_population()` with the same draws. `main()` sets `a75ss.search_cell` to the S4 version
  and calls the unchanged `a75ss.main` (checkpoint pin, outputs, TOP writer). The filter is turned on by the cfg key
  `exclude_lane_sets: [[1, 2]]`. With the key absent or empty, the code runs exactly as a75ss. `res['a85']` records
  the filter, the rejected seeds, the random/mutation rejection counts, the per-generation generated counts and the
  sha of a85s4.py.
- `a75ss.py`, `a75_wmforms.py` and `wm_fix_crn.py` in a85ship/ are byte-identical copies.
- `DIFF_vs_S2b.txt` contains the function diff (a75ss.search_cell vs a85s4), the launcher diff and the cfg diff.

## 3. Hashes (sha256)
| file | sha256 |
|---|---|
| a85ship/a85s4.py (NEW) | 7ca82976e7183ce205d0591470a60281b553c2d03e8643d479d739b86c419261 |
| a85ship/a75ss.py (unchanged) | cae54dc11be5f5220f19eb7399efac2699c7cdf363a90b7fa202766d5b906bc0 |
| a85ship/a75_wmforms.py (unchanged) | 3af8f45d9b0a90d0ff3fb1e9931bb2a1af1f2d3b8759f6d2ba6c9c1908e476bd |
| a85ship/wm_fix_crn.py (unchanged) | 62e8e38f9ce7894d956314e6e030c03ed54f4becf1bef450da3c4b949960f46b |
| cfg/A85S4_SEARCH_CFG.json | 0a80c0f4c6a8858faf4430c8e55be04a9abc77779337cd91921e1df873b82f5e |
| S2b cfg A75S2B_SEARCH_CFG.json (ref) | 8f71eb5534093333c4fa042143daf4b66de81b727580975aed72ea398fc2491c |
| box/run_search_s4.sh | 293ed0367b297bb4c1c36ad1e0cd7710270f56af744baec6ac21d267d1590726 |
| box/queue_s4.sh | 58389938f9983c72bbe51c57bff5e6a58b9e96b2ce76f35155c45e965cf6ea3c |
| box/setup_s4.sh | b8543e7704df46789b0a4185d7c44c2107575b58da0c3a1af939b1f778295a09 |
| box/check_s4_gen0.py | e1ded2ab4463278d6f9f20b795c2550b5e503851ad2a5b1399480a2070c23f6d |
| a85_s4_pkg.tar (all of the above + MANIFEST_A85.sha256) | d007eb0bc2495500379012bd76fed5158573e65796c39bed94527698a70264e6 |
| a75_search_box.tgz (bundle; same file as box a75/ and a75p20/) | 9eb32080ea789219202f61f1819944870bb28c428f4d417762ca310631a63277 |
| strategy_search.py / wm_backend.py / regime_STATES.json (bundle) | 02e119c1... / 728fc917... / 74aef8f8... |
| wm_s.pt | 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 |
| S2b launcher run_search_s2b.sh / chain_s2b.sh (ref_box/) | f6c4afeb... / e18d1569... |
| tests/test_a85s4.py | 7f73c63bdc7551d00009479a6ecf2e3ded2121a28e328fe71ae476ca5c6c6c3a |
| DIFF_vs_S2b.txt | b029720e0889b0f7ece9eed30a9c762635e0a0be03f6627a80c0f88a6e6ff8a4 |

## 4. Tests (CPU, StubWM fakes; `python.exe -I -B -m pytest -q -p no:cacheprovider tests`): 10 passed
- The pinned inputs match. The cfg differs from S2b only in `seed` and `exclude_lane_sets`.
- Lane semantics: lane 0 is the closing lane, the duo12 plans have lane set {1, 2} and the comparators have {2}.
- **Filter off reproduces the S2b generation-0 keys.** All 229 keys equal the recorded S2b `screened[:229]` in order.
- Filter off: `a85s4.search_cell` and `a75ss.search_cell` give byte-identical JSON on a reduced StubWM search, for
  both the S2b cfg and the S4 cfg without the filter key.
- Filter on, full-scale generation 0: no plan has lane set {1, 2}. All 216 random plans are kept (35 redraws). The
  11 seeds listed above are removed.
- Filter on, generations 1-4 at full scale (60 children + 36 immigrants each): no {1, 2} plan appears, and every batch
  has exactly 96 candidates.
- Filter on, StubWM search_cell: no evaluated candidate and no non-comparator final has lane set {1, 2}.
  `generated_per_generation` = [seeds + n_init, 18, 18] at the reduced sizes.
- `main` routes through the S4 search_cell.
- Generation-0 lane-set mix (tests/GEN0_SUMMARY.txt). S2b: {1,2} 41 of 229 admitted. S4 with the filter on:
  221 admitted, 0 with {1,2}.
- **Box check** (setup_s4.sh; CPU only, CUDA_VISIBLE_DEVICES='', nice 19). Output:
  `GEN0_OK s2b_keys_reproduced 229 s4_pop 233 s4_admitted 221 rejected_seeds 11 rejected_random 35`, then SETUP_OK.
  This means the box tree reproduces S2b generation 0 from /root/autodl-tmp/a75/out_s2b (keys read only), the
  bundle MANIFEST passes (514 files), wm_s.pt is pinned and MANIFEST_A85 passes.

## 5. Box install (done)
- /root/autodl-tmp/a85/: a85_s4_pkg.tar + a75_search_box.tgz (two scp calls), extracted to a75/ (its own bundle copy,
  like a75p20/a75), a85ship/, cfg/, run_search_s4.sh, queue_s4.sh, setup_s4.sh, check_s4_gen0.py and setup_s4.out.
  `file` reports the .sh scripts as plain ASCII bash; there are 0 CR bytes. Size 19 MB; 50 G free.
- The launch mirrors run_search_s2b.sh line for line (see DIFF_vs_S2b.txt). Differences: A=/root/autodl-tmp/a85/a75,
  ship /root/autodl-tmp/a85/a85ship, `-m a85s4`, plus an extra sidecar out_s4.meta/SHIP_SHA256. The MPS client env is
  the same as chain_s2b.sh (`CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_a75_pipe`, `CUDA_MPS_LOG_DIRECTORY=/tmp/mps_a75_log`).
  The MPS daemon (pid 5702) was alive at 21:20.
- queue_s4.sh behaviour:
  - Takes the lock dir queue_s4.lock and refuses if out_s4 or out_s4.meta exists.
  - Polls every 60 s for `^S3b rc` in /root/autodl-tmp/a75/chain_s2b.log.
  - If the chain pid 5708 (identity-checked: `bash .../chain_s2b.sh`) is gone with no rc line, it logs `S4 NOT STARTED`
    and exits. It never starts without the rc line.
  - If the S3b rc is not 0, it logs a NOTE and starts anyway.
  - Waits a 30 s grace, logs whether MPS is alive and the GPU memory, writes `S4 start <date>`, runs S4 once and writes
    `S4 rc <n> <date>` and `QUEUE_DONE` to /root/autodl-tmp/a85/chain_s4.log.

## 6. Launch (coordinator; do not start before registration)
```bash
timeout 30 ssh -o BatchMode=yes -p 37084 root@connect.bjb1.seetacloud.com 'setsid -f nohup bash /root/autodl-tmp/a85/queue_s4.sh > /root/autodl-tmp/a85/queue_s4.nohup 2>&1 < /dev/null' < /dev/null
```
Check: `cat /root/autodl-tmp/a85/chain_s4.log`; progress `wc -l /root/autodl-tmp/a85/out_s4/LOG_D2800_p60.jsonl`
(the generation-0 line, then 4 more lines; then the finals). To withdraw before S4 starts, stop the queue by its own pid
in /root/autodl-tmp/a85/queue_s4.lock/pid.

## 7. Wall time and disk
- S2b: T_END - T_START = 17,121 s (4 h 45 min), 607 evaluated, 29,518 WM rollouts, 14 workers under MPS, alone on the
  GPU.
- S4 has the same population sizes and the same states, so expect **about 4.5-5.5 h**. Plans with a lane-0 role need
  CAVs in more lanes inside the zone, so they activate less often; that can lower the rollout count, while plans with
  more roles can raise it.
- S3b started at 20:06. Its LOG already showed 613 evaluated at about 21:14 (finals stage), so it may finish well
  before 00:30. S4 then starts within about 1.5 min of the rc line. If S3b ends at about 22:00-01:00, S4 ends at
  about 03:00-06:30 local.
- Disk: S2b out 1.1 MB + meta 24 KB; the install is 19 MB. S4 needs less than 50 MB, against 50 G free.

## 8. Copy outputs back (after `S4 rc 0` in chain_s4.log)
```bash
mkdir E:/research-dense/runs/active/C4-V7FT-20261017/a85_s4
timeout 300 ssh -o BatchMode=yes -p 37084 root@connect.bjb1.seetacloud.com 'tar -C /root/autodl-tmp/a85 -cf - out_s4 out_s4.meta chain_s4.log' < /dev/null > E:/research-dense/runs/active/C4-V7FT-20261017/a85_s4/a85_s4_out.tar
# then on the laptop: tar xf a85_s4_out.tar; verify: cd out_s4 && sha256sum -c ../out_s4.meta/SHA256SUMS
```
The files are STRATEGY_SEARCH_D2800_p60.json, TOP_OFFER_CONDITIONS_D2800_p60.json (the top 12, unchanged writer),
ARMS.json, LOG_D2800_p60.jsonl and crn_log/. res['a85'] in the STRATEGY_SEARCH file holds the filter counters.

## 9. Draft registration paragraph
> **S4 (A85, WM-only plan search outside the two-through-lane form; zero SUMO).** Question: when the S2b WM search
> (A75 hold-forms search, cell D2800_p60, 60 % CAVs, map wz3to2) cannot use plans whose roles sit only in the two
> through lanes (role-lane set {1, 2}, the form of the hand-written two-CAV hold, which 43 of the 63 S2b finalists
> used), which plans does the frozen WM rank highest? Changed variables vs S2b (closest prior: S2b, out_s2b, seed
> 20261176):
> - a generation-time filter rejects any candidate with role-lane set exactly {1, 2} (initial population, immigrants,
>   mutation children) and replaces it with a fresh draw from the same RNG stream, so the configured population sizes
>   are kept;
> - the 11 hand-library seeds with lane set {1, 2} leave the initial population; comparators are unchanged;
> - search seed 20261185.
>
> Everything else is unchanged: code a75ss cae54dc1 / strategy_search 02e119c1 / a75_wmforms 3af8f45d / wm_fix_crn
> 62e8e38f plus the new wrapper a85s4.py 7ca82976; cfg A85S4_SEARCH_CFG.json 0a80c0f4; WM wm_s.pt 6da9d333 (frozen,
> no training); states regime_STATES.json 74aef8f8 (18 valid D2800_p60 states); 14 workers, MPS.
> Budget: one run on the A800 GPU after S3b, about 5 h, no retry, zero SUMO sim-seconds. Output: a85/out_s4, with
> TOP_OFFER_CONDITIONS (top 12) by the unchanged writer. Any SUMO check of S4 plans is a separate registration.
> Tests: 10/10 CPU tests pass. With the filter off the code reproduces the S2b generation-0 keys (229/229) and
> a75ss.search_cell byte for byte; the box setup check passes.

## 10. Doubts
1. **The filter removes only the exact set {1, 2}.** Through-lane-only plans in a single lane stay allowed: {1} or {2}
   with 2-4 roles, e.g. g2_stagger40 (lane-2 two-CAV stagger) and its dense variants. In S4 generation 0 that is 16 of
   221 plans (7 {1}, 9 {2}). If the owner means "must include the closing lane", set
   `exclude_lane_sets: [[1], [2], [1, 2]]` in the cfg. That is a one-line cfg change with no code change, but the
   cfg sha and manifest must be regenerated.
2. A rejected mutation child redraws its parent as well as the mutation. This treats the parent choice as part of the
   child draw. A same-parent redraw would also satisfy the brief. Parents never have lane set {1, 2}, so neither
   choice can loop.
3. Mutations that return None and run()-level drops (duplicates, < min_roles, banned) are not replaced, as in S2b.
   S2b lost 3/2/0/1 children per generation this way, so the screened count per generation can still be below 96.
4. The novelty reference set still contains the removed duo12 seeds, as specified ("nothing else changes"). Plans
   far from duo12 keep getting the novelty bonus.
5. MPS: S4 uses the same MPS client env. If the daemon (pid 5702) has gone by then, CUDA runs without MPS; the queue
   logs a NOTE, and the run may be slower or use more memory. The queue does not start or stop MPS.
6. Provenance cosmetics: the unchanged a75ss.main labels the output `a75` and argparse prog `python -m a75ss`.
   res['a85'] and out_s4.meta/SHIP_SHA256 record the S4 wrapper.
7. Unpacking the bundle prints "time stamp in the future" warnings (the bundle mtime is 2026-11-14 04:00). These are
   harmless and the manifest passes.
8. Tests use the CPU StubWM only. The real WM path (ParallelWM, a75_wmforms) is the unchanged S2b path and was not
   exercised here.
