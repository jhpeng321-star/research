# A86 S5 prep notes (2026-11-13, ~22:45 UTC+8): closing-lane WM plan search with structural niching, gpuR

Status: PREPARED and INSTALLED on gpuR (/root/autodl-tmp/a86, SETUP_OK). NOT LAUNCHED. This task ran zero SUMO,
zero GPU, zero training and zero WM rollouts. Nothing was written into /root/autodl-tmp/a75. The only thing read from
there was a copy of /root/autodl-tmp/a75_search_box.tgz, verified by sha. No process was signalled, the A800 and the
CPU boxes were not touched, and REGISTRATION and the ledger were not edited.

## 1. What S5 is (differences vs S4, a85_s4_diffsearch_20261113)
1. **Lane-set filter `exclude_lane_sets: [[1], [2], [1, 2]]`.** The S4 filter code is unchanged; only its cfg value
   changes. Every candidate must have at least one role in lane 0 (the closing lane). Initial plans, immigrants and
   mutation children are redrawn from the same RNG stream, as in S4.
   - **Hand seeds:** all 28 hand-library seeds that survive hold_only and search_grid are removed, because none has a
     lane-0 role. Their lane sets are {2} 16, {1, 2} 11 and {1} 1. Removed: wall_g_solo2, wall_g_duo12,
     wall_g_solo2_gate0, wall_m_solo2, wall_g_solo1, wall_g_duo12_gate0, wall_g_alt21(_gate0), wall_g_relay21/12,
     g2_stagger40, wall_g_duo12_stagger40, g2_gap1(_gate0), solo2_v18_h90/h120(_gate0), duo12_v18,
     g2_stagger20/40_v18, solo2_v20_h90/h120(_gate0), duo12_v20, g2_stagger20/40_v20.
     **None remain**, so generation 0 consists of n_init = 72 random plans.
   - **Comparators** (wall_g_solo2, solo2; lane set {2}): they are appended to the finals and flagged
     `comparator=True` by key, exactly as in S4 (the `for c in SS.COMPARATORS` lines are unchanged). They are never
     candidates. They are excluded from best/beats/TOP, and res['comparators'] reports them. One difference from S4:
     wall_g_solo2 was also a hand seed in S4, so it was screened there too. In S5 it is scored only in the finals.
     This is tested.
2. **Seed 20261187.**
3. **Output /root/autodl-tmp/a86/out_s5** (sidecars in out_s5.meta). The run uses its own bundle copy,
   /root/autodl-tmp/a86/a75.
4. **Size and workers** (section 4). Settings: `--workers 3`, no MPS. Search: n_init 72 (S4 216), generations 3 (4),
   n_children 20 (60), n_immigrants 12 (36). The immigrant share is 12/32 = 37.5 %, the same as S2b's 36/96.
   n_elite 6, finalists 18, the 18 states, K_screen 4, K_final 6, H 300 and --max-rows 64 are unchanged. Nothing that
   scores a single plan changed.
5. **DISCLOSED DESIGN DIFFERENCE (owner, via coordinator): structural niching.** a85s4.py is no longer the entry
   point. The new wrapper `ship/a86s5.py` is a copy of a85s4.py. Every changed line in an existing function is marked
   `# A86-S5`, and new functions are marked on their `def` line. Niching is switched on by the cfg key `niching`; with
   the key absent the code is identical to a85s4 (tested).
   - **Niche** = (form family, sorted role-lane tuple), from plan structure only. The form family is the A80
     FORM_RULE: `a80_s2b.form_family`, copied verbatim (body equality is tested against
     experiments/a82_p20_fourarm_20261113/ship/a80_s2b.py, 48c5c448). The rule takes the first match in this order:
     gap (a role with start=gap) > pulse > taper > zone (mid/merge/custom) > rolling (2 or more roles with 2 or more
     distinct onsets) > short (every hold_s <= 45) > other.
     Under the filter there are 7 forms x 4 lane sets ({0}, {0,1}, {0,2}, {0,1,2}) = 28 possible niches.
     **Generation 0 covers 15 niches** (72 plans): gap 17, pulse 20, zone 15, taper 10, rolling 9, other 1, short 0.
     Per niche: gap:L01 4, gap:L012 13, other:L012 1, pulse:L01 3, pulse:L012 13, pulse:L02 4, rolling:L01 2,
     rolling:L012 4, rolling:L02 3, taper:L01 3, taper:L012 3, taper:L02 4, zone:L01 1, zone:L012 4, zone:L02 10.
   - **Parents** (`parents: true`). The global elite list (top n_elite by the unchanged rank) and each archive's top
     list are taken in niche round-robin order of the same ranked list. That order is the best plan of each niche
     (niches ordered by their best plan), then each niche's second best, and so on. The round-robin runs inside each
     activation tier, so activated plans stay first (rank_activated_first). The counts are unchanged. The test shows
     that the 6 elite parents come from 6 different niches in every generation.
   - **Finalists** (`finalists: true`). The same round-robin order is applied to the 18 score finalists and the
     per-archive finalists. The cap of 4 x 18 is unchanged, and finals scoring is unchanged.
   - **TOP-12** (`top_cap: 2`). The unchanged a75ss.main writer runs first. a86s5.main then checks that the writer's
     list equals the recomputed global top, and renames that file to
     `TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED_D2800_p60.json` (bytes unchanged). It then writes
     **`TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json`**:
     - the same eligible plans as the writer (not comparator, not banned, activated; in final order);
     - taken niche-best round-robin, then second best;
     - at most 2 per niche, up to 12 rows;
     - in the identical row format.
     `NICHE_TOP_D2800_p60.json` lists both selections with their niche labels. **The NICHECAP2_FOR_SUMO file is the
     one that goes to SUMO.** The plain `TOP_OFFER_CONDITIONS_D2800_p60.json` name no longer exists in out_s5, on
     purpose.
   - Applied to the real S2b finals as a demonstration (no new evaluation), the global top 12 held 4 niches, all with
     lane set {1,2}: zone:L12 4, rolling:L12 3, pulse:L12 3, other:L12 2. The capped top 12 held 12 different niches.
   - res['a86'] records the definition, the niching flags, the generation-0 niche counts, the parent niches per
     generation, the finalist niches, the niche of every final plan and the a86s5 sha. res['a85'] still records the
     filter counters; its sha field is now named `a86s5_sha256`.

## 2. Hashes (sha256)
| file | sha256 |
|---|---|
| ship/a86s5.py (NEW wrapper, entry point `-m a86s5`) | 71dda3e8436d1202ef54235f1c02db0cd38e485329cf17a036cd36bd491e152e |
| ship/a85s4.py (unchanged; used only by the checks) | 7ca82976e7183ce205d0591470a60281b553c2d03e8643d479d739b86c419261 |
| ship/a75ss.py / a75_wmforms.py / wm_fix_crn.py (unchanged) | cae54dc1... / 3af8f45d... / 62e8e38f... (full values in pkg MANIFEST_A86.sha256) |
| cfg/A86S5_SEARCH_CFG.json | 3adc266397fad96a11430e4251f6155eaf4a8ad90b47ffe7b0ba50b1e9b9c215 |
| cfg/S4_REF/A85S4_SEARCH_CFG.json (ref) | 0a80c0f4c6a8858faf4430c8e55be04a9abc77779337cd91921e1df873b82f5e |
| cfg/S2B_REF/A75S2B_SEARCH_CFG.json (ref) | 8f71eb5534093333c4fa042143daf4b66de81b727580975aed72ea398fc2491c |
| ref/S2B_GEN0_KEYS.json (S2b screened[:229] keys) | cacb89d42b95e8f97bbbd279caa6f0405291218a5ed1d52e1a5f2b661a592044 |
| box/run_search_s5.sh | 2d3ecca6cb96fad09d948e7aec9284cebe46aabfc2a01befbda3b192fce28649 |
| box/start_s5.sh | f494996da6686ac3124e2947b9f6e4df5e5cc4df256ac50d2a6745996099175f |
| box/setup_s5.sh | c7afd1d2cb375d55de280fd2c8e89e87bb9d67d4e50fc5ffdf9106da27ae2c1e |
| box/check_s5_gen0.py | d1239ddb5d4e72aa3a9baa60886b9f947f0d31c058ed30227efd2a89f0f5c4e5 |
| a86_s5_pkg.tar (all of the above + MANIFEST_A86.sha256) | 5d5e316043ccca7ad09604a23687e2fa147ed32f3d53c17985462111fe674f3e |
| tests/test_a86s5.py | 537c2a658346d950e85968a855c79d6deb3b46866baa27e08413e0f79d6eadeb |
| DIFF_vs_S4.txt | 8ba808accad74f5929052bb4847e21332d98365db845f8937164a58682f7483b |
| S5 generation-0 key list (json of 72 keys) | 275c80a9e53ffe65126f25aa69246bcc6ef9f5302413081b00c6c76800bda364 |
| gpuR, verified: a75_search_box.tgz | 9eb32080ea789219202f61f1819944870bb28c428f4d417762ca310631a63277 |
| gpuR, verified: wm_s.pt | 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 |
| gpuR, verified: regime_STATES.json / strategy_search.py / wm_backend.py | 74aef8f8... / 02e119c1... / 728fc917... (match S4 notes; a75 MANIFEST passes) |

## 3. Tests: 20 passed
Command: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I -B -m pytest -q -p no:cacheprovider tests`. CPU
only, StubWM fakes. The tests extend the S4 suite.
- The pinned inputs match, including the A80 file, and form_family is verbatim from A80.
- The cfg differs from S4 only in seed, exclude_lane_sets, niching and the 4 size keys. All scoring keys are
  unchanged. The immigrant share is at least S2b's.
- **Niching off reproduces the earlier runs:**
  - the S2b generation-0 keys (229/229);
  - the S4 generation 0 (221 admitted, same rejections as a85s4);
  - a86s5.search_cell gives byte-identical JSON to a85s4.search_cell for the S2b cfg, the S4 cfg and the S5 cfg
    without niching (sha field excluded).
- **S5 filter:**
  - Generation 0 at full scale is 72 random plans, every one with a lane-0 role and none in an excluded set.
  - All 28 seeds are rejected.
  - Each later generation has exactly 32 children and immigrants, all with a lane-0 role.
  - In a StubWM search_cell, no comparator is screened, both comparators are in the finals with comparator=True,
    every non-comparator final has a lane-0 role, generated_per_generation = [n_init] + [n_children + n_immigrants] per generation, and the
    whole run is deterministic under the seed.
- **Niching:**
  - The round-robin order and tiers are correct.
  - **The parents span niches:** the 6 elite parents come from 6 different niches in every generation, and the
    total span is at least the span with niching off.
  - **The capped top has at most 2 plans per niche.** On the real S2b finals: the recomputed global top equals the
    S2b writer file; the capped top has 12 rows, at most 2 per niche, is deterministic and contains only eligible
    plans.
  - Synthetic cap and fill order are correct.
  - write_niche_top: the renamed global file is byte-identical, and the capped rows have the writer's row keys.
  - main routes through the S5 search_cell and post-processes only when niching is set.
  - cfg validation works.
- Box check (setup_s5.sh, CUDA_VISIBLE_DEVICES='', nice 19):
  `GEN0_OK s2b_keys_reproduced 229 s4_admitted 221 s5_pop 72 s5_admitted 72 rejected_seeds 28 rejected_random 12
  niches 15 gen0_key_sha 275c80a9e53ffe65`, then IMPORT_OK and SETUP_OK. The bundle MANIFEST, wm_s.pt and
  MANIFEST_A86 all pass.

## 4. Workers, size and wall-time estimate
- **Workers 3.** gpuR has a history of running out of memory on exactly this search code
  (/root/autodl-tmp/a75/out_s1*.meta, read only):
  - S1 with 10 workers OOM'd at screening.
  - S1b with 5 workers and max-rows 64 OOM'd after 46 min. The processes held 3.8-5.0 GB each, and the failing
    process asked for another 1.0 GB on a 23.5 GB card.
  - With 3 workers the expected peak is about 15-18 GB, under the ~20 GB limit. 4 workers would be 20-24 GB.
  - --max-rows was not lowered, so the batching and CRN path is identical to S2b and S4.
- **Rate basis (gpuR's own log, same states / K 4 / H 300 / max-rows 64, no MPS).** The S1b crn_log shows 46
  completed calls: 1,300 actions x K 4 = 5,200 rollouts in about 63-66 min from T_START, so about 1.3-1.4 rollouts/s
  with 5 workers. With 3 workers the rate is between 0.8 rollouts/s (linear in workers, worst case) and about
  1.35 rollouts/s (GPU-bound). For reference on the A800 with 14 workers and MPS: S2b ran at 1.72 rollouts/s
  (29,518 rollouts in 17,121 s), and S3b at 2.03 rollouts/s.
- **Work.** Rollouts = sum over states of (number of activated distinct specs + base + incumbent) x K.
  - Generation 0 is **exactly 2,004 rollouts**. This is computed on CPU from the real activation code (tests/
    STUB_ROLLOUT_COUNT.txt). The same tool reproduces S2b's 7,008 exactly.
  - Generations 1-3 are 3 x 32 new plans at about 46 rollouts per plan, the S2b real average, so about 4,400. The
    upper bound is 72 per plan, about 6,900.
  - Finals are at most 74 plans x 18 states x K 6. S2b's 63 finals cost 5,298, so expect about 4,000-6,000.
  - Central total is **about 11,000 rollouts**; the worst case is about 15,000.
- **Estimate: about 2.5-4 h** (central 11,000 rollouts: 4.0 h at 0.8 rollouts/s, 2.3 h at 1.35 rollouts/s, plus
  about 5 min of startup). The worst case is about 5 h (15,000 rollouts at 0.8 rollouts/s).
- **On the "S3b 20 %" guide:** S3b is the **20 % penetration cell (D2800_p20)** at FULL S2b size (n_init 216,
  4 generations, 60+36), not a 20 % size reduction (a82/s3b/STRATEGY_SEARCH_D2800_p20.json config). It ran 86 min
  because p20 states need fewer rollouts (10,458). So it could not be used as a size template, and the reduction
  above is derived from rollout counts instead.
- Disk: install 19 MB. Expected output under 50 MB, against 28 GB free.

## 5. Launch (coordinator; only after registration)
```bash
timeout 30 ssh -o BatchMode=yes gpuR 'setsid -f nohup bash /root/autodl-tmp/a86/start_s5.sh > /root/autodl-tmp/a86/start_s5.nohup 2>&1 < /dev/null' < /dev/null
```
start_s5.sh refuses to start in any of these cases: no SETUP_OK, the lock /root/autodl-tmp/a86/start_s5.lock exists,
out_s5 or out_s5.meta exists, or any compute process is on the GPU. It writes `S5 start <date>`, runs once, then
writes `S5 rc <N> <date>` and START_DONE to /root/autodl-tmp/a86/chain_s5.log. It never retries.
Check progress with `cat /root/autodl-tmp/a86/chain_s5.log; wc -l /root/autodl-tmp/a86/out_s5/LOG_D2800_p60.jsonl`
(1 + 3 lines, then the finals) and with `nvidia-smi`. To withdraw it, stop it by its own pid
(start_s5.lock/pid and its children); do not kill by name.

## 6. Copy back (after `S5 rc 0`)
```bash
mkdir E:/research-dense/runs/active/C4-V7FT-20261017/a86_s5
timeout 300 ssh -o BatchMode=yes gpuR 'tar -C /root/autodl-tmp/a86 -cf - out_s5 out_s5.meta chain_s5.log' < /dev/null > E:/research-dense/runs/active/C4-V7FT-20261017/a86_s5/a86_s5_out.tar
# laptop: tar xf; cd out_s5 && sha256sum -c ../out_s5.meta/SHA256SUMS
```
Files: STRATEGY_SEARCH_D2800_p60.json (res['a85'] filter, res['a86'] niches),
TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json (**for SUMO**),
TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED_D2800_p60.json, NICHE_TOP_D2800_p60.json, ARMS.json, LOG_D2800_p60.jsonl and
crn_log/.

## 7. Draft registration paragraph
> **S5 (A86, WM-only closing-lane plan search with structural niching; zero SUMO).** Question: when every candidate
> must hold at least one CAV in the closing lane 0, and the search keeps several structural plan types alive instead
> of converging on one, which plans does the frozen WM rank highest, and which 12 structurally different plans should
> go to SUMO admission? Closest prior: S4 (A85, a85/out_s4, filter {1,2} only, seed 20261185) and S2b.
>
> Changed variables vs S4:
> - **Lane filter** `exclude_lane_sets [[1],[2],[1,2]]` (unchanged filter code). All 28 hand seeds leave
>   generation 0, because none has a lane-0 role. The comparators wall_g_solo2 and solo2 are still scored in the
>   finals as labelled comparators.
> - **Seed** 20261187.
> - **Disclosed design change: niching** (wrapper a86s5.py 71dda3e8, identical to a85s4 when the key is absent).
>   - Niche = (A80 FORM_RULE form family {gap>pulse>taper>zone>rolling>short>other}, sorted role-lane set). It uses
>     structure only. There are 28 possible niches; generation 0 covers 15.
>   - Parents and finalists are taken in niche round-robin order (best of each niche first, inside activation
>     tiers), with unchanged counts.
>   - The SUMO candidate file TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO takes the eligible finals niche round-robin,
>     at most 2 per niche, up to 12. The unchanged global top 12 is kept as TOP_OFFER_CONDITIONS_GLOBAL_UNCAPPED.
> - **Size** reduced for a 24 GB card and about 4 h: n_init 72, generations 3, 20 children + 12 immigrants (immigrant
>   share 37.5 %, as S2b), 3 workers, no MPS.
>
> Unchanged: per-plan scoring (score formula, novelty_weight 40, archives); 18 D2800_p60 states; H 300; K_screen 4;
> K_final 6; finalists 18; max-rows 64; code a75ss cae54dc1 / strategy_search 02e119c1 / a75_wmforms 3af8f45d /
> wm_fix_crn 62e8e38f; WM wm_s.pt 6da9d333 (frozen, no training); states regime_STATES.json 74aef8f8; cfg
> A86S5_SEARCH_CFG.json 3adc2663.
>
> Budget: one run on gpuR (RTX 4090), estimated 2.5-4 h (worst about 5 h), no retry, zero SUMO sim-seconds. Output:
> a86/out_s5. Any SUMO check of S5 plans needs a separate registration.
>
> Tests: 20/20 CPU tests pass. With niching off, the code reproduces the S2b (229) and S4 (221) generation-0 keys and
> a85s4.search_cell byte for byte. The box setup check passes on gpuR.

## 8. Doubts
1. **Out-of-memory risk remains.** gpuR OOM'd at 10 and at 5 workers on this code. 3 workers should peak around
   15-18 GB, but there is no measurement at 3 workers. A failure ends with `S5 rc != 0` and no retry; the coordinator
   decides what follows.
2. **The time estimate is wide (2.3-4 h central, about 5 h worst).** The gpuR rate comes from a single run that
   failed (S1b, 5 workers), and scaling from 5 to 3 workers is unknown. Generation 0's rollout count is exact; later
   generations and the finals are extrapolated from S2b's per-plan averages. Lane-0 plans need CAVs in more lanes, so
   they activate less often (generation 0: 27.8 rollouts per plan, against 30.6 in S2b).
3. **Novelty is effectively inactive in S5.** Every generation-0 plan is at or above the novelty cap (minimum
   distance to a hand seed 2.575, cap 2.0). The bonus is therefore a constant 80 veh-s for every plan and does not
   change any ranking. It was 93 % at the cap in S4 and 90 % among the S2b finals, so it was nearly inactive before
   too. I kept novelty_weight at 40 as instructed. Making novelty act would need a different reference, for example
   distance to already evaluated plans; that changes the rank formula and is the owner's call.
4. **No hand-written starting points.** All seeds are removed, so S5 starts from 72 random plans only. Mutation can
   still reach seed-like shapes as long as they have a lane-0 role.
5. **Niche choices.**
   - The form rule is first-match, so a plan with both gap and pulse roles counts as gap.
   - The 'short' family does not appear in generation 0, because the earlier rules almost always match first.
   - The number of roles and the trigger are not part of the niche. Adding them would give many small niches; the
     simplest definition that uses the existing classifier was chosen.
   - Round-robin is applied to each parent list separately (global elite and each archive), so duplicates across
     lists can still occur, as in S2b.
6. **Downstream file name.** Downstream tools (for example an A82-style admission) that read
   TOP_OFFER_CONDITIONS_D2800_p60.json will not find it in out_s5. They must pin the NICHECAP2_FOR_SUMO file, which
   is the intended behaviour. That file is in round-robin order, not score order, so the row order (draw_rank) is
   niche-best first. If fewer than 12 eligible plans fit under the cap, it has fewer rows, which the A82 rule allows
   (1-12).
7. **No MPS.** gpuR has no MPS daemon running and I did not start one. The rate basis (S1b) was also measured
   without MPS.
8. **Marking convention.** Changed lines in existing functions carry `# A86-S5`. The bodies of the new functions
   (form_family, niche, niching_cfg, niche_order, capped_top, write_niche_top) are marked on their def line only.
9. **Test coverage.** The tests use the StubWM only. The real WM path (ParallelWM, a75_wmforms, CRN) is the
   unchanged S2b/S4 path and was not exercised. Stub counts beyond generation 0 are not representative, because the
   scores are fake.
10. The provenance cosmetics from S4 remain: a75ss.main labels its output 'a75'. res['a85'] and res['a86'] carry
    the a86s5 sha.
