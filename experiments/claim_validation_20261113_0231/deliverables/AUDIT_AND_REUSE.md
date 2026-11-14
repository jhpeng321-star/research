# AUDIT_AND_REUSE: claim validation 20261113_0231 (Amendment 72) and post hoc items

Built from the record only (zero simulation, no ssh). Sources:
- `PROTOCOL.md` sha256 6c5790173c4be915c10c9fe7d5f0a7ea1e6c92b580035fc3be5e94299596270c
- `PROTOCOL.json` 537bc8fa79cf327ccc043bb067fb324591b3a97f2ad027598bb237f5526d2885
- `BUILD_NOTES.md` a14723db14585ff7742c1198730a1107dae2b219054b2ef25c8dd56db4fcd526
- `AMENDMENT_72_BLOCK.md` 0526acf6dd10618a2b98243248ecd14c7193032011edcd178f216b31475e10b8
- `REG` = `runs/active/C4-V7FT-20261017/REGISTRATION.md` (append-only; line numbers as of 11,683 lines).
Paths without a prefix are relative to `experiments/claim_validation_20261113_0231/`.

## 1. Audit before the study

| Item audited | Recorded result | Source |
|---|---|---|
| Closest prior experiments | A47 (WM search, D2800_p60), A42 (four-plan bank F), A68 (multi-plan bank, failed confirmation) | `REG:11485` |
| Unanswered difference | (E1) WM ranking of a FIXED random pool vs a random order of the same pool; (E2) F vs a resource-matched uniform hold chosen on Development worlds | `REG:11486-11487` |
| Frozen identities | WM `wm_s.pt` 6da9d333...c063; `strategy_search.py` 02e119c1...4fc; start states `regime_STATES.json` 74aef8f8...54e3 (18 valid); search cfg `A47_SEARCH_CFG.json` f5636a96...4be9; scorer `wm_pool_score.py` 94824b8a...c1f; bank F `A42_BIG_BANK_D2800_p60.json` 7e7dc76b...73ff; U0 d9042cc5...9ca | `REG:11492-11499`; `PROTOCOL.md` section 1 |
| Registered identities found by the build | U0, WM ckpt, bank F, executor `oracle_plan.py` 73cb1ca6..., wrapper `a37_cellbank.py` 5d9811a0... (A37 digest 0a902437), runner 62496e86... / `nm_finaleval.py` 9f289431... | `BUILD_NOTES.md` section 1 |
| SUMO determinism prerequisite | IDENTITY_A68 = PASS (difference 0.0; decompressed log compared, not gzip container), IDENTITY_A67 = PASS | `BUILD_NOTES.md` section 1 |
| Paper text vs code | legal grid larger than four knobs; all four F plans have non-zero onsets (S differs from F for every plan; no H equals F); no bank-level activation cap; WM scoring cost | `BUILD_NOTES.md` section 2 |
| Bank files | Box 3 crosscheck PASS for all 8 bank files; Box 3 dry build PASS for the Development block | `REG:11536` |
| Unit tests at registration | "30 local tests pass" | `REG:11536` (see flag C2) |
| Random arm isolation | permutation is a pure function of (sorted pool hashes, seed); module imports only hashlib/random; audit hook shows no file opened; frozen before any WM score | `BUILD_NOTES.md` section 4; `PROTOCOL.md` section 2; `REG:11539` |
| Seed blocks | see section 3 | |

## 2. Reuse versus newly run

### E1 (registered)
| Element | Reused or new | Source |
|---|---|---|
| Pools (6 seeds x 120 plans) | NEW, generated from the search config only; frozen (POOL shas in PROTOCOL.json) | `PROTOCOL.md` section 2 |
| Random orders | NEW, frozen before any WM score | `PROTOCOL.md` section 2; `REG:11539` |
| WM, search cfg, 18 start states | REUSED unchanged (A47/A68 states; states not regenerated) | `REG:11492-11499` |
| WM scores | NEW (zero SUMO) | `REG:11596` |
| Verification episodes | NEW: union of both top sixes + U0 on fresh per-seed 24-world blocks; a plan in both arms runs once; 1,848 episodes, 2,217,600 sim-s | `PROTOCOL.md` section 2; `REG:11597`; `REG:11603` |
| Confirmation | NEW banks on the fresh 96-world panel 990900-990995; CONF_U0 wave REUSED from E2 (shared); identical banks share one run; empty bank = U0 (no run: WM seed 42) | `REG:11606`; `BUILD_NOTES.md` section 4; `E1_ANALYSIS.json` |

### E2 (registered)
| Element | Reused or new | Source |
|---|---|---|
| Bank F | REUSED byte-identical (A42, 7e7dc76b...) | `PROTOCOL.md` section 3; `REG:11546` |
| S, H18..H22 banks | NEW, derived from F (onsets / v_w only; checker refuses other changes) | `BUILD_NOTES.md` section 4; `REG:11546` |
| Tuning worlds | REUSED, already-viewed Development panel a35_D2800_p60 worlds 0..23 (998200-998223, pin 425e8a8d...) | `REG:11516`; `REG:11545` |
| Identity re-check | REUSED stored A42 records (U0 + F on 998600-998603) as the comparison target; NEW 9,600 sim-s per box | `REG:11544`; `REG:11548` |
| Confirmation | NEW on 990900-990995 (U0 wave shared with E1) | `REG:11560` |
| Secondaries re-analysis | REUSED E2C_RESULTS.json; reproduction byte-identical to the sealed E2_ANALYSIS.json | `REG:11671`; `e2_secondary/E2_SECONDARY_REPORT.md` |

### Sensitivity / repair (post-registration)
| Item | Reused | New | Source |
|---|---|---|---|
| WM-ACT | registered screen scores and finalist list; 10 already-verified candidates; reuse-mode final rows kept only as probe record | mode-`all` final scores; 26 new candidates verified; U0 re-runs checked for exact equality vs Box 4 U0 | `REG:11609`; `REG:11615`; `REG:11627`; `REG:11639` |
| E1-FIX (A74) | WM ckpt, pools, cfg, states, seeds, budget 6, verification panels, admission rule, bank builder, 96-world panel, U0, Random arm and its confirmation outcomes; identical plan x world episodes from E1 or WM-ACT not re-run | repaired scoring (GPU R, launched 13:21, zero SUMO) | `REG:11644`; `REG:11649`; `REG:11682` |

### Post hoc items
| Item | Reused | New | Source |
|---|---|---|---|
| A69 (60% lane change) | stored A68 MAPPO60C references for admission and A68C arms on 996200-996295 for confirmation | admission 6 plans x 24; LC_BANK arm x 96 | `REG:11445`; `REG:11567`; `REG:11570` |
| A70 (small banks) | A68 admission data for plan choice (confirmation panel not used for choice); stored A66/A67 records as identity target | 4 arms x 96 fresh worlds 996300-996395 | `REG:11450-11455`; `REG:11552` |
| A71 (wide admission) | 12 A68 admission verdicts and A68 MAPPO60C admission references; A70 MAPPO60C and DUO_HOLD arms on 996300-996395 (panel already viewed for SMALL1/SMALL2) | 51 plans x 24 admission worlds; WIDE_BANK arm x 96 | `REG:11467-11472`; `REG:11584`; `REG:11589` |
| A73 (20% lane change) | 30 native TRAIN tapes of 11-08 as WM start states (no new SUMO); harm calibration fitted at p60 | search; mechanics; admission incl. own U0 on 992524-992547 | `REG:11578-11579`; `REG:11635`; `REG:11665` |

## 3. Seed and panel disjointness, as recorded

- A72 blocks (each reserves its full hundred): e1v_seed11 993800-993823; e1v_seed23 994100-994123; e1v_seed42 994500-994523;
  e1v_seed67 995400-995423; e1v_seed89 999300-999323; e1v_seed103 999800-999823; conf 990900-990995; reserved unused
  990500-990599 and 999400-999499 (`REG:11524-11528`; `PROTOCOL.md` section 6).
- Confirmation: "CONFIRMED BY MAIN 2026-11-13 (independent grep ... zero seed hits; two decimal-fraction false positives in
  RESULTS_SUMMARY.md)" (`REG:11524`). Evidence scope: REGISTRATION.md (03:17 version), code tree, ledger, C4 run directory,
  c4/* branch tips, A70/A71 code and tables, Box 3 /root/a70 and /root/a71 tables; all blocks outside the banned ranges
  (`REG:11529-11530`; `PROTOCOL.md` section 6).
- Panel pins (file sha / panel_sha256) recorded before any SUMO (`REG:11545`).
- A68: 992500-992599 and 996200-996299 UNUSED; host run roots not scanned (`REG:11426`).
- A70: 996300-996399 UNUSED; audit 118a8f6e... (corrected sha and hit count at `REG:11461`).
- A73: seed audit `ops/a73/SEED_AUDIT.json` 4fc82b0a, verdict UNUSED; new confirmation block 996400-996499 (`REG:11635`).
- Stated limit, recorded in each: a text search cannot prove a seed was never simulated by an unrecorded run
  (`REG:11530`; `PROTOCOL.md` section 6).

## 4. Recorded as unverified or not done

| Item | Status in the record | Source |
|---|---|---|
| `box_driver.py` against the live tree | later: crosscheck PASS (8 banks), dry build PASS | `BUILD_NOTES.md` section 5.1; `REG:11536` |
| Seed-table edits + identity re-check | later: PASS on Box 4 and Box 3 (max diff 0.0) | `BUILD_NOTES.md` 5.2; `REG:11548`; `REG:11597` |
| Replay mechanics check (9,600 sim-s) | NOT RUN; replay itself not run | `BUILD_NOTES.md` 5.3; `REG:11560` |
| WM scorer chunk-equality check before chunk < 120 | no record found of it being run; probe later NOT identical on 5 seeds; determinism check 0 of 12 identical | `BUILD_NOTES.md` 5.4; `REG:11614`; `REG:11626` |
| Cause of non-reproducible WM scores | "not determined" (09:02, 10:06); A74 build names unseeded global torch RNG for merge-noise keys as "likely cause" | `REG:11614`; `REG:11626`; `REG:11680` |
| WM scorer backend path | "has not been executed yet" at registration | `REG:11537` |
| A69 empty-plan identity half | not run (recorded as not run, not replaced) | `REG:11427` |
| E1-FIX code tree identity | "checked through these hashes only" | `REG:11676` |
| Replay commands | `drv prepare/dry` on replay waves and U0 reuse not verified without a run | `e2_secondary/E2_SECONDARY_REPORT.md` |
| Raw logs | A68/A70 (Box 3) and A71/A68-mirror raw per-episode dirs deleted with owner approval; per-vehicle reports and results kept | `REG:11638`; `REG:11659` |

## 5. Conflicts flagged (not resolved)

- C1. `REG:11524` says "independent grep of the seven hundreds"; the same block lists nine reserved hundreds (6 e1v + conf
  + 2 reserved) and `PROTOCOL.md` section 6 says "no number of the nine hundreds appears". Which seven were grepped by
  MAIN is not stated.
- C2. Test count: `BUILD_NOTES.md` "29 SUMO-free tests pass" vs `REG:11536` "30 local tests pass" (BUILD_NOTES predates
  registration; a later test may have been added; not stated).
- C3. Mechanics budget: `PROTOCOL.md` section 5 and `BUILD_NOTES.md` 5.3 tie 9,600 sim-s to the replay mechanics check;
  addendum P1 (`REG:11544`) spends 9,600 per box (19,200 total) on identity re-checks "inside the total cap".
- C4. `BUILD_NOTES.md` section 2.5: "never-activated descriptors rank last"; as run they scored 0.0 and outranked negative
  scores (`REG:11598`).
- C5. `BUILD_NOTES.md` section 2.4 estimates scoring every pool member under the final protocol (8,640 rollouts per seed);
  the registered rule final-scores only the top 12 (`PROTOCOL.md` section 2). Design change before registration, not an error.
- C6. `BUILD_NOTES.md` section 6 replay command uses `--block e2r`; e2_replay tags waves `block='dev'`
  (`e2_secondary/E2_SECONDARY_REPORT.md`).
- C7. `REG:11539`: WM scoring "starts only after search L (A69) ends"; `REG:11596`: scoring overlapped the A73 search from
  05:26 (a different search; search L had ended 04:57, `REG:11564`). No conflict on A69; recorded for completeness.
