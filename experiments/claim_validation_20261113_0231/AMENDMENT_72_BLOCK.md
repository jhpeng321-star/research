
## AMENDMENT 72 - claim validation 20261113_0231 (E1 WM order vs random order; E2 bank F vs matched uniform hold)

Registered 2026-11-13 before any formal result of this round exists (no WM score of an E1 pool, no SUMO episode).
Full protocol: `experiments/claim_validation_20261113_0231/PROTOCOL.md` and `PROTOCOL.json`
(PROTOCOL.json sha256 537bc8fa79cf327ccc043bb067fb324591b3a97f2ad027598bb237f5526d2885;
PROTOCOL.md sha256 6c5790173c4be915c10c9fe7d5f0a7ea1e6c92b580035fc3be5e94299596270c).

Closest prior: A47 (WM search, D2800_p60), A42 (four-plan bank F), A68 (multi-plan bank, failed confirmation).
Unanswered difference: (E1) whether the WM ranking of a FIXED random pool yields more admissible plans than a random
order of the same pool; (E2) whether F beats a resource-matched uniform hold chosen on Development worlds.

Cell D2800_p60, map wz3to2, 1200 s. Reference U0 = frozen speed-only base MAPPO d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca.
Sign: treatment - reference; negative total delay = improvement.

Frozen inputs (sha256):
- WM checkpoint wm_s.pt 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063
- strategy_search.py (code_v12r7b) 02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc
- start states data/regime_STATES.json 74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3 (18 valid)
- search config inputs/A47_SEARCH_CFG.json f5636a96759d567b6d9510fc6f38c2391b43a019464dc0e3fe53bb4109044be9
  (A47 values: K_screen 4, n_screen_states 4, finalists 12, K_final 4, final states all 18, H 300, lam 0.5)
- scorer wm_pool_score.py 94824b8a4b916d67e0cbeb785af84c745825c8601aa64d341d80d495b0c3cc1f
- bank F inputs/A42_BIG_BANK_D2800_p60.json 7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff

E1. Search seeds 11, 23, 42, 67, 89, 103 (RNG seeds). Per seed a fixed pool of 120 legal hold-only plans, 30 per
size 3/4/5/6 (POOL file sha256: 11 3f124460..., 23 cf73db58..., 42 af8f1c44..., 67 4351ef1b..., 89 be51a2d8...,
103 b2a90a79...; full values in PROTOCOL.json).
- Random arm: seeded shuffle of the pool hashes, top six. Already drawn and frozen before any WM score.
- WM arm, two-stage: screen all 120 on 4 seeded start states (K 4); final-score exactly the top 12 on all 18 states
  (K 4); rank by final summary.score, ties by screen rank; top six. No archive leaders or comparators (difference
  from the A47 search, stated). Zero SUMO; at most 2,960 WM rollouts per seed.
- Verification: union of both top sixes, each plan alone + U0, 24 worlds per seed.
- Admission: 95% CI upper bound of (plan - U0) total delay < 0 and every group mean (L0, L1, L2, HDV, CAV) <= +10 s.
- Banks: admitted plans among each arm's first six; none admitted -> U0. No --max-plans. Stated risk: holds of a
  multi-plan bank stack (A68).
- PRIMARY E1: admitted count at budget 6, WM - Random per seed, exact paired permutation over 6 pairs, two-sided.
- Secondary: bank total delay WM vs Random and each bank vs U0 on the 96 confirmation worlds.

E2. Arms F (original bank), S (synchronised onsets), H18..H22 (uniform hold).
- Tuning: five H speeds + U0 on 24 Development worlds (a35_D2800_p60 worlds 0..23, seeds 998200-998223, already viewed).
- Rule registered before tuning: H* = lowest mean total delay on the Development worlds; ties -> lower speed.
- Confirmation: U0, F, S, H* on the 96 confirmation worlds (U0 wave shared with E1).
- PRIMARY E2: total delay F - H*, paired percentile bootstrap, two-sided.

Multiplicity: Holm over the two primaries, two-sided, alpha 0.05. Stated limit: the E1 permutation test has a
minimum p of 2/64 = 0.03125, so E1 can pass only if E2's p <= 0.025.

World-seed blocks (each reserves its full hundred): CONFIRMED BY MAIN 2026-11-13 (independent grep of the seven hundreds in REGISTRATION.md, ledger, ops/tools/protocols/docs and the C4 run dir: zero seed hits; two decimal-fraction false positives in RESULTS_SUMMARY.md)
- e1v_seed11 993800-993823; e1v_seed23 994100-994123; e1v_seed42 994500-994523; e1v_seed67 995400-995423;
  e1v_seed89 999300-999323; e1v_seed103 999800-999823
- conf 990900-990995 (96 worlds, shared E1 + E2)
- reserved, unused: 990500-990599 (Development fallback), 999400-999499 (spare)
Evidence: no standalone-integer hit in REGISTRATION.md (03:17 version), the code tree, the ledger, the C4 run
directory, c4/* branch tips, A70/A71 code and tables. A text search cannot prove absence of unrecorded use.

Budget cap (simulated seconds): E1 verification 2,246,400; E1 confirmation 1,382,400; confirmation U0 115,200;
E2 confirmation 345,600; E2 tuning 144,000 + Development U0 28,800; replay 115,200; mechanics 9,600.
Total cap 4,387,200. Allocations are recorded when each starts.

Status at registration: code built and unit-tested (30 local tests pass); Box 3 crosscheck PASS for all 8 bank
files; Box 3 dry build PASS for the Development block. The WM scorer's backend path has not been executed yet.
