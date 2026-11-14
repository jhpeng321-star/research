# Claim validation 20261113_0231 - protocol

Frozen 2026-11-13 before viewing any formal result of this round: at freeze time no E1 pool has a world-model (WM)
score and no SUMO episode of this round exists. Machine-readable copy with every file hash: `PROTOCOL.json`
(sha256 537bc8fa79cf327ccc043bb067fb324591b3a97f2ad027598bb237f5526d2885).

World-seed blocks in section 6 are PROPOSED. They are not used until MAIN confirms them.

Cell D2800_p60, map wz3to2, 1200 s episodes. Reference U0 = frozen speed-only base MAPPO
(sha256 d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca).
Sign: difference = treatment - reference; negative total delay = improvement.

## 1. Frozen models, code and config

| item | sha256 |
|---|---|
| WM checkpoint `wm_s.pt` | 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 |
| `tools/wzprop/strategy_search.py` (code_v12r7b) | 02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc |
| start states `data/regime_STATES.json` (A42/A47/A68; 18 valid states) | 74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3 |
| search config `inputs/A47_SEARCH_CFG.json` | f5636a96759d567b6d9510fc6f38c2391b43a019464dc0e3fe53bb4109044be9 |
| scorer `wm_pool_score.py` | 94824b8a4b916d67e0cbeb785af84c745825c8601aa64d341d80d495b0c3cc1f |
| original bank F `inputs/A42_BIG_BANK_D2800_p60.json` | 7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff |

The search config is the `config` object of the A47 result, unchanged: K_screen 4, n_screen_states 4,
finalists 12, K_final 4, n_final_states 30 (only 18 valid states exist, so all 18), H 300 s, lam 0.5,
seed 20261113. All other file hashes are in `PROTOCOL.json` -> `file_sha256`.

## 2. E1 - does the WM order find more admissible plans than a random order?

**Pool (done, frozen).** Search seeds 11, 23, 42, 67, 89, 103 (RNG seeds, not world seeds). Per seed 120 distinct
legal hold-only plans, 30 per group size 3/4/5/6, drawn uniformly from the registered descriptor space
(`e1_pool.py`, stream `claim_validation.e1_pool.v1`). The generator reads only the search config.

| seed | POOL file sha256 | pool doc sha256 |
|---|---|---|
| 11 | 3f124460a045d95b0c1cbef53c3d582adcc78c28a017fca55a7646ba4d0defb4 | c9739a70840f... |
| 23 | cf73db5844a8b10bd1ddba0515c4cdea634e9b66787f1165f447f1136f418a03 | 8ab7b51d70f5... |
| 42 | af8f1c445d98ec7d4bd68524e4b050dc30dce96db61450ccec9eedfe361ef51c | 29c2ff9a73df... |
| 67 | 4351ef1b4fb7580d80c6ea767f7a75dd5e72be27a72736911fe984b500c22a03 | 1698dd8b6ac9... |
| 89 | be51a2d8fbfc8f773f5ac668913bf0c33dbfc139dc1088799e3901b856e20f7e | a95931a555d2... |
| 103 | b2a90a7992ecde63c47c143aa1adfeb86367e82cb7efbc61a9c023bb510aa058 | 0be576d8a37a... |

**Random arm (done, frozen before any WM score).** The pool's descriptor hashes, sorted, shuffled by
`random.Random(sha256(stream|seed))`, stream `claim_validation.e1_random_arm.v1`; top six = first six.
`e1_random_arm.py` has no file access. Files `out/e1/order/ORDER_RANDOM_seed<N>.json` (hashes in `PROTOCOL.json`).

**WM arm (two-stage scoring, the default of `wm_pool_score.py`).**
1. Screen: all 120 plans on `random.Random(search_seed).sample(valid_states, 4)` with K 4.
2. Finalists: exactly the top 12 by screen `summary.score`.
3. Final: the 12 finalists on all 18 valid states with K 4.
4. Rank finalists by final `summary.score` descending, ties by screen rank. WM top six = first six.

`summary.score` is the search's own score (`strategy_search.summarize`): 95% lower bound of the gain against base
minus lam x disagreement / sqrt(n states). Differences from the A47 search, stated: the pool is fixed, there is no
mutation, and no archive leaders or comparators are added to the finalists. The scorer never reads a SUMO outcome.

**Verification.** Per seed, the union of both arms' top six (at most 12 plans), each run alone, plus U0, on that
seed's 24 verification worlds. A plan in both arms runs once.

**Admission** (the rule that admitted the original bank, `a37_cellbank.unc_verdict`): PASS iff the 95% CI upper
bound of (plan - U0) total delay is below 0 and each of the five group means (L0, L1, L2, HDV, CAV) is at most +10 s.

**Banks.** Per seed and arm: the admitted plans among the arm's first six. No admitted plan -> the bank is U0.
`--max-plans` is not used. Risk stated in advance: the executor has no bank-level activation cap, so the holds of a
multi-plan bank stack (A68: an 8-plan bank held about 44 times per world and failed confirmation).

**Primary E1.** Admitted-plan count at budget 6, WM minus Random per search seed; exact paired sign-flip
permutation over the six seed pairs, two-sided.
Secondary: total delay of the WM bank against the Random bank, and of every bank against U0, on the 96
confirmation worlds. Budgets 2 and 4 are descriptive.

## 3. E2 - is the original bank F better than a matched uniform hold?

Arms: F (original four-plan bank, byte-identical), S (same plans, synchronised onsets), H18..H22 (uniform hold at
one common speed, resource matched).

- Tuning: the five H speeds plus U0 on 24 Development worlds.
- Selection rule, registered before tuning: H* = the speed with the lowest mean total delay on the Development
  worlds (the strongest uniform hold); exact ties -> the lower speed. Nothing else is tuned.
- Confirmation: U0, F, S, H* on the 96 confirmation worlds (the same U0 wave E1 uses).

**Primary E2.** Total delay F minus H* on the 96 confirmation worlds; paired percentile bootstrap, two-sided.
Secondary: F - S, every arm - U0, executed control effort, replay on at most 24 Development worlds.

## 4. Multiplicity

Holm over the two primaries, two-sided, alpha 0.05.
Caveat stated before any result: an exact permutation over six pairs has a minimum two-sided p of 2/64 = 0.03125.
So E1 can be significant only if E2's p is at most 0.025 and E1's p is at most 0.05.

## 5. Sizes and budgets (simulated seconds)

| part | size | budget |
|---|---|---|
| E1 verification | 6 seeds x (<= 12 plans + U0) x 24 worlds | <= 2,246,400 |
| E1 confirmation | 12 banks x 96 worlds | <= 1,382,400 |
| Confirmation U0 (shared E1 + E2) | 96 worlds | 115,200 |
| E2 confirmation | F, S, H* x 96 worlds | 345,600 |
| E2 tuning | 5 speeds x 24 worlds + U0 | 144,000 + 28,800 |
| Replay | <= 24 Development worlds | <= 115,200 |
| Mechanics | | 9,600 |
| **Total cap** | | **4,387,200** |

WM scoring uses zero SUMO: at most 2,960 WM rollouts per seed (screen 1,952 + final 1,008), expected 750 to 1,350.

## 6. World-seed blocks - PROPOSED, not used until MAIN confirms

| block | worlds | use |
|---|---|---|
| e1v_seed11 | 993800-993823 | E1 verification |
| e1v_seed23 | 994100-994123 | E1 verification |
| e1v_seed42 | 994500-994523 | E1 verification |
| e1v_seed67 | 995400-995423 | E1 verification |
| e1v_seed89 | 999300-999323 | E1 verification |
| e1v_seed103 | 999800-999823 | E1 verification |
| conf | 990900-990995 | confirmation, shared by E1 and E2 |
| (fallback) | 990500-990599 | Development fallback; not needed |
| (spare) | 999400-999499 | spare |

Each block reserves its full hundred. Development = worlds 0..23 of the existing, already-viewed panel
a35_D2800_p60 (998200-998223, panel sha256 425e8a8d...); used for tuning, replay and mechanics only.

Unused-seed evidence: no number of the nine hundreds appears as a standalone integer in REGISTRATION.md (03:17
version, 4,099,714 bytes), the code tree `ops/` and `tools/`, `protocols/experiment_history.json`,
`runs/active/C4-V7FT-20261017`, `a42-run`, any `c4/*` branch tip, the A70/A71 code (they use 992500, 996300,
998600) or the Box 3 `/root/a70` and `/root/a71` tables. All blocks are outside the banned ranges.
A text search cannot prove a seed was never simulated by an unrecorded run.
