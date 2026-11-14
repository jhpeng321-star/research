# Claim validation build 20261113_0231 (E1 ordering, E2 timing / resource controls)

Status: CODE BUILT, NOTHING RUN. Zero SUMO, zero GPU, zero WM scoring, no ssh, no world seed chosen, no existing
file edited. Local checks only: 29 SUMO-free tests pass; local dry-runs of e1_pool, e1_order (random, wm-jobs),
e2_build and e2_replay plan pass. `box_driver.py` and `wm_pool_score.py` are written but have never been executed
(see "Not verified").

Cell: map wz3to2, D2800_p60. Reference arm everywhere: U0 = frozen speed-only base MAPPO (not MAPPO60C).
Sign convention everywhere: diff = treatment - reference, negative = less delay.

## 1. Registered identities found

| item | where | sha256 |
|---|---|---|
| U0 base MAPPO | `nm_finaleval.FROZEN_MAPPO_SHA` | d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca |
| WM checkpoint `ckpt/ckpt/wm_s.pt` (GPU box) | A68 WM scorer receipts | 6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063 |
| WM ckpt.tar / code_v12r7b.tar | same | f9b3a854... / 320e19f3... |
| original four-plan bank | E:/agent-work/a42-run/offload/root/a42_banks/A42_BIG_BANK_D2800_p60.json | 7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff |
| search config (grid) | E:/agent-work/a42-run/a47_search/STRATEGY_SEARCH_D2800_p60.json, key `config` | read, not pinned here |
| executor `tools/coopmwm/oracle_plan.py` | code tree E:/agent-work/a54-mappo60b (HEAD d9507eff, read only) | 73cb1ca6... |
| executor wrapper `tools/coopmwm/a37_cellbank.py` (A37 digest prefix 0a902437) | same | 5d9811a0... |
| runner `tools/coopmwm/nm_eval_runner.py` / `ops/nm_finaleval.py` | same | 62496e86... / 9f289431... |
| driver pattern `ops/a42_bigp60.py`, `ops/a68_research60c.py` | same | 6b117a94... / 7e404fc3... |
| WM search code `tools/wzprop/strategy_search.py` (code_v12r7b) | GPU box tree | 02e119c1... |

Full constants are in `cvlib.py`; every stage that reads the bank checks its sha.
SUMO determinism: IDENTITY_A68 = PASS (stored J reproduced, difference 0.0; the gzip container sha differs, the
decompressed vehicle log is what must be compared), IDENTITY_A67 = PASS. The replay design relies on this.

## 2. Paper text versus code (to settle before the experiments are described)

1. **Legal grid is larger than the paper's four knobs.** Per role: v_w {18,19,20,21,22}, dmax {1.0,1.5},
   hold {90,120,180} s, onset {0,20} s. The code also draws: zone (full / near / far), spread {200,300} m,
   trigger (`wall` with probability 0.4 when the plan has <= 3 roles, else `lanes`), a (lane, rank) slot per role,
   and cooldown is derived (max over roles of onset + hold), not searched. `e1_pool.py` samples this real space.
2. **Onsets.** All four original plans have non-zero onsets (20 s): rand_3a452b1348b01083 (5 roles, 2 non-zero),
   mut_2922c89d0b31d4c8 (6, 2), rand_bb1cb39d47f0a524 (5, 1), mut_d7251d07b4f749e0 (6, 3). So the synchronized
   copy S differs from F for every plan, and no H arm equals F.
3. **Activation cap.** There is no bank-level cap. Each plan has its own cooldown (`next_ok = t + cooldown_s`,
   oracle_plan.py:219), one slot, member exclusivity, 300 s warm-up. Priority = verified lb95 gain descending,
   then key. A bank of N plans therefore holds roughly N times as often as one plan (A68: 8-plan bank about 44
   holds per world against 5 to 10 for one plan, failed confirmation). `e1_bank.py` keeps this registered
   behaviour by default and exposes one explicit option, `--max-plans N` (first N admitted plans in executor
   order). Using it is a registration decision.
4. **"120 plans / 944 rollouts".** The search evaluated 120 descriptors but screened them on 4 states and fully
   scored only 12 finalists. E1 needs every pool member scored under the final protocol: about
   120 x 18 states x K 4 = 8,640 rollouts per seed (estimate about 80 min per seed, about 8 h for six seeds on one
   GPU; estimate from A68 timings, not measured for this script).
5. **WM ranking** uses the raw `summary.score` of the frozen search objective. No archive / fairness filter is
   applied before ranking (the search finals did not apply one either); never-activated descriptors rank last.

## 3. Files

| file | role |
|---|---|
| `cvlib.py` | shared mirrors (descriptor, key, legality, bank document), sealing, write-once, seed-ban check, `crosscheck` |
| `e1_pool.py` | per search seed: 120 distinct legal hold-only descriptors, 30 per size 3/4/5/6, dedupe by sha256 of normalised descriptor, generation order + every attempt + rejection reason + category kept |
| `e1_random_arm.py` | Random arm: `random_order(pool_hashes, seed)`; imports only hashlib and random |
| `e1_order.py` | `random`, `wm-jobs` (job files + exact GPU commands), `wm-ingest` (rank), `show` |
| `wm_pool_score.py` | GPU-box scorer of a fixed pool with the frozen WM (chunked, resumable) |
| `e1_verify.py` | verification plan: union of both arms' top six, 24 worlds, paired with U0, cache keys |
| `e1_bank.py` | `admit`, `build` (banks per seed x arm x budget 2/4/6, FROZEN_BANKS.json, empty bank := U0), `analyse` |
| `e2_build.py` | S (onsets zeroed) and H18..H22 (one common speed, onsets zeroed) banks, descriptor-diff checker, onset report |
| `e2_replay.py` | same-state replay: `plan`, `select`, `check`, `analyse` |
| `e2_analyse.py` | F vs S, F vs each H (worst case over the five speeds), each arm vs U0, actual control effort, Holm |
| `stats.py` | paired bootstrap (10,000), seed x world bootstrap, exact permutation (64 flips), Holm, CVaR90 on common vehicles, hard-brake vehicle-seconds, control effort, log prefix digest |
| `box_driver.py` | on-box adapter to the registered runner (A68 pattern, U0 reference) |
| `patches/SEED_TABLES_TEMPLATE.txt`, `patches/REGISTRATION_TEMPLATE.json` | not applied; table edits and registration MAIN must fill |
| `tests/test_claim_validation.py` | 29 tests |

Every stage writes once, keeps identical files on re-run (resume), refuses different content, and has `--dry-run`.

## 4. Design points worth a decision

- **E1 Random isolation.** The permutation is a pure function of (sorted pool hashes, seed). Tests: the module
  imports only hashlib / random and names no file API; an audit hook shows the function opens no file; the
  `random` command opens only POOL_seed<N>.json even with decoy score / result files beside it.
- **E1 verification blocks.** One block of 24 worlds per search seed (six settings), because the registered
  runner selects one setting per process and each setting owns a 100-seed range. A candidate chosen by both arms is
  verified once. Budget if no overlap: 6 seeds x (12 plans + U0) x 24 worlds x 1,200 s = 2,246,400 sim-s maximum.
- **E1 admission** = the rule that admitted the original bank (`a37_cellbank.unc_verdict`): CI95 upper of
  plan - U0 < 0 and each of L0 / L1 / L2 / HDV / CAV group means <= +10 s, groups from the reference run.
- **E1 confirmation.** Banks with identical content share one run; empty banks are U0 (difference exactly 0).
  Confirmation block size and whether budget 6 is the single primary are MAIN's to register
  (`e1_bank.PRIMARY_PREFIX = 6` is a default label only).
- **E2 S / H.** Same plans, same order, same lb95 priority, same cooldown, same members, holds and dmax.
  S changes onsets only; H changes onsets and v_w only. The checker refuses anything else. Keeping the cooldown
  means S and H plans finish up to 20 s earlier within the same cooldown; changing cooldown instead would change
  activation counts. Actual executed effort is reported per arm, so the match is checked on receipts.
- **E2 replay needs no executor change and no state saving.** Discover: run each world under the original bank
  and read the first launch (t*, plan P, members), kept if t* <= 600 s. Branches, each re-run from t = 0:
  U0; a one-plan bank holding only P with cooldown 100,000 s (one launch possible); the same with P's
  synchronized copy. Because nothing is commanded before t*, the three runs share the state at t*.
  `check` requires, per world: zero launches in U0, exactly one launch in each plan branch at the same t* with
  the same members, and identical decompressed vehicle-log prefix before t*. Worlds failing are excluded and
  counted, not silently used. At most 24 states. Maximum budget: 28,800 (discover) + 86,400 (branches) sim-s.

## 5. Not verified (needs the box or a small registered budget)

1. `box_driver.py` has never been imported against the live tree (local checkout is sparse: `a32_arms.py` is
   absent, so `crosscheck` returns UNAVAILABLE here). Run `crosscheck` and `dry` on the box first. Locally only
   the four registered strategy keys of the original bank are reproduced by the mirror.
2. Seed tables: the tree needs table-only edits (new ranges and settings) and panel files; then the tree's unit
   tests and an identity re-check on published worlds, because runner bytes change.
3. Replay mechanics check (proposed, not run): 2 worlds x 4 runs x 1,200 s = 9,600 sim-s, passing iff
   `e2_replay.py check` reports ok for both. This is the test of "same state at t*" and "exactly one launch".
4. WM scorer mechanics check (GPU, no SUMO): score one small chunk twice, as one call and split, and require
   equal scores, before using `chunk` < 120. Also confirm the scoring config file on the box is the A47 search
   config (`--cfg-name`).

## 6. Commands (PY = the box python; OUT = one shared out directory copied between hosts)

Local, already valid:

    python e1_pool.py --cfg <STRATEGY_SEARCH_D2800_p60.json> --out OUT
    python e1_order.py random --out OUT
    python e1_order.py wm-jobs --out OUT            # writes wm_jobs/ and prints the GPU commands
    python e2_build.py --bank <A42_BIG_BANK_D2800_p60.json> --out OUT --n-worlds <N_m>
    python e2_replay.py plan --bank <bank> --out OUT --n-worlds <n <= 24>

GPU box (one line per seed, printed by wm-jobs; from code_split/code_v12r7b):

    WZPROP_SPLIT_ACTIONS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. python -B \
      <dir>/wm_pool_score.py <cfg>/A47_SEARCH_CFG.json <dir>/wm_jobs/WM_POOL_seed<N>.json <dir>/wm_scores \
      <ckpt>/wm_s.pt 10 120

Then locally:

    python e1_order.py wm-ingest --out OUT --scores-dir <wm_scores>
    python e1_verify.py plan --out OUT

Box 3 (after seed tables + REGISTRATION.json; one block per process):

    PY box_driver.py crosscheck --registration R --banks-dir OUT/e1/banks OUT/e2/banks
    PY box_driver.py prepare|dry|run --registration R --block e1v_seed<N> --waves OUT/e1/verify/VERIFY_PLAN.json \
       --final-root ROOT --banks-dir OUT/e1/banks --workers 46 --chunk 96
    PY box_driver.py collect ... --kind e1verify --out RES/VERIFY_RESULTS_seed<N>.json

    python e1_bank.py admit --out OUT --results-dir RES
    python e1_bank.py build --out OUT --confirm-worlds <N_c> [--max-plans N]
    PY box_driver.py run     ... --block e1c --waves OUT/e1/bank/FROZEN_BANKS.json
    PY box_driver.py collect ... --kind e1confirm --out RES/CONFIRM_RESULTS.json
    python e1_bank.py analyse --out OUT --results-dir RES

    PY box_driver.py run     ... --block e2m --waves OUT/e2/E2_MANIFEST.json --banks-dir OUT/e2/banks
    PY box_driver.py collect ... --kind e2main --out RES/E2M_RESULTS.json
    python e2_analyse.py --bank <bank> --out OUT --results RES/E2M_RESULTS.json [--primaries e1=p e2=p]

    PY box_driver.py run     ... --block e2r --waves OUT/e2/replay/REPLAY_PLAN.json        # discover
    PY box_driver.py collect ... --kind e2discover --out RES/DISCOVER_RESULTS.json
    python e2_replay.py select --out OUT --results RES/DISCOVER_RESULTS.json
    PY box_driver.py run     ... --block e2r --waves OUT/e2/replay/REPLAY_SELECTION.json   # branches
    PY box_driver.py collect ... --kind e2branch --selection OUT/e2/replay/REPLAY_SELECTION.json --out RES/BRANCH_RESULTS.json
    python e2_replay.py check|analyse --out OUT --results RES/BRANCH_RESULTS.json

Exact file names under OUT are printed by each stage; check them with `--dry-run` before a charged run.

## 7. Open decisions for MAIN

Seed blocks (9 blocks, none chosen here) and the table edits; confirmation and E2 block sizes; which two contrasts
are the Holm primaries; whether `--max-plans` is used; confirmation of the WM scoring config on the GPU box;
ledger registration (stable IDs, budgets) before any episode.

Tests: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I -m pytest -q -p no:cacheprovider tests` -> 29 passed.
