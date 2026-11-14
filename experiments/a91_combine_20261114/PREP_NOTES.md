# A91: can two plans run at the same moment? Five-arm fresh panel at 60 % (D2800_p60), prepared and NOT launched

Prepared 2026-11-14, 00:20 to 00:45 Box 3 local time. Zero SUMO, zero training. The `pin` stage and a zero-SUMO
`e2e` analysis test ran once, both charged 0. `/root/a91/root` has no `CHARGES.jsonl` and no `evaluation/`. The queue
`/root/a91queue.sh` is installed and NOT started. REGISTRATION.md and the ledgers were not edited. Nothing was committed.

## 1. Code check (Box 3 `/root/a89/src`, executor digest d1af7e0f; a75_holdforms.py file sha 72046db6)

Files: `pytools/tools/coopmwm/` `oracle_plan.py` (OP), `a75_holdforms.py` (H), `a32_arms.py`, `a44_lcvariant.py`.

**a. Arbitration.** A bank runs `HoldFormCellBankCore` (H:421). It has no `step` of its own, so it uses
`OracleCore.step` (OP:479-499). That step builds one core per plan (H:431-438): plain plans get `LcStrategyCore`,
extended plans get `HoldFormCore`.
- **Each plan keeps its own session ledger.** Slots and cooldown are checked per plan: StrategyCore OP:210-219 and
  HoldFormCore H:385-394 (`busy < slots`, `t >= next_ok`, `next_ok = t + cooldown_s`). This is documented at OP:17-21
  and OP:435-441.
- **There is no global lock, bank cooldown or lane lock.** The only cross-plan rule is about CAVs. For plan i, `_busy`
  is the set of CAVs in any other plan's session (OP:487). A selection that contains a busy CAV is refused as
  `member_busy`: a32_arms.py:76-78 for plain plans, H:411-413 for extended plans. As a backstop, a CAV in two sessions
  raises an error (OP:495-496).
- **So two plans can be active at once, on disjoint CAVs.**
- **When two offer conditions are true in the same second,** plans are tried in a fixed priority order (OP:484). The
  order comes from the `a75_bank_plans` sort (H:218): situational rows first, then −lb95, key, family. The first plan
  takes its CAVs.
  - The later plan still runs its own `select_members`. Ranks are counted over all zone CAVs, including busy ones
    (free list H:387 / OP:212; ranking H:231-243).
  - If the later plan picks a CAV the first plan has just taken, it is refused for that second. It does not fall back
    to another CAV. If its picked CAVs are disjoint, it fires in the same second.
- **What this means for L2 + TAPER.** Both plans want the lane-2 rank-0 CAV of the `far` zone. The L2 row has priority
  (lb95 2917.05 > 0.0). While one plan holds that CAV, and the CAV is still the lane-2 lead in the zone, the other plan
  is refused. The plans overlap only when a different lane-2 CAV has become the zone lead during the other plan's
  session. The analysis measures this; it is not assumed.

**b. A bank-file-only way to run them together.**
- **Yes, the existing two-plan bank already allows overlap** (point a). That is the BANK_TURNS arm, so "turns" is a
  misnomer.
- **A one-plan union is expressible but can never fire.**
  - Schema: one plan row can mix a plain "rolling" role with "taper" roles. The new role fields are per role
    (ROLE_DEFAULTS H:39-41, spec_of H:97-106, canonical H:62-80). Any extended role makes the whole plan run under `HoldFormCore` (H:83-85, H:431-437),
    with default-valued fields for the plain role. The executor admits the union: there is no duplicate-role refusal in
    `holdform_refusal` (H:120-164).
  - Fatal problem: the union has two roles with lane 2, rank 0 (the L2 role and TAPER role 0). `select_members` maps
    both to the same CAV and returns None on the duplicate-id check (H:244). The plan never activates. This is shown in
    the test `test_union_plan_is_not_executable`, which passed on Box 3 with the real executor.
  - The offer conditions also differ: spread 200 vs 300 m, cooldown 30 vs 140 s, and TAPER needs 4 CAVs. One
    `offer_condition` per plan cannot express "both conditions required". Picking one condition would also change
    selection.
  - Making the union fire would need a renumbered L2 rank, which is retuning. **So there is no CONCURRENT arm.**

**c. Minimal code change for forced co-activation (NOT made).**
- Option 1: in `select_members` (H:224-251), rank each role only among CAVs not already chosen for an earlier role in
  the same plan, and not busy in another plan. In a bank: filter `_busy` out of `free` before ranking, at H:387 and
  OP:212. Then the union, or the bank, picks the next lane-2 CAV instead of failing.
- Option 2: a bank-level "joint" rule. When plan A fires, plan B is offered in the same second with A's members
  excluded.
- Either option changes the executor digest (d1af7e0f), so it would need a new executor plus mechanics. Not done.

## 2. Package (`/root/a91`; `ship/` = shipped sources + SHIP_FILES.sha256)

- **Driver `ops/a91_combine.py`** (aed9bd9d). It is a thin wrapper over A81 `a81_bigconfirm.py` (c755a551,
  unchanged), using the A89 re-pointing pattern.
  - Its own pieces: `prepare` (3 bank copies), `waves`, `mech_compare`, `analyse` / `analyse_core`, `dry`, and the
    analyse CLI with its own `summary()` line. That last item fixes the A89 crash: KeyError 'primary' at
    a81_bigconfirm.py:677.
- **Tree:** a real copy of `/root/a89/src`. The only file changed is the runner: 982ec0cf + `A91_RANGES` →
  **91e40a94**. New files: `a91_combine.py`, `test_a91_combine.py`, `a91_analyse_e2e.py`, and `ops/a91/` (the builder,
  the patch, the seed audit, and two banks).
- **Supporting files:** `run_a91.sh` 9caad4cf, `setup_a91.sh` e418d9d3, `build_a91_turns.py` 7a82062c,
  `patch_tables_a91.py` 96fdafd1, env 478e6744.

| arm | runner arm | definition |
|---|---|---|
| L2_ONLY | A75_PLAN | family a89l2_d86b1d6ac653e8ad of the A89 arms bank **de9a30a9** (byte copy; same spec as A89 ROLL_L2_ONLY except job_id / bank path) |
| BANK_TURNS | A75_BANK | the two-row bank **A91_TURNS f10e7689**: L2 row + TAPER row mut_1fca373046a3c66e copied unchanged (canonical-equal proofs in `box_outputs/build_turns.out`). Executor priority: L2 first (LcStrategyCore), then TAPER (HoldFormCore) |
| ROLL_BOTH | A75_PLAN | d2_d86b1d6ac653e8ad of de9a30a9 (= A89) |
| DUO_HOLD_60C, MAPPO60C | DUO_HOLD / U0 | the A81 arms, unchanged |

- **Not run, by instruction:** CONCURRENT (see 1b), TAPER alone, ROLL_L0_ONLY.
- **Dry spec diff vs L2_ONLY:**
  - ROLL_BOTH differs only in `a75_family` and `job_id`.
  - BANK_TURNS differs only in arm / `a75_arm` / `a75_bank` / `a75_family` / `job_id`.
- **Panel:** `A91_PANEL_C.json`, seeds **988600-988695**. 988696-988699 are reserved and unused.
  - file sha256 **0f483ac58d98bccdae11819e1d75ab5ee63b9a81ee6af20390f22a3db380a896**
  - panel_sha256 **e5d51c5ad882c161bf134e080474aae0892a74c3e98a53a32433d8903042d386**
- **Seed audit** (`ship/SEED_AUDIT_A91.json` 12d13b9d; `audit/seed_audit_box3_988600.json` 709f6054):
  - REGISTRATION.md: 0 hits. One regex match sits inside a hex digest.
  - experiment_history.json: 0 hits.
  - Box 3 `/root` + `/home/ubuntu` (29,669 files, 506 panels): 0 panel overlaps and 0 runner overlaps. The only code
    hit is a hex string in CPython's svnmap.txt.
  - Box 3 `/root/*/root/CHARGES.jsonl`: 0 hits.
  - Not done: a broad laptop working-tree grep (denied earlier; not attempted), git branch tips, Box 4.

## 3. Budget (1,200 sim-s per episode)

- **mech:** 12 episodes = **14,400**, all on A70 worlds 996300/996301:
  - A81 identity for A75D2_BANK_60C, DUO_HOLD_60C and MAPPO60C (vs stored A70 / A75-D2 records).
  - Identity of L2_ONLY and ROLL_BOTH vs the stored **A89 mechanics records** (receipts 30269ad4 / 1699e048). The
    stored-side report is recomputed read-only from the sealed log, so nothing is written under `/root/a89`.
  - BANK_TURNS rule checks: per-family `plan_checks`, zone, executor, digest, no CAV in two sessions, 0
    collisions/teleports.
- **confirm:** 5 × 96 × 1,200 = **576,000**.
- **Phase cap:** **590,400**, enforced in code. No retry reserve.
- **Runtime:** A89 ran 576 confirm episodes in about 50 min with 46 workers. Expected: mech about 8-10 min, confirm
  480 episodes about 42-50 min, analyse under 1 min. **Total about 55-60 min.**

## 4. Launch (prefix `timeout 60 ssh -o BatchMode=yes -i ~/.ssh/rented_box ubuntu@124.220.72.157 '<cmd>' < /dev/null`)

Register first, then start the queue once:
```
sudo -n bash -c "setsid -f nohup bash /root/a91queue.sh < /dev/null > /root/a91queue.nohup 2>&1"
```

The queue script is `/root/a91queue.sh` (1d00da3e). What it does:
- Before waiting, it refuses unless pinned, and refuses if mech was already launched or CHAIN_STOPPED exists.
- It waits for `A86Q_DONE` in `/root/a86queue.log`.
- If `/root/a85queue.log` exists, it then waits for `A85Q_DONE`. If that log is still absent at 01:40 box time, it
  stops waiting and proceeds.
- It runs `run_a91.sh mech`, then `run_a91.sh confirm` (analysis included), once each with 46 workers and no retry.
- A failure writes `/root/a91/CHAIN_STOPPED`.
- `/root/a91queue.log` ends with `A91Q_DONE`. The confirm chain file ends with `A91_CONFIRM_DONE`.
- To stop between rounds: `sudo -n touch /root/a91/root/STOP_A91`.

Manual alternative, as for A89: `A91_WORKERS=46 A91_TURNS_SHA256=f10e7689… A91_PANEL_SHA256=e5d51c5a… bash
/root/a91/run_a91.sh mech`, then `… confirm`.

## 5. Registered analysis (in the driver, zero SUMO)

All numbers come from the a42 `D._contrast` → `nm_finaleval._bank_compare`, as in A81/A89: paired percentile bootstrap
10,000, seed 20261104; diff = treatment − reference, negative = less delay. This panel only, never pooled.

- **PRIMARY: L2_ONLY − DUO_HOLD_60C.** SUPPORTED iff CI95 upper < 0 AND `groups.all_pass`, with the five-group 10 s
  rule computed relative to **DUO_HOLD_60C**. `_bank_compare` takes any reference (`co_primary_groups(...,
  reference=reference)`, nm_finaleval:2867), and A81's primary used DUO_HOLD_60C the same way.
- **SECONDARY:** BANK_TURNS − L2_ONLY; L2_ONLY − ROLL_BOTH (a replication of A89).
- **Descriptive:**
  - L2_ONLY, BANK_TURNS, ROLL_BOTH and DUO_HOLD_60C − MAPPO60C.
  - CVaR90 (`tail_cvar90`), hard brakes, TTC, collisions and teleports for every contrast.
  - Arm means and safety totals.
  - Activation counts per arm, plus per family within BANK_TURNS (`activations_by_family`, `offers` fire /
    member_busy).
  - BANK_TURNS overlap: concurrent seconds with both plans active, activations started while the other plan was
    active, worlds with any overlap, and CAV-in-two-sessions (must be 0).

## 6. Receipts

| what | path |
|---|---|
| pin and e2e receipts | `/root/a91/chain_pin.out` (PIN_OK, charged 0), `chain_e2e.out` (ANALYSE_E2E_OK, charged 0), `dry_i.out`, `dry_c.out`, `pin_prepare.out`, `build_turns.out`, `/tmp/a91_setup.out` (copies in `box_outputs/`) |
| after the run | `root/A91_MECH.json`, `root/A91_RESULTS_D2800_p60.json`, `analyse.out`, `chain_{mech,confirm}.out`, `root/A91_LOG.jsonl`, `root/CHARGES.jsonl`, `root/WAVE_A91*_*.json`, `/root/a91queue.log` |

Tests: `run_a91.sh test` passed **10/10** on Box 3, including the turns-bank derivation and the union-plan test. The
e2e analysis ran on the A89 confirm layout copied to /dev/shm. It printed structure only, and the /dev/shm copy was
removed.

## 7. Doubts and disclosures

1. **BANK_TURNS overlap is opportunistic, not forced.** With the shared lane-2 lead and L2 priority, overlap may be
   rare. If it is near zero, BANK_TURNS answers "bank arbitration" rather than "simultaneous". Forcing co-activation
   needs a code change (1c).
2. **Priority comes from carried fields.** It is set by the lb95 values carried in the rows (L2 2917.05 from c325ec1d;
   TAPER 0.0 from the E1R bank). They were not chosen for A91.
3. **The primary is post hoc.** L2_ONLY was picked as the best A89 arm, so this is a replication on fresh worlds.
4. **The e2e test read the A89 confirm records.** It used them as stand-ins (TAPER → BANK_TURNS) to exercise the code.
   No number was printed or used.
5. **The a85 wait can stall.** If `/root/a85queue.log` exists but the A85 queue never writes `A85Q_DONE` (for
   example, it is REFUSED before its wait loop), the A91 queue waits indefinitely. This follows the brief.
6. **Files left on Box 3:**
   - outside `/root/a91`: `/root/a91_ship/`, `/tmp/a91_ship/`, `/tmp/a91_setup.out`, `/tmp/a91_seed_audit*.json`,
     `/tmp/a91_seed_audit_box.py`, `/tmp/a91_probe.py`.
   - inside `/root/a91`: `chain_e2e.out`, created by the e2e stage.

## 8. Draft registration paragraph

> **AMENDMENT 91 - A91** (can two plans run at the same moment, D2800_p60; POST-HOC, exploratory). Registered `<time>`
> before any A91 SUMO second.
>
> - **Question.** Does the A89-best single lane-2 hold (L2_ONLY) beat DUO_HOLD_60C on fresh worlds? Does running it in
>   a two-plan bank with the TAPER plan, under the unchanged executor arbitration that allows concurrent sessions on
>   disjoint CAVs, change it?
> - **Panel.** 988600-988695 (panel e5d51c5a, file 0f483ac5; audit 12d13b9d). Box 3 `/root/a91`, phase combine_a91.
> - **Arms** (all on MAPPO60C 2abed317; same worlds):
>   - L2_ONLY and ROLL_BOTH: A75_PLAN on de9a30a9.
>   - BANK_TURNS: A75_BANK on f10e7689.
>   - DUO_HOLD_60C and MAPPO60C: as A81.
>   - No union arm: not executable without a code change.
> - **Mechanics** (14,400; PASS required): A81 identity, plus identity of L2_ONLY / ROLL_BOTH vs the A89 mechanics
>   records, plus BANK_TURNS per-plan rule checks.
> - **Confirm:** 5 × 96 × 1,200 = 576,000 (cap 590,400).
> - **Analysis** (a42 `_contrast`; paired percentile bootstrap 10,000, seed 20261104; negative = less delay):
>   - PRIMARY: L2_ONLY − DUO_HOLD_60C, SUPPORTED iff CI95 upper < 0 AND the five-group 10 s rule vs DUO_HOLD_60C.
>   - SECONDARY: BANK_TURNS − L2_ONLY; L2_ONLY − ROLL_BOTH.
>   - Descriptive: every arm − MAPPO60C, CVaR90, safety, activations per plan, BANK_TURNS overlap.
> - **Code:** driver a91_combine.py aed9bd9d (wrapper over A81 c755a551); runner 91e40a94; run_a91.sh 9caad4cf; queue
>   1d00da3e.
> - **Launch rule:** one launch per stage, no retry.
