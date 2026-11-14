# A94 / A95: searched bank vs hand bank on a fresh 192-world panel (D2800_p60). Prepared, NOT launched

Prepared 2026-11-14, 01:57 to about 02:30 Box 3 time. Zero SUMO, zero training. Rules: Amendment 94 (01:56), overridden by
Amendment 95 (02:09): BEST_SINGLE dropped, BANK_HAND added, primary = BANK_SEARCHED - BANK_HAND. Box 3 `/root/a94`.
`/root/a94/root` has no `CHARGES.jsonl` and no `evaluation/`. REGISTRATION.md and the ledgers were not edited. Nothing
was committed. `/root/a94queue.sh` is installed and NOT started.

## 1. Package

- It is a close copy of A91. Driver `ops/a94_bank.py` is a thin wrapper over the unchanged A81 driver `a81_bigconfirm.py`
  (c755a551), copied from `a91_combine.py` (aed9bd9d). It keeps the A91 fix (its own analyse CLI summary).
- Tree: a real copy of `/root/a91/src`. The only file changed is the runner: 91e40a94 + `A94_RANGES = ((988700, 988899),)`
  -> **47fb9f76**. New: `a94_bank.py`, `test_a94_bank.py`, `a94_analyse_e2e.py`, `a94_mech_e2e.py`, `ops/a94/`
  (`build_bank_a94.py`, `patch_tables_a94.py`, `SEED_AUDIT_A94.json`, plus the built banks).
- The arm set is fixed at import by `A94_HAND_SHA256`:
  - a sha gives 4 arms: BANK_SEARCHED, BANK_HAND, DUO_HOLD_60C, MAPPO60C.
  - `none` gives 3 arms (no hand plan admitted); the primary falls back to BANK_SEARCHED - DUO_HOLD_60C.
  - unset: refused.
- Both bank arms are A75_BANK on MAPPO60C (U0 base). DUO_HOLD_60C and MAPPO60C are the A81 arms, unchanged.

**Ship hashes** (`ship/SHIP_FILES.sha256`, c91f4e82):

| file | sha256 |
|---|---|
| a94_bank.py | c8b05145 |
| test_a94_bank.py | 18905f5c |
| a94_analyse_e2e.py | 647e2626 |
| a94_mech_e2e.py | 9d932ea5 |
| build_bank_a94.py | 0e6c35b4 |
| patch_tables_a94.py | ea35b96a |
| run_a94.sh | 008e6983 |
| setup_a94.sh | f1a2bc61 |
| SEED_AUDIT_A94.json | cec7acd7 |

Other files: queue `/root/a94queue.sh` **770a1cc9**; env `env_a94.sh` 91df1f9b.

- Setup ran once with the first ship (A94-only arms; kept in `/root/a94_ship_v1/`). After Amendment 95, the changed files
  were byte-copied into `/root/a94/src`, `/root/a94/run_a94.sh` and `/root/a94_ship/`, and checked with `cmp`. The
  current `setup_a94.sh` reproduces that tree. It adds `a94_mech_e2e.py` and imports with `A94_HAND_SHA256=none`.

## 2. Panel and seed audit

- `A94_PANEL_C.json`: seeds **988700-988891**. 988892-988899 are reserved and unused.
  - file sha256 **720d19e1df5d95c1b769170b62e234c1498c1c616fafa209c1880b5a474095ac**
  - panel_sha256 **d85999ddaaf3eb583775d66dbabca168b4432966bc01e7193dfc79f244f7946c**
  - Built by `run_a94.sh panel` (A81 `build_panel_c`, the same path as A91). PANEL_OK, charged 0.
- Seed audit (`ship/SEED_AUDIT_A94.json`; `audit/seed_audit_box3_988700.json` 066b92d4):
  - Box 3 `/root` + `/home/ubuntu`, using A91's tool (9655906c): 32,148 files, 555 panels, **0 panel overlaps, 0 runner
    overlaps**.
  - The only code hits are hex strings in CPython `svnmap.txt`.
  - `/root/*/root/CHARGES.jsonl`: 0 delimited tokens.
  - REGISTRATION.md: only the A94 paragraph itself.
  - experiment_history.json: 0 hits.
  - Not done: the laptop working tree, git branch tips and Box 4.

## 3. Banks

- **BANK_SEARCHED** (built for real, `run_a94.sh bank-searched`; BANK_OK):
  - file `src/ops/a94/A94_BANK_SEARCHED_D2800_p60.json` **40b16ed95add6fcd63385dabcbf7e8a03fe8dbf20d0418e58e5aa8feee30ee04**
    (build receipt d04b72c9; copies in `pinned/`).
  - 6 plans, 0 duplicates. Rows and templates are canonical-equal to the S5 bank (577a0564, pinned) and the S4 bank
    (a428cd1d, pinned).
  - Executor priority: deba0680 > rand_8b08 > 576d526e > 28cad0ce > 069966 > 97dda2be.
- **BANK_HAND** (`build_bank_a94.py hand --hand ROOT=SHA|none [...]`):
  - One root: its rows are used unchanged, and the output is checked byte-equal to the source.
  - Two roots: the admission code's own `a75_d2forms.bank_choice` runs unchanged (a80 FORMS, MAX_BANK 4) on the union of
    both receipts' verdict rows.
    - **Statistic:** verdict field `ci95[1]`, the admission CI95 upper of (plan - MAPPO60C) total delay. It equals
      -`lb95_veh_s` in the row. The most negative wins; ties go by strategy `key`.
    - Root 1 comes before root 2 in the duplicate order.
    - The chosen rows are copied unchanged from their own root's bank file.
  - No plan admitted anywhere: the build writes NO_HAND, and you then use `A94_HAND_SHA256=none`.
  - Stand-in tests passed:
    - one root (S4 bank as the fake hand bank);
    - two roots (S5 + S4);
    - a synthetic no-plan root;
    - a wrong sha stops the build.

## 4. Analysis (driver, zero SUMO)

All numbers come from a42 `D._contrast`: paired percentile bootstrap 10,000, seed 20261104; diff = treatment - reference,
negative = less delay.

- **PRIMARY: BANK_SEARCHED - BANK_HAND.** SUPPORTED iff CI95 upper < 0 AND the five-group rule relative to BANK_HAND.
  `direction` is one of:
  - SEARCHED_BETTER;
  - NOT_SEPARATED;
  - REFERENCE_BETTER (CI95 lower > 0).
- **Fallback primary** (no hand plan): BANK_SEARCHED - DUO_HOLD_60C.
- **SECONDARY:** BANK_SEARCHED - DUO_HOLD_60C; BANK_HAND - DUO_HOLD_60C; BANK_SEARCHED - MAPPO60C; BANK_HAND - MAPPO60C.
- **Descriptive:**
  - DUO_HOLD_60C - MAPPO60C;
  - CVaR90 and safety per contrast; arm means and safety totals;
  - per bank arm: activations per plan, `offers_by_plan` (fire / member_busy), and the A91 overlap block (concurrent
    seconds, starts while another plan was active, CAV in two sessions, worlds with overlap) plus seconds by number of
    plans active;
  - **CAVs engaged at once**: a histogram of seconds by k, and max k.

## 5. Tests run (all zero SUMO, charged 0)

- `run_a94.sh test`: **12/12 passed**.
- `run_a94.sh drytest`: **DRYTEST_OK**, for both the 4-arm layout and the 3-arm layout. It ran:
  - prepare, dry i and dry c;
  - mech rule-check e2e on the A91 mechanics records: MECH_E2E_OK;
  - analysis + summary e2e on the A91 confirm layout in /dev/shm: ANALYSE_E2E_OK.
- A91 outcome records were used only as stand-ins. Only structure was printed.
- The `/root/a94/root` file list was unchanged.
- Output: `box_outputs/drytest.out`.

## 6. Budget and time

| layout | mech | confirm | cap |
|---|---|---|---|
| 4 arms | 10 episodes, 12,000 (cap 14,400) | 4 x 192 x 1,200 = 921,600 | 936,000 |
| 3 arms | 8 episodes, 9,600 | 691,200 | 705,600 |

- Each stage launches once, with no retry, on 46 workers. A failure writes `/root/a94/CHAIN_STOPPED`.
- Expected time: mech about 10-13 min (A91: 12 episodes in 13 min). Confirm takes about 64 min for 768 episodes (A91:
  480 in 40 min), or about 48 min for 3 arms. Analysis takes under 1 min. **Total about 75-80 min** (about 60 with 3
  arms).

## 7. Exact command sequence after the hand admission(s) finish

Prefix every command with `timeout 120 ssh -o BatchMode=yes -i ~/.ssh/rented_box ubuntu@124.220.72.157 '<cmd>' < /dev/null`.

1. **Build BANK_HAND** (zero SUMO). Give one root, or two:
   ```
   sudo -n env A94_HAND_ROOTS="/root/a95adm/root=<sha of /root/a95adm/root/a80_banks/A80_BANK_D2800_p60.json, or none>" bash /root/a94/run_a94.sh bank-hand
   ```
   - For two roots, use `A94_HAND_ROOTS="/root/a95adm/root=<sha|none> /root/a95adm_b/root=<sha|none>"`.
   - Read the result with `sudo -n cat /root/a94/chain_bank-hand.out`. It shows the BANK_HAND plan ids and either
     `sha256 A94_BANK_HAND_D2800_p60.json <H>` or `... none`.
   - If the builder STOPs, nothing is written and the stage can be re-run. See `/root/a94/bank_hand.out.stopped_*`.
2. **Register** the bank shas: BANK_SEARCHED 40b16ed9..., BANK_HAND `<H>` or none, and panel d85999dd....
3. **Pin** (zero SUMO):
   ```
   sudo -n env A94_BANK_SHA256=40b16ed95add6fcd63385dabcbf7e8a03fe8dbf20d0418e58e5aa8feee30ee04 A94_HAND_SHA256=<H|none> bash /root/a94/run_a94.sh pin
   ```
   Then check `chain_pin.out` for PIN_OK and check `/root/a94/PINNED_SHAS`.
4. **Launch, once:**
   ```
   sudo -n bash -c "setsid -f nohup bash /root/a94queue.sh < /dev/null > /root/a94queue.nohup 2>&1"
   ```
   - The queue waits for nothing. It refuses unless pinned, with the panel equal to d85999dd.
   - It takes the shas from PINNED_SHAS and runs mech, then confirm (analysis included).
   - `/root/a94queue.log` ends with `A94Q_DONE`.
   - To stop between rounds: `sudo -n touch /root/a94/root/STOP_A94`.

## 8. Receipts

| when | what |
|---|---|
| now | `/root/a94/chain_panel.out`, `panel_c.out`, `chain_bank-searched.out`, `bank_searched.out`, `src/ops/a94/A94_BANK_SEARCHED_BUILD.json`, `/tmp/a94_setup.out`, `/tmp/a94_drytest2.out`, `/tmp/a94_seed_audit.json` (laptop copies in `box_outputs/`, `pinned/`, `audit/`) |
| after the run | `chain_bank-hand.out`, `bank_hand.out`, `chain_pin.out`, `PINNED_SHAS`, `root/A94_MECH.json`, `root/A94_RESULTS_D2800_p60.json`, `analyse.out`, `chain_{mech,confirm}.out`, `root/A94_LOG.jsonl`, `root/CHARGES.jsonl`, `/root/a94queue.log` |

## 9. Disclosures / doubts

1. BANK_SEARCHED was built after the S4 outcome was seen. This is the Amendment 95 rule change, already disclosed there.
2. The mechanics have no identity check for the bank arms. No stored records of these plans exist on 996300/996301.
   The bank arms get the A91-style per-plan rule checks only, run per family with each plan's own zone (the 6 plans have
   4 different zones).
3. The two-root hand merge offsets draw_rank by 1000 per root, in a copy used only for the duplicate order. This is my
   reading of "same statistic and tie-break".
4. Files outside `/root/a94`: `/root/a94_ship`, `/root/a94_ship_v1`, `/root/a94queue.sh`, `/tmp/a94_ship*`,
   `/tmp/a94_*.out`, `/tmp/a94_seed_audit*`.

## 10. Addendum 03:00: S5-only searched-bank option (owner may choose; zero SUMO)

- **Code change** (logged in `/root/a94/UPDATES.log`; the old files are kept as
  `run_a94.sh.v008e6983_20261114_025736` and `build_bank_a94.py.v0e6c35b4_*`):
  - `build_bank_a94.py` **c8746b4a**: `searched --s5-bank` alone means S5 rows unchanged. The receipt records `mode`.
  - `run_a94.sh` **f223c8ad**: the switch is `A94_SEARCHED_MODE=s5only`.
  - `test_a94_bank.py` **980daa34**: 12/12 passed.
- **What `bank-searched` does in s5only mode:**
  - It runs once, and only before pin.
  - It first moves aside the two-search bank 40b16ed9, its build receipt, `chain_bank-searched.out`, `bank_searched.out`
    and the pid file as `*.twosearch_<time>`. Nothing is deleted. A marker `BANK_SEARCHED_S5ONLY_ASIDE` is written.
- **Dry build:** 3 plans, sha **577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6**. It is byte-identical
  to the S5 bank.
- **drytest with s5only:** DRYTEST_OK for both the 4-arm and 3-arm layouts, charged 0
  (`box_outputs/drytest_s5only.out`). Arm names, primary, caps and mech (10 / 8 episodes) are unchanged.
- **Commands:**
  1. Rebuild the bank:
     `sudo -n env A94_SEARCHED_MODE=s5only bash /root/a94/run_a94.sh bank-searched`
  2. Pin with `A94_BANK_SHA256=577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6`.
