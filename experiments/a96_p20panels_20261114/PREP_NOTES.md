# A96 parts (c) and (d): two 20 % four-arm panels (D2800_p20). Prepared, NOT launched

Prepared 2026-11-14, 02:40 to about 03:20 Box 3 time. Zero SUMO, zero training. The rule is Amendment 96 (02:39), parts
(c) and (d).

- New bases: `/root/a96pd` (part d) and `/root/a96pc` (part c). The ship is in `/root/a96_ship`.
- Queues `/root/a96pd_queue.sh` and `/root/a96pc_queue.sh` are installed and NOT started.
- `/root/a82four` was only read. `/root/a94`, the a95/a96adm dirs and running processes were not touched.
- REGISTRATION.md and the ledgers were not edited. Nothing was committed.

## 1. Route: extend A82 part B, the code that changes least

The driver is `ops/a96_p20.py`, a copy of the never-launched A82 part-B driver `a82_fourarm.py` (5181bf4b).

**Unchanged from A82 part B:**
- the arm builders:
  - bank arms = A75_BANK on U0 d9042cc5, built as `a76_d2p20.confirm_waves`;
  - DUO_HOLD = `a32_specs('DUO_HOLD')` (duo12_v20) on U0;
  - MAPPO20C = `a77_mappo20c.bind` plus `A52.with_executor`;
- the p20c setting, the A82 identity mechanics, `_contrast`, and drop-raw.

**Changed:**
- BANK_HAND_P20 added.
- U0 dropped from the panel. It stays in the identity mechanics.
- Per-plan rule checks for the bank arms.
- A94-style contrasts, a direction label, and the descriptive block (overlap, activations / member_busy per plan, CAVs
  engaged at once).
- Part parameters (`A96_PART`).
- Its own summary line (the A91 fix).

**Tree:** a real copy of `/root/a82four/src`.
- Part d: the runner is unchanged (2de0f1c0; it already has `A82_RANGES`).
- Part c: the runner gets `A96C_RANGES = ((984400, 984599),)` and becomes **7f34dadb**.

**Ship hashes** (`ship/SHIP_FILES.sha256`):

| file | sha256 |
|---|---|
| a96_p20.py | b29c45d4 |
| test_a96_p20.py | d07740f6 |
| a96_e2e.py | f2fd9518 |
| build_bank_a96.py | 11585bd4 |
| patch_tables_a96c.py | 902f36b3 |
| run_a96.sh | 82c24b57 |
| setup_a96.sh | 43457023 |
| SEED_AUDIT_A96C.json | e6ab4a0a |

| part | queue | env |
|---|---|---|
| d | bfef5d8e | d1ea146a |
| c | ce7970c8 | bf62e89f |

**Arms and the registered analysis.** diff = treatment − reference, negative = less delay; paired percentile bootstrap
10,000, seed 20261104. The arm set is fixed by `A96_HAND_SHA256`: a sha gives 4 arms, `none` gives 3 arms.

| | part d (`/root/a96pd`) | part c (`/root/a96pc`) |
|---|---|---|
| searched arm | A82_BANK = S3b bank 3609634d (file `/root/a82p20/root/a82_banks/A82_BANK_D2800_p20.json`) | BANK_SEARCHED_P20 = S4-p20 rows + S5-p20 rows, unchanged, dedupe identical keys |
| panel | 988200-988391, byte copy of the staged A82 192 panel. File 1648eb4b, panel **3c879f6f...6aa6485**. Never the 384 file. | 984400-984591 (registered). File **0bb84f3c...45c56**, panel **6c161a56e1c913e35441570774f1df6ee62999a0ca9e11b27a309ab3b0dfc5ff** |
| other arms | BANK_HAND_P20, DUO_HOLD, MAPPO20C | same |
| PRIMARY | searched − BANK_HAND_P20. SUPPORTED iff CI95 upper < 0 AND the five-group rule vs BANK_HAND_P20. `direction` is SEARCHED_BETTER, NOT_SEPARATED or REFERENCE_BETTER. | same |
| fallback (hand = none) | 3 arms; primary = searched − DUO_HOLD | same |
| SECONDARY | each bank − DUO_HOLD, each bank − MAPPO20C | same |
| descriptive | DUO_HOLD − MAPPO20C; CVaR90 / safety; per bank arm: activations and offers (fire / member_busy) per plan, overlap block, CAVs engaged at once | same |

## 2. Mechanics: what each arm gets (registered cap 14,400; 1,200 sim-s per episode)

**A82 identity** (8 episodes, PASS required). Each record must equal its stored record:
- U0, on 997300/997301, against A76C_COMP.
- A75_BANK with the A76 bank, on 997300/997301, against A76C_BANK. This proves the executor path both bank arms use.
- The MAPPO20C path (u0 bound as IDENT_U0 through a77 bind + with_executor), on 997300/997301, against A76C_COMP. This
  proves the path, not the MAPPO20C file.
- DUO_HOLD, on 993100/993101, against the A32 chunk 007cef74 (no stored report, so the report check is not gating).

**Rule checks only** (no stored records exist): each bank arm that runs gets 2 episodes on 997300/997301. Checks:
- `a80_s2b.plan_checks` per plan;
- U0 executor and a75 d1af7e0f;
- zone ±1 m per plan;
- no CAV in two sessions, no refusal;
- 0 collisions and 0 teleports.

| layout | mech | confirm | total cap |
|---|---|---|---|
| 4 arms | 12 episodes, 14,400 | 4 × 192 × 1,200 = 921,600 | **936,000** |
| 3 arms | 10 episodes, 12,000 | 691,200 | 705,600 |

The cap is enforced in code. Each stage launches once on 46 workers. A failure writes `CHAIN_STOPPED`.

**Expected time** (A82 estimate: 192 p20 episodes in about 12 min on Box 3 alone; A91: 480 episodes in 40 min): mech
about 8-10 min, confirm about 45-50 min, analysis under 1 min. **About 60 min per panel** (about 45 with 3 arms).

## 3. Tests run (zero SUMO, charged 0)

- `run_a96.sh test`: part d **13 passed + 1 skip** (the part-c runner check); part c **14 passed**.
  - Covered: layouts of both parts × both arm sets (fresh interpreters), env refusal, block / patch, identity rows, the
    primary direction, overlap / engaged, the summary.
  - Builder stand-ins: synthetic p20 roots made from the 60 % S4/S5 admissions, test only. Cases: two sources merged; one
    empty → the other alone, byte-equal; all empty → exit 8 NO_SEARCHED; hand one root; hand none → NO_HAND; a wrong sha
    stops.
- `run_a96.sh stage`: STAGE_OK for both parts (panel into the root, IDENT_U0 bound, charged 0).
- `run_a96.sh drytest`, for each part, in `/dev/shm` roots, for the 4-arm and 3-arm layouts. Stand-ins: searched = the
  S3b bank, hand = the A76 bank, MAPPO20C = the IDENT_U0 binding. Steps:
  - prepare, dry i (identity + rule-check waves, stored references), dry c (waves 192 each);
  - mech rule-check e2e on the A91 mechanics records;
  - analysis e2e on the A91 confirm layout. Only structure was printed.
  - Results: `box_outputs/drytest_{d,c}.out`.

## 4. Exact command sequence

Prefix every command with
`timeout 120 ssh -o BatchMode=yes -i ~/.ssh/rented_box ubuntu@124.220.72.157 '<cmd>' < /dev/null`.
`<p>` is `d` or `c`, and `<P>` is `D` or `C`.

0. **MAPPO20C files.** Put `policy_00084.pt`, `PLAN.json` and `BOUND_00084.json` in `/root/a82four/inputs/`. Record the
   policy sha. The bind checks that BOUND_00084 is the training receipt of PLAN 57f51364.
1. **Hand bank**, once per base. Both bases build identical bytes from the same source:
   ```
   sudo -n env A96_HAND_ROOTS="/root/a96adm_hand/root=<sha256 of its a80_banks/A80_BANK_D2800_p20.json, or none>" bash /root/a96p<p>/run_a96.sh bank-hand
   ```
   - Read the result in `/root/a96p<p>/chain_bank-hand.out`: `sha256 A96<P>_BANK_HAND_D2800_p20.json <H>` or `... none`.
   - The builder expects the A80 layout: `ROOT/A80_ADMISSION_D2800_p20.json` and `ROOT/a80_banks/A80_BANK_D2800_p20.json`.
   - The hand file is then `/root/a96p<p>/src/ops/a96/A96<P>_BANK_HAND_D2800_p20.json`.
2. **Searched bank** (part c only):
   ```
   sudo -n env A96_SEARCHED_ROOTS="/root/a96adm_s4/root=<sha|none> /root/a96adm_s5/root=<sha|none>" bash /root/a96pc/run_a96.sh bank-searched
   ```
   - Exit 8 (NO_SEARCHED) means the panel is not run.
   - The file is `/root/a96pc/src/ops/a96/A96C_BANK_SEARCHED_D2800_p20.json`. Its sha `<S>` is in `chain_bank-searched.out`.
3. **Register** the shas (searched, hand or none, MAPPO20C). The panel shas are 3c879f6f / 6c161a56. Then **pin**
   (zero SUMO):
   ```
   sudo -n env A96_BANK_FILE=<part d: /root/a82p20/root/a82_banks/A82_BANK_D2800_p20.json | part c: /root/a96pc/src/ops/a96/A96C_BANK_SEARCHED_D2800_p20.json> \
     A96_BANK_SHA256=<d: 3609634dace1d68778ec89392bc125f41ebd05b409a4e5777e6e80932283feb1 | c: S> \
     A96_HAND_FILE=/root/a96p<p>/src/ops/a96/A96<P>_BANK_HAND_D2800_p20.json A96_HAND_SHA256=<H|none> \
     A96_MAPPO20C_POLICY=/root/a82four/inputs/policy_00084.pt A96_MAPPO20C_SHA256=<registered> \
     A96_TRAIN_RECEIPT=/root/a82four/inputs/BOUND_00084.json A96_TRAIN_PLAN=/root/a82four/inputs/PLAN.json \
     bash /root/a96p<p>/run_a96.sh pin
   ```
   - When hand = none, omit `A96_HAND_FILE`.
   - Check for `PIN_OK` in `chain_pin.out`, and check `/root/a96p<p>/PINNED.env`.
4. **Launch, once:**
   ```
   sudo -n bash -c "setsid -f nohup bash /root/a96p<p>_queue.sh < /dev/null > /root/a96p<p>_queue.nohup 2>&1"
   ```
   - The queue waits for nothing. It refuses unless pinned, with the registered panel.
   - It runs mech, then confirm with analysis.
   - The log `/root/a96p<p>_queue.log` ends with `A96PD_DONE` / `A96PC_DONE`.
   - To stop between rounds: `touch /root/a96p<p>/root/STOP_A96P<P>`.

## 5. Receipts

| when | what |
|---|---|
| now | `/root/a96p<p>/{chain_stage.out, stage.out, SETUP_OK, PART, env_a96.sh}`, `/tmp/a96_setup_<p>.out`, `/tmp/a96_drytest_<p>.out`, `/tmp/a96c_seed_audit_984400.json` (laptop copies in `box_outputs/`, `audit/`, `pinned/`) |
| after the run | `chain_bank-*.out`, `bank_*.out`, `chain_pin.out`, `PINNED.env`, `pin.out`, `dry.out`, `root/A96<P>_MECH.json`, `root/A96<P>_RESULTS_D2800_p20.json`, `analyse.out`, `chain_{mech,confirm}.out`, `root/A96<P>_LOG.jsonl`, `root/CHARGES.jsonl` |

## 6. Estimate only (not built): splitting part (c) across Box 3 and Box 4, 96 worlds each

1. **Code.** Add a world-range parameter to `confirm_waves` (`A96_WORLD_SLICE=0-95|96-191`) and an `import-half` stage.
   - With chunk 96, each wave already publishes `__0000-0095` / `__0096-0191`.
   - `import-half` copies Box 4's `WAVE_A96CC_*__0096-0191.json`, `RESERVED_*`, `evaluation/<job>/vehicles_report.json`
     and CHARGES lines into the Box 3 root. It verifies seals, shas and job ids, then runs the unchanged `analyse`.
   - About 40 lines and 1 h, including tests.
2. **Box 4.** It needs the same tree and root.
   - That means `/root/a96pc` plus `/root/a76p20/root` (for identity) and the panel / banks / MAPPO20C files.
   - The hosts were hash-identical on 11-13. This has to be re-verified.
3. **Mechanics.** Run them on both boxes (+14,400). That exceeds the 936,000 cap, so it needs a registration line. The
   alternative is identity-only on Box 4 using the 8 A82 identity episodes.
4. **Gain.** About 25 min of wall time. Risk: cross-host nondeterminism must be zero (the identity check covers it).

## 7. Doubts and disclosures

1. **The S3b bank under the rule checks.** It (part d) was admitted by the A76-code admission vs U0 on the reused A76
   worlds, as registered.
2. **The searched merge order is S4 then S5**, following the registered text. Only the first-kept duplicate depends on
   it. Executor priority comes from the carried lb95 values.
3. **Rule-check worlds.** The bank-arm rule checks run on the A76 confirmation worlds 997300/997301, the same ones the
   A82 identity uses. They are not panel worlds.
4. **The hand-root layout is assumed.** It is the A80 code's own layout: `A80_ADMISSION_D2800_p20.json`, `a80_banks/`.
   If the other agent's root differs, step 1 stops; nothing is written, and it can be re-run.
5. **Part d block row.** The panel reuses the A82 block row 988200-988599, of which 988200-988391 is used. A89's
   988400-988499 lies inside that row but is not used by this panel. This is the same as in A82.
6. **Files outside the bases:** `/root/a96_ship`, `/tmp/a96_*`, `/root/a96p{d,c}_queue.sh`.
