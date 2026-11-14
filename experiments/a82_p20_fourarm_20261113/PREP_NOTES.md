# A82 at 20 % (D2800_p20): S3b admission + four-arm confirmation, prepared and NOT launched

Prepared on 2026-11-13, 20:34 to 21:18 local (laptop `date`, UTC+8). Zero SUMO seconds, zero training, zero WM rollouts. No
`CHARGES.jsonl` exists in any of the 4 new roots, and `evaluation/` is empty in each. Nothing has been registered,
committed or pushed. REGISTRATION.md, ledgers and shared docs were not edited.

## 1. Design

**Part A, "A82-S3b admission" (Box 3 `/root/a82p20`).** This is the 20 % version of A80.
- Candidates: the S3b `TOP_OFFER_CONDITIONS_D2800_p20.json` (1 to 12 rows). It is pinned by file and sha and mapped
  losslessly with A80's unchanged MAPPING_RULE (`a80_s2b.map_rows`), using cell D2800_p20.
- Worlds: the same 24 admission worlds as A76 (997200-997223). `A76_PANEL_V.json` is byte-copied into the root. Its panel
  sha `02e20925b23cb1ad8d0c91d2d536a6750d9a1e93960e937180e46b50e25879d4` is checked against the sealed `A76_PINS.json`.
- Reference: the stored A76 comparator wave `A76V_COMP` (U0 d9042cc5 on those 24 worlds), read-only from
  `/root/a76p20/root` and NOT re-run.
  - Its wave receipt `4c11584f1dc2ee30ebfb137a5d31876986f64e936bc0965336b93d4412a50561` and its 24 DROPPED_RAW-sealed
    per-vehicle reports are pinned at `pin`.
  - A reference episode whose raw log is still present makes the run refuse, so nothing is ever cached into the A76 root.
- Admission and bank: `a76_d2p20.admit` itself, which is A76's ADMIT_RULE and BANK_RULE with `a75_d2forms.bank_choice`
  / `bank_plans`. It is imported unchanged, and its module names are re-pointed to the A82 root (the same pattern A80
  used for A75-D2).
  - Form families follow A80's FORM_RULE: 7 families including "other".
  - Output: `A82_ADMISSION_D2800_p20.json` plus `a82_banks/A82_BANK_D2800_p20.json` (no bank if nothing is admitted).
- Mechanics: the first 2 verification episodes (997200/997201) of EVERY candidate. They are checked with A80's
  `plan_checks` plus U0 executor, a75 d1af7e0f, zone ±1 m, and 0 collisions/teleports. Tokens that never fire count as
  NOT EXERCISED. These episodes are the same job ids as the verification episodes, so they cost nothing extra.
- Equivalence proof (zero SUMO, `a82_replay_a76.py`): I fed A76's own 36 plan waves plus the stored reference through
  this exact re-pointed path into a throwaway `/dev/shm` root. The result matches the sealed A76 admission receipt:
  **ALL_EQUAL**. That means 36/36 plans equal on ci95 / PASS / group rule / duplicate / IN_BANK / per_seed, the same
  chosen bank, and bank file sha = f8b5237e. Only booleans were printed; no outcome number was shown.

**Part B, "A82 four-arm confirm" (Box 3 `/root/a82four`).** One fresh panel, four arms on the same worlds, all run in
this root:

| arm | definition (frozen, reused code) |
|---|---|
| A82_BANK | runner arm A75_BANK on a frozen bank file (parameter), on U0 d9042cc5. Specs are built exactly as `a76_d2p20.confirm_waves` builds A76C_BANK. |
| DUO_HOLD | hand-written two-CAV hold, see section 2 |
| MAPPO20C | A77 iteration-84 export. Bound by `a77_mappo20c.bind` (unchanged: file sha = registered sha = BOUND_00084 receipt of PLAN 57f51364, speed-only), then `A52.with_executor` on the U0 specs (the A77 evaluation path). |
| U0 | frozen speed-only transferred MAPPO d9042cc5 (`a76_d2p20.u0_specs`) |

- Contrasts: diff = treatment − reference, negative = improvement. All use the A42 driver's `_contrast` (the code
  A76/A77/A81 used): paired percentile bootstrap 10,000, seed 20261104; total delay, five groups vs 10 s, CVaR90
  (most-delayed 10 %), hard brakes, TTC events, collisions, teleports.
  - **PRIMARY:** A82_BANK − DUO_HOLD. SUPPORTED iff CI95 upper < 0 AND the five-group 10 s rule holds vs DUO_HOLD.
  - **Secondary:** A82_BANK − MAPPO20C, DUO_HOLD − MAPPO20C, MAPPO20C − U0.
  - **Descriptive:** A82_BANK − U0, DUO_HOLD − U0.
- Mechanics / identity (8 episodes, 9,600 sim-s, before any panel episode). Each arm type must reproduce a stored record
  on 2 worlds:
  - U0, A75_BANK (with the A76 bank) and the MAPPO20C path (u0 d9042cc5 bound as IDENT_U0 through a77 bind +
    `with_executor`) on 997300/997301, compared with A76C_COMP / A76C_BANK (receipts 4c6eb3d5… / 4da36413…).
  - DUO_HOLD on 993100/993101 (the established A35/A50/A77 identity worlds), compared with the A32 chunk receipt
    `WAVE_DUO_HOLD__0000-0047.json` (receipt b991b0bd…, file sha 007cef74…). Its a32 digest 1c51f22c and executor
    d9042cc5 equal this tree's.
  - Rule: J exactly equal, safety equal, same world, executor = d9042cc5, receipts equal, and the report equal for the
    A76 references. A32 has no stored report, so for DUO the report check is not gating.
  - Dry run: spec differences vs the stored drivers' own specs are only paths, labels and job ids (U0: binding_path; bank:
    a75_bank.path + binding_path; IDENT_U0: binding fields + label; DUO: binding_path).
- Raw logs are dropped per round under the A75-D2 DROP_RULE (prefix a82fc_). There is no automatic replay. One launch per
  stage.

## 2. The hand-written arm (exact definition and source)

**DUO_HOLD = `a32_ablation.a32_specs('DUO_HOLD', panel, root)`.** It is the A32 executor arm "DUO_HOLD" (a32_arms digest
1c51f22c…): duo bank `a32_banks/GUIDANCE_C4SIT_priv_UNION_UNC.json`, sha 39f384a2…; rule row duo12_v20, started
whenever feasible; on the frozen U0 executor d9042cc5. It is not edited.

Sources:
- REGISTRATION l.10570 (A31): "DUO_HOLD: hand-written two-CAV hold duo12_v20 (the rule row of UNION_UNC 39f384a2),
  started whenever feasible".
- Used by A32/A34 on the p20c panel and registered for the A73 20 % confirmation ("DUO_HOLD duo12_v20/39f384a2").
- Code path: `ops/a73_lcside20.py::ref_specs('DUO_HOLD')`.

It is **not H20**. H20 is the A72-E2 uniform-hold arm chosen in e2tune at D2800_p60 (60 %), not a 20 % arm. At 60 %
the same rule bound to MAPPO60C is DUO_HOLD_60C.

## 3. Fresh world block: 988200-988599 (400 seeds; use the first 96, 192 or 384)

Evidence that it is unused (`ship/SEED_AUDIT_A82.json`, sha f52f65ed…):
- REGISTRATION.md + experiment_history.json: every token and range parsed; nothing in the block except
  decimal/hash fragments. Neighbours: 988000-988071 is an older TRAIN block; 987200-987599 is A77 TRAIN.
- git grep over all 911 branch-tip trees (py/sh/md/yaml/toml): 0 hits.
- Box 3 full scan (307 panel files, 29 runner ranges, code/docs; evaluation dirs pruned): 0 overlaps. The only token
  hits are inside hex hashes of CPython's `svnmap.txt`.
- Box 4 light check only (runner/adapter/config files): 0 hits.
- **NOT DONE:** the broad grep of the laptop working tree was denied by the permission classifier. I stopped there and
  did not work around it.

Avoided: 964000-964199, 962xxx, 977xxx, 989600-989699, all training and evaluation blocks. The runner gets one additive
table row (A82_RANGES), and `block_free` is code-enforced.

Staged panels (built zero-SUMO, write-once, `/root/a82four/panels/`):

| N | seeds | file sha256 | panel_sha256 |
|---|---|---|---|
| 96 | 988200-988295 | a4b8e9a1076f3dbbfeec477255715314d670dd7c04d0f7bec603ef789c78c76a | 5a59adccb04cf145fa98ab41724db9126f64f524f0eed610c57eaf889a5c0ee7 |
| 192 | 988200-988391 | 1648eb4b6d578a0256896120f49531d99edc4badf1e9e87e21762eb432462a89 | 3c879f6ff47fa75ebbf5f5055f4ed35ad7e54be96a704dffa743287b86aa6485 |
| 384 | 988200-988583 | 7841a6b2ee7935b3ec3809b411e9e08f4dc02b27275827b40b10462b6f027b38 | d1012d149e03ddbe3e7e1c0298f65bc1bff19d03b10995f563c69de4f2e95fa2 |

## 4. Cost (1,200 sim-s per episode)

| | per arm | 4 arms | + mech 9,600 | wall estimate on 46 workers* |
|---|---|---|---|---|
| N=96 | 115,200 | 460,800 | **470,400** | ~25 min |
| N=192 | 230,400 | 921,600 | **931,200** | ~50 min |
| N=384 | 460,800 | 1,843,200 | **1,852,800** | ~1 h 40 min |

Part A: at most 12 × 24 × 1,200 = **345,600** (the 24 mechanics episodes are inside this; the actual cost is
n_candidates × 28,800). Wall time is about 20 to 25 min plus about 3 min for mechanics.

\* The wall estimate is extrapolated from the A76 confirm (192 episodes in about 12 min alone on Box 3). It is slower if
the box is shared.

Recommendation: N=192. 384 matches the 60 % A81 panel if the owner prefers the same power.

## 5. Bundle and installed hashes

- `a82_ship.tgz` = `4ca7d7ac875fc80fbfb4ee1f9f7c68745ef876140e688fd7e1dc37428cee38d8`. Box 3 copy is in `/root/a82_ship`.
  Per-file hashes are in `ship/SHIP_FILES.sha256`.
- Key shipped files:
  - `a82_s3b20.py` 2b33146b…
  - `a82_fourarm.py` 5181bf4b…
  - `a82_replay_a76.py` 5c322732…
  - `run_a82.sh` 6f6ce9eb…
  - `setup_a82.sh` b332e6b1…
  - `patch_tables_a82.py` 70dbfec2…
  - tests 479aaf17… / fe135858…
  - `a80_s2b.py` 48c5c448… (= registered A80 driver)
  - `a77_mappo20c.py` 308894c6… (= A77 eval ship)
  - SYNTH top 6c7edebf… (= A80 synthetic rows)
- Trees: real copies (`cp -a`, no hard links) of `/root/a76p20/src`; all 1,491 files were verified identical first.
  - Runner, part A: A76 runner **79fc9efd**, unchanged.
  - Runner, part B: 79fc9efd + A82_RANGES = **2de0f1c0…**, a one-line additive diff.
  - Root bindings are copied, and `cmp`-equal to both the p20c bank root and the A76 root.
- After `SETUP_OK` I replaced only my own A82 driver/test/run files several times while fixing dry-run bugs. Every
  replacement is logged with old → new sha in `<base>/UPDATES.log`.
- Zero-SUMO artefacts already written:
  - `/root/a82four/root/BOUND_A77_IDENT_U0.json` (cd77d1e2…, binding aaeb93a6…)
  - `/root/a82four/root/a82f_banks/A82F_IDENT_A76BANK_D2800_p20.json` (f8b5237e…)
  - the 3 staged panels
  - the synth roots under each base

## 6. Tests and dry runs (Box 3, nice 19)

- Unit tests:
  - Part A: 16 passed, 2 skipped. The skips are the part-B host checks.
  - Part B, with `A82F_TEST_ROOT`: 18 passed, including the host check that U0 vs the MAPPO20C path differ only in the six
    executor fields.
- Part A synthdry: the synthetic 12-row top file was pinned (gap 4, pulse 2, rolling 3, taper 2, zone 1). Then pins,
  prepare, and dry: 12 waves × 24, 345,600 sim-s, mech 28,800. The reference loaded (24/24, reports OK). **charged 0.**
- Part B synthdry: N=192 panel, IDENT_U0 bind, prepare (A76 bank as placeholder), dry i (8 episodes, 9,600; stored refs
  2+2+2+2 loaded; spec diff above), dry c (4 × 192, 921,600). **charged 0.** MAPPO20C was substituted by the IDENT_U0
  binding in the synth dry only; this is flagged and refused for anything else.
- Replay-equivalence of part A admission: ALL_EQUAL (section 1).
- `/root/a76p20` was verified untouched: 0 files newer than my setup. I started no processes that are still running;
  the /dev/shm temp dirs were removed.

## 7. Launch commands (fill the `<...>` values)

Prefix for every Box 3 command:
`timeout 60 ssh -o BatchMode=yes -i ~/.ssh/rented_box ubuntu@124.220.72.157 '<cmd>' < /dev/null`.
Copy files box to box via the laptop, one remote file per scp.

**A1. S3b top file (after S3b exits rc 0).**
- Copy A800 → laptop: `runs/active/C4-V7FT-20261017/a75p20_s3b/TOP_OFFER_CONDITIONS_D2800_p20.json`.
- Compare `sha256sum` on both sides.
- Copy laptop → Box 3 `/tmp/`, then:
  `sudo -n bash -c "mkdir -p /root/a82p20/inputs && cp /tmp/TOP_OFFER_CONDITIONS_D2800_p20.json /root/a82p20/inputs/ && sha256sum /root/a82p20/inputs/TOP_OFFER_CONDITIONS_D2800_p20.json"`

**A2. Pin (zero SUMO; register first):**
`sudo -n bash -c "A82_TOP_FILE=/root/a82p20/inputs/TOP_OFFER_CONDITIONS_D2800_p20.json A82_TOP_SHA256=<TOP sha> bash /root/a82p20/run_a82.sh pin"`
- Expect `PIN_OK`.
- A mapping refusal writes `A82_MAPPING_REFUSED.json`, and nothing runs.

**A3. Mechanics** (Box 3 must be free of A80):
`sudo -n bash -c "A82_WORKERS=46 setsid -f nohup bash /root/a82p20/run_a82.sh mech < /dev/null > /root/a82p20/nohup_mech.out 2>&1"`
Wait for `A82_MECH_DONE` in `chain_mech.out`.

**A4. Admission runs:**
`... A82_WORKERS=46 setsid -f nohup bash /root/a82p20/run_a82.sh admit ...`
Wait for `A82_VERIFY_DONE`.

**A5. Bank (zero physics):** `sudo -n bash -c "A82_WORKERS=1 bash /root/a82p20/run_a82.sh bank"`

**B0. Decide the bank by the pre-registered rule.** No outcome is needed, only whether a bank file exists:
- If `/root/a82p20/root/a82_banks/A82_BANK_D2800_p20.json` exists, use `A82_BANK_FILE=` that file,
  `A82_BANK_SHA256=` its sha256 (must equal `bank.file_sha256` in the A82 admission receipt), and `A82_BANK_SOURCE=A82_S3B`.
- Otherwise use `A82_BANK_FILE=/root/a76p20/root/a76_banks/A76_BANK_D2800_p20.json`,
  `A82_BANK_SHA256=f8b5237eacfe2fecc04ab0c3d7f08432b5eda5e1825805316a98b05c94f22d80`, and `A82_BANK_SOURCE=A76`.

**B1. MAPPO20C files (after TRAIN_DONE on Box 4):**
- Copy Box 4 → laptop, one file per call:
  - `/home/ubuntu/a77/run84/production/policy_00084.pt`
  - `/home/ubuntu/a77/run84/PLAN.json` (sha must be 57f51364…)
  - `BOUND_00084.json` from `launch.py bind --run-dir run84`. I expect it at `/home/ubuntu/a77/run84/BOUND_00084.json`;
    run `ls` there first.
- Copy laptop → Box 3 `/tmp`, then `sudo cp` into `/root/a82four/inputs/`.
- MAIN records the MAPPO20C sha in REGISTRATION (A77 rule).

**B2. Pin (zero SUMO):**
`sudo -n bash -c "A82_N_WORLDS=<96|192|384> A82F_PANEL_SHA256=<panel_sha256 from section 3> A82_BANK_FILE=<..> A82_BANK_SHA256=<..> A82_BANK_SOURCE=<A82_S3B|A76> A82_MAPPO20C_POLICY=/root/a82four/inputs/policy_00084.pt A82_MAPPO20C_SHA256=<registered sha> A82_TRAIN_RECEIPT=/root/a82four/inputs/BOUND_00084.json A82_TRAIN_PLAN=/root/a82four/inputs/PLAN.json bash /root/a82four/run_a82.sh pin"`
Expect `PIN_OK`. The dry i/c outputs are in `/root/a82four/dry.out`.

**B3. Mechanics (9,600):**
`sudo -n bash -c "A82_WORKERS=8 setsid -f nohup bash /root/a82four/run_a82.sh mech < /dev/null > /root/a82four/nohup_mech.out 2>&1"`
Wait for `A82F_MECH_DONE`. FAIL means STOP.

**B4. Confirm + analyse:**
`sudo -n bash -c "A82_WORKERS=46 A82_BANK_SHA256=<same> A82F_PANEL_SHA256=<same> setsid -f nohup bash /root/a82four/run_a82.sh confirm < /dev/null > /root/a82four/nohup_confirm.out 2>&1"`
- Wait for `A82F_CONFIRM_DONE`.
- Results go to `/root/a82four/root/A82F_RESULTS_D2800_p20.json`.

To stop between rounds: `touch /root/a82p20/root/STOP_A82` or `/root/a82four/root/STOP_A82F`.

## 8. Caps to register

- **holdforms_a82s3b20:** verification 345,600 (mechanics inside).
- **fourarm_a82p20:** mech 9,600 + confirm 4 × N × 1,200. That is 460,800 / 921,600 / 1,843,200, for totals of
  470,400 / 931,200 / 1,852,800. The code takes N from the pinned panel.
- **Both parts with N=192:** at most 1,276,800 sim-s.

## 9. Draft registration paragraph

> **AMENDMENT 82 - A82 (20 % S3b admission + four-arm confirmation at D2800_p20; POST-HOC, exploratory) - registered
> <time> before any A82 SUMO second.**
>
> **Question.** At 20 % CAVs, does a searched hold-form bank beat the hand-written two-CAV hold on fresh worlds? How do
> both compare with MAPPO trained at 20 % (MAPPO20C) and the transferred U0? Closest prior: A76 (A76 bank − U0 −1,993.4
> [−3,908.8, −102.6], 96 worlds; no hand-written arm) and A81 (60 %, bank − DUO_HOLD_60C SUPPORTED on 384 worlds).
> Changed variable: CAV share 60 → 20 % for the A81 design, plus the MAPPO20C arm.
>
> **Part A (Box 3 /root/a82p20, phase holdforms_a82s3b20, cap 345,600).**
> - The A79 S3b TOP_OFFER_CONDITIONS_D2800_p20.json (sha <TOP sha>) is pinned and mapped by the unchanged A80
>   MAPPING_RULE.
> - Each plan is run once on the 24 A76 admission worlds 997200-997223 (panel 02e20925). The reference is the stored
>   A76V_COMP wave (receipt 4c11584f…, 24 reports pinned; not re-run).
> - Mechanics: first 2 verification episodes of every plan (receipt invariants, zone, U0 executor, 0 collisions /
>   teleports); PASS required.
> - Admission and bank: a76_d2p20.admit unchanged (A76 ADMIT_RULE: CI95 upper < 0 AND five-group 10 s; BANK_RULE), with
>   A80 FORM_RULE families. Re-pointing verified zero-SUMO: it reproduces the A76 admission receipt exactly (36/36).
> - Driver a82_s3b20.py 2b33146b; ship a82_ship.tgz 4ca7d7ac.
>
> **Part B (Box 3 /root/a82four, phase fourarm_a82p20).** One fresh panel 988200-<last> (N = <N>, panel sha <..>; block
> 988200-988599 audited UNUSED, SEED_AUDIT_A82 f52f65ed). Arms on identical worlds:
> - A82_BANK = A75_BANK on <bank: A82 part-A bank if it admits anything, else the A76 bank f8b5237e; fixed before
>   launch: <file sha>>, on U0;
> - DUO_HOLD = a32 DUO_HOLD duo12_v20 / UNION_UNC 39f384a2 on U0 (the registered 20 % hand-written arm);
> - MAPPO20C = A77 policy_00084 sha <..> (a77 bind, BOUND_00084 of PLAN 57f51364) via A52.with_executor;
> - U0 = d9042cc5.
>
> **Identity (9,600 sim-s, PASS required).** U0 / A75_BANK (A76 bank) / the MAPPO20C path (u0 bound as IDENT_U0)
> reproduce the stored A76C records on 997300/997301, and DUO_HOLD reproduces the A32 records on 993100/993101 exactly.
>
> **Confirm.** 4 × <N> × 1,200 = <..> (cap <..> incl. identity).
>
> **Analysis.** PRIMARY A82_BANK − DUO_HOLD: SUPPORTED iff CI95 upper < 0 AND the five-group 10 s rule. Secondaries:
> A82_BANK − MAPPO20C, DUO_HOLD − MAPPO20C, MAPPO20C − U0. Descriptive: bank − U0, DUO − U0. Paired percentile bootstrap
> 10,000, seed 20261104; diff = treatment − reference, negative = less delay; CVaR90, hard brakes, TTC, collisions,
> teleports; never pooled. Driver a82_fourarm.py 5181bf4b; runner 2de0f1c0 (A76 runner + A82_RANGES).
>
> **Disclosures.** Bank/admission selection-biased; coordination arms run on U0, not on MAPPO20C; A76 admission worlds
> reused for part A; the MAPPO20C arm runs in the A76 (A69-based) tree; the A77 registered evaluation (997600-997695) is
> separate. Each stage launched once, no retry.

## 10. Doubts and deviations (please read)

1. **"H20" vs DUO_HOLD.** The brief named H20 or DUO_HOLD. H20 is a 60 % E2 arm. The registered 20 % hand-written arm is
   DUO_HOLD duo12_v20, and that is what I used.
2. **Base policy of the coordination arms.** As in every registered 20 % arm, the bank and DUO_HOLD sit on U0, not on
   MAPPO20C. So bank − MAPPO20C and DUO − MAPPO20C mix "coordination" with "base policy". Coordination on top of
   MAPPO20C would be a new, unfrozen arm, and I did not build it.
3. **Part A reference is not re-run.** It uses the stored A76 U0 wave, which saves 28,800 sim-s. Equivalence is proven by
   replay, but the reference runs were produced by A76's process, not this root.
4. **Part A mechanics differ from A80.** All candidates are checked, not only those with uncovered tokens. It costs
   nothing extra. PASS does not require any token to be exercised.
5. **Form families.** I used A80's 7-family FORM_RULE instead of A76's 6, because WM plans can combine forms.
6. **Cosmetic labels from reused code.** Part A verdict rows say "A76 (new run, D2800_p20)" with schema
   `coopmwm.a76.admission.v1`, and the setting labels say "amendment 76". MAPPO20C binding files are named
   `BOUND_A77_*.json` and carry A77 authority text. All of this comes from the unchanged A76 and A77 code. It is disclosed
   in `A82_ADMISSION_EXTRA.json`.
7. **MAPPO20C identity covers the path, not the file.** No stored MAPPO20C record exists, so identity proves the
   `with_executor` binding path with the u0 file. MAPPO20C also runs in the A76 tree, not the A66 tree A77e would use.
8. **DUO_HOLD identity reference.** It is the A32 chunk receipt (laptop copy, gpuB run) on the p20c worlds
   993100/993101. These worlds are the long-standing identity worlds, but they belong to that earlier p20c
   confirmation panel. I verified only hash fields (a32 digest, executor, seeds) and opened no outcomes. There is no
   stored report, so the DUO report check is not gating.
9. **Seed audit gap.** The laptop working-tree grep was denied by the classifier and not retried. Box 4 got only a light
   check.
10. **A77's registered evaluation.** It (a77e on 997600-997695 with U0, MAPPO_CT, MAPPO20C; cap 480,000) is still
    separate and not set up. A82 does not replace it.
11. **Run order and capacity.** Part A must wait for A80 to finish on Box 3; part B confirm must wait for MAPPO20C (about
    03:10) and the part-A bank decision. Box 3 free disk is 31 GB. Raw logs are dropped per round, so the peak is a few
    GB.
12. **Files replaced after SETUP_OK.** I replaced my own files after `SETUP_OK` (logged in each base UPDATES.log: a82p20 and a82four; final shas above).
    A clean re-setup from the final ship is possible if preferred.
13. **Leftover synth roots.** The `synth/` roots in both bases are throwaway zero-SUMO fixtures. Leaving them causes no
    problem.

## Files
- `ship/`: shipped sources.
- `a82_ship.tgz`: the bundle.
- `tools/`: `free_blocks.py`, `seed_audit_box.py`, `update_a82_files.sh`.
- `audit/`: box audit JSON and git tree list.
- `box_outputs/`: setup, test, dry, replay and UPDATES logs from Box 3.
