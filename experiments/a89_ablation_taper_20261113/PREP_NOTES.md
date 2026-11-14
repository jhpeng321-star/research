# A89: six-arm fresh panel at 60 % (D2800_p60), prepared and NOT launched

Prepared 2026-11-13, 22:51 to 23:06 local. Zero SUMO and zero training. The `pin` stage (zero SUMO: panel, bank copies,
dry runs) was run once and charged 0. `/root/a89/root` has no `CHARGES.jsonl` and no `evaluation/`. REGISTRATION.md
and the ledgers were not edited. Nothing was committed.

## 1. Design

- **Driver `ops/a89_sixarm.py`** (sha 758c8daa). It is a thin wrapper that imports the registered A81 driver
  `a81_bigconfirm.py` (c755a551) unchanged. `point()` re-points A81's module names (amendment, phase, caps, block,
  panel / log / mech / results file names, prefixes `a89i_` / `a89c_`, `STOP_A89`) at A89 values, the same way A80
  re-points A75-D2. Only these functions are A89's own:
  - `core`, `register_c_setting` and `_runner_check`: labels, block 988400-988499, `A89_HOST_BASE`.
  - `prepare`: byte copies of two pinned banks.
  - `waves`, `mech_compare`, `analyse`, `dry`.
  Panel-c, mech/confirm execution (`D2.execute`), caps, OWNER.lock, drop-raw and status all run through A81's own
  `main`.
- **Tree `/root/a89/src`:** a real copy (`cp -a`) of `/root/a81big/src`, verified identical before patching.
  - Changed: the runner 538bd0bf gets `A89_RANGES` added (one-line additive tail plus the new constant) and becomes
    **982ec0cf**.
  - New: `a89_sixarm.py`, its test, `a80_s2b.py` (byte copy 48c5c448, used only for `plan_checks`; its `_point_d2` is
    never called), and `ops/a89/`.
- **Arms, all on the MAPPO60C executor 2abed317, setting a70c, plant 5ea22964:**

| arm | runner arm | definition |
|---|---|---|
| ROLL_BOTH | A75_PLAN | family d2_d86b1d6ac653e8ad: template and row copied unchanged from the A75-D2 bank c325ec1d (canonical-JSON equal, same serializer and depth). Roles: L0 rank 1, 24 m/s, 30 s, onset 0; L2 rank 0, 16 m/s, 20 s, onset 20; lanes, far, spread 200, cooldown 30 |
| ROLL_L0_ONLY | A75_PLAN | a89l0_d86b1d6ac653e8ad (key 928dfa5951710cb2): the same template with only the L2 role deleted |
| ROLL_L2_ONLY | A75_PLAN | a89l2_d86b1d6ac653e8ad (key 0bbb70f7956dae14): only the L0 role deleted, onset 20 kept |
| TAPER | A75_PLAN | mut_1fca373046a3c66e: template and row copied unchanged from the E1R candidate bank `/root/a80e1r/root/a80_banks/A80_CANDS_D2800_p60.json` (057c1f06), the bank the E1R A75_PLAN episodes used. 4 roles, extended (HoldFormCore) |
| DUO_HOLD_60C | DUO_HOLD | the A81 arm unchanged |
| MAPPO60C | U0 | the A81 arm unchanged |

- **Arms bank** `A89_ARMS_D2800_p60.json` = **de9a30a944eda470777cbd3247c6ca81db2745547549cde3e2bff154e23824c2**.
  - It was built deterministically by `build_a89_arms.py` (99cce7e5).
  - Its proofs are all true: ROLL_BOTH and TAPER template/row equality; the single-role arms changed nothing except the
    roles list; the kept role is identical.
  - The executor accepts every family with one plan. Copies: `pinned/` and `/root/a89/src/ops/a89/`.
- **Dry spec diff:** each plan arm differs from ROLL_BOTH only in `checkpoint.a75_family` and `job_id`.

## 2. Panel and seed audit

- **Panel:** `A89_PANEL_C.json` with seeds 988400-988495 (96 worlds). 988496-988499 are registered in the runner row but
  reserved unused. It was built by `nm_finaleval.build_panel` as A81 did.
  - file sha256 **635b9aba204e7190dd809e5bd3e3cf5a5fba9c39f027e620246cc1c7721eb77c**
  - panel_sha256 **b543b55b895809c82aaf06ab741b182899180140a23ef801bf312b7d7d83d8dd**
  - bootstrap contract: paired_percentile, 10,000, seed 20261104
- **Audit** (`ship/SEED_AUDIT_A89.json`, 5a115c80) on 988400-988499:
  - REGISTRATION.md and experiment_history.json: 0 hits.
  - A token grep of the laptop `experiments/` and `runs/active/C4-V7FT-20261017`: 0 hits.
  - Box 3 `/root` and `/home/ubuntu` scan with the A82 `seed_audit_box.py`: 25,969 files, 432 panels, 0 code hits. The
    only overlaps are A82 part-B staging: the unrun write-once 384-world option panel `/root/a82four/panels/A82F_PANEL_C_384.json`
    (988200-988583) and the A82 tree's runner row 988200-988599. **That 384 panel must never be launched.**
  - Not done: a broad laptop working-tree grep (not attempted; it was denied for A82), git branch-tip grep (A82
    covered the superset block), Box 4.

## 3. Budget (1,200 sim-s per episode)

- **mech:** 14 episodes = **16,800**. That is the A81 identity for MAPPO60C, DUO_HOLD_60C and the A75-D2 bank
  (6 episodes, vs stored A70 / A75-D2 records), plus 4 plan arms × 2 worlds (8 episodes). All run on A70 worlds
  996300/996301.
- **confirm:** 6 × 96 × 1,200 = **691,200**.
- **Phase cap:** **708,000**, enforced in code (`STAGE_CAP`, `CAP_SIM_S`). No retry reserve.

## 4. Runtime estimate (Box 3, 48 cores, 46 workers)

- A81 confirm ran 1,152 episodes in about 95 min, about 5.0 s per episode. A80/E1R plan waves ran at about 6 s per episode.
- Expected: mech about 8-12 min (one round of 14), confirm 576 episodes about 48-60 min, plus analysis about 1-2 min.
- **Total about 60-75 min.** That fits the window only if it starts by about 01:55.

## 5. Launch commands (each launched once; prefix `timeout 60 ssh -o BatchMode=yes -i ~/.ssh/rented_box ubuntu@124.220.72.157 '<cmd>' < /dev/null`)

The pin stage is already done (`/root/a89/chain_pin.out`: PIN_OK, charged 0). Do not relaunch it; it refuses anyway.

mech (register first):
```
sudo -n bash -c "A89_WORKERS=46 A89_ARMS_SHA256=de9a30a944eda470777cbd3247c6ca81db2745547549cde3e2bff154e23824c2 setsid -f nohup bash /root/a89/run_a89.sh mech < /dev/null > /root/a89/nohup_mech.out 2>&1"
```
Wait for `A89_MECH_DONE` in `/root/a89/chain_mech.out`. A failure writes `CHAIN_STOPPED` and STOP.

confirm (and the analysis that follows it):
```
sudo -n bash -c "A89_WORKERS=46 A89_ARMS_SHA256=de9a30a944eda470777cbd3247c6ca81db2745547549cde3e2bff154e23824c2 A89_PANEL_SHA256=b543b55b895809c82aaf06ab741b182899180140a23ef801bf312b7d7d83d8dd setsid -f nohup bash /root/a89/run_a89.sh confirm < /dev/null > /root/a89/nohup_confirm.out 2>&1"
```
Wait for `A89_CONFIRM_DONE`. To stop between rounds: `sudo -n touch /root/a89/root/STOP_A89`.

How the chain behaves:
- Each stage logs start and end lines to `/root/a89/chain_<stage>.out`.
- An existing chain file makes the stage refuse to start.
- Any failure writes `/root/a89/CHAIN_STOPPED` (JSON: stage, time, reason, charged), and later stages refuse while it
  exists.
- There is no retry.

## 6. Receipts and outputs

| what | path |
|---|---|
| mechanics verdict | `/root/a89/root/A89_MECH.json` (sealed receipt_sha256; identity rows plus plan_checks) |
| results | `/root/a89/root/A89_RESULTS_D2800_p60.json` (sealed analysis_sha256); the summary line is in `/root/a89/analyse.out` |
| per-episode log, charges, drop manifest | `root/A89_LOG.jsonl`, `root/CHARGES.jsonl`, `root/A89_DROPPED_RAW.jsonl` |
| sealed wave receipts | `root/WAVE_A89{I,C}_*.json` |
| progress | `mech.out`, `confirm.out`, `chain_*.out` |

## 7. Analysis (in the driver, zero SUMO; runs automatically after confirm, or `run_a89.sh analyse`)

All numbers come from the A81 code: the a42 `D._contrast` → `nm_finaleval._bank_compare` (paired percentile bootstrap
10,000, seed 20261104, total delay, five groups vs 10 s, CVaR90 most-delayed 10 %) plus hard brakes / TTC / collisions /
teleports via `fe.paired_stats`. diff = treatment − reference; negative = less delay. This panel is never pooled with
any other.

- **PRIMARY ablation:** ROLL_BOTH − ROLL_L0_ONLY and ROLL_BOTH − ROLL_L2_ONLY. COOPERATION SUPPORTED iff both CI95
  uppers are < 0.
- **PRIMARY taper:** TAPER − MAPPO60C. CONFIRMED iff CI95 upper < 0 AND `groups.all_pass` (the A81 five-group rule).
- **Secondary:** TAPER − DUO_HOLD_60C.
- **Descriptive:** ROLL_BOTH / L0 / L2 − MAPPO60C, DUO_HOLD_60C − MAPPO60C (sanity replicate of A81), and the
  interaction (BOTH − M) − [(L0 − M) + (L2 − M)]. The interaction is computed as `fe.paired_stats(BOTH+M, L0+L2)` with
  the same bootstrap.
- CVaR90 is reported in every contrast row. Arm means, safety totals and activation counts are also reported.

## 8. Tests and checks

- `run_a89.sh test` passed **7/7** on Box 3. It covers caps, re-pointing, block_free, verdict logic, interaction
  identity, the contrast list, and the arms-bank derivation.
- Setup identity check: a75 digest d1af7e0f, `A70_RANGES` / `A81_RANGES` / `A89_RANGES`. The tree diff is only the
  runner plus the A89 files.
- Dry i: 7 waves × 2, 16,800. Stored refs: receipts f235a47d / 0af22b0c / b4a17a6b, reports ok.
- Dry c: 6 × 96, 691,200, charged 0.

Files: `ship/` (shipped sources + SHIP_FILES.sha256), `pinned/` (arms bank, panel, pin), `box_outputs/` (setup, pin,
dry, build logs), `audit/`, `ref_a81/` and `ref_src/` (read-only reference copies).

## 9. Doubts and deviations

1. **The single-role arms change which situations activate.** With both roles, an activation needs a rank-1 CAV in L0
   and a rank-0 CAV in L2 within 200 m. The single-role arms activate whenever their one role is fillable, so they may
   fire more often and at other times. That is part of the ablation, but it is not a pure "remove one action".
   - L0_ONLY keeps rank 1 (the second-most-downstream lane-0 CAV in the zone). Ranks were not renumbered (deletion
     only).
   - Activation counts per arm are reported.
2. **Plan-arm mechanics are checks, not identity.** No stored record of these four plans on 996300/996301 exists.
   The plan arms get receipt invariants, zone ±1 m, executor/digest, and 0 collisions/teleports. Exact identity covers
   only MAPPO60C, DUO_HOLD_60C and the A75-D2 bank. TAPER was not identity-checked against its E1R records on 992500/1,
   to save time.
3. **TAPER runs in the A81 (a75d2c-derived) tree; E1R ran it in the a69-derived a80e1r tree.** The a75 executor
   digest is the same (d1af7e0f); the runner differs (982ec0cf vs ef8788ff).
4. **The analysis is unit-tested only at the verdict / pure level.** No synthetic end-to-end analyse run was done.
   An exception in `analyse` would write CHAIN_STOPPED after confirm. The data would be kept, and
   `run_a89.sh analyse` could be run once after a fix.
5. **Cosmetic A81 labels remain** in some sealed schemas from the unchanged code (`coopmwm.a81.dropped_raw.v1`; the
   status/CLI help text).
6. **Seed audit caveat.** My laptop grep (`grep -r` over `experiments/` and `runs/active/C4-V7FT-20261017`) applied
   the banned-word path filter only to its output, so it may have opened files under such paths. It matched and
   printed nothing. The top-level directories carry no banned words. Disclosed here.
7. Two primaries with no multiplicity correction (stated in the results). The rolling plan's own selection history
   (A75-D2/A81) and TAPER's (1 of 12 random plans, 24 worlds) are post hoc.
8. Left on Box 3 outside `/root/a89`: `/tmp/a89_ship/`, `/tmp/a89_setup.out`, `/tmp/a89_seed_audit*.json`,
   `/tmp/a89_seed_audit_box.py`, and the ship copy `/root/a89_ship/` (all mine).

## 10. Draft registration paragraph

> **AMENDMENT 89 - A89 (role ablation of the rolling plan + fresh-panel check of E1R taper plan, D2800_p60; POST-HOC,
> exploratory) - registered <time> before any A89 SUMO second.**
>
> **Question.** (i) Is the benefit of the A75-D2 rolling plan d2_d86b1d6ac653e8ad cooperation? That is, are both
> roles together better than either role alone? (ii) Does E1R random-draw plan mut_1fca373046a3c66e beat MAPPO60C on
> fresh worlds?
>
> **Closest prior.** A81 (bank of rolling + pulse vs DUO_HOLD_60C SUPPORTED, 384 worlds); E1R admission (24 worlds).
>
> **Panel and setup.** One fresh panel 988400-988495 (96 worlds; panel sha b543b55b, file 635b9aba). The block
> 988400-988499 was audited, SEED_AUDIT_A89 5a115c80; 988496-499 is reserved unused. Box 3 /root/a89, phase
> sixarm_a89.
>
> **Arms** (all on MAPPO60C 2abed317; same worlds):
> - ROLL_BOTH, ROLL_L0_ONLY, ROLL_L2_ONLY and TAPER: A75_PLAN on arms bank de9a30a9.
>   - ROLL_BOTH is byte-copied from c325ec1d.
>   - The single-role arms are deterministic deletions of the L2 or L0 role.
>   - TAPER is byte-copied from the E1R candidate bank 057c1f06.
> - DUO_HOLD_60C and MAPPO60C: as A81.
>
> **Mechanics** (16,800; PASS required). A81 identity of MAPPO60C / DUO_HOLD_60C / A75-D2 bank on 996300/1, plus A80
> plan_checks, zone, executor and 0 collisions/teleports for the four plan arms.
>
> **Confirm.** 6 × 96 × 1,200 = 691,200 (cap 708,000 incl. mechanics).
>
> **Analysis** (a42 `_contrast`; paired percentile bootstrap 10,000, seed 20261104; diff = treatment − reference,
> negative = less delay; never pooled).
> - PRIMARY ablation: COOPERATION SUPPORTED iff ROLL_BOTH − ROLL_L0_ONLY and ROLL_BOTH − ROLL_L2_ONLY CI95 uppers are
>   both < 0.
> - PRIMARY taper: CONFIRMED iff TAPER − MAPPO60C CI95 upper < 0 AND the five-group 10 s rule.
> - Secondary: TAPER − DUO_HOLD_60C.
> - Descriptive: each rolling arm − MAPPO60C, the interaction, and DUO_HOLD_60C − MAPPO60C (replicate).
> - CVaR90 / hard brakes / TTC / collisions / teleports for every contrast.
>
> **Code.** Driver a89_sixarm.py 758c8daa (thin wrapper over A81 c755a551); runner 982ec0cf (A81 runner + A89_RANGES);
> run_a89.sh b534b42a; env 76030bfc.
>
> **Launch rule.** One launch per stage, no retry; a failure writes CHAIN_STOPPED.
