# A96(b) 20 % admission (D2800_p20) vs MAPPO20C: prep notes (2026-11-14 ~02:55 box time, prep agent)

Rule: REGISTRATION.md AMENDMENT 96 (line 12039 ff.), part (b). This prep ran zero SUMO; every root has charged 0 and no
CHARGES.jsonl. Nothing was launched. Untouched: a85adm, a86adm, a91, a93adm, a94, a82p20, a82four, and all running
processes. One read-only copy was taken: /root/a95adm/top/TOP_OFFER_CONDITIONS_D2800_p60.json (sha 1a9166db). No A94 or
A95 outcome file was read. Nothing was added under /root/a82four; the queue only reads its inputs/ files.

## Worlds (registered by the coordinator)
- Admission worlds are 988900-988923. The runner block is 988900-988999, with 988924-988999 reserved and unused.
- Audit: the Box 3 `seed_audit_box.py` (the A82/A91 tool) over /root and /home/ubuntu scanned 34,643 files and 604 panels.
  - 0 panel overlaps, 0 runner overlaps, 0 CHARGES hits.
  - Code hits are only the out-of-scope boundary seed 988999 in test_clean4_train_world_cpu.py (train scope starts at
    989000) and one hex fragment in svnmap.txt.
  - A free-block scan of REGISTRATION.md and experiment_history.json found the block free.
  - Audit file: ship/SEED_AUDIT_A96.json (3cfe27f2). The laptop working-tree scan was not done.
- Panel (/root/a96ref/root/A96_PANEL_V.json):
  - file sha 282b9da369e1b3c7b4626a269c017605b82f228fc97c008bc97aa12b03ab13da
  - panel_sha256 969fa8f9a646fd025575494b1d7e43fe09a6ca0f32a16f0f80a6bb6fe4c47bbf
  - A96_REF_PINS.json 989f5696...
  - Bootstrap is the registered paired percentile, 10,000 resamples, seed 20261104 (checked at load).

## Design (60 %-specific items and how they were handled; the rule is unchanged)
I did NOT use a80_s2b's own admit. It is hard-wired to 60 %:
- CELL D2800_p60
- the a68v panel 992500-992523
- the STORED A68V_MAPPO60C wave plus the ref68 reports
- the A69 tree / settings
- a mechanics check that every record loaded MAPPO60C

Instead I used its registered 20 % version, the A82 part-A driver `a82_s3b20.py` (2b33146b), imported unchanged. It
already has CELL D2800_p20 (p20c plant / bindings), A80 MAPPING_RULE / FORM_RULE / plan_checks / mech_verdict, the
a76_d2p20.admit ADMIT_RULE (CI95 upper < 0 AND five-group <= +10 s) and BANK_RULE (best admitted per form, A80 families),
and the A75-D2 execute / drop rule. The new driver `a96_p20adm.py` (85cdef84) re-points four things process-locally,
using the same pattern A82 used:
1. **Worlds:** the A96 block replaces the reused A76 worlds. Runner = A76 runner 79fc9efd + one additive row
   `A96_RANGES = ((988900, 988999),)` (patch_tables_a96.py), giving runner sha 0c5c2a4b. The panel is built once in the
   reference root and byte-copied into every admission root, with the sha pinned.
2. **Reference:** MAPPO20C is run ONCE in /root/a96ref as wave A96V_MAPPO20C.
   - Specs = a76_d2p20.u0_specs through A52.with_executor with the a77_mappo20c-bound executor. This is the same
     path as the a82_fourarm MAPPO20C arm and a77 confirm_specs.
   - Raw logs are dropped. `ref-seal` writes A96_REF.json (wave receipts, 24 DROPPED_RAW-sealed reports, executor).
   - Admission roots read the reference read-only through the A82 stored-reference path, with pinned receipts and
     reports. I checked zero-SUMO that the reference specs rebuilt in an admission root are byte-identical to the
     specs the reference root runs (SPECS_EQUAL).
3. **Executor check of a76 admit:** plan records must carry U0 d9042cc5, because plans run on U0 as registered.
   Reference records must carry the bound MAPPO20C sha.
4. **Labels / file names:** A96_* files, phase holdforms_a96p20, job ids a96v_ / a96r_.
   - Cosmetic leftovers from the unchanged code: wave names A82V_*, verdict rows with schema coopmwm.a76.admission.v1
     and source "A76 (new run, D2800_p20)", and BOUND_A77_* binding names with A77 authority text.

**Mechanics:** I used the A82 20 % rule: the first 2 verification episodes (988900/988901) of every candidate, which
are the same job ids as verification and so cost nothing extra. A80 "mech where required" means plans with tokens
outside the 60 % fixture; at 20 % no fixture covers anything, so every plan is checked. This is stricter for the hand
plans, which A80 at 60 % did not run. Flag it if you prefer A80-literal.

No threshold is read from any 60 % receipt. I see no blocker to a faithful p20 reproduction.

## Bases (setup_a96.sh e3d2c6cc, ship /root/a96_ship_v2 = a96_ship_v2.tar f22bcbda; each a real cp -a of /root/a76p20/src, all files verified)
| base | role | state |
|---|---|---|
| /root/a96ref | reference root (MAPPO20C x 24, cap 28,800) | SETUP_OK, panel built; MAPPO20C NOT bound yet (queue binds it) |
| /root/a96adm_hand | HAND-p20 (10 plans, cap 288,000 inside the code cap 345,600) | PIN_OK |
| /root/a96adm_s4 | S4-p20 top 12 | SETUP_OK, awaiting TOP file |
| /root/a96adm_s5 | S5-p20 top 12 | SETUP_OK, awaiting TOP file |
| /root/a96synth_ref, /root/a96synth_adm | throwaway zero-SUMO synth roots | IDENT_U0 dry OK; driver replaced there twice while debugging (UPDATES.log) |

Hand pin details:
- TOP = the A95 file 1a9166db, mapped at D2800_p20: 10/10, forms other 5 / rolling 5.
- A96_CANDIDATES.json 609bd4ae...; doc_sha256 30037d41...; A96_PINS.json 0ffb4c10...
- A96_CANDS bank 2cb51277...; comparator receipt c5914225...
- dry: 10 waves x 24 episodes, 288,000 sim-s, mech 24,000 (inside), plan executor d9042cc5, charged 0.

## S4 / S5 pin (zero SUMO; map-check happens here; a refusal writes root/A96_MAPPING_REFUSED.json and nothing runs)
Copy the TOP file laptop -> Box 3 /tmp (one file per scp), then:

    sudo -n bash -c "mkdir -p /root/a96adm_s4/top && cp /tmp/<S4 TOP>.json /root/a96adm_s4/top/TOP_S4P20.json && sha256sum /root/a96adm_s4/top/TOP_S4P20.json"
    sudo -n bash -c "A80_TOP_FILE=/root/a96adm_s4/top/TOP_S4P20.json A80_TOP_SHA256=<sha> bash /root/a96adm_s4/run_a96.sh pin"

Expect PIN_OK. For S5, use the same commands with s5 / TOP_S5P20.json.

## Queues (written, NOT started)
| script | sha256 | what |
|---|---|---|
| /root/a96q_ref_hand.sh | 6799f830ff055731f6fcf35d7a139b65315472b8d3679eee7c7d3a6bbded894b | waits for A94Q_DONE + /root/a82four/inputs/{policy_00084.pt, PLAN.json, BOUND_00084.json} with policy sha == A96_MAPPO20C_SHA256; binds MAPPO20C (zero SUMO, A77 PLAN 57f51364 check); busy check; ref (28,800); hand pin-ref; busy check; hand mech -> admit -> bank; log /root/a96q_ref_hand.log, last line A96Q_REF_HAND_DONE |
| /root/a96q_s4.sh | 81baa978d7411657a1360034d1a8055c5c327f1378c2b3fb45c86a7982ae9dc6 | no wait; refuses unless pinned, not launched, reference sealed; busy check; pin-ref; mech -> admit -> bank; log /root/a96q_s4.log, last line A96Q_S4_DONE |
| /root/a96q_s5.sh | 49fffe12e55e9871a30223ba706c4c67be75ec0eb4b78ceb8306e7c3ab0d9740 | same for s5 (A96Q_S5_DONE) |

All three use 46 workers (the reference and mech stages run on min(46, 24) workers). Each stage runs once, and any
failure ends the chain.

The busy check uses `ps` with these patterns, excluding lines that name the queue script itself:
- `[/]root/a94/run_a94\.sh`
- `[a]94queue\.sh`
- `[/]root/a82four/run_a82\.sh`
- `[/]root/a96[a-z0-9_]*/run_a96\.sh`
- `[a]96q_[a-z0-9_]*\.sh`
- plus /root/a82four/chain_confirm.out without A82F_CONFIRM_DONE or a FAILED line.

If busy, the queue writes "A96Q_*_SKIPPED busy" and exits. I tested it: it reported BUSY on another agent's running
`/root/a96pd/run_a96.sh drytest`. That other agent's A96 roots (/root/a96pd, /root/a96pc) use the same script name, so
even their zero-SUMO stages cause a skip.

Launch commands:

    sudo -n env A96_MAPPO20C_SHA256=<registered sha> setsid -f nohup bash /root/a96q_ref_hand.sh > /root/a96q_ref_hand.nohup 2>&1 < /dev/null
    sudo -n setsid -f nohup bash /root/a96q_s4.sh > /root/a96q_s4.nohup 2>&1 < /dev/null
    sudo -n setsid -f nohup bash /root/a96q_s5.sh > /root/a96q_s5.nohup 2>&1 < /dev/null

## Receipts / bank files
- Reference: /root/a96ref/root/A96_REF.json (seal); /root/a96ref/chain_ref.out (A96_REF_DONE); BOUND_A77_MAPPO20C.json.
- Per admission root R in {a96adm_hand, a96adm_s4, a96adm_s5}:
  - admission receipt: /root/R/root/A96_ADMISSION_D2800_p20.json, plus A96_ADMISSION_EXTRA.json
  - bank file: /root/R/root/a96_banks/A96_BANK_D2800_p20.json (written only if a plan is admitted)
  - mechanics: root/A96_MECH.json
  - reference pin: root/A96_REFPIN.json
  - chain logs: /root/R/chain_{mech,admit,bank}.out
- Budgets: reference 28,800; hand 288,000; s4/s5 at most 345,600 each.

## Box 4 for /root/a96adm_s5 (estimate only; not built; Box 4 was not touched)
Steps:
- After TRAIN_DONE: scp a96_ship_v2.tar.
- Verify that /root/a76p20/src, the p20c bank root /root/autodl-fs/nm_finaleval_p20c_bank, /root/a66/venv,
  /home/ubuntu/e3final and the MAP_SPEC exist and are hash-identical (cross-host check of 11-13).
- `A96_BASE=/root/a96adm_s5 setup_a96.sh`.
- After the Box 3 reference seal, copy /root/a96ref/root to Box 4 at the SAME path: the panel, REF pins, BOUND files,
  A96_REF.json, the 24 reference evaluation dirs (reports + DROPPED_RAW, a few MB), and the published wave receipts.
  The admission root pins the reference by path and receipt.
- Then pin, then a96q_s5.sh with the busy patterns adjusted for Box 4.

Effort: about 20-30 min of zero-SUMO work. Physics: about 25 min for 345,600 sim-s on 46 workers, if Box 4 has
comparable CPUs (not measured).

## Files (laptop, this folder)
- ship/: a96_p20adm.py, patch_tables_a96.py, setup_a96.sh, run_a96.sh, SHIP_FILES.sha256, plus the unchanged
  a80_s2b / a77_mappo20c / a82_s3b20.
- a96_ship_v2.tar
- box3/: queue scripts, busy.inc, setup / pin / dry / panel / synth outputs.
- readonly/: copies of the A76 / A82 code that was read.
- cmp_refspecs.py: the synth spec-equality check.

## BOX 4 (ubuntu@49.235.140.169): the whole A96(b) chain runs here (owner yes, 03:0x); Box 3 roots kept, unlaunched
- **A76 tree.** Box 4 had no /root/a76p20. It is a byte copy of Box 3's, moved via the laptop as tgz 87dfaf7c:
  src, env_a76.sh, SETUP_OK and the 8 root binding files (marker file /root/a76p20/COPIED_FROM_BOX3.txt). The src tree
  hash 6fe13463 is equal on both boxes.
- **Shared inputs, equal on both boxes:**
  - p20c bank root files (BOUND_U0 d19c74cd, checkpoint 5ea22964; setup cmp passed)
  - venv python d3ccb5b8, sumo binary 118bd817, torch 2.8.0+cpu, numpy 2.3.2, traci sources fbf69740
  - MAP_SPEC 386400b7, WM_PACKAGE target
- **Roots.** Ship /root/a96_ship_v2 (f22bcbda) was installed with setup_a96.sh for /root/a96ref, /root/a96adm_hand,
  /root/a96adm_s4 and /root/a96adm_s5: all SETUP_OK, runner 0c5c2a4b (equal to Box 3).
- **Panel.** A96_PANEL_V.json and A96_REF_PINS.json were byte-copied from Box 3 (282b9da3 / 989f5696):
  panel_sha256 969fa8f9...
- **Hand pin.** PIN_OK. Every value equals Box 3:
  - 10/10 mapped, other 5 / rolling 5
  - A96_CANDIDATES 609bd4ae, doc 30037d41, A96_PINS receipt f5ffaae9, A96_CANDS 2cb51277
  - dry 288,000 sim-s (mech 24,000), charged 0
  - The only difference is the A96_COMPARATOR.json file hash (54398362 on Box 4 vs Box 3's), because it is a sealed
    publish (timestamp); the comparator record itself is the same.
- **Not yet done.** MAPPO20C is bound by the queue after TRAIN_DONE (reads /home/ubuntu/a77/run84 directly, nothing
  copied). The bind step runs ref-dry, and a failure stops the chain.

### Box 4 queues (NOT started)

| script | sha256 |
|---|---|
| /root/a96q_ref_hand.sh | 8be41f60da33079b0cb53848e7fc4d382e1f2503a2466d8ca078e935327df1a9 |
| /root/a96q_s4.sh | ecdc874b2af5d1f342a7b693f64ac077da37e722bebbd72d98c4bb43db330279 |
| /root/a96q_s5.sh | 0bb8f5301d33ec4897ca0d01aefc580fe8b1c7e5c4f67a10150027da0ad0a0bc |

- a96q_ref_hand.sh waits for /home/ubuntu/a77/TRAIN_DONE and the run84 policy_00084.pt (sha == given), PLAN.json and
  BOUND_00084.json, then allows 60 s of grace. It then binds MAPPO20C, runs busy check, ref, hand pin-ref, busy check,
  then hand mech, admit and bank.
- Box 4 busy patterns:
  - `[g]o_train\.sh`
  - `[w]ms\.control\.mappo\.train`
  - `[/]root/a96[a-z0-9_]*/run_a96\.sh`
  - `[a]96q_[a-z0-9_]*\.sh`
  - `[/]root/a6[89]/run_a6[89]\.sh`
- Tested: the check reports BUSY on the running A77 trainer.

Launch on Box 4:

    sudo -n env A96_MAPPO20C_SHA256=<sha> setsid -f nohup bash /root/a96q_ref_hand.sh > /root/a96q_ref_hand.nohup 2>&1 < /dev/null

S4 / S5: pin as in the Box 3 section, on Box 4. Then:

    sudo -n setsid -f nohup bash /root/a96q_s4.sh > /root/a96q_s4.nohup 2>&1 < /dev/null

(and the same for s5).

### Cross-host
No Box 3 A96 episode exists, so there is no episode-level comparison. Box 3 / Box 4 identity rests on the earlier
11-13 A69 cross-host check (box3_verify.tgz on Box 4) plus the equal hashes above.

### Expected physics on Box 4
48 cores, 46 workers. This is extrapolated from Box 3: about 3 min per 1,200 s episode round. Box 4 speed has not been
measured.

| stage | episodes | estimate |
|---|---|---|
| reference | 24 (24 workers, one round) | about 4 min |
| hand mech | 20 | about 4 min |
| hand admit | 220 | about 15 min |
| hand bank | - | under 1 min |
| S4 or S5 (12 plans), mech | 24 | about 4 min |
| S4 or S5 (12 plans), admit | 264 | about 18 min |
