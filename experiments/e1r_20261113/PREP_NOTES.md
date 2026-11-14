# E1R prep notes (2026-11-13 20:45 UTC+8): WM-ranked 12 vs random 12 from the S2b pool

Status: PREPARED, NOT LAUNCHED. Zero SUMO, zero WM rollouts, zero training. No REGISTRATION or ledger edits.
No A80 outcome (WAVE_*, admission receipt, verify/admit/confirm logs, CHARGES) was opened.

## 1. Pool: all 607 full definitions recovered (zero WM, zero SUMO)
- The search output stores full strategies only for the 63 `final` plans; `screened` holds key/name/gain only.
- `replay_pool.py` replays the S2b candidate generator (a75ss cae54dc1 over strategy_search 02e119c1, single
  `random.Random(20261176)` stream) on the laptop, with no WM. Generation 0 depends only on the seed. In generations
  1-4 the parent of each child is chosen from WM-ranked lists that are not re-run. The replay therefore solves for the
  parent: each child must reproduce the NEXT recorded screened key (16-hex sha256 strategy key), and filtered
  children are budgeted from LOG_D2800_p60.jsonl (3/2/0/1 per generation).
- Result: all 607 keys were reproduced in the recorded order. `strategy_key(strategy) == key` holds for every plan,
  and all names match. The 61 non-comparator `final` strategies are byte-equal after canonicalisation. Output:
  `POOL_REPLAY_ALL607.json` (sha256 b0f28551...).
- Code used (read-only copies, hashes in `src_ref/`): a75ss.py cae54dc1, strategy_search.py 02e119c1 (the same as the
  repo's tools/wzprop), families/families_lc/geometry/wm_backend/fakes, and the vendored lcm_pipeline (pins checked
  by families_lc.verify).
- Pool = 607 - the 12 pinned TOP plans (these include duo12_v20 and duo12_v18) - comparators (solo2 and wall_g_solo2
  are not among the screened plans) = **595 plans**, sorted by key. `E1R_POOL.json`: canonical pool sha256
  31ebf2c958fe22f77e396422d55c4aa8f21cfd9ff9824862e83cf7e72eea227f; file sha256 d7dfdea2...
- The pool still contains 11 other hand-written rule seeds (wall_g_*, g2_*, wall_g_duo12_stagger40, ...). The brief
  named only the two duo12 plans, so these were not removed. It also contains 50 plans with WM screen feasible = 0
  (they never activated on the 18 WM screen states). The real TOP writer would never pick those, but the brief's pool
  definition includes them.

## 2. Draw (made once, `draw_e1r.py`, refuses to overwrite)
numpy 2.2.6 `default_rng(20261113).choice(595, 12, replace=False)` over the key-sorted pool. `E1R_DRAW.json` sha256
10aaf9b6353f4fb84f66fa982ac08b08bd135a034e15a12149db4a32817a68e3.

| # | plan | trigger / zone | roles | A80 form (FORM_RULE) | mech | WM screen feasible |
|---|------|----------------|-------|------|------|------|
| 1 | mut_3cc4a6cac9ecbbdc | wall / full | 2 | rolling | no | 1.00 |
| 2 | mut_2d63cd1d82262ffe | wall / mid | 2 | zone | yes | 0.89 |
| 3 | rand_d86c59950501b868 | wall / full | 3 (taper, pulse, dv_rel) | pulse | yes | 1.00 |
| 4 | mut_1fca373046a3c66e | lanes / far | 4 (dv_rel, taper) | taper | yes | 0.33 |
| 5 | rand_fc669b0881c3202a | wall / mid | 2 (dv_rel, taper) | taper | yes | 0.89 |
| 6 | mut_b80c232749756e25 | lanes / merge | 3 (dv_rel) | zone | yes | 0.22 |
| 7 | rand_d93996c482e115fc | lanes / full | 4 (2 gap-start roles) | gap | yes | 0.17 |
| 8 | rand_c558b66748f20a32 | lanes / far | 2 (pulse) | pulse | yes | 0.94 |
| 9 | mut_f1f5d50d9cc5e257 | wall / full | 2 | rolling | no | 1.00 |
| 10 | rand_8c424a6c16394b5e | lanes / full | 4 | rolling | no | **0.00** |
| 11 | mut_356e1670d007b933 | lanes / custom [400,200] | 4 (dv_rel) | zone | yes | 0.11 |
| 12 | mut_2bb5c7d8fd9aec2e | lanes / merge | 4 (dv_rel, pulse, taper) | pulse | yes | 0.17 |

(The feasible column is the WM screen activation fraction. It is not a SUMO outcome.)

## 3. TOP file
`write_top_e1r.py` imports `a75ss.offer_condition` / `is_extended` unchanged. Self-check: the same row builder on the
12 pinned `final` records reproduces the real TOP file byte for byte (2ea90165...).
`E1R_TOP_OFFER_CONDITIONS_D2800_p60.json` sha256 **03e80fe28e95b1beedc0488b733328c4067c21e277aa22cdc7324490158fadd7**
(LF bytes). The WM fields of the random rows are as follows: `score` comes from the archive listing when present,
else null; `gain_vs_base` and `feasible` come from the screened record; `gain_vs_base_lcb` is null. The driver uses no
WM number in any SUMO rule.

## 4. Box 3 base /root/a80e1r (built 20:41-20:43 while A80 confirm ran; nice 19 / ionice idle; zero SUMO)
- Setup: `sudo -n env A80_BASE=/root/a80e1r nice -n 19 ionice -c3 bash /root/a78s2b_ship/setup_a80.sh` -> SETUP_OK
  (log /root/a80e1r/setup_e1r.out). This is the documented path: a hard-link clone of /root/a69/src, it writes only
  $BASE, and it ran without changes.
- `diff -rq /root/a78s2b/src /root/a80e1r/src` shows 0 differences; ref68/ and panels/ are identical. The only
  difference in env_a80.sh is BASE/SRC/A80_HOST_BASE=/root/a80e1r. run_a80.sh is byte-identical (44fbf6a7).
- Driver a80_s2b.py **48c5c448c2a85d4e2020def54cb17259979c89f2e0cc341829d6e99de0d0e289** (unchanged). Also unchanged:
  a75_d2forms 8a2533ef (ADMIT_RULE), a75_holdforms 72046db6 (a75 digest d1af7e0f), runner ef8788ff, settings
  49b6d048, REF_PINS 9741e2e3.
- TOP: copied laptop -> /tmp -> /root/a80e1r/top/TOP_OFFER_CONDITIONS_D2800_p60.json; the sha256 on the box matches
  03e80fe2...
- `run_a80.sh pin`: PIN_OK. **12/12 accepted by MAPPING_RULE, 0 refused.** 9 plans need mechanics: #2-8, #11 and #12
  in the table. 3 plans are covered: #1, #9 and #10. by_form gap 1 / pulse 3 / rolling 3 / taper 2 / zone 3;
  candidates doc_sha256 8ae44753...
- dry v: 12 waves x 24 episodes, worlds 992500-992523, arm A75_PLAN, sim_s 345,600, **charged 0**, panel_sha256
  989248f9... (the same as the A80 dry), reference = stored A68V_MAPPO60C receipt 1a6d31dd... with 24 reports_ok (the
  same as A80, reused and not re-run). dry c: "no admission receipt".
- **CHARGES.jsonl does not exist in /root/a80e1r/root.**
- Copies of pin.out, dry_v/c.out, setup_e1r.out and A80_CANDIDATES.json are in `box3_pin/`.
- Disk: 32 G free (the stages check for >= 2 G for mech and >= 4 G for admit).
- Files I left outside the base: /tmp/E1R_TOP_OFFER_CONDITIONS_D2800_p60.json and /tmp/e1r_setup.out (mine).

## 5. Sim-second budget (E1R)
- mech: 9 plans x 2 worlds x 1,200 = **21,600** (these count inside the verify cap)
- admit: remaining 12 x 24 - 18 = 270 episodes x 1,200 = **324,000**
- total = **345,600** = the A80 verify cap. Zero charge for bank. **Do NOT run `confirm`.** E1R is admission only; the
  bank stage will write a bank and its receipt says "A80_BANK_60C runs on the A70 panel", but that is not part of E1R.
- Wall time reference: the A80 mech + admit took about 30 min with 46 workers.

## 6. Launch commands (wait until the A80 chain has finished: no live pid 746789 / chain_confirm finished)
```bash
ssh -o BatchMode=yes -i ~/.ssh/rented_box ubuntu@124.220.72.157
# check A80 is done without reading outcomes:
ps -p 746789 >/dev/null && echo "A80 chain still running" || echo "A80 chain finished"
# E1R chain (mech -> admit -> bank; NO confirm):
sudo -n bash -c 'cd /root/a80e1r && export A80_BASE=/root/a80e1r A80_WORKERS=46 && setsid -f nohup bash -c "bash /root/a80e1r/run_a80.sh mech > /root/a80e1r/nohup_mech.out 2>&1 && bash /root/a80e1r/run_a80.sh admit > /root/a80e1r/nohup_admit.out 2>&1 && bash /root/a80e1r/run_a80.sh bank > /root/a80e1r/nohup_bank.out 2>&1" > /root/a80e1r/nohup_chain.out 2>&1 < /dev/null'
# or stage by stage:
sudo -n bash -c 'export A80_BASE=/root/a80e1r A80_WORKERS=46; setsid -f nohup bash /root/a80e1r/run_a80.sh mech > /root/a80e1r/nohup_mech.out 2>&1 < /dev/null'
#   after "A80_MECH_DONE" in /root/a80e1r/chain_mech.out:
sudo -n bash -c 'export A80_BASE=/root/a80e1r A80_WORKERS=46; setsid -f nohup bash /root/a80e1r/run_a80.sh admit > /root/a80e1r/nohup_admit.out 2>&1 < /dev/null'
#   after "A80_VERIFY_DONE" in /root/a80e1r/chain_admit.out:
sudo -n bash -c 'export A80_BASE=/root/a80e1r A80_WORKERS=46; setsid -f nohup bash /root/a80e1r/run_a80.sh bank > /root/a80e1r/nohup_bank.out 2>&1 < /dev/null'
# stop between rounds: sudo -n touch /root/a80e1r/root/STOP_A80
```
Receipt for the analysis: /root/a80e1r/root/A80_ADMISSION_D2800_p60.json. The A80 receipt has the same name under
/root/a78s2b/root/.

## 7. Analysis (`analyse_e1r.py`, zero SUMO; tested on synthetic input only, `SELFTEST_ANALYSE.txt`)
`python -I analyse_e1r.py A80_ADMISSION.json E1R_ADMISSION.json E1R_DRAW.json --out E1R_ANALYSIS.json`
- Primary: admitted count WM-12 vs random-12 with a two-sided Fisher exact test. Drawn plans with no verdict count as
  not admitted and are listed.
- Secondary: per-plan `mean` (plan - MAPPO60C total delay over 24 worlds, veh-s, negative = less delay) with a
  two-sided Mann-Whitney test (exact when there are no ties).
- Both are reported including and excluding duo12_v20/duo12_v18. The script refuses if panel, rule, reference or
  seeds differ between the two receipts.
- Self-test: the Fisher and Mann-Whitney values match scipy 1.15.3 exactly.

## 8. Deviations / doubts
1. The driver labels every candidate `stratum='wm_top'`, the A80 phase name `holdforms_a80s2b` and the `a80v_` job
   prefix. These come from the frozen driver. They are cosmetic, and the E1R root and HOST_BASE are separate. The
   ledger must record that E1R charges carry these labels.
2. The pool includes 50 feasible=0 plans and 11 other hand-written seeds, per the brief's definition (see 1). One drawn
   plan (#10 rand_8c424a6c16394b5e) has WM feasible 0.00. The real writer's `activated` filter would have excluded it
   from any WM TOP file. It is kept, not redrawn.
3. Multiplicity and post-hoc caveats are the same as A80, on the same 24 already-used worlds.
4. The replay treats the parent-list length as a free nuisance (n_parents 9/9/9/19 are equivalence representatives of
   rng consumption). Correctness rests on the 607 sha256-key matches, not on reproducing the WM ranking.
