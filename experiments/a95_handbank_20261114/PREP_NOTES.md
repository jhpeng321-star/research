# A95(a) hand-written plan admission: prep notes (2026-11-14 ~02:30, prep agent)

Rule: REGISTRATION.md AMENDMENT 95 (line 12031), part (a). This prep ran zero SUMO and zero WM. Nothing was
launched. I did not touch a85adm, a86adm, a91, a82four, a93adm or a94, signalled no process, and read no A93 outcome
file.

## Candidates (a95_hand.py)
The candidates are the 17 plans of `strategy_search.seeds()` (02e119c1), in order, followed by `duo12_v20` from
`dense_seeds()`. All are unchanged: a75ss is not installed, no S4 cfg is used, and there is no uncond, lane-set or
min_roles filter. They are written with the A93 writer (`a75ss.offer_condition` / `strategy_key` / `is_extended`) into
HAND_ALL.json, with score, gain_vs_base, gain_vs_base_lcb and feasible null. The zero-SUMO map check is
`a80_s2b.map_row` 48c5c448 in /root/a95adm/precheck. FORM is the A80 form_family; mech means the plan has an
uncovered token.

Notation: G22 = v 22, dmax 1.0, hold 120 s; M18 = v 18, dmax 1.0, hold 90 s; L = SUMO lane (0 = closing lane);
zone full; unconditional unless stated.

| # | plan | description | map | form | mech |
|---|---|---|---|---|---|
| 1 | wall_g_solo2 | wall, L2 G22 (incumbent) | ok | other | no |
| 2 | wall_g_duo12 | wall, L1 G22 + L2 G22 | ok | other | no |
| 3 | wall_g_solo2_gate0 | wall, L2 G22, start_gate empty100 | REFUSED: start_gate = empty100 (search is unconditional) | - | - |
| 4 | wall_m_solo2 | wall, L2 M18 | ok | other | no |
| 5 | wall_g_solo1 | wall, L1 G22 | ok | other | no |
| 6 | wall_g_duo12_gate0 | wall, L1+L2 G22, start_gate empty100 | REFUSED: start_gate | - | - |
| 7 | wall_g_alt21 | wall, L2 G22 / L1 G22 alternating, phases [[0],[1]] | REFUSED: phases | - | - |
| 8 | wall_g_alt21_gate0 | as 7 + start_gate empty100 | REFUSED: start_gate | - | - |
| 9 | wall_g_relay21 | wall, L2 v22 h90 + L1 G22 onset 60 | ok | rolling | no |
| 10 | wall_g_relay12 | wall, L1 v22 h90 + L2 G22 onset 60 | ok | rolling | no |
| 11 | g2_stagger40 | lanes, L2 r0 G22 + L2 r1 G22 onset 40 | ok | rolling | no |
| 12 | wall_g_duo12_stagger40 | wall, L2 G22 + L1 G22 onset 40 | ok | rolling | no |
| 13 | g2_gap1 | lanes, L2 G22 + L1 M18 onset 20 | ok | rolling | no |
| 14 | g2_gap1_gate0 | as 13 + start_gate empty100 | REFUSED: start_gate | - | - |
| 15 | g2_lc01 | lanes, L2 G22 + L0 G22 lane change to 1 | REFUSED: lane-change role (search is hold-only) | - | - |
| 16 | lc01_g22 | lanes, L0 G22 lane change to 1 | REFUSED: lane-change role | - | - |
| 17 | g2_gapopen12 | lanes, L2 G22 + L1 r1 M18 onset 20 lane change to 2 | REFUSED: lane-change role | - | - |
| 18 | duo12_v20 | wall, L1+L2 v20 dmax1 h120 (DUO_HOLD_60C baseline) | ok | other | no |

- 10 plans are mappable and 8 are refused (5 by start_gate/phases, 3 for a lane-change role).
- The mappable plans have only 2 distinct forms: other (5) and rolling (5). Under the bank rule (best admitted plan
  per form), BANK_HAND therefore has at most 2 plans.
- No plan is extended and none is a mechanics plan. mech runs nothing: the code takes the 'nothing to run' path and
  mech-compare gives PASS. The mech stage has not been run, so this is from reading the code. dry v shows mech
  sim_s 0.

## Box 3: one root, /root/a95adm (10 plans, so there is no a95adm_b)
- Setup used the same recipe as a93adm: `A80_BASE=/root/a95adm bash /root/a78s2b_ship/run_a80.sh setup`, with ship
  SHA256SUMS OK, and gave SETUP_OK.
- Pin gave PIN_OK.
  - TOP sha 1a9166dbd81cdf03ef747310bfb829ee70377fed30d79a9cf55901d4bf24caee, the same on the laptop, in top/ and in
    the pinned copy.
  - doc_sha256 38428a9d95d81222506cc39a003dce9fe57b788970b6b5895fa1d5447b1a74dc.
  - A80_CANDS seal db8b8d29f6bed6146172adb9a9e5be0f34c4502eab8d6dc76de99b4b134e30f1.
  - A80_CANDIDATES.json sha 878642ed3f6327c8e984b01c7cd4289114f8d29eca416202e20377912cb0f064.
- dry v: worlds 992500-992523, sim 288,000 (10 x 24 x 1200), mech 0, charged 0, panel 989248f9, reports_ok. dry c
  charged 0. There is no CHARGES.jsonl.
- The pinned doc carries the fixed A80 strings `stratum='wm_top'` and `world_model=...WM search file`. For the hand
  plans these are inaccurate; disclose them.
- Receipts (after bank):
  - admission receipt: /root/a95adm/root/A80_ADMISSION_D2800_p60.json, plus A80_ADMISSION_EXTRA.json
  - bank file: /root/a95adm/root/a80_banks/A80_BANK_D2800_p60.json
  - mech receipt: root/A80_MECH.json
  - chain logs: /root/a95adm/chain_{mech,admit,bank}.out
- Prep receipts: /root/a95adm/{setup_a95.out,pin.out,dry_v.out,dry_c.out} and
  precheck/{HAND_ALL.json,mapcheck_a95.py,MAPCHECK.json}. Laptop copies are in box3/.

## Queue (written, NOT started): /root/a95queue.sh sha256 493b063b65cf5bf020cd08c861a7fecf8cce3ec65023a30fc715eff6054261f2
- Refuses unless every root is pinned and its mech stage has not been launched.
- Waits for A93Q_DONE or A93Q_SKIPPED in /root/a93queue.log.
- a82-busy check before each root:
  - (a) `ps -eo pid,args | grep -E '[r]un_a82\.sh|[a]82four'`, or
  - (b) /root/a82four/chain_confirm.out exists without A82F_CONFIRM_DONE or a FAILED line.
  - If either holds, it writes "A95Q_SKIPPED busy" and exits.
- Then mech, admit and bank, once each, with 46 workers, for a95adm and then for a95adm_b if that root exists (it
  does not). Log /root/a95queue.log; the last line is A95Q_DONE.

Launch (Box 3):

    sudo -n setsid -f nohup bash /root/a95queue.sh > /root/a95queue.nohup 2>&1 < /dev/null

## Hashes (sha256)
- a95_hand.py 3e1efb1415e8bf81f2335cac7389f5887fdcb5835f8f67180ee3b384af14fd04
- mapcheck_a95.py 3fa0e353ea5b608f60f869a1702e45207d37a6c13e89810f70996910e5bf69c4
- HAND_ALL.json bfb1f576378f5e87aceaa52f615f60544d65142a91e2eea230814ae2a41d21e6
- MAPCHECK.json 213dc31ed5fa7f96de28067519ee4a72d188f863f24e85cbe79aeb55614113c0
- TOP_OFFER_CONDITIONS_D2800_p60.json 1a9166dbd81cdf03ef747310bfb829ee70377fed30d79a9cf55901d4bf24caee
- box3/a95queue.sh 493b063b...
- box3/pin.out 791cfbcd...
- box3/dry_v.out 159ee2ec...
- box3/dry_c.out 487c9ae8...
- box3/setup_a95.out 79daeedf...
