# A101 S-E1-RANK admission prep notes (2026-11-14 ~05:40, prep agent)

Rule: REGISTRATION.md AMENDMENT 101. Zero SUMO, zero WM. Nothing launched. Other Box 3 roots only read
(a86adm for comparison, a95adm/a93adm logs); no process signalled.

## Top files (a101_top.py, sha 2fbe56ed...)
- Rows = a86_s5/STRATEGY_SEARCH_D2800_p60.json (30f10455) list `final`, matched by strategy.name, written with the
  verbatim row expression of a86s5.write_niche_top (71dda3e8) using ship a75ss.offer_condition / is_extended
  (cae54dc1, = search provenance code.a75ss). Lane fields (sumo_lane = role lane, lc_target_sumo = role lc_target)
  come from that function; nothing hand-converted.
- Self-check: the same expression over the 12 shortlist rows reproduces
  a86_s5/TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json (22097910) byte for byte (and content-equal).
  strategy_key recomputed with a75ss.strategy_key equals the search key for all 43.
- All 31 are non-comparator, non-banned, activated (feasible > 0), i.e. pass the TOP-writer eligibility filter.
- Order inside each file = A101_GROUPS.json order (score descending).
- A 11 rows sha 556ed105448a880d66dd4f59be8c1eacc409ea5c0edadd4d07357a96d80250df
- B 10 rows sha a9d73bde85282eeaff92e7e67bcabf98146bdb30e64ff0543416475ade8d4c88
- C 10 rows sha 7d11e42670c5b5141ede67cd088203dd49ec2c10dcfcdf6e3ed5e5435e006942
  (laptop == box top/ == pinned root/A80_TOP_OFFER_CONDITIONS_D2800_p60.json)

## Box 3 roots (same recipe as a86adm / a95adm)
`A80_BASE=<B> bash /root/a78s2b_ship/run_a80.sh setup` (ship SHIP_FILES.sha256 OK; setup_a80.sh 2d4a24c5,
run_a80.sh 44fbf6a7) -> SETUP_OK; then `A80_BASE=<B> A80_TOP_FILE=<B>/top/... A80_TOP_SHA256=<sha> run_a80.sh pin`.
Setup log shows policy pin 2abed317 (MAPPO60C policy_00084.pt), runner A80 ef8788ff, settings 49b6d048.
src tree hash, run_a80.sh, env_a80.sh (modulo base path) and ref68 (24 files) equal to /root/a86adm.

| root | pin | plans | mech plans | by_form | doc_sha256 | A80_CANDS seal | dry v sim (mech) |
|---|---|---|---|---|---|---|---|
| a101adm_a | PIN_OK | 11 | 11 | gap1 pulse3 rolling2 short1 taper2 zone2 | a10bee54... | 046dcb93... | 316,800 (26,400) |
| a101adm_b | PIN_OK | 10 | 10 | gap3 pulse1 rolling1 taper3 zone2 | e48409fe... | 08c6f4f7... | 288,000 (24,000) |
| a101adm_c | PIN_OK | 10 | 7 | gap2 other2 pulse2 rolling2 taper1 zone1 | 5522b42c... | e23860bd... | 288,000 (16,800) |

pin-top refused none of the 31 (no A80_MAPPING_REFUSED). All dry v: worlds 992500-992523, charged 0, cap 460,800,
panel 989248f9, reference A68V_MAPPO60C receipt 1a6d31dd reports_ok. dry c charged 0. No CHARGES.jsonl.
Total dry-v sim = 892,800 = the registered cap exactly. A80_CANDIDATES.json shas: a 11ee3c8e..., b d2b1d68b...,
c 1a894720...

## Queue (written, NOT started): /root/a101q.sh sha256 d3def857220d426afad3faa4eefb1f8310659000f98f31b84d530c1bc574a201
Refuses if any root unpinned or any chain_{mech,admit,bank,confirm}.out exists. No waits/busy checks. For a, b, c:
mech && admit && bank once each, A80_WORKERS=46; failed root logs rc and the queue continues. Log /root/a101q.log
("<time> <root> start (ALLOCATION START)", "<time> <root> end rc N", final A101Q_DONE).
Launch: `sudo -n setsid -f nohup bash /root/a101q.sh > /root/a101q.nohup 2>&1 < /dev/null`

## Deviations / disclosures vs A90/A95
- Three roots instead of one (12-plan A80 cap); otherwise identical chain.
- Pinned docs carry the fixed A80 strings stratum='wm_top' / world_model=WM search file; here the plans are WM
  finalists but not the WM top list, so these strings are slightly inaccurate.
- Box receipts copied to box3/ (setup, pin, dry v/c per root, queue).
