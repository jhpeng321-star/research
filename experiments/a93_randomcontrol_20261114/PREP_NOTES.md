# A93 random-draw control: prep notes (2026-11-14, prep agent)

Rule: REGISTRATION.md AMENDMENT 93 (line 12023). This prep ran zero SUMO, zero WM and zero training. Nothing was
launched. /root/a85adm, /root/a86adm, /root/a91 and /root/a82four were not written, and no process was signalled. No
admission outcome file of the running job was read.

## Draw (laptop, `a93_draw.py`)
- Generator: `strategy_search.random_strategy` (02e119c1) as patched by `a75ss.install(cfg)` (cae54dc1), with the S4
  cfg `A85S4_SEARCH_CFG.json` 0a80c0f4 merged with SS.DEFAULTS. The script asserts this cfg equals the `config` key
  of a85_s4/STRATEGY_SEARCH_D2800_p60.json. Only that key is read; no score or result field is read.
  Globals are set as in search_cell (set_globals, ONSET_MAX, SEARCH_GRID). The RNG is `random.Random(20261114)` and
  the draws are taken in generation order. The search's screen-state sample is not drawn, because no states are used.
- Checks, applied in the S4 search order: (1) the generator's validate (a ValueError would reject the draw);
  (2) the S4 lane-set exclusion {1,2} (a85s4.LaneFilter.bad, 7ca82976); (3) the run() filters: SS.uncond (cfg
  unconditional: gate/start_gate/phases set to None), min_roles 2, banned v_w, and a duplicate strategy key;
  (4) the zero-SUMO A80 pin mapping, `a80_s2b.map_row` 48c5c448. This last check ran on Box 3 over the first 40
  passing plans (mapcheck_a93.py, /root/a93adm/precheck/). It refused 0 of 40.
- 16 draws were used. 4 were rejected, all for excluded_lane_set {1,2}: draw 1 rand_84f2b34a3563c9b5, draw 5
  rand_98770a4a7f3cd262, draw 7 rand_fdba7617dbd1c41e, draw 15 rand_0351f0fd6ef5a767. Every draw is listed in
  DRAWS.json (STREAM_DRAWS.json has all 50 draws to the 40th pass).
- Rows are in the a75ss writer format. score, gain_vs_base, gain_vs_base_lcb and feasible are null, because the pin
  only copies them into `wm` (a report field) and no rule reads them. extended comes from a75ss.is_extended. The
  family is the generator's rand_<key> name. Disclosure: the pinned doc carries the fixed strings
  `stratum='wm_top'` and `world_model='plans and their order come from the WM search file'` from the unchanged
  A80 code. For A93 these strings are inaccurate: the order is the draw order.

## The 12 plans (all are extended, all have uncovered tokens and so are mechanics plans; dmax per role in the JSON)
1. rand_2b84ba100dca39c9 lanes/near cd200: L1r1 gap 20 s (v20, on20, 3 events, clear, drop600) + L0r1 v12 h30 on40
2. rand_f48526f5527aee64 lanes/full cd200: L1r0 v16 h45 on60 dv6 + L1r1 v0 h60 on20 dv8 + L0r0 v12 h180 on20 dv4
3. rand_f1f3c30f83d035fe lanes/far cd140: L1r0 gap 15 s on40 + L1r1 v16 h90 on20 dv4 + L0r0 v12 h60 on0 taper-out10 + L2r1 v12 h180 on60 dv6
4. rand_b9a542e05b0287be lanes/full cd140: L1r1 gap 20 s (3 events, drop150, dv4, taper-out10) + L0r0 v12 h60 on40 pulse5/5
5. rand_3426823506bdb95f lanes/near cd200: L2r1 v20 h180 on40 + L0r1 v16 h60 on40 dv2 + L2r0 v12 h120 on60 dv6 taper-out5
6. rand_324930af7618b8aa lanes/full cd60: L1r1 v18 h20 on20 + L0r1 v16 h10 on60 + L0r0 v12 h30 on0 dv6
7. rand_954827ee338796cd lanes/near cd60: L0r1 v14 h20 on40 + L1r0 gap 15 s on20 dv4 + L1r1 v24 h60 on60
8. rand_a4404139342ec3ad lanes/full cd200: L0r0 v12 h60 on0 dv6 + L2r0 v12 h90 on20 dv6
9. rand_e60ee926d34eb631 wall/mid cd90: L2 v14 h120 on60 + L0 v24 h30 on20 taper10/20 + L1 gap 20 s (3 events)
10. rand_752ec63f1e216873 wall/mid cd200: L1 gap 5 s (clear, dv8) + L0 v22 h120 on20
11. rand_ac77454ba4dc3772 wall/far cd140: L0 v12 h60 on60 + L1 gap 20 s (v24, clear)
12. rand_855593233fe40415 lanes/far cd90: L1r0 v14 h30 on40 taper-in5 pulse5/20 + L2r1 v16 h90 dv6 pulse5/20 + L1r1 v12 h60 on20 + L0r0 v18 h90 on20 pulse20/20

Forms: gap 7, rolling 3, taper 1, pulse 1. The trigger is never gated (all plans are unconditional).

## Box 3 (/root/a93adm)
- Setup: `A80_BASE=/root/a93adm bash /root/a78s2b_ship/run_a80.sh setup` (ship SHA256SUMS OK; setup_a80.sh
  2d4a24c5, run_a80.sh 44fbf6a7) gave SETUP_OK. The src tree hash (1491 files) equals /root/a85adm. env_a80.sh
  equals a85adm's apart from the base path. run_a80.sh is byte-equal to a85adm's.
- Pin: `A80_TOP_FILE=/root/a93adm/top/TOP_OFFER_CONDITIONS_D2800_p60.json A80_TOP_SHA256=238b0cbe... run_a80.sh pin`
  gave PIN_OK with 12 candidates. doc_sha256 cbc37fb0a45e943ae3e1d7f10352ecef917a853f469a0b75dea399cef553300e. The
  A80_CANDS seal is 2d1a18edffb25d499b68e625810aa10e18830004df57c7fb0eef36a0638c5596. The A80_CANDIDATES.json file
  sha is 2ee6035a633898f8e412f13ef6f1dc731d0900323ece313e4661686a9276cdfb.
- dry v: worlds 992500-992523, sim 345,600, mech 28,800, charged 0, panel 989248f9, reference A68V_MAPPO60C receipt
  1a6d31dd with reports_ok. dry c charged 0. No CHARGES.jsonl.
- Receipts on the box: /root/a93adm/{setup_a93.out,pin.out,dry_v.out,dry_c.out}, root/A80_CANDIDATES.json,
  root/A80_TOP_OFFER_CONDITIONS_D2800_p60.json, precheck/{STREAM_PASS.json,mapcheck_a93.py,MAPCHECK.json}.
  Laptop copies are in box3/.

## Queue (written, NOT started): /root/a93queue.sh sha256 c87e8855bc5de8609ca7fb0cbb11a6283381ccaed5d683f084fc25a59d9c5565
The script refuses unless the plans are pinned and not yet launched. It waits for A85Q_DONE in /root/a85queue.log,
then checks whether the a82 chain is busy. Busy means either (a) `ps -eo pid,args | grep '[a]82'` finds a process
whose command line contains "a82", or (b) /root/a82four/chain_confirm.out exists without A82F_CONFIRM_DONE or a
FAILED line. If busy, it writes "A93Q_SKIPPED busy" and exits. Otherwise it runs mech, then admit, then bank, each
once, with 46 workers, no confirm, logging to /root/a93queue.log; the last line is A93Q_DONE.
Caveat: check (a) also matches any unrelated command (for example a monitoring ssh) that mentions an a82 path at that
moment.
Launch (on Box 3):
    sudo -n setsid -f nohup bash /root/a93queue.sh > /root/a93queue.nohup 2>&1 < /dev/null

## Hashes (sha256, laptop)
a93_draw.py ebcf1f17bd42fd897a02a78e22828298c1577756780de525819f349b91b13bdd
mapcheck_a93.py 67a98c6c78fe0d9cc1d554c179fd8a0f708016f0308d6a6f9a6323f8a0043e5b
STREAM_PASS.json b59d29977f82f93d107e9df9a451888993ecd534e89f435a28a4cfac52c16b93
STREAM_DRAWS.json d5455bb0178f59c8ee4175ac4039bba509a9f3af5c1387f0462d0479507de4b5
MAPCHECK.json 098e074c76ecededa2b56c5f9789874f732bca6cdfeb7ae72810d5fb3f24639e
TOP_OFFER_CONDITIONS_D2800_p60.json 238b0cbef482d402bdf0156edd978af88809738ae43f59273aaf6989b6c4c054 (== box top + pinned copy)
DRAWS.json b6d366f7f5e16b03ce657eb311ecfde733a6dd48aa51e81c7f258b68ba963d7d
box3/a93queue.sh c87e8855... ; box3/pin.out c65ab816... ; dry_v.out a35f7647... ; dry_c.out 487c9ae8... ; setup_a93.out 1c0f1f68...
