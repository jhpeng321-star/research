# A103 S-E1-RAND36 admission prep notes (2026-11-14 ~07:08, prep agent)

Rule: REGISTRATION.md AMENDMENT 103. Zero SUMO, zero WM, zero training. Nothing launched. Box 3 was idle before prep
(load 0.00, A101Q_DONE, no run_a80/nm_eval process); other roots not touched; no process signalled.

## Generator / config evidence (S5 = Amendment 88)
- Generator = a86s5.LaneFilter.random (a86s5.py 71dda3e8) = strategy_search.random_strategy (02e119c1) patched by
  a75ss.install(cfg) (cae54dc1), with exclude_lane_sets [[1],[2],[1,2]]; cfg A86S5_SEARCH_CFG.json 3adc2663 merged with
  SS.DEFAULTS, asserted equal to the 'config' key of a86_s5/STRATEGY_SEARCH_D2800_p60.json (30f10455).
- Proof: `a103_draw.py verify` re-runs a86s5.initial_population (seed 20261187, screen sample of the 18 D2800_p60 states
  74aef8f8 first) and reproduces the S5 generation 0 exactly: 72 random plans, keys == recorded screened[:72] in order,
  key-list sha 275c80a9e53ffe65... (= the S5 box GEN0_OK value), 12 random rejects, 28 seeds dropped (as recorded).
- Only 'config' and plan keys are read from the S5 output; no score/outcome field.

## Draw (laptop, a103_draw.py)
- random.Random(20261115), generation order; the LaneFilter redraw loop is unrolled so each excluded draw is recorded.
- Checks in S5 order: validate; S5 lane-set exclusion; uncond + min_roles 2; banned; duplicate key; then A103 rule: key in
  the S5-evaluated set (167 keys: screened 165 + final + archives + best) or any A93 draw key (40 keys, all 50 A93 draws);
  then zero-SUMO A80 mapping (a80_s2b.map_row) on Box 3 over the first 60 passing plans (mapcheck_a103.py = mapcheck_a93.py
  verbatim apart from the docstring; /root/a103adm_a/precheck/): refused 0 of 60.
- 42 draws used, 6 rejected, all excluded_lane_set: d5 rand_6a0fa7359de324a0 {1,2}; d15 rand_2b15b95cf4906aba {1};
  d24 rand_bcf5dbb83f8c027d {1,2}; d26 rand_b56337c0ef09d091 {2}; d32 rand_99189b7560870324 {1,2};
  d39 rand_930566a6df5e0297 {1,2}. No S5-key, A93-key, duplicate, min_roles, banned or mapping rejects.
- Kept 36 (draw order). A80 forms: gap 11, pulse 11, taper 6, zone 4, rolling 4 (a: gap5 pulse3 taper2 zone1 rolling1;
  b: pulse4 rolling3 taper2 zone2 gap1; c: gap5 pulse4 taper2 zone1). Role-lane sets: L012 23, L01 7, L02 6.
  Roles per plan: 2 x7, 3 x16, 4 x13. Score fields null (as A93).
- key list sha256 (json.dumps of 36 keys) c45f3a3aa8aa6da03bfdb16df5158c6b4083a6796c6ba25e89455620fb7c4c3d;
  TOP_ALL36 7e721bfb62258c699790f93780cb00daced1b70afcfaf80e4ff1f5449014f44d.

## Box 3 roots (same recipe as a93adm / a101adm_*)
`A80_BASE=<B> bash /root/a78s2b_ship/run_a80.sh setup` -> SETUP_OK (policy pin 2abed317, runner ef8788ff, settings
49b6d048); `A80_BASE=<B> A80_TOP_FILE=<B>/top/TOP_OFFER_CONDITIONS_D2800_p60.json A80_TOP_SHA256=<sha> <B>/run_a80.sh pin`.
src tree (1491 files), run_a80.sh 44fbf6a7, env_a80.sh (modulo base) and ref68 equal to /root/a101adm_a.

| root | draws | pin | top sha | mech plans | by_form | doc_sha256 | A80_CANDS seal | A80_CANDIDATES.json | dry v sim (mech) |
|---|---|---|---|---|---|---|---|---|---|
| a103adm_a | 1-12 | PIN_OK | 9ed1a87d... | 12 | gap5 pulse3 rolling1 taper2 zone1 | 796ee055... | 3c73378e... | b32fdda4... | 345,600 (28,800) |
| a103adm_b | 13-24 | PIN_OK | 1aa81eb0... | 11 | gap1 pulse4 rolling3 taper2 zone2 | ec568cc1... | d992a1ce... | 232fbe9d... | 345,600 (26,400) |
| a103adm_c | 25-36 | PIN_OK | 5d92dc9c... | 12 | gap5 pulse4 taper2 zone1 | 21b77bc0... | 56e517b8... | 0a0bf5a6... | 345,600 (28,800) |

All dry v: 12 waves, worlds 992500-992523, charged 0, root cap 460,800, panel 989248f9, reference A68V_MAPPO60C receipt
1a6d31dd reports_ok. dry c charged 0. No CHARGES.jsonl. Total dry-v sim 1,036,800 (mech inside) = registered cap.

## Queue (written, NOT started): /root/a103q.sh sha256 4d70940745191df90cf38c47fc16cf66f8a18d7939780e74a05af5326fd4ae20
= a101q.sh with a101->a103 (only comment lines differ otherwise); bash -n OK. Refuses if any root unpinned or launched.
a -> b -> c; mech && admit && bank once each, A80_WORKERS=46; failed root logs rc, queue continues; log /root/a103q.log;
end marker A103Q_DONE.
Launch: `sudo -n setsid -f nohup bash /root/a103q.sh > /root/a103q.nohup 2>&1 < /dev/null` (redirection must run as root:
`ssh ... "sudo -n bash -c 'setsid -f nohup bash /root/a103q.sh > /root/a103q.nohup 2>&1 < /dev/null'"`).

## Deviations / disclosures
- Pinned docs carry the fixed A80 strings stratum='wm_top' / world_model='WM search file'; wrong for random draws (labels only).
- Three roots (12-plan A80 cap per root).
- A93-key exclusion uses all 40 keys in A93 STREAM_DRAWS.json (superset of the 12 A93 kept plans).
- Root b has 11 mechanics plans (one plan fully covered by existing tokens), so mech sim differs per root.

## Hashes (sha256)
a103_draw.py 4a858193f67d613f54397298a3dc292bc044297cfb4dd464aa633850b6b94476
mapcheck_a103.py fda95fca170d119b52865334249b58def2ac2b9068ed773af7b7d67d38ec7ec0
STREAM_PASS.json 70aaf8390cf8a056ab9b30cb4f982bf893a7ec046d069beb0808c03b5bc2f7ec
STREAM_DRAWS.json acdedaa229bf33ead3547a8b6055faad5c4248eae172018a0fcaf824d160fd5b
MAPCHECK.json b9ce257d0e734baa20361311ca50441043d7e2a22e453f9e2b1356758e8e197c
DRAWS.json 44152bb57651315819c265aadfc4b38be5f6b2d6ee7c78ec06cf551f04fd41e4
TOP_OFFER_CONDITIONS_A103A 9ed1a87da644ddcb3a9d156bdaf2f064e4df14b6b99d23ec81ac23a53a2b1890
TOP_OFFER_CONDITIONS_A103B 1aa81eb0f81335d55f04ddcdf21de598001898a62093dda8b45128a144ba45dd
TOP_OFFER_CONDITIONS_A103C 5d92dc9ca4227cf3b4c6b80dd281ced478b0963bf43ec96f9b7f28abcbd3daba
