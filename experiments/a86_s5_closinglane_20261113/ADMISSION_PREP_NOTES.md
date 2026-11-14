# A86 S5 / A85 S4 SUMO admission prep on Box 3 (2026-11-14 ~00:00 UTC+8)

Prep agent. Zero SUMO run. Nothing launched. /root/a89 not read or touched; no process signalled.

## Build (identical to /root/a80n1)
- Both bases built with the unchanged A80 ship: `A80_BASE=<base> bash /root/a78s2b_ship/run_a80.sh setup`
  (ship SHA256SUMS verified first; setup_a80.sh 2d4a24c5, run_a80.sh 44fbf6a7). Logs: /root/a86adm/setup_a86.out,
  /root/a85adm/setup_a85.out. Hard-link clones, 54 MB each; 31 GB free after.
- Source tree hash (sha256 of sorted per-file sha256, excl. __pycache__ / WM_PACKAGE symlink; 1491 files):
  2cffea4f... for /root/a80n1, /root/a86adm and /root/a85adm (equal). WM_PACKAGE symlink target equal.
- Equal to a80n1 byte for byte: run_a80.sh, a68_files.list, A68_SYNC.list, driver ops/a80_s2b.py 48c5c448,
  runner ef8788ff, settings 49b6d048, panels (a68c 91f48bc8, a68v trainblock 5caece78), ref68 (24 MAPPO60C reports),
  root binding mirror; env_a80.sh equal modulo the base path.

## S5 pin (/root/a86adm), zero SUMO
- List: /root/a86adm/top/TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json sha 2209791012e41ce35f184cd4c155632e65bdb7b59442495f005c5a76b0fbf6fb
  (laptop == box; pinned copy root/A80_TOP_OFFER_CONDITIONS_D2800_p60.json same sha).
- pin-top accepted all 12 (no MAPPING_REFUSED). doc_sha256 ff91796e3d9d3fa4924cb395ad77c71bc8283588af993e2d112fe28a93b5ad73;
  A80_CANDS seal d0e0548b07a29cc547682f0a624b8ddfa8e176bb55a899df0f8346ba6da12dfb; A80_CANDIDATES.json file sha cea459f9020a264ef229e2f527b9c3f783168433d04eaa238672a8339b8eb716.
  by_form gap 2, pulse 2, rolling 3, taper 3, zone 2; n_extended 10; mechanics plans 10 (all but mut_06996649, rand_cf50c22e).
- dry v: 12 waves x 24 worlds 992500-992523, sim 345,600, charged 0, panel 989248f9..., reference A68V_MAPPO60C receipt
  1a6d31dd..., reports_ok. mech 10 waves, 24,000 sim-s (inside the 345,600). dry c: charged 0 (no confirm). No CHARGES.jsonl.
- Duplicate check by strategy key against the TOP files of a78s2b (A80), a80e1r, a80e1b, a80n1, a80n2, a82p20: 0/12 seen before.

## Queue scripts (written, NOT started)
- /root/a86queue.sh sha f3609f6e...: refuses if unpinned/launched; waits for A89Q_DONE in /root/a89queue.log; mech->admit->bank, 46 workers; log /root/a86queue.log, ends A86Q_DONE.
- /root/a85queue.sh sha 8d1a245a...: refuses unless /root/a85adm/root/A80_CANDIDATES.json exists (checked at start and after the wait); waits for A89Q_DONE and, if /root/a86queue.log exists, A86Q_DONE.
- Laptop copies: box3_adm/.

## S4 pin (to run when the file lands; S4 SUMO list = out_s4/TOP_OFFER_CONDITIONS_D2800_p60.json)
    scp <file> to /tmp, then sudo cp to /root/a85adm/top/TOP_OFFER_CONDITIONS_D2800_p60.json, then
    cd /root/a85adm && A80_BASE=/root/a85adm A80_TOP_FILE=/root/a85adm/top/TOP_OFFER_CONDITIONS_D2800_p60.json A80_TOP_SHA256=<sha> bash /root/a85adm/run_a80.sh pin
