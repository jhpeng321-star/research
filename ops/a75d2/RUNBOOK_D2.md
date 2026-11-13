# A75 stage D2 (S-HOLDFORMS-D2) - box RUNBOOK (built 2026-11-13; NOTHING run)

New A75 hold forms (gap-start, short, custom/mid/merge zones, pulsed, tapered, rolling) tested directly in SUMO at
D2800_p60 on MAPPO60C. Driver `ops/a75_d2forms.py`; executor `a75_holdforms.py` (a75_digest `d1af7e0f...`, commit
921a850e); runner arms added on the box by `patch_runner_a75.py` + seed block by `patch_tables_d2.py`.

**v2 (2026-11-13, after the v1 setup failed on Box 3 with zero SUMO):** v1 cloned `/root/a68/src`, which has no
`a44_lcvariant.py` and no A44 runner (the A75 executor and runner patch build on both). v2 clones `/root/a69/src`
(= A68 tree + A44 files + merged runner 62496e86, pinned) into the NEW base `/root/a75d2b` (the partial v1
`/root/a75d2` is never touched). Runner and `ops/nm_finaleval.py` get new inodes and additive patches: the registered
A70 table / a70c setting edit (needed by confirm, same edit as D1's confirm tree), A75 arms, D2 block; the expected D2
runner sha for 997400 is `ec095e50...` (checked off-box: equal to the HEAD runner except the seed tables). Plus
`a70_smallbank.py` (72af622e) and the A70 panel (835e423a). Worlds, panels, bindings and the stored A68 MAPPO60C
reference are unchanged; the mech identity stage still proves A75_PLAN on a plain plan == the stored A68 record.
**v3:** new base `/root/a75d2c` (v2 `/root/a75d2b` untouched). Setup hard-links every file of `/root/a68/src` missing
from the A69 clone (A68 run-time files such as `ops/a68/CANDIDATES_D2800_p60.json`, list in `A68_SYNC.list`), requires
every shared file byte-identical (runner excepted), pins the A68 candidates (json bff1a7bb, pin be1a5df3, doc 194610bd)
and checks `A68.candidates()` loads. Test file: the synthetic-traffic helpers are vendored (v2's lazy import made pytest
skip the whole file during fixture discovery on a tree without test_a44_lcvariant_cpu); `test` runs with `-rs`.
Also v2: drop-raw pointer check = sha256 + size of the DECOMPRESSED RESULT (D1 fix 498f273c).

## Before launch (coordinator)
1. Register the mechanics block: `A75D2_MECH_SEED0=997400` proposed (SEED_AUDIT.json; never 964000-964199, 962xxx,
   977xxx, 989600-989699). Unset -> setup and every stage refuse.
2. Record the candidate doc sha `9025e3b1...` and bundle sha (below) in the registration.
3. Box needs: `/root/a69` (completed A69 setup), `/root/a68` (completed A68 setup), `/root/a66` (MAPPO60C binding), the D1 install `/root/a75/ref68`
   (or `ref68_reports.tar` as setup arg). Confirm stage also needs the A70 references (default D1 install paths
   `/root/a75/a70ref/root`, `/root/a75/ref70`; override `A75D2_A70_ROOT`, `A75D2_REF70`).

## Commands (as root; each stage runs ONCE: its chain file exists -> refuse)
```
mkdir -p /root/a75d2c_ship && tar -xzf a75d2_ship_v3.tgz -C /root/a75d2c_ship && cd /root/a75d2c_ship
sha256sum -c SHIP_FILES.sha256
export A75D2_MECH_SEED0=997400 A75D2_WORKERS=46          # workers: box cores - 2
bash run_a75d2.sh setup [ref68_reports.tar]               # zero SUMO, ~1 min, writes /root/a75d2c only
B=/root/a75d2c/run_a75d2.sh
$B test                                                   # unit tests (expect 25 passed, 0 skipped), ~1 min
$B dry                                                    # banks, mech panel (997400-997403), dry i/m/v + reference binding
setsid nohup $B mech    > /root/a75d2c/nohup_mech.out 2>&1 &     # 20 ep, 24,000 sim-s
setsid nohup $B admit   > /root/a75d2c/nohup_admit.out 2>&1 &    # 864 ep, 1,036,800 sim-s (needs MECH PASS)
$B bank                                                   # zero physics: admission + bank rule, < 2 min
setsid nohup $B confirm > /root/a75d2c/nohup_confirm.out 2>&1 &  # 96 ep, 115,200 sim-s + analyse
```
Progress: `tail /root/a75d2c/chain_<stage>.out`; status: `cd /root/a75d2c/src && $PY ops/a75_d2forms.py status --stage
v --final-root /root/a75d2c/root ...`. Stop between rounds: `touch /root/a75d2c/root/STOP_A75D2`.

## Durations at ~13,000 sim-s/min (throughput-bound; mech is parallelism-bound)
| stage | episodes | sim-s (cap, enforced in code) | expected |
|---|---|---|---|
| mech | 4 identity + 16 forms | 24,000 | 2 min of throughput; wall ~ 2 episode lengths (4 then 16 parallel), ~15-25 min |
| admit | 36 x 24 | 1,036,800 | ~80 min + drop-raw per round |
| bank | 0 | 0 | < 2 min |
| confirm | 96 | 115,200 | ~9 min + analyse ~5 min |

## Rules fixed before outcomes (texts in the driver; sealed into every receipt)
- Draw: DRAW_RULE (seed 20261113, stream a75d2-holdforms); 36 plans: gap 12 (4 alone, 4 +1 role, 4 +2 roles),
  short 6, zone 6 (2 mid, 2 merge, 2 custom), pulse 4, taper 4, rolling 4; roles 1-4.
- Mechanics: MECH_RULE (A75D2_MECH.json PASS/FAIL; FAIL = stop). Deviations from MECH_FIXTURE.md: identity runs on
  the first 2 A68 admission worlds (so the stored A68 record is also bound); F2 gap uses 4 p60 worlds (no p20 plant
  in this root). Same 20 episodes / 24,000 sim-s.
- Admission: ADMIT_RULE = A71/D1 rule unchanged (plan - MAPPO60C CI95 upper < 0, bootstrap 10,000 seed 20261104,
  AND five group means <= +10 s); reference = stored A68V_MAPPO60C (pinned, not re-run).
- Bank: BANK_RULE = D1 rule with the hold_s class replaced by the form family (duplicates excluded; best admitted per
  family by CI95 upper; best 4 of those by CI95 upper; priority lb95 = -CI95 upper).
- Confirm: PRIMARY_RULE A75D2_BANK_60C - MAPPO60C on the A70 96 worlds; vs DUO_HOLD_60C descriptive. Outputs: delay
  diff + CI, worst-10% delay (CVaR90), group means, hard braking, per-form activation counts.

## Disk
Default: after each dispatch round, every published admit/confirm episode gets its per-vehicle report written and
re-verified, a sealed DROPPED_RAW.json, then RESULT.json.gz / vehicles.jsonl.gz / observable.jsonl.gz of THAT episode
are removed (only `a75d2v_*` / `a75d2c_*` dirs directly under /root/a75d2c/root/evaluation). Peak ~2-3 GB (one round
of 96 episodes). Mechanics episodes are never dropped (~0.4 GB). `A75D2_KEEP_RAW=1` disables (needs ~20 GB/1,000 ep).

## sha256 (files in the bundle; SHIP_FILES.sha256 is authoritative)
see SHIP_FILES.sha256; candidate doc_sha256 9025e3b1b09c7f05c99621f16e99a905098080b529bfa5e9e4bb8ec5364dae8d.
