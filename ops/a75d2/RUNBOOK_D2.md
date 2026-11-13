# A75 stage D2 (S-HOLDFORMS-D2) - box RUNBOOK (built 2026-11-13; NOTHING run)

New A75 hold forms (gap-start, short, custom/mid/merge zones, pulsed, tapered, rolling) tested directly in SUMO at
D2800_p60 on MAPPO60C. Driver `ops/a75_d2forms.py`; executor `a75_holdforms.py` (a75_digest `d1af7e0f...`, commit
921a850e); runner arms added on the box by `patch_runner_a75.py` + seed block by `patch_tables_d2.py`.

## Before launch (coordinator)
1. Register the mechanics block: `A75D2_MECH_SEED0=997400` proposed (SEED_AUDIT.json; never 964000-964199, 962xxx,
   977xxx, 989600-989699). Unset -> setup and every stage refuse.
2. Record the candidate doc sha `9025e3b1...` and bundle sha (below) in the registration.
3. Box needs: `/root/a68` (completed A68 setup), `/root/a66` (MAPPO60C binding), the D1 install `/root/a75/ref68`
   (or `ref68_reports.tar` as setup arg). Confirm stage also needs the A70 references (default D1 install paths
   `/root/a75/a70ref/root`, `/root/a75/ref70`; override `A75D2_A70_ROOT`, `A75D2_REF70`).

## Commands (as root; each stage runs ONCE: its chain file exists -> refuse)
```
mkdir -p /root/a75d2_ship && tar -xzf a75d2_ship.tgz -C /root/a75d2_ship && cd /root/a75d2_ship
sha256sum -c SHIP_FILES.sha256
export A75D2_MECH_SEED0=997400 A75D2_WORKERS=46          # workers: box cores - 2
bash run_a75d2.sh setup [ref68_reports.tar]               # zero SUMO, ~1 min, writes /root/a75d2 only
B=/root/a75d2/run_a75d2.sh
$B test                                                   # unit tests (expect 23 passed), ~1 min
$B dry                                                    # banks, mech panel (997400-997403), dry i/m/v + reference binding
setsid nohup $B mech    > /root/a75d2/nohup_mech.out 2>&1 &     # 20 ep, 24,000 sim-s
setsid nohup $B admit   > /root/a75d2/nohup_admit.out 2>&1 &    # 864 ep, 1,036,800 sim-s (needs MECH PASS)
$B bank                                                   # zero physics: admission + bank rule, < 2 min
setsid nohup $B confirm > /root/a75d2/nohup_confirm.out 2>&1 &  # 96 ep, 115,200 sim-s + analyse
```
Progress: `tail /root/a75d2/chain_<stage>.out`; status: `cd /root/a75d2/src && $PY ops/a75_d2forms.py status --stage
v --final-root /root/a75d2/root ...`. Stop between rounds: `touch /root/a75d2/root/STOP_A75D2`.

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
are removed (only `a75d2v_*` / `a75d2c_*` dirs directly under /root/a75d2/root/evaluation). Peak ~2-3 GB (one round
of 96 episodes). Mechanics episodes are never dropped (~0.4 GB). `A75D2_KEEP_RAW=1` disables (needs ~20 GB/1,000 ep).

## sha256 (files in the bundle; SHIP_FILES.sha256 is authoritative)
see SHIP_FILES.sha256; candidate doc_sha256 9025e3b1b09c7f05c99621f16e99a905098080b529bfa5e9e4bb8ec5364dae8d.
