# A98: two add-on arms on the A94/97 panel worlds (D2800_p60). Pinned (zero SUMO), NOT launched

Box 3 `/root/a98`, prepared 2026-11-14 03:05-03:12. `/root/a94` was only read: its code, binding files and panel file,
by hash. No A94 outcome file was read. Nothing was committed.

## Package

- **Tree:** a copy of `/root/a94/src`, hash-checked:
  - runner 47fb9f76
  - `a94_bank.py` c8b05145
  - `build_bank_a94.py` c8746b4a
  - `a81_bigconfirm.py` c755a551
- **Driver `ops/a98_addon.py`** (c5e341aa) is a thin wrapper over the unchanged A81 driver and the pure helpers of
  `a94_bank.py`. Bank arms use the exact A94 BANK_SEARCHED executor path: A75_BANK on MAPPO60C / U0, a75 d1af7e0f.
- **Other files:**

| file | sha256 |
|---|---|
| a98_e2e.py | 4c73abca |
| test_a98_addon.py | 1b97e40f |
| run_a98.sh | ce8ca7b5 |
| setup_a98.sh | 69d41f75 |
| `/root/a98queue.sh` | **a733e130** |

- **Panel:** byte copy of the A94 panel. File 720d19e1, panel sha d85999dd, 192 worlds 988700-988891. It uses the A94
  setting and runner block.
- **Banks** (pinned in code; byte copies in `root/a98_banks/`):
  - BANK_S4 = `/root/a85adm/root/a80_banks/A80_BANK_D2800_p60.json`, **a428cd1d…**. 3 plans: deba0680, 576d526e, 28cad0ce.
  - BANK_S4S5 = rebuilt in `/root/a98/src/ops/a98/` with `build_bank_a94.py` in two-search mode from S5 577a0564 + S4
    a428cd1d. Result **40b16ed95add6fcd63385dabcbf7e8a03fe8dbf20d0418e58e5aa8feee30ee04**, confirmed equal to the
    registered value. 6 plans.

## Budget

| stage | episodes | sim-s |
|---|---|---|
| mech | 2 arms × 2 A70 worlds (996300/996301), rule checks only | **4,800** |
| confirm | 2 × 192 × 1,200 | **460,800** |
| total cap (enforced in code) | | **465,600** |

Each stage runs once on 46 workers.

## Analysis

Runs inside the confirm stage, only after A94Q_DONE and A94_CONFIRM_RUN_DONE. Otherwise CHAIN_STOPPED says "A94 not
finished"; run `run_a98.sh analyse` later.

1. It copies the four A94 arms' wave receipts and cached reports into `/root/a98/root/ref94`. Seals, worlds, the
   `a94c_` job-id prefix and the panel are checked. `/root/a94` is only read.
2. Paired contrasts, all secondary, using `D._contrast` (bootstrap 10,000, seed 20261104, five-group rule):
   - BANK_S4 − BANK_HAND
   - BANK_S4S5 − BANK_HAND
   - BANK_S4S5 − BANK_SEARCHED
   - BANK_S4 − BANK_SEARCHED
   - each new arm − DUO_HOLD_60C
   - each new arm − MAPPO60C
3. Per plan: activations, offers (fire / member_busy), the overlap block, and CAVs engaged at once.

## Tests (zero SUMO, charged 0)

- `run_a98.sh test`: 4/4 passed.
- `run_a98.sh drytest`: DRYTEST_OK. It covered:
  - prepare;
  - dry i (4 episodes, 4,800);
  - dry c (384 episodes, 460,800; specs differ only in bank file / path / job ids);
  - e2e: copy_a94 on a fake A94 root built from A91 records, analysis + summary, and rule checks on the A91 mechanics
    records.
- `run_a98.sh pin`: **PIN_OK, charged 0.**

## Launch

```
sudo -n bash -c "setsid -f nohup bash /root/a98queue.sh < /dev/null > /root/a98queue.nohup 2>&1"
```

The queue runs these steps:
1. Wait for A94Q_DONE.
2. Refuse if `/root/a94/CHAIN_STOPPED` exists or there is no A94_CONFIRM_DONE.
3. Wait while a SUMO stage script runs on the box. The patterns are `/root/a94/run_a94.sh`, `/root/a96*/run_a96.sh`,
   `/root/a82four/run_a82.sh` and `/root/a95adm*/run_*`. Queue wait-loops are not counted, so it cannot deadlock.
4. Run mech, then confirm, then analysis, once.
5. End the log `/root/a98queue.log` with `A98Q_DONE`.

## Ordering with the 20 % queues (not edited, just read)

- The `busy()` in `a96q_ref_hand.sh`, `a96q_s4.sh` and `a96q_s5.sh` matches only a94 / a82four / a96 / a96q patterns. It
  will **NOT see a98**, because `run_a98.sh` and `a98queue.sh` match none of them.
- `a96q_ref_hand.sh` also waits only for A94Q_DONE plus the MAPPO20C files. Both queues could therefore start together
  right after A94Q_DONE, and only a98 would hold back:
  - If a96 starts its ref stage first, a98 sees `/root/a96ref/run_a96.sh` and waits (the safe direction).
  - If a98 starts first, a96 runs on top of it.
- **Recommended:** launch a98 now. Launch `a96q_ref_hand.sh` only after `A98Q_DONE` is in `/root/a98queue.log`, then
  s4 / s5, then the a96pd / a96pc panels.
- Alternatively, launch both now and accept that they may share the box.

Expected time: mech about 5 min; confirm 384 episodes is about 32 min (A91: 480 in 40 min on 46 workers); analysis
under 1 min. **About 40 min.**
