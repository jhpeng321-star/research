# MAIN08:16/08:25 continuation — 2026-11-03 08:34 UTC+8

- Full brief read through line671; latest eight-entry threshold is applied.
- Threshold code/registration:37723cb8; live watcher publication:0d679b3b.
- Watcher2926173/start68009586/bootb8bc1fcb, immutable replaystartcpu4,
  control`/home/ubuntu/coopmwm20261102_warm1/replay_start_0816_v3`.
- Full Linux qualification:706 PASS/0 skips/85.010s/0 SUMO.
- Actual watcher status:8 admitted/3 cooperative/threshold8. Remaining gates:
  four joint-release GPU slots, then qualified collector boundary activation.
- V5 rollout controller exited on a stale READY owner after gpuR handover.
  Original`LS_NICHENULL_ROLL_V5.log` retained; its last assertion is gpuR@0.
- Fix5295b6ed:refresh host inventories after handover before admission; unchanged
  live identity/gate checks. No numerical/search/executor change.
- Verification:`python -m pytest -q tools/coopmwm/test_niche_roll_cpu.py tools/coopmwm/test_niche_drain_cpu.py tools/coopmwm/test_source_install_cpu.py`
  =>13 passed in0.36s.
- Registered V6 amendment818e14acaf073715d1603518c665da71db4a453b430567af1bdf8c54515c175c.
- New controller10148/create1793665996.30793, expected continue_roll argv verified.
  Existing activation39660/create1793662027.9718082 and observer40668/create1793658957.771496 retained/live.
- Adopted gpuR766170/start4229162208:2/8 gates, aliveR/no failure. Peer623287
  remains READY8/serving. Completed gpuA gates retained. Never duplicate owners.
- Bank owner2388293/start66646196 live:73 usable/35 confirmed/8 admitted/3 cooperative,
  no pending confirmations (BANK_BUILDER_STATUS_1793666039521061900.json).
- unix1793666051 health:STOPnone,A48/B3/R3,charged5582380,disk16464134144B.
- No new physics/reservation/query replay; no B_REPLAY_STARTED receipt yet.
