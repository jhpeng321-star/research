"""WM-C: world-model PROPOSER for SUMO-verified cooperation-plan mining on the wz3to2 lane drop.

The world model proposes, SUMO decides. At a bottleneck moment the proposer searches lane-drop
cooperation moves (abreast walls, closing-lane pairs, executable N9 families) over members, timing
and strength, ranks them by WM-predicted synergy (group minus best solo, paired per WM draw, with
the draw spread as uncertainty) and hands the top-k to the unchanged N3/PLANGEN verification funnel
(per-seed 3 replicates, later-harm check, safety) through the plug-in.

    geometry.py      wz3to2 constants, WM-frame projection, seed/path guard
    families.py      candidate moves (members x timing x strength) in N3 variant format
    wm_backend.py    WM contract; StubWM (CPU, planted) and the WM-A binding (WMA_READY.json)
    proposer.py      staged WM search, uncertainty-aware ranking, diversity, optional W3 re-rank
    n3.py            the vendored N3/PLANGEN pipeline (n3_pipeline/, sha-pinned), bound to a root
    verify_plugin.py moments/natives/screen/solos/rep1/rep2/analyze stages for WM proposals
    runner.py        precharge-ledger job runner (pg_run semantics, configurable root/core)
    compare.py       matched-effort comparison: WM vs structured vs random proposals
    wallfit_bridge.py  W3 head as optional re-ranker + wz3to2 label builder for training it later
    fakes.py         planted fake world + fake SUMO core (CPU dry runs and tests; no SUMO)

Sign convention (veh-s, J = road speed deficit + source waiting): advantage T = J(base) - J(group);
synergy S = min_i J(solo_i) - J(group); margin M = min(J(base), min_i J(solo_i)) - J(group).
Positive = the group is better. World configuration labels (cell/demand/penetration/world seed)
are never proposer inputs.
"""
