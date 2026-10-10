"""Probe: parameterized VAR_LEADER subscription vs direct getLeader(vid, 200).

TraCI's vehicle.subscribe accepts a ``parameters`` map for variables that
need extra arguments (VAR_LEADER needs the lookahead dist). This script
subscribes every vehicle with VAR_LEADER + dist=200.0 and compares the
subscription result against traci.vehicle.getLeader(vid, 200.0) at the same
simulation state for every vehicle over many dense-traffic steps.

Run: conda run -n rl_driving python -u tools/probe_leader_subscription.py
"""

import sys

sys.path.insert(0, ".")

import traci
import traci.constants as tc

from config.defaults import merge_config
from env.sumo_utils import close_sumo, start_sumo

cfg = merge_config({"sumo_config": "osm.dense.sumocfg"})
port = start_sumo(
    sumo_binary=cfg["sumo_binary"],
    sumo_config=cfg["sumo_config"],
    project_root=cfg["project_root"],
    port=None,
)

LOOKAHEAD = 200.0
subscribed = set()
mismatches = 0
comparisons = 0
none_mismatch = 0
examples = []

for step in range(120):
    traci.simulationStep(traci.simulation.getTime() + 1.0)
    for vid in traci.simulation.getDepartedIDList():
        if vid not in subscribed:
            try:
                traci.vehicle.subscribe(
                    vid,
                    [tc.VAR_LEADER, tc.VAR_SPEED],
                    parameters={tc.VAR_LEADER: ("d", LOOKAHEAD)},
                )
                subscribed.add(vid)
            except Exception as exc:
                print(f"subscribe failed for {vid}: {exc}")
    results = traci.vehicle.getAllSubscriptionResults()
    for vid in traci.vehicle.getIDList():
        rec = results.get(vid)
        if rec is None or tc.VAR_LEADER not in rec:
            continue  # subscribed this step; results start next step
        sub_leader = rec[tc.VAR_LEADER]
        try:
            direct = traci.vehicle.getLeader(vid, LOOKAHEAD)
        except Exception as exc:
            direct = f"ERR {exc}"
        comparisons += 1
        # Normalize: legacy None vs ("", -1)
        sub_norm = None if sub_leader in (None, ("", -1), ("", -1.0)) else tuple(sub_leader)
        dir_norm = None if direct in (None, ("", -1), ("", -1.0)) else direct
        if sub_norm is None and dir_norm is None:
            continue
        if sub_norm is None or dir_norm is None:
            none_mismatch += 1
            if len(examples) < 8:
                examples.append((step, vid, sub_norm, dir_norm))
            continue
        sid, sgap = sub_norm
        did, dgap = dir_norm
        if sid != did or abs(sgap - dgap) > 1e-9:
            mismatches += 1
            if len(examples) < 8:
                examples.append((step, vid, sub_norm, dir_norm))

print(f"steps=120 vehicles_subscribed={len(subscribed)}")
print(f"comparisons={comparisons} none_mismatch={none_mismatch} value_mismatch={mismatches}")
for e in examples:
    print("  example:", e)
close_sumo()
print("PROBE DONE")
