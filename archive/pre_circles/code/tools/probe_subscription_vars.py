"""Probe: which vehicle variables are subscribable in SUMO 1.27, and do
subscription results match the direct getters at the same simulation state?

Run: conda run -n rl_driving python -u tools/probe_subscription_vars.py
"""

import sys

sys.path.insert(0, ".")

import traci
import traci.constants as tc

from config.defaults import merge_config
from env.sumo_utils import close_sumo, start_sumo

cfg = merge_config({"sumo_config": "osm.medium.sumocfg"})

CANDIDATES = {
    "VAR_SPEED": tc.VAR_SPEED,
    "VAR_ACCELERATION": tc.VAR_ACCELERATION,
    "VAR_ROAD_ID": tc.VAR_ROAD_ID,
    "VAR_LANE_ID": tc.VAR_LANE_ID,
    "VAR_LANE_INDEX": tc.VAR_LANE_INDEX,
    "VAR_LANEPOSITION": tc.VAR_LANEPOSITION,
    "VAR_POSITION": tc.VAR_POSITION,
    "VAR_TIMELOSS": tc.VAR_TIMELOSS,
    "VAR_TYPE": tc.VAR_TYPE,
    "VAR_SIGNALS": tc.VAR_SIGNALS,
    "VAR_LEADER": tc.VAR_LEADER,
    "VAR_FOLLOWER": tc.VAR_FOLLOWER,
    "VAR_LENGTH": tc.VAR_LENGTH,
    "VAR_MINGAP": tc.VAR_MINGAP,
    "VAR_ANGLE": tc.VAR_ANGLE,
}

port = start_sumo(
    sumo_binary=cfg["sumo_binary"],
    sumo_config=cfg["sumo_config"],
    project_root=cfg["project_root"],
    port=None,
)

# Warm up until vehicles exist.
for _ in range(120):
    traci.simulationStep(traci.simulation.getTime() + 1.0)
    ids = list(traci.vehicle.getIDList())
    if len(ids) >= 3:
        break

ids = list(traci.vehicle.getIDList())[:3]
print(f"vehicles: {ids}")

# Subscribe each candidate separately to see which ones fail.
ok_vars = {}
for name, var in CANDIDATES.items():
    if var is None:
        print(f"{name}: constant missing in traci.constants")
        continue
    try:
        traci.vehicle.subscribe(ids[0], [var])
        ok_vars[name] = var
        print(f"{name} (0x{var:02x}): subscribe OK")
    except Exception as exc:
        print(f"{name} (0x{var:02x}): subscribe FAILED: {exc}")

# Step once and read all subscription results.
traci.simulationStep(traci.simulation.getTime() + 1.0)
results = traci.vehicle.getAllSubscriptionResults()
print("\n-- getAllSubscriptionResults for probe vehicle --")
for k, v in sorted(results.get(ids[0], {}).items()):
    print(f"  0x{k:02x}: {v!r}")

# Compare subscription values with direct getters.
print("\n-- direct vs subscription --")
vid = ids[0]
checks = [
    ("speed", tc.VAR_SPEED, lambda: traci.vehicle.getSpeed(vid)),
    ("road", tc.VAR_ROAD_ID, lambda: traci.vehicle.getRoadID(vid)),
    ("lane", tc.VAR_LANE_ID, lambda: traci.vehicle.getLaneID(vid)),
    ("laneidx", tc.VAR_LANE_INDEX, lambda: traci.vehicle.getLaneIndex(vid)),
    ("lanepos", tc.VAR_LANEPOSITION, lambda: traci.vehicle.getLanePosition(vid)),
    ("pos", tc.VAR_POSITION, lambda: traci.vehicle.getPosition(vid)),
    ("timeloss", tc.VAR_TIMELOSS, lambda: traci.vehicle.getTimeLoss(vid)),
    ("type", tc.VAR_TYPE, lambda: traci.vehicle.getTypeID(vid)),
    ("signals", tc.VAR_SIGNALS, lambda: traci.vehicle.getSignals(vid)),
    ("accel", tc.VAR_ACCELERATION, lambda: traci.vehicle.getAcceleration(vid)),
    ("length", tc.VAR_LENGTH, lambda: traci.vehicle.getLength(vid)),
    ("mingap", tc.VAR_MINGAP, lambda: traci.vehicle.getMinGap(vid)),
]
sub = results.get(vid, {})
for name, var, getter in checks:
    if var is None:
        continue
    try:
        direct = getter()
    except Exception as exc:
        direct = f"ERR {exc}"
    print(f"  {name}: direct={direct!r}  sub={sub.get(var, 'N/A')!r}")

# VAR_LEADER comparison: subscription (if OK) vs getLeader(vid, 200).
if "VAR_LEADER" in ok_vars:
    try:
        direct_leader = traci.vehicle.getLeader(vid, 200.0)
    except Exception as exc:
        direct_leader = f"ERR {exc}"
    print(f"  leader: direct(200m)={direct_leader!r}  sub={sub.get(tc.VAR_LEADER, 'N/A')!r}")

# Arrived vehicles: what happens to their subscriptions?
print("\n-- departed/arrived handling --")
all_subbed = list(traci.vehicle.getIDList())
for v in all_subbed:
    try:
        traci.vehicle.subscribe(v, [tc.VAR_SPEED])
    except Exception as exc:
        print(f"  subscribe {v}: {exc}")
for _ in range(5):
    traci.simulationStep(traci.simulation.getTime() + 1.0)
res = traci.vehicle.getAllSubscriptionResults()
live = set(traci.vehicle.getIDList())
arrived = set(traci.simulation.getArrivedIDList())
print(f"  live={len(live)} arrived_last_step={sorted(arrived)}")
print(f"  sub results keys={len(res)}; arrived-in-results={sorted(arrived & set(res))}")
print(f"  results-not-live={sorted(set(res) - live)[:5]}")

close_sumo()
print("PROBE DONE")
