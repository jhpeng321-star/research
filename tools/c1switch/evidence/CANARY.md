# Native restore gate PASS

TRAIN world969100, p=.05, demand1900; native warmup300 + live90 + restored90.
480 precharged = 480 actual sim-s. Own gpuR PID523094 exited normally.
Full two-physics-step history tape digest is identical:
`571a4f28e738e1a080d4bba1109054d726e11984f43354560fbf047f8cf40589`.
Both paths: team2941.159261686626 = road2675.0010976866265 + source266.15816400000006
veh-s, collisions0, teleports0, TTC<3 exposure67 vehicle-seconds, hard braking2 vehicle-seconds.
Exact metrics/tapes/ledger in CANARY_STATUS.json; no failed simulator attempt.
TraCI's two normal connection-start `Retrying in 1 seconds` messages are not job retries.
No alternate RNG seed was used. This task-specific state passes; it does not
invalidate C1FLOOR's separately observed native restore failure on another world.
Step0 native repeats at all12 worlds remain required; no WM checkpoint was chosen.
