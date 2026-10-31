# S-SUMOCORE handoff

- Owner task: observable-start SUMO 1.27.1 physics ceiling; branch `c4/sumocore`, base `f0db3c82`.
- Read owner WM-PHYS registration (2026-10-31 23:12), both top-down reviews and WMBREAKDOWN.
- No simulation or inference dispatched yet; cap 100,000 actual sim-s.
- cpuQ: 32 CPUs, load ~1.4 at first check; no GPU. gpuR load ~17/32, GPU free; CPU-only planned.
- Existing WMDATA observable views and exact native/MAPPO/PI wrappers located on cpuQ.
- Tooling: `tools/sumocore/remote.py` is an allowlisted SSH inspection transport.
- No shared model files changed; no switches added. No hidden state/type/parameter files read.
- Next: implement safe observable extraction, state reconstruction, encoder mapping and capped ledger.
- Fresh-session resume: git HEAD f0db3c82, previous HANDOFF/remote.py preserved. No task jobs on cpuQ (32 CPUs; load .08/.74/1.00), no GPU installed; gpuR has no `python3` on SSH PATH, so cpuQ only for now.
- Exact WMDATA wrappers inspected read-only remotely. Existing observer exports causal history and due-queue counts; no original scenario/state/type data needed. Do not call CavSumoEnv.reset (generates hidden driver assignments).
- Shared obsstate exporter not present yet; avoid duplicating helper work. Build independent runner/controller and inference adapters first.
- Check hosts with `python tools/sumocore/check_hosts.py` (gpuR currently reports interpreter failure); no signals, dispatches or sim-s spent.
