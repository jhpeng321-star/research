# S-RB-INT handoff

Updated 2026-10-31 23:17 +08:00; branch `c4/rb-int`, base `f0db3c82`.

## Status
- Read owner authorization at active C4 registration line 5682 and all four WHY reports.
- Integrator worktree initially clean. No physics branches merged yet.
- Initial poll: `c4-rb-{lane,merge,cf,eval}/HANDOFF.md` absent; SUMOCORE handoff absent.
- CPU: 22 logical processors, initial load 40%; local NVIDIA driver unavailable.
- No training, downloaded code, scientific rollout, or peer-process action performed.

## Integration contract for workers
- Please commit usable increments and state exact commit, switch names, test command, and known gaps here in your own handoff.
- Target A/B: SUMO IDM + LC2013 + SUMO merge/insertion; every learned motion/lane/merge correction OFF.
- A keeps the existing motion-history encoder settings; B uses population defaults, including new entrants.
- Loading the existing checkpoint must preserve encoder weights; switches must not silently substitute an untrained encoder.
- Shared-file changes stay opt-in/default-OFF. Existing checkpoint behavior must remain unchanged without the switches.
- Evaluation must bind a fixed permitted FIT panel before scoring; never open prohibited paths or truth assignments.
- Need finite/conserved 60 s rollouts from two recorded FIT starts before the registered A/B distributional panel.

## Tests / results
- None yet. Next: establish available Python/runtime and integration verification tools.

## Known gaps
- Worker APIs and registered evaluation panel are not yet available at first poll.
- No measured WM-PHYS efficacy; historical WM2 statistics are references, not matched-panel results.
