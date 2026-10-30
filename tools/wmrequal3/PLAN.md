# S-WMREQUAL3 — 2026-10-30

Read WMREQUAL2.md and wmrequal2/AUDIT.md first. Base 3f40ba6c, branch
c4/wmrequal3. No push, deletion, SUMO, training, assignment/truth files or excluded
paths. All remote input opens guarded; owned output is wmrequal3_20261030.

1. Default-OFF scorer native timeline ownership repair. Preserve the historical
   scorer path exactly OFF, including both-mix default. Native masks combine
   originals and entrants. Policy callbacks retain explicit crossings. CPU checks
   compare frozen OFF and exercise the 44 validation original ramp CAVs.
2. Frozen checkpoint/manifest contract: ours_c1 (64f70af4) and coopoff_c1
   (e1fa4881) are residual_on_native, lane_control=native, no lane head. Both are
   eligible, 72 contrasts / 36 worlds. MAPPO (aecd7d70) has a lane head and is
   diagnostic only: original policy lanes are absent from the packed timeline.
3. WM2 b8797242, K8, mix ON only, 600 steps, all-fleet road + source waiting.
   Native reference gets original and entrant native lanes. Retain MAPPO
   separately without claiming execution-matched lane-policy accuracy.
4. gpuA only after fresh nvidia-smi room check; total owned usage <=12 GB,
   allocator limit reserves context headroom. Otherwise gpuR <=6 GB, else wait
   >=600 seconds. Pilot processes are never inspected through /proc or signalled.
5. Commit CPU repair, complete raw panel evidence, then analysis/report. Report
   <=60 lines, same horizon metrics/calls/additive decomposition as WMREQUAL2.
   Unknown perturbation sigma remains unknown; no new policy authority implied.
