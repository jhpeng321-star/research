# Recovery and owner17:55 hold — verified 2026-11-01 18:21 +08

Read HANDOFF.md's top section; all older dispatch instructions are superseded.
No new GPU admission until MAIN explicitly appends `S-RB2-INT RESUME`.
Do not start another coordinator, collector or interim publisher.

## Existing live ownership

- gpuA CPU admission513448/start890664618 and late514854/start890706569 bothT.
- gpuR CPU admission948148/start4215302457 and late949247/start4215344299 bothT.
- Own evaluate v5_969439/517852 and wm2_969554/952289 remainR at18:21. They drain
  naturally; all claims preserved. v5_969435/514556 already finished naturally.
- Sole collector25240/create134379970282244431 LIVE; original Nov2 09:40 deadline.
- Sole cache-only publisher23664/create134380020249390235 LIVE; original Nov2
  09:55 deadline. Exact previous publisher44988 exited with the expected source
  guard STOP before successor takeover. Old late waiter9724 also safely EXITED.
- `runs/rb2int_interim_hold_v1/PROCESS.json` and START bind successor; exclusive
  status receipt `OWNER_1755_STATUS01.json` binds all local and remote incarnations.
- No GPU process/evaluate/peer signal or inference dispatch in this recovery.
  Only exact owned CPU admission/standby pidfds receivedSIGSTOP. No automatic resume.

## Current scientific deliverable

Published report `docs/paper_results/WMPHYS_V5.md`, interim commitb9aaed90.
Eight matched worlds, four replicated cells, allfivearms/K8, not the reserved8.
v5−v4 delay CRPS120/300/600+240/+1350/+4461veh-s;
simultaneous95% intervals[118.5,361.4]/[533.0,2167.0]/[2613.3,6307.7].
HDV60+0.5m,CI[-4.5,5.5];300s rate-distance+2.50pp,CI[-22.54,21.42].
0.8calls v4win44.4/57.1/58.3% vs forecast92.4/90.2/93.8%;v5win62.5/70.0/77.8%
vs forecast90.6/92.5/93.1%. Range coverage v4=37.5/57.5/72.5%,v5=25/50/67.5%.
Detailed simultaneous intervals/controller signs/short bounds are in the report.
Short total-team signs remain partially identified, not replaced by road signs.
No final gate, switching, or cooperation/planner qualification is established.

At18:21 28/216new worldjobs collected,10matched v5/v4worlds;8are in currently
replicated cells and two are excluded from that scoped CI view. The final58/
primary54/fourmodel plan and modelb2df7458 remain unchanged, never outcome-selected.
First successor snapshot is complete, not a claim that reserved8 is complete.
New publisher triggers immediately on full reserved8 and on verified collection
of all three original hold-time children, then approximately3h per view.
The collector alone performs incremental verified collection every20min.

## Verification and receipts

Fresh-session git/status clean05d9b3f0 preserved; all four current worker handoffs,
common rules, morning/noon credibility evidence and registered design read.
Current worker adverse development evidence stays disclosed; no new model change.
CPU15interim/ownership tests PASS1.72s;21hold/collector/CI/credibility tests PASS1.12s.
Initial new-test import-path collection error was corrected before all21passed.
Ruff PASS with existing E402/E702 scripting exceptions. Registration line added
to shared owner registry; shared FULL-v5 READY note updated with hold and limits.
