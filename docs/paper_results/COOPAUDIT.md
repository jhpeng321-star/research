# COOPAUDIT — measured cooperation in real SUMO

S-COOPMEASURE; owner2026-10-31 15:45/17:20. Status: ARTIFACT HOLD; dispatch paused.
No complete-panel controller cooperation estimates or95% CIs exist yet.
D(S)=J(native)-J(S); C_G=D_GROUP-sum_i D_ONLY:i; positive is beneficial interaction.

## Frozen audit
- 28 fresh TRAIN worlds:969650-969677;24 in .2/.4 x2400/2700 plus4 coverage worlds.
- Canary969699; both frozen seed guards PASS29/29 cases on cpuQ and gpuR.
- Primary: MAPPO speed-only iter169 d9042cc5 + wall-v2.1 closure + S-SUMOCF trained policy when available.
- Native/IDM means no controller acceleration request, native lane choice, and native release after window.
- Exact saved-state/RNG branches, stable identities and masked controller memory; all branches continue natively after intervention.
- Old grid6 binding superseded: raw-owner-rule winner grid26/all-zone/60s is under artifact investigation.
- Group candidates include all observable-zone CAVs; no artificial four-member cap. Zone s[-150,75.44]m.
- Team fleet+source J300/J600 and merge-local J; raw C_G, D_G, sumD_i, shareC_G>0.
- 95% stratified world-bootstrap CIs:5000draws, seed20261031; tiny groups retained, missing never zero.
- Full-panel census/cap admission precedes branches; a panel exceeding500000sim-s stops without truncation.

## Real-SUMO mechanics canary (not panel inference)
- Live native / loaded B / loaded B_repeat: bit-identical trajectories AND metrics.
- Wall GROUP / GROUP_repeat and MAPPO GROUP / GROUP_repeat: bit-identical.
- Wall assembled at307s;106 commands; native safety handback at353s when a member left coverage.
- Wall C_G180=-154.834veh-s; coordinated mechanism detected, beneficial synergy NOT established.
- MAPPO C_G180=0 on this one geometry-selected pair; no efficacy selection or sample replacement.
- No collisions, teleports or dropped vehicle steps in any canary branch.
- Canary actual2280sim-s, charged2580; failed setup reservation300 retained at0actual.
- One documented pre-observation PATH repair used a new attempt identity; no outcome retry.

## Evidence, continuation and behaviour handoff
- Design: AUDIT_DESIGN.md; CPU contracts: tools/coopaudit/test_cpu.py (all pass on gpuR).
- Receipt: tools/coopaudit/evidence/CANARY_VERIFIED.json; code commits c422c3a2/c3d5068d/52546a7c.
- Raw canary: gpuR /root/coopaudit20261031/jobs/canary_s969699_attempt2/.
- Traces: that directory's traces/*.jsonl.gz; members and nearby ramp vehicles, t0-30 through window end+30,0.5s frames.
- Existing main969650 traces: /root/coopaudit20261031/jobs/main_s969650/traces/*.jsonl.gz (old two-member design).
- Schema and identity/subset mapping: tools/coopaudit/TRACE_HANDOFF.md for S-COOPBEHAV.
- Continuation entry: python -B -m tools.coopaudit.continue_audit (exclusive lock;120s sequential polling).
- Owned continuation39620 stopped; gpuR YIELD set and owned idle worker516243 paused;16/16 original CPU tests passed.
- Canary +28 census +1 original five-arm world preserved: actual21180, charged21480/500000sim-s.
- Further SUMO requires explicit S-SUMOCF ARTIFACT_CHECK.md PASS and corrected design/cap binding; file absent at recovery.
- PI/CIRCLES/CTRL0 are optional only; existing old-design observations are historical, not corrected-panel estimates.
- Completion and blockers are recorded in tools/coopaudit/evidence/STATUS.json and CONTINUATION_FAILURE.json.
- Seed scan disclosure: an initial broad scanner opened legacy parameter JSON; no values used; corrected guards prune these paths.
- Full exposure disclosure: tools/coopaudit/evidence/EXPOSURE_NOTE.md.542 allowed runtime files verified;1 restricted-name file deliberately unopened.
- Fleet improvement or visually coordinated behaviour is not evidence of beneficial C_G; await the complete fixed panel.
