# Owner requirement (via main, 2026-10-01): the action descriptor is the PLANNED SCHEDULE. Applies to Part B v0.4b AND the WM-N1 evaluation

1. **Action descriptor = the planned schedule, not the arm name.**
   - The per-CAV (or per-CAV-group) controller for each short segment over
     the horizon. This is the same plan tensor as WM-N1: 30 s segments over
     570 s (19 segments).
   - Plus the wall hold, speed and cap (hold_cap under v2.2).
   - "Damp 60 s then native" and "damp 570 s" MUST be distinguishable inputs.
2. **Disclosure.** The training data is dominated by whole-window actions.
   The switching arms (per-CAV schedules) and the walls supply the start/stop
   variation. Both registration and report must state this.
3. **Stratified reporting.** Accuracy (sign accuracy and ΔV CRPS vs the
   zero / cell-mean baselines) is REPORTED STRATIFIED:
   - by action duration/onset: ≤ 60 s, 60–240 s, whole window;
   - by horizon τ.
   The planner re-decides every 60 s (commit 60 s).
4. **A-priori representation analysis** (if feasible without delay; zero SUMO,
   no outcomes):
   - Count TRAIN pairs whose schedules differ only in the first 60 s ("60 s
     then revert").
   - Also count by duration stratum, to show how well short commitments are
     represented.
