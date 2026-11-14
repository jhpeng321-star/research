# IDM v5.6.1 early low-authority formation no-go diagnosis

Status: **development-only no-go before treatment implementation**

The frozen treatment-blind scan found no common early assembly start across
860/865/870/875 s.  Seed 4402 was feasible at 865, 870, and 875 s, but 4401,
4404, and 4405 failed every grid value.  The common-start selector therefore
returned null.

The dominant rejection class is assembly-start front/rear safety: all candidate
two-lane combinations in the failing seeds either violate the 20 m gap or the
3 s front/4 s rear TTC gate.  Other rejections include temporary-envelope or
monotone-convergence failures and, in isolated combinations, speed-spread
failures.  These are predecision geometry/safety facts from damping-only
trajectories, not treatment outcomes.

This resolves the v5.6 Candidate-A 4405 formation-veto hypothesis narrowly:
more time upstream cannot be claimed as a repair under the present road,
coverage, and safety contract because there is no shared safe early cohort.
It does not prove early formation is impossible under a materially different
road geometry, CAV placement, member-selection rule, or intervention family.

No v5.6.1 controller state machine, wall outcome branch, fresh seed, predictor
dataset, predictor, or RL policy was created.  The correct next action is an
owner-approved, separately defined route—such as treatment-blind revised
geometry or support-CAV feasibility development—not tuning this grid or
relaxing the final 50 m compactness/safety criteria after the result.

Artifacts:

- `runs/probe/idm_v5_6_1_early_formation_scan_p30_s4401_4402_4404_4405/`
- `experiments/idm/run_idm_v5_6_1_early_formation_scan.py`
