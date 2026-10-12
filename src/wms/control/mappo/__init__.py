"""Model-free MAPPO baseline for WM-S C4 (docs/WM_C4_DESIGN.md section 3, step 4b).

A FAIR universal MAPPO: the SAME actor (``CavPolicy`` with the lane head + ``RsuGroupEncoder``
on CAV reports and the RSU field), observations, lane feasibility / dwell rules, safety
layer, team reward and TTC Lagrangian as the C4 policy, executed step-for-step like the C4
deployment (``wms.control.policy.PolicyExecutor``); PPO over the joint per-CAV action
(acceleration + stay/left/right) with the privileged ``TeamCritic`` (CTDE), trained
directly in SUMO with penetration and demand sampled per episode.  See MAPPO_NOTES.md.
"""
