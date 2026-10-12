"""Model-free MAPPO baseline for WM-S C4 (docs/WM_C4_DESIGN.md section 3, step 4b).

Parameter-shared per-CAV actor (the SAME ``CavPolicy`` architecture class as the
C4 policy, injected) + centralised critic on privileged state, trained directly
in SUMO on the same observations / actions / team reward as C4.  All partner
modules (``wms.control.obs`` / ``reward`` / ``policy`` / ``safety`` and the SUMO
``traffic_shepherd.c4.CavSumoEnv``) are injected by name and imported lazily so
this package can be developed and unit-tested against fakes while the parallel
workstreams land (see MAPPO_NOTES.md).
"""
