"""W3 wall-fit: a small synergy head on top of the FROZEN world model (option b).

The WM mis-ranked SUMO-verified walls (COOP_PROBE_REPORT 2026-11-06: WM synergy -21.6 / -8.4
for walls SUMO measured at +336.7 / +271.4 veh-s). This package learns a correction from the
labelled (situation, joint move) pairs in the mining / coop_probe ledgers:

    guard.py     seed + path refusals (TRAIN worlds only; never 962xxx/964000-964199/977xxx;
                 never holdout/sealed/truth paths)
    dataset.py   label builder: SUMO group/solo/base outcomes + synergy, joined to the observable
                 situation, the WM query and the WM's own prediction; split by world seed
    features.py  observable-only situation/move features, WM-prediction and WM-latent features
    latent.py    WM latent belief extraction (frozen driver encoder; gpuA runtime, no SUMO)
    head.py      SynergyHead (residual on the WM's own prediction; zero-init = WM ranking)
    evaluate.py  rank correlation + precision@k on test seeds vs the current WM ranking
    rank_api.py  probe/mining-compatible ranking call; DEFAULT OFF (env WALLFIT_HEAD unset)

Sign convention (all targets, veh-s): advantage = J(policy) - J(move); synergy =
J(best single-member solo) - J(group). Positive = the group is better. Nothing here runs SUMO,
retrains the WM, or trains a black-box comparator.
"""
