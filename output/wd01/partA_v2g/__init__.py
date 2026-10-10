"""WD-01 partA_v2g: the general rule-based Part A v2 predictor.

Owner decision 2026-09-29 (PARTA_V2_GENERAL_DESIGN.md): Treiber IDM +
MOBIL + a mandatory merge rule, ONE nominal parameter set
(params_nominal.json), deterministic, tracks every given vehicle, no
insertions, CAVs follow the commanded action.  predict() implements the
frozen contract wd01_predict/2 with M = 1 plus the v2.1 tracked-scope
fields (PREDICT_CONTRACT_v2_1.md).
"""
