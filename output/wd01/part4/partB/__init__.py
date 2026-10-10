"""Part B: long-term delay estimator skeleton (DESIGN_PARTS_A_B.md §3).

Code skeleton only -- synthetic-data tests, no training on real branch
data (the WM-36 campaign does not exist yet).
"""
from .featurize import (CAV_RAW, CTX_RAW, DEMAND_RAW, LMAX_BINS, N_LANES,
                        WALL_RAW, PartBFeatures, collate, featurize)
from .model import DelayEstimator, DelayEstimatorConfig, LaneFieldEncoder, \
    count_parameters
from .losses import (gaussian_nll, near_zero_weight, pairwise_ranking_hinge,
                     partb_loss)
from .ensemble import DelayEnsemble, resolve_device, seed_everything

__all__ = [
    "CAV_RAW", "CTX_RAW", "DEMAND_RAW", "LMAX_BINS", "N_LANES", "WALL_RAW",
    "PartBFeatures", "collate", "featurize",
    "DelayEstimator", "DelayEstimatorConfig", "LaneFieldEncoder",
    "count_parameters",
    "gaussian_nll", "near_zero_weight", "pairwise_ranking_hinge",
    "partb_loss",
    "DelayEnsemble", "resolve_device", "seed_everything",
]
