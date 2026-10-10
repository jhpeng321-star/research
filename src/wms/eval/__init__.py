"""Evaluation of WM-S rollouts: states, prediction, metrics, fingerprints, comparison.

Import order is acyclic: ``errors`` / ``normalize`` / ``stats`` -> ``metrics`` ->
``states`` / ``rollout_inputs`` / ``backend`` -> ``predictor`` -> ``score`` -> ``report``
-> ``ranking`` / ``fingerprints`` -> ``compare``.  Evaluation never imports ``train``.  ``compare`` and ``ranking`` are submodules
(import their functions from ``wms.eval.compare`` / ``wms.eval.ranking``).
"""

from .compare import ModelResult, write_comparison
from .fingerprints import best_vs_last, sliced_fingerprint, state_fingerprint
from .predictor import RolloutFlags, WmsPredictor
from .report import blowup_rate, model_report, paired_bootstrap
from .score import ScoreSpec, evaluate_state, score_states

__all__ = [
    "ModelResult", "RolloutFlags", "ScoreSpec", "WmsPredictor", "best_vs_last",
    "blowup_rate", "evaluate_state", "model_report", "paired_bootstrap",
    "score_states", "sliced_fingerprint", "state_fingerprint",
    "write_comparison",
]
