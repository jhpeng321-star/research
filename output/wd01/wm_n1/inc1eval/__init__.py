"""WM-N1-INC1: the zero-shot incident evaluation hookup.

This package is the bounded evaluation scaffolding registered in
``REGISTRATION_WM_N1_INC1.md`` (commit c221984, LF-sha256 d5282ecb...).  It is
deliberately SEPARATE from the training/eval pieces so that the registration's
"evaluated once, sealed" contract is enforced in code rather than convention.

Pieces
------
``obstacle``       fixed stopped-obstacle slot + injection into a rollout world;
``rollout_baseline`` the Part A v2g rollout baseline with obstacle support;
``incident_ctrl``  R1/R2 controller mirrors (partA_ctrl style) + R3/R4 adapters;
``predictor``      per-arm delay / TTC<3 / hard-brake prediction from a t0 state;
``metrics``        registration section 5 metrics + the run-once gate;
``model_lock``     INC1_MODEL_LOCK.json writer/verifier (seal, section 6);
``zeroshot``       event_mask / obstacle / setStop zero-shot assertion.

Zero SUMO, zero real outcomes; every unit test in ``inc1eval/tests`` runs on
synthetic states.
"""
from __future__ import annotations

__all__ = ["obstacle", "rollout_baseline", "incident_ctrl", "predictor",
           "metrics", "model_lock", "zeroshot", "CLI"]
