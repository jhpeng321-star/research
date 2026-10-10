"""WM-N1 evaluation harness (design WM_N1_DESIGN.md sections 6-7).

Package layout:

* :mod:`evalh.protocol`   -- the abstract ``Predictor`` protocol and row types;
* :mod:`evalh.climatology`-- the trivial reference (climatology) predictor;
* :mod:`evalh.predictors` -- predictor loading by ``module:attr`` import path;
* :mod:`evalh.sources`    -- the evaluation data source protocol + synthetic;
* :mod:`evalh.metrics`    -- the registration metrics and the decision rule;
* :mod:`evalh.seals`      -- CAL / chunk-5 TEST / HOLDOUT seal enforcement;
* :mod:`evalh.run_eval`   -- the ``--split {train_oof,cal,final_test}`` CLI;
* :mod:`evalh.timing_gate`-- the 1-hour GPU timing gate (design section 7).

CPU-only, zero SUMO, synthetic tests.  The model package (e1) is built in
parallel, so this harness codes against an abstract ``Predictor`` protocol.
"""
from __future__ import annotations

__all__ = ["protocol", "climatology", "predictors", "sources", "metrics",
           "seals", "run_eval", "timing_gate"]
