# Prospective CI implementation clarification — before final evaluation

The frozen18-contrast breakdown-distance family compares absolute **group mean**
WM–real rate biases. Absolute value is non-differentiable at calibrated zero
bias; a direct nonlinear bootstrap cannot automatically be called valid there.

Implementation therefore obtains joint simultaneous95% max-stat bands for the
12 signed WM–real rate biases (four fixed models ×three horizons), with paired
whole-world and within-world K8 variation. It propagates those joint intervals
conservatively through absolute value and subtraction for all18 registered
distance contrasts. Joint coverage of the signed biases implies simultaneous
coverage of every distance contrast, including at zero. Known probability support
clips signed intervals to[-1,1]. Missing/unreplicated/zero-estimable support stays
NA, not an artificial zero-width factual CI.

Panel, model comparisons, point estimates, bootstrap count/seed and switching
conditions are unchanged. This correctness clarification precedes any new final
score; it is not a response to evaluation outcomes. Bootstrap signed-moment bands
remain finite-sample approximations, and small replicated cells do not establish
precision or full12-cell qualification. Regression covers the zero-bias kink.
