# Finite physical-profile population and driver posterior

Approximate population search support learned from gaussian_v1. Mixture masses
and residual scale maximize the full driver-sequence Gaussian likelihood by EM.
Use `MotionFit` to draw ONCE from the finite joint support, not the log-normal
moment approximation. See HANDOFF.md for measured validation/adoption status.
