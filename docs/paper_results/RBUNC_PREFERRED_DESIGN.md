# Frozen preferred-posterior comparison — 2026-11-01 01:35 +08

Completed raw and encoder panels are retained. No previous job is restarted.
Purpose: determine whether FIT's already motion-calibrated `shrinkage_v1`
changes the **legacy deterministic IDM/MOBIL reference** forecast relative to
the completed raw posterior, with arrivals and newcomer population held fixed.
This does not qualify integrated WM-PHYS or select a new scale from outcome labels.

- Same FIT32 native worlds969400–969431, t0=300, authentic60s histories,
  K8, H120 physical rollouts; same seed20261031 and observable numeric VIDs.
- Same `observed_arrivals.json` bytes from completed addendum_v2; exact same
  hidden-rate draws, request clocks and joint newcomer population. No refit.
- Change only existing-driver finite posterior to FIT's `shrinkage_v1`.
  Optional `lane_v2` latent diagnostic remains disconnected from physical lanes.
  LC2013 candidate is not adopted in these legacy physical dynamics.
- Preserve source ablations on969400: both/drivers/arrivals/fixed. Expected280
  physical futures. Fixed-source tapes must remain bit-identical within K.
- Reuse original world factual targets and identity bindings exactly. Verify
  complete32/K8 coverage, conservation, no dropped requests, unchanged global RNG.
- Report source SD and real across-world SD without targeting their equality;
  bias, empirical CRPS, range coverage and paired CRPS change versus raw, with
  10000 whole-world bootstrap resamples seed20261031. No posterior/scale sweep.
- New exclusive remote root `cpuQ:/home/ubuntu/rbunc20261031/preferred_v1`;
  four disjoint nice19/one-thread CPU workers, CUDA hidden. Admit only with
  load<24/32 and all prior addendum workers terminal rc0. No retries.
- Publish actual artifact/source hashes, dispatch/PIDs/terminal receipt and
  separate preferred raw/summary artifacts. Failure or incomplete coverage is
  reported, never relabeled as the completed raw panel.
- Integrated EVAL has no completed candidate traffic score at freeze; calibrating
  a spread multiplier from that absent evidence is unsupported. FIT's motion-only
  learned shrinkage is consumed exactly. INT/EVAL retain physical qualification.
