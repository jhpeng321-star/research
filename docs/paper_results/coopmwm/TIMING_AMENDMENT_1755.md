# Availability-qualified GPU timing — 2026-11-01 18:20

- Cross-host H60/120/300/600 credit parity PASS: maximum coordinate error
  0.000012040138244628906veh-s. Original H600 K8+single canary wall:
  gpuA484.54s, gpuR431.95s. No simulator physics in GPU qualification.
- Original timing fixture had no available upstream cooperative option. Its
  ~0.003s timings and apparent693queries/s are INVALID for capacity fitting;
  `DUAL_1755_THROUGHPUT.json` is preserved, never used for sampling.
- Prospectively registered six new K8 H300 timing requests on the first original
  B-u1 TRAIN roster with a complete upstream group and inherited option available.
  Selection uses observable geometry/availability only: world986000,t550,
  availableO1/O2/O4,setting0. Immutable IDs: `TIMING_REQUESTS_1755.json`.
- Existing dual service queue handles them; no service restart, duplicate forecast
  or SUMO rerun. Use the first valid completion per host for the pre-registered fit.
- Budget remains4,918,800allocated/6,000,000cap; added timing is GPU-only overhead
  inside the24h/host service wall cap. Primaryu1 A/B receipts already published;
  u2 waits for numeric registration and payload hash binding.
- First E2 exact window closed all four twins; charged79,200sim-s/66closedjobs.
  C_G=0 in both WM and exact at all horizons: degenerate, not evidence for useful
  cooperation discrimination or variance reduction on nonzero interactions.
