# S-C1FLOOR receipts

Remote immutable outputs: gpuR `/root/c1floor20261031`.
The registration and reference-tape inputs are copied through SSH without
allocating SUMO on cpuQ. `PREFLIGHT_BINDING.json` binds the actual staged code;
timestamped preflights retain earlier zero-SUMO mechanics checks.
`accounting.jsonl` is precharged and fsynced before worker launch, never refunded.
`CANARY.json` gates the remaining 23 worlds. Each attempt retains its own
START, exact original/reference tape comparison, original saved state/RNG,
RNG-only state variant, full branch tapes, and final/failure receipt.

The final report must distinguish pairwise stochastic divergence from a formal
Bayes-risk lower bound: a stochastic SUMO-vs-SUMO curve is a reference floor,
not a theorem that every optimal deterministic predictor must exceed it.
