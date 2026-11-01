# GPU-only inferred source-counter ABI repair — 2026-11-01 18:28

- First real H300 timing request failed BEFORE physical WM rollout:
  `LearnedArrivalFit._observation`: invalid learned request counts/exposure.
  The adapter supplied donor-prior rate times t550, generally fractional.
- Decoder now nearest-integer encodes inferred prior sufficient statistics;
  integer-valid prior inputs are unchanged. Rate perturbation <=0.5/t veh/s.
  These are **inferred current-clock FIT pseudo-counts**, not observed source
  requests. Current-clock pseudo-exposure also differs from retaining the
  donor's original posterior exposure; this weaker uncertainty contract is
  disclosed, not claimed to equal a causal full-history posterior.
- Fixed FULL-v4 weights/physics, K8 CRN producer, option executor, SUMO grading,
  PPO, beta and outcomes remain unchanged. No teacher/source truth is consulted.
- CPU21:50PASS/0skips/1.71s/0SUMO; fractional t550, finite prior validation,
  nonmutation, MAPPO parity, paired-future proposal parity all covered.
- Original dual1 services/master ended on error; both remote process inventories
  show no COOPMWM GPU process. Other GPU PIDs left untouched.
- Register fresh dual2 immutable source/service receipts; reuse original queue,
  completed forecasts/exact twins/checkpoints and failed-before-rollout requests.
  Same first availability-qualified H300 K8 request on both hosts supplies new
  cross-host/per-future parity and conservative K8+single-canary throughput.
- At most six timing requests were registered; once both hosts have real timing,
  explicitly mark unused timing requests skipped BEFORE inference, keep all files.
  Timing is infrastructure-only and never consumed by PPO or trust estimation.
- Original24h/host GPU deadline retained (first dispatch+86400s), not reset.
  Total allocated4,918,800/6,000,000sim-s unchanged; charged83,200sim-s/70closed
  jobs at repair. Primaryu1 preserved; u2 remains at numeric sampling gate.
