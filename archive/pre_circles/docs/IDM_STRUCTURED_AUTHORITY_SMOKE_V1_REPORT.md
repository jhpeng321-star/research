# IDM structured-authority instrumentation smoke v1

Status: integrity passed; full development-only authority screen authorized.
This report does not authorize RL training.

The smoke used demand 2200, seeds 4107 and 4109, four deterministic landmarks,
24 branches, three workers, gain values 0.75/1/1.25 and reference offsets
-1/0/+1 m/s. The protocol hash is
`abbdbbd5eb2e8b4ad93de43ba30f87304052982d314ad18d333a1efe306df5e6`.

All predecision and discovery fingerprints passed. Nominal replay was exact,
all 60-second horizons completed, timestamps were causal at one-second
intervals, immediate followers were on the correct lane and behind the
absorber, the full analytical-to-projected command chain was present, and
there was no action leakage, collision, or teleport.

The complete smoke occupied 2.3 MB. Mean branch JSON was 29.9 KB and mean
compressed trace size was 64.7 KB. Three workers were stable. The larger screen
therefore remains frozen at three workers.

The smoke is too small to evaluate the three-independent-seed family gates.
Its early action signal was weak: gain median projection survival was 22.4%,
reference-offset survival was 36.5%, no helpful effect appeared, and only one
reference arm exceeded the deadband (harmful). These outcomes are preserved but
are not used to stop before testing the predeclared extreme gain and reference
values. The full screen is authorized solely because every instrumentation and
execution-integrity gate passed.
