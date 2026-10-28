# S-LRDIAG-1 analysis contract

Registered append-only in the shared REGISTRATION.md before inspecting outcomes.
Base: c4/wmcombined b74837d0; WM2 SHA256
b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436.

Only S-E3DATA `validation` worlds may supply numerical verdict evidence. Retain
native, mappo, ours_c1; coopoff is not needed for the requested three-arm analysis.
Use K=8 coupled draws from the exact warmup-end states and recorded commands,
300 s at dt=0.5 s. No new simulation, training, or GPU computation.

COOPMEAS documents behavioural and wave-damping detectors, but no collapse
detector. Define breakdown without a hand-set speed threshold: use 10 s bins of
mainline speed in the upstream and merge regions, with the 60 s pre-t0 baseline.
Compare constant/linear models against a downward step plus a shared linear
trend; choose by BIC, counting the searched change-point as a parameter. Require
three bins on either side for estimability. A selected step at t>=0 is a new
breakdown; earlier steps are explicitly pre-existing. Apply the identical detector
to recorded and imagined observations. Save step size, BIC difference, onset and
per-draw occurrence; confusion uses ensemble majority and onset error uses only
true positives. The detector identifies discrete speed-loss events, not a physical
speed cutoff. Report its limitations.

The delay objective is the existing horizon-gate t0-HDV delay proxy, arm-native,
not source-inclusive full-fleet travel time. All five ablations share initial
states, commands and randomness with baseline. Population-default ablation must
replace inferred physical driver parameters while preserving learned embeddings.
Residual-off means pure longitudinal IDM physics with inferred driver settings:
zero the attention/temporal acceleration residual, courtesy, yield and ramp
adaptation; learned lanes and learned merge remain. Other switches remove just
their named longitudinal contribution. This clarification was made before any
full-horizon scoring (only a 10 s plumbing smoke had run).

At a single world there is one observation per arm contrast: a within-world
regression slope for MAPPO-native alone is unidentifiable. Save its predicted/real
ratio (with no clipping), and a within-world two-arm through-origin slope;
pooled arm-specific slopes use OLS with intercept across validation worlds.
Bootstrap worlds, never stochastic draws as independent worlds.

Pulse_m/pulse_p qualification pairs are observationally eligible for mechanistic
diagnostics only. If not validation worlds, label them non-verdict and never use
their numbers to support the final verdict. Multi-controller paired trajectories
are not isolated speed-pulse interventions and cannot prove string stability.

Occurrence inference uses mean K8 event probabilities, not the frequency of
majority decisions; the requested confusion table separately uses majority >0.5.
Low predicted-on-real slope is decomposed into Pearson correlation times the
SD ratio, rather than interpreted as a physical perturbation gain. Paired-speed
signals retain both mean path RMS and RMS of the mean response, with survival
support. The full-road region/class timeline includes arrivals and is distinct
from the gate's t0-cohort objective; baseline metadata-only repeats must preserve
the original J and all-class speed/count series. No extra K draws are pooled.

CPU jobs initially ran nice 19, then (before any full-horizon result) nice 10
because of severe analysis contention; <=24 owned workers per host, with explicit owned PID
receipts and a collection-yield guard. Never signal another job. Source gpuR and
gpuA are read-only. Reject prohibited path components and evaluation seeds before
opening data. Preserve source hashes and use allowlisted converted arrays only.
