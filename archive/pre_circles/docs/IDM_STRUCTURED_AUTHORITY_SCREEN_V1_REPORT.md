# IDM structured-authority screen v1 report

Status: completed development-only screen. No action family passes every frozen
gate; PPO and action-local prediction are not authorized.

## Cohort and integrity

The screen used 2200-demand seeds 4107--4112 and 2700-demand seeds 4113--4116.
Seed 4108 produced no eligible landmark. Fifteen eligible decisions generated
150 branches under protocol hash
`3e075699c06640a1defda161233940db777764e83ce03bd2e765557c920fbe0e`.

All predecision/discovery identities, duplicate nominal replay, horizons,
timestamps, follower/lane associations, command-chain instrumentation and
action-leakage gates passed. There were no collision or teleport observations.
The complete compact artifact occupied 14.52 MB.

## Gain family

Forty of 60 nonnominal gain arms were dose-qualified. Nine were helpful, six
harmful and 25 neutral. Helpful outcomes spanned five seeds, four values won in
different states, 14/40 effects exceeded replay noise, spatial response reached
beyond the absorber in 36/40 arms, and executed-dose/effect-magnitude Spearman
rho was 0.532. Median projection survival was only 0.390, below the frozen 0.50
gate. Two stronger-gain arms in the seed-4110 2200 wave case also lost more than
one merge discharge.

At 2200, only gain 0.5 produced one helpful seed (4112); the other values had no
helpful seed. At 2700, gains 1.25 and 1.5 were helpful in seeds 4113, 4114 and
4116. The pooled diversity therefore does not establish the required
near-critical 2200 action diversity.

## Reference-offset family

Forty of 60 nonzero reference-offset arms were dose-qualified. Ten were
helpful, ten harmful and 20 neutral. Helpful outcomes spanned four seeds, three
values won, 19/40 effects exceeded replay noise, spatial response reached beyond
the absorber in 36/40 arms, median projection survival was 0.595, and
executed-dose/effect-magnitude Spearman rho was 0.454.

Every offset in the seed-4110 2200 wave case lost 2--5 merge discharges, so the
frozen service gate fails. At 2200, helpful labels occurred only in seed 4112;
mean effects for every value were nonpositive. At 2700, negative offsets were
helpful in seeds 4113 and 4116, while +2 was helpful in seeds 4113 and 4115.
This is genuine physical authority, but it is largely demand-regime separated
and cannot support a 2200 action-value predictor or near-critical damping policy.

## Decision

Neither gain, the gain-derived mild/nominal/strong modes, nor reference offset
passes every frozen family gate. Do not train PPO and do not fit a pooled
classifier whose signal could be demand separation.

The remaining structured option is safe absorber choice. The current selector
chooses the nearest runway-feasible CAV before TTC admission; if that candidate
is unsafe it does not try the second candidate. The configured
`minimum_absorber_spacing_m` is not currently enforced by selection. Exposing a
ranked set of candidates that each independently passes runway, front/rear TTC,
secure-gap and influence-separation checks could improve dose and placement
authority at 2200, but it materially changes absorber semantics. Its exact
candidate-set definition must be discussed and frozen before implementation.
