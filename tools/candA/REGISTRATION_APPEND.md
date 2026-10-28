# S-INTEGRATE design notes for MAIN to append to REGISTRATION.md (2026-10-29 06:00)

Frozen before any run; not executed by S-INTEGRATE. Append verbatim if MAIN approves.

---

## A-LONGX design note — executable STOP band and fork procedure (registered before any run)

Authority: OWNER 02:40 approval (exploratory long-credit arm + separate 21,600 sim-s) +
MAIN 06:00 (iii). Runs ONLY if the chosen WM does NOT qualify CCT-2 (if it does, A itself
carries the gain-only long credit and A-LONGX is not run; budget unused).

**Fork procedure.** Candidate A runs with `checkpoint_every=1` (configs/c4_policy14_candA.toml).
A-LONGX forks from A's `policy_000050.pt` (update 50, full actor/Adam/RNG state): a new phase
on the OTHER GPU resumes it with `--new-phase` (auditfix refuses any silent trainer/reward
override), identical recipe/seed lineage, PLUS the gain-only long credit forced to
`H_long = 300 s` for the remaining 100 updates, labelled EXPLORATORY in every
receipt/summary. Time-box: must finish by Friday 06:00 (+08:00); at the qualified 350.8
s/update the 100 updates project ~9.7 h. Required pre-run amendment (registered here, not
yet coded, default-OFF discipline): an explicit `long_credit.exploratory_unqualified = true`
switch that bypasses ONLY the CCT-2 file check in `gated300._install_gain_only`, keeps the
gain-only row rule (K8 median < -2*sigma AND q10 improvement > 0), and stamps
`EXPLORATORY_UNQUALIFIED_LONG_CREDIT` on the run; no other behaviour may change.

**Executable STOP band (frozen).** One monitoring window = one completed training update.
Windows with zero gated long-credit rows are skipped (do not count, do not reset). For each
window w with |G_w| > 0 gated rows, compute with the S-EXPLOIT machinery
(`exploit.joint_credit` / `ensemble_tail_credit` metrics, detached):

1. *K8-q10 band.* Per gated row r, from its own K8 joint credits `joint[:, H_long, r]`
   (signed paired objective, reward = -J): band_lo = q10, band_hi = q90 (linear
   interpolation, the registered antiexploit QUANTILE=0.10 convention). The row's imagined
   improvement is `-joint` (positive = predicts gain).
2. *Cross-WM agreement.* Score the SAME rows and commands on the frozen incumbent WMs
   (WM2 `b8797242…`, WM3 `41e18f09…`, plus the best non-chosen overnight WM if loadable);
   agreement_w = fraction of rows where EVERY member's q10 also predicts a gain (q10
   improvement > 0), i.e. no member's pessimistic tail sees harm.

**STOP condition.** Window w trips iff
`median_r(-joint[K8, H_long, r]) > band_hi median` (the claimed improvement exceeds the
row's own K8 uncertainty band — optimism) OR `agreement_w < 0.80`. HALT A-LONGX when 3
CONSECUTIVE qualifying windows trip (STOP file, own run only; A is not stopped). All window
metrics are appended per-update to `exploit_windows.jsonl`; the halt is mechanical, never
efficacy-based, and the stopped fork's checkpoints are retained.

---

## S-EVAL-A frozen design note (owner 23:55 + 02:40; canary amendment per MAIN 06:00 iii)

Budget: 86,400 sim-s HARD (24 worlds x 4 arms x 900 incl. 300 s warm-up), separate cap;
A-LONGX fifth arm 21,600 sim-s (24 x 900) separate. No refunds/retries/top-ups without a
new amendment. Charge every allocation BEFORE dispatch.

**Exposure audit (before any allocation).** Seeds 977300-977323 from the registered final
block 977000-977999: (a) complete-token (not substring) search of REGISTRATION.md and all
task MD/JSON evidence under E:/agent-work for `9773xx` tokens; (b) pruned names-only
filesystem audit on cpuP, cpuQ, gpuA, gpuR owned roots (`/home/ubuntu`, `/root/autodl-tmp/c4`)
for `9773` filenames, excluding scalar-hash false positives by full-token recheck; (c) the
runtime seed guards admit EXACTLY the 24 pairs and refuse 964000-964199 and 973xxx training
seeds. If any prior exposure is found: prospectively take the NEXT UNUSED 24 seeds in
977000-977999 (re-audit, register), never silent substitution.

**Canary allocation (inside the cap).** Canary = the native arm of seed 977300, run first,
serially, 900 sim-s charged inside the 86,400 cap, adopted once on mechanical PASS
(initial-state CRNs, group/request populations, clocks, additive/conservation accounting,
zero collisions/teleports/overflow). The remaining 23 worlds x 4 arms release only after
PASS; a mechanical failure stops all dispatch (no retry authority). Additionally, one
native rerun of stored E3-final world 977000 on the frozen b3c031ad runtime must remain
byte-identical to the stored native result before ANY S-EVAL-A job (zero charge: reuses
the S-E3-FINAL/S-CTRLARM canary receipt if still valid for the unchanged runtime; else
MAIN re-authorises one 900 s check charged to this cap, leaving 23 evaluation worlds).

**Arms.** native; CTRL0 (switch-only control, E3-final runtime, residual-aware
BranchController path as in S-CTRLARM); MAPPO `aecd7d70…` (E3-final arm settings);
ours-A (the new policy under the always-commanded law, mode 2, `--population textbook`,
deploy observer); A-LONGX arm only if A-LONGX ran. Same demand/penetration sampling and
cell order as E3-final; 300 s warm-up + 600 s continuous intervention, cumulative
120/300/600 s at absolute 420/600/900 s; same CRNs across arms.

**Analysis (frozen).** Primary estimands (600 s fleet+source total delay incl. entrance
waiting, treatment - reference, negative = better): ours-A - MAPPO, ours-A - CTRL0,
ours-A - native; exploratory: A-LONGX vs each. 2000-draw world-cluster percentile
bootstrap, seed 0, one shared resampling plan across arms/metrics/clocks; 95% CIs, means,
both sign conventions. Safety (collisions/teleports/braking/TTC<3 s), throughput,
per-source/road and fairness CVaR90 exploratory. Equal-weight paired worlds; no optional
stopping, no outcome-dependent exclusions, no multiplicity adjustment beyond the
pre-registered primary set; report achieved uncertainty. Freeze all design/source/actor/WM/
execution-law hashes (candA authority JSON) BEFORE the first SUMO second.
