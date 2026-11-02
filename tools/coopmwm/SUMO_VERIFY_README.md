# SUMO verification of WM top-k plans (`sumo_verify.py`; owner option 1, MAIN 2026-11-02 10:20)

**Design.** When the guide returns an accepted proposal at t0, the episode blocks. `verify` builds
k+1 sealed branch specs from one decision-time snapshot: `policy` (group follows its policy) plus
`cand1..candk` (top-k profiles via `proposal['topk']`, executed through the unchanged Controller ->
`requests` -> safety -> CommandOwnership chain). All branches: replay the live recorded samples
300..t0-1; check fingerprints 300..t0 bitwise against the live episode (any `PrefixMismatch` ->
status `stopped`, group follows policy, nothing imputed); at t0 reuse the live t0 sample, restore the
AR noise memory and the torch generator state captured at the decision, then consume ONE draw
(= the Controller's beta draw if accepted / the hook's pad draw if rejected), so branches and live
share one noise stream (CRN). Plans committed at t0 continue their profiles (minus those an accepted
start truncates per `overlap_recheck`); plans that would start after t0 are not simulated; no WM
query inside branches. Window: t_end = min(t0 + max_i D_i + 120, 1200), same for all branches;
J = increase of `team_delay_vs` over [t0, t_end]. Choose argmin J (ties: candidate order); accept
only if J_best < J_policy (a tie is rejected). Exact grade = J_policy - J_chosen (positive = helped).
Receipt (`<root>/verifications/<id>.json`, no-overwrite, SHA256-sealed; `decision_sha256` covers the
deterministic core) holds all k+1 J values, branch receipts, candidate SHAs, charges.

**Cost.** Each branch charged like an accel_plant job: t_end sim-s from t=0, all k+1 precharged
before physics (no refunds; `BudgetBlocked` -> status `budget_blocked`, no physics, policy).
Per decision = (k+1) x t_end: k=4, D=300 -> 5 x (t0+420): t0=300 -> 3600, t0=600 -> 5100,
t0>=780 -> 6000 (cap); CLI `sumo_verify quote --t0 600 --k 4 --max-duration 300`. Wall ~ slowest branch.

**Arm R.** Same `verify`, `proposer='random'`: candidates = the matched random plan + k-1
`sample_unscored` plans around the same centre, seeds derived from the plan digest only (no extra
live generator draws). `plan_client` adds `verify_context` (group/centre/bounds/regime/duration) to
the R result only when `job['sumo_verify']` is set.

**Integration (live agent).**
1. Merge branch `c4/sumo-verify`. `accel_plant.episode` and `plan_client` changes are guarded by
   `job['sumo_verify']` (default absent -> byte-identical behaviour).
2. Register the budget phase (default `sumo_verification`; set `phase` to whatever MAIN registers in
   the accounting PHASES under the 8.5M cap) and set per job:
   `sumo_verify=dict(root=<verify root on cpuQ>, run_root=<run root>, phase=..., k=4, workers=5, timeout_s=3600)`.
3. Episodes must run where branches can launch (cpuQ): `SubprocessBranchRunner` spawns
   `python -m tools.coopmwm.sumo_verify run-branch` per branch (nice 19, PIDs in `WORKERS.jsonl`,
   only own Popen objects are waited on/terminated on timeout; never kills by name).
4. Quota: an accepted verification returns the proposal with the SUMO-chosen plan (counted executed
   via `guide.completed`); rejected/stopped/failed returns `accepted=False` -> quota DECISION
   executed=False. Check worker budget: 24 episodes x 5 branches concurrently.
5. DECIDE: the Controller still applies the trust beta draw after acceptance; with beta<1 a
   SUMO-verified plan can still be dropped. Pin beta=1 for verified plans if that is intended.
6. Smoke before B switches (no SUMO run here): one decision on cpuQ; all 5 branches pass the prefix
   check; chosen-branch fingerprints equal the live continuation up to the next query window (CRN).

**Tests.** `tools/coopmwm/test_sumo_verify_cpu.py` (10, fake deterministic plant): identical-prefix
specs/window cap, best choice + grade, policy fallback (incl. tie), mismatch -> stopped (no
imputation, charges kept), parallel determinism, budget arithmetic + BudgetBlocked, R variant,
overlap guard, live hook (choice swap, RNG pad alignment, quota passthrough, R arm).
`VerifyController`/`LocalBranchRunner` (real plant path) are not exercised on CPU.
