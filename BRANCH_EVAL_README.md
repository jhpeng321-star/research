# S-COOPMWM paired branch test (co-primary #1/#2): harness
Branch `c4/branch-eval` from c4/coopnative 978f44bf. Code: `tools/coopmwm/branch_eval.py`; tests: `tools/coopmwm/test_branch_eval_cpu.py`. No SUMO, GPU or sealed data were used.

## Design
- Panel: the 96 sealed FINAL worlds 989600-989635 + 990600-990659 (metadata from `accel_protocol.worlds('final_test')`; registered SHA d74f685e... checked).
- Branch moments: 6 per world at T = 300, 420, 540, 660, 780, 900 s. This is a fixed even grid in [warmup, 1200-300], the same for every world and fixed before any outcome. Each window is H = 300 s and ends by 1200 s.
- Metric: J_b = team_delay_vs(T+300) - team_delay_vs(T), meaning road speed deficit plus source waiting in veh-s. This is the same J as `coopnative` comparator/evaluation. d = treatment - comparator, so negative is better. % reduction = 100*(sum J_comp - sum J_treat)/sum J_comp, so positive is better. Safety and road/source deltas are reported as secondary outcomes.
- Arms: MAPPO, A, B now (fixed update24, deterministic, no guidance). R and native are added later under the same plan. Every arm runs from the same state.
- Inference: world-cluster percentile bootstrap. It draws 10,000 whole-world resamples with `numpy.default_rng(20261101)`, and a world keeps all 6 branches. P-values are two-sided by CI inversion. Holm alpha = 0.05 (FWER) over the 4-hypothesis family {B-MAPPO, B-A} x {all branches (#1), focus stratum (#2)}. A-MAPPO and the later contrasts are secondary and not part of the Holm family.
- Focus stratum: demand 2400-2700 x CAV share 0.2-0.4. That is 4 cells x 8 = 32 worlds. Sources: c4-coopbandit `COOPBANDIT.md` (high-demand .2/.4 dense states) and c4-coopplan `COOPPLAN.md` (planner gains). The full-panel J600/J900 results stay secondary.
- Plan: `python -m tools.coopmwm.branch_eval --out PLAN.json` writes the full JSON (worlds, times, prefix, arms, metric, analysis, RNG, budget) with `plan_sha256`. `verify_plan` rejects any edit made after hashing. Default draft SHA: eb04b9e8ebf66c594287b867e498436463cdc4f548027ad7c6b15b5fc43edd72.

## Prefix choice and fairness
- Shared prefix policy: frozen speed-only MAPPO169, i.e. the zero-residual update0 `Policy()`, from the native warmup to T.
- Every arm restarts SUMO from t=0 through the same `accel_plant.episode` code path. `TakeoverControllerMixin` suppresses the arm's residual requests while t < T. Exact zero abstains, so the commands are bitwise MAPPO (this parity is CPU-tested upstream). The arm still observes the prefix, so it keeps one event per second.
- Gate: the fingerprints for every second 300..T must match the MAPPO reference episode bitwise. accel_plant's own `branch.prefixes` check and `check_prefix` both enforce this. Cumulative totals at T must also be identical. Any mismatch raises `PrefixMismatch`, which means STOP with no retry.
- Why this is fair:
  - The prefix policy is the comparator itself and does not depend on A, B or R.
  - At T, all arms share the same physical state, roster, MAPPO executor memory and command history.
  - A, B and R are residuals on MAPPO, so MAPPO-induced states are within their support.
  - The times are fixed in advance, so no arm is favoured by state selection.
  - Each arm's own-trajectory performance is still measured by the registered full-episode J600/J900.
- MAPPO's branch outcome is the MAPPO reference episode itself (MAPPO prefix plus MAPPO continuation). Its prefixes are recorded every second, and totals are decoded from `fingerprint['metrics']` hex floats.

## Budget quote (exact reruns, warmup and prefix charged in full, no saved-state discount)
- Formula: per run-arm = worlds x sum_T(T+300) = 96 x 5,400 = **518,400 sim-s**. The exact machinery starts fresh SUMO for every job and has no prefix cache, so prefixes cannot be shared across arms or branches.
- 96 x 6 x {MAPPO, A, B}: **1,036,800 sim-s** if the final-test MAPPO169 episodes (`jobs/final_mappo_<seed>/RESULT.json`, already charged) are reused as the reference. It is **1,152,000** if the reference must be rerun (96 x 1,200).
- Adding R: **1,555,200** (1,670,400). Adding R and native: **2,073,600** (2,188,800).
- Reusing MAPPO saves 518,400 compared with branch-rerunning MAPPO (naive 1,555,200 for the 3 arms).
- The owner's ~0.5M estimate (518,400 = 96 x 6 x 3 x 300) only holds with saved-state restore. Restore is physically exact but not bitwise (S-ENSPROBE insertion-cache residual), so it would need a separate amendment.
- Throughput: about 6.9 CPU-h at COOPPLAN's measured 41.7 sim-s/wall-s for the A+B quote. Throughput of the accel executor has not been measured.

## Integration recipe (live S-COOPMWM agent)
1. Merge this commit. Build the plan and append its `plan_sha256` to the registry BEFORE any B/R outcome or final-test result is opened.
2. After the final MAPPO169 and A/B update24 waves, construct the runner:
   `AccelPlantRunner(root, b['runtime'], b['checkpoint'], weights={'MAPPO':Policy().state_dict(),'A':..,'B':..}, reference_results={seed: lambda s=seed: read(root/'jobs'/f'final_mappo_{s}'/'RESULT.json')}, extras=dict(queue_root=str(root), accel_bounds=b['accel_bounds'], gpu_deadline=b['gpu_deadline']))`.
3. Get the jobs from `be.jobs(plan)`. Precharge their `sim_s` with the existing `reserve()`/`admission()`, then run the `mode=='run'` jobs in spawn workers (the takeover patch is scoped per process). Apply `check_prefix`/`window` exactly as `be.execute` does, and keep the job IDs unique (`branch_<arm>_<seed>_t<T>`).
4. Before the panel, do a mechanics check on one world, one branch and arm A. It costs T+300 sim-s and must pass the bitwise prefix gate against the reused reference. If it fails, either rerun the reference (115,200) or STOP.
5. Run `be.analyze(records, plan)` on the complete panel. It returns d, CI, p, Holm, % reduction, focus results and per-time means.

## Tests (CPU, fake deterministic runner)
- `rl_driving` python: `python -m pytest -q tools/coopmwm/test_branch_eval_cpu.py` gives **14 passed**. On system py3.14 without torch: 13 passed, 1 skipped (the real-Controller takeover test). ruff is clean.
- Coverage:
  - plan hash stability and tamper detection, branch time validity, final panel SHA, focus subset (32 worlds/4 cells)
  - the exact 4-member Holm family, budget arithmetic (quote equals job sum equals fake actual sim-s)
  - identical prefix across arms, plus a leak that is rejected, a per-second prefix gate, and rejection of incomplete or duplicate panels
  - end-to-end known effects (-450/-600/-450/-150 veh-s), Holm textbook values and step-down stop, world-cluster bootstrap matching a manual resample and wider than iid
  - known 10% reduction, null and worse cases, takeover suppression (mixin and real Controller), and restoration of the patch
