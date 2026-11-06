# R9_GATE2 — Astra stage-2C checklist items 7–8

## Implementation

- `configs/c4_wm_v10round9.toml`: `model.merge_pivot_state_st=true`;
  `physics.ramp_lane_end_stop=true` remains enabled.
- `scripts/round9_gate.sh` invokes real CUDA training (`--mode run`) at
  `curriculum.unroll_s=[120.0,300.0]`, `stage_steps=[2,2]`, with batch 4
  **at both stages**. The smoke overrides `optim.batch_size_per_stage=[4,4]`
  because the full-run preset otherwise overrides the scalar batch size with `[4,2]`.
  The preset's TBPTT chunk lengths `[120,60]` are retained; each smoke step executes
  backward and Adam over its rollout, rather than a forward-only probe.
- The trainer logs `merge_pivot_grad_norm`: detached pre-clip L2 norm across pivot
  parameter gradients, zero when none exist. No gradients, losses, clipping, or
  optimizer updates are modified by this diagnostic.
- The existing per-step `cuda_peak_mb` CUDA allocator high-water mark is required
  and its maximum is recorded in `smoke_r9.gate.json` (MiB). This is peak allocated
  tensor memory, not total device/reserved memory.
- The shared verdict in `src/wms/train/round9_gate.py` fails closed on:
  - Fewer than four optimizer rows; noninteger, duplicate, skipped, or out-of-order
    step numbers; fewer than two rows at each exact 120 s / 300 s stage.
  - Missing/null/nonfinite/nonnumeric `step`, `loss`, `stage_s`, or required metrics.
    Booleans are not numeric evidence. Every loss must be finite.
  - Nonpositive total on-policy events or TF-positive-mass events.
  - Either TF-zero-mass or on-policy-zero-mass total being nonzero. Negative counts
    are rejected separately, so cancellation cannot hide failures.
  - Nonpositive scored seconds over rows with `stage_s>=120` or `stage_s>=300`.
  - Any nonpositive/nonfinite tau or width; invalid gradient norms or an all-zero
    pivot-gradient total; missing/invalid GPU peak-memory evidence.
  - A nonzero training exit, unreadable/malformed logs, or `Traceback` in either log.
- Before launching the full round, the verifier actually reloads
  `wm_s_stage120.pt`, `wm_s_stage300.pt`, and `wm_s.pt` on CPU. It checks exact,
  finite weights (including pivot weights), stage/step metadata, final history,
  both batch sizes, reconstructed state-ST setting, and lane-end physics. SHA-256
  digests and reload results are included in the JSON report. The existing `wm_s/1`
  checkpoint format saves model weights; this is not an Adam-state resume claim.
- Existing smoke/full-run directories, log, verdict, JSON report, and gate lock
  are refused, including dangling symlinks. An atomic retained lock prevents
  concurrent/repeated launches; shell noclobber and exclusive JSON creation prevent
  evidence overwrites. Only a successful verdict permits the full run.
- Touched source/config/test/report files are pinned to LF in `.gitattributes`.

## Verification — 2026-10-22

Local CPU dependencies were installed under ignored `runs/active/r9_gate2_deps`;
test temporaries and caches stayed in this worktree. PowerShell setup:

```powershell
$env:PYTHONPATH="$PWD\runs\active\r9_gate2_deps;$PWD\src"
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q tests/test_round9_gate_verdict.py tests/test_round9_gate_launcher.py tests/test_round9_gate_training.py tests/test_merge_pivot_training.py tests/test_r9_lane_end_stop.py tests/test_r9_risk_retain.py --basetemp=runs/active/r9_gate2_verified
```

**289 passed**: 201 synthetic verdict/CLI cases, 9 isolated Bash launcher cases,
17 config/gradient/checkpoint cases, and 62 existing pivot/lane regression cases.
The gradient tests compare exact gradients and Adam-updated parameters against an
independent reference with clipping both enabled and disabled. Launcher tests use
a stub executable and fresh workspace roots, including paths with spaces.

Additional successful checks:

```text
bash -n scripts/round9_gate.sh
python -m ruff check src/wms/train/round9_gate.py src/wms/train/trainer.py tests/test_round9_gate_verdict.py tests/test_round9_gate_launcher.py tests/test_round9_gate_training.py
git diff --check
```

## Execution boundary

The real-data 120/300 s CUDA smoke was **not run** in this CPU-only test runtime;
there is no measured real-run GPU peak or launch approval from this session.
No remote execution or push occurred. On the intended authorized CUDA installation,
after the existing support gate passes, run `bash scripts/round9_gate.sh REPO_DIR`
with fresh artifact paths. Inspect `smoke_r9.verdict` and `smoke_r9.gate.json` for
the actual peak memory, metrics, and checkpoint reload evidence. `R9_ROOT` and
`R9_PYTHON` can select an installation; failures retain their artifacts and lock.
