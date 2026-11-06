# S-R14B independent review (2026-10-25)

Reviewer: an independent Claude Opus subagent, read-only. It reviewed `83ce0ec1..8242051f` on `c4/v10r14b`, which includes cherry-pick `e4572409` of S-TTC `68366f88`. It ran the smoke's exact pytest list (223 passed) and five R14/loop14/R13E suites (80 passed).

It confirmed the following:
- The cherry-pick is byte-identical to `68366f88` in `src/`, `tests/` and `tools/`. Only the loop14 config was dropped.
- The only recipe changes vs `83ce0ec1` are `emergency_onset="driver_b"` in both stages and stage-A `init_from = runs/c4_wm_v10_round13e_b2/wm_s.pt`.
- `driver_b` reaches the flat config, TrainSettings and build_physics.
- The round-9 context arm validates only with `onset=plant`.
- The same-flags reference copies `emergency_onset`.
- STOP signalling uses the owned pid, start ticks, exact `--out` argv and pidfd.
- The launch-guard plus kill-window race is covered.
- A check ERRORs, and r13c "measurement error (fail closed)" STOPs, never signal B.
- There are no name-based kills, and the guards are read-only.

Verdict: **GO**, once blocker 1 is fixed and the host preconditions are checked.

## Findings and resolution

| # | Finding | Resolution |
|---|---|---|
| B1 | `autochain ticks()` treated a zombie b2 as still running, so the chain would wait 16 h and launch nothing. | **Fixed.** State `Z`/`X` now counts as gone. |
| B2 | Host preconditions: recheck `checkpoints.r10b.sha256`, b2 `INIT_FROM.json` `sha256`, offline report present, no prior r14 artifacts. | **Verified on gpuR 18:10.** Recheck r10b = `94e3ce9b…` = b2 INIT_FROM sha = sha256 of `b/wm_s_stage120.pt`. `diag_r14/R14_OFFLINE_REPORT.json` is present (`a03a51b1…`). There are no `registration_r14*`, `smoke_r14*`, `production_r14*`, `chk_r14`, `.round14*` or `runs/c4_wm_v10_round14*` artifacts. |
| S3 | Watchers had no liveness check. | **Fixed.** `watch` writes `chk_r14/<name>_watch_ready.json`. The autochain refuses to continue without the b2 watcher (exit 12), and launch-b refuses to exec B without the A watcher (fail closed). |
| S4 | An unreadable (half-written) RESULT was dropped and never retried. | **Fixed.** It stays pending and is retried up to 10 polls before it is recorded as UNREADABLE. |
| S5 | The B disk guard (1000 MB) left about 0.05 GB of margin. | **Fixed.** It is now 1500 MB. |
| S6 | CPU/RAM overlap: A checks, B midrun, the b2 postrun and B training run at once. | **Accepted.** gpuR has 754 GB RAM (677 GB available) and 224 cores. Checks use 4 workers at nice 19. |
| S7 | Every exception becomes ERROR, so a diverged (NaN) B would never be stopped. | **Fixed.** A non-finite candidate on a finite same-panel reference is now a STOP. Other exceptions stay ERROR, as the spec requires. |
| N | Stale comments (`repo_r14_`, init text, "B's midrun consumes A's RESULT"). | **Fixed.** The dangling R13E docstring in `config/train.py` is left as is: it is cosmetic and keeps the cherry-pick byte-identical. |
| N | `driver_b` follows the comfortable plan when it is feasible, even if the plant-limit plan is stricter. This is SUMO-like by design. The same-flags stop rule cannot catch regressions caused by the onset change itself. | **Noted.** R14B scoring decides this. |

## Fast-path stub (coordinator request)

The 13E stage-A FINAL was run through `d11_run5.py` on 3 states at k1 over 120 steps, single worker, with `WMS_SFAST_TEST_SEED=7`. Stub and stock outputs are byte-identical across all npz arrays. Stock-vs-stock without a seed is NOT deterministic, because the tool samples unseeded.

There was no speedup: 12.4 s stock vs 12.3 s with the stub, so the stub is **not used**. Checks now run in parallel with B, so their speed no longer idles the GPU.
