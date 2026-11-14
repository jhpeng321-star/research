# Idea A PRE-FLIGHT GATES (owner request, 2026-10-01). REQUIRED before READY

The owner (via main) runs these in the morning. Each gate needs one exact
command, printing PASS/FAIL, and a JSON report under
`output/wd01/wm_n1/preflight/reports/`.

The package is `output/wd01/wm_n1/preflight/` (piece P11), invoked as
`python -m preflight <gate>` from `output/wd01/wm_n1`. Only TRAIN worlds are
used. It is CPU only, and runs no SUMO.

| gate | name | what it checks |
| --- | --- | --- |
| G1 | `zero-residual` | With the HDV residual forced to 0 and arrivals OFF, the Idea A rollout (P3 + P1 + baseline + validity) reproduces the Part A v2g rollout on the same J4 TRAIN window. Per vehicle per tick, positions and speeds must agree within 1e-6, or an explained, documented tolerance. |
| G2 | `ctrl-parity` | The P1 CAV mover reproduces the recorded SUMO CAV trajectories (teacher data) on ≥ 2 TRAIN windows for each of native, damping, PI and wall v2.1 (the J4 law), to a stated tolerance. P1 must support BOTH wall v2.1 (J4 parity) and v2.2 (the new rule). |
| G3 | `label-leak` | The main-config HDV network input schema holds no action, strategy or CAV-flag channel (asserted on the tensor schema). A perturbation test changes the labels, flags and strategies and requires an identical output. |
| G4 | `overfit` | A single member drives the total loss (including the contrast and rollout terms) near zero on 2 tiny TRAIN windows, on CPU. The report shows the threshold and the loss curve. |
| G5 | `e2e-smoke` | A 60 s J4 TRAIN world run through the GPU-queue commands: convert → timing (30 s) → train (a few steps) → checkpoint → resume → eval, all on CPU. It also runs a torch 2.5.1 compatibility check: either a pinned env run, or an explicit API scan plus a `torch.__version__`-gated test. |
| G6 | `heldout-dryrun` | Held-out configs (i)–(iii) build the correct train/eval splits. It asserts that the excluded arms (J4B, PI, the highest J4 wall speed) are ABSENT from training. |

`python -m preflight all` runs G1–G6 and writes `PREFLIGHT_SUMMARY.json`.

**Reviewers:** a DeepSeek first pass, then a GLM second pass on P1, P3 and P7.
