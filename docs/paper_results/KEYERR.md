# S-KEYERR — 2026-10-30

Final tested code/verification SHA: `be3350b855d1806822a801b771d308350c2efb24`.
Final delivery SHA: `git log -1 --format=%H -- docs/paper_results/KEYERR.md`.

## Root cause and fix
With `cav_exec_cap='idm'`, `step` set `exec_cap=None` exactly when
`(present & is_cav).any()` was false across the whole batch. The callback still
runs every physics tick; its vectorized `project_native` required the missing key.
CPU reproduction: both CAVs exit, HDVs remain; the exact production traceback fails
in the initial full no-grad `Replay.forward`, before zero-skip/prefix planning.
No-leader and entrant-only states do not suppress the cap. The supplied evidence
has no failing-state/tick dump, so the exact exit tick is not recoverable.
Fix: compute the real `_exec_cap` whenever an enabled-cap callback runs, including
empty-CAV ticks. Existing CAV masks still govern motion. Always-commanded execution,
SUMO safe-speed cap, leader rule, objective and credits retain their original laws.
All previously working nonempty-CAV steps take identical arithmetic/gradient paths.

## CPU verification
**482 passed; 5 CUDA-only skipped**, including LONGGRAD/LONGSKIP/LONGMEM,
execution/dead-zone/NANFIX, candidate-A/zero-start/integration, auditfix and LOOPLONG.
Before fix: two regressions fail with the exact KeyError; after fix: eight cap tests
pass, including exits, empty starts, entrant-only, re-entry and segment-2 backward.
Nonempty forward tensors, actor gradients and RNG are bitwise equal against the
actual pre-fix step; twelve continuation tests include trained-head/Adam equality.
Ruff, shell syntax and diff checks pass. Commands/results: `tools/keyerr/CPU.md`.

## Exact relaunch (handoff only; no remote/GPU/SUMO action performed)
Stage this final tree at the runtime below; preserve the existing byte-identical
`authority_gen/BOUNDARY_WM2.json`, WM/data/mix/driver/panel/authority inputs. Refresh
SOURCE.json and build/verify the source manifest with the same combined entry config.
`tools/candA/stepK.sh` contains the complete, literal environment and CLI arguments:
```sh
export RUNTIME=/root/autodl-tmp/c4/candA_keyerr
OUT=/root/autodl-tmp/c4/candA_prod_keyerr
test ! -e "$OUT" || exit 1
nohup setsid bash "$RUNTIME/tools/candA/stepK.sh" "$OUT" \
  > "$OUT.log" 2>&1 < /dev/null &
```
It resumes `/root/autodl-tmp/c4/candA_prod_184542ab_lgs/policy_000189.pt`, the
highest complete checkpoint in this failure, with the same
`configs/c4_policy14_candA_longgrad_skip.toml` and serial mode-2/pairbias/segment-2
recipe. Ordinary full resume preserves Adam and the trained head; no new-phase
reset/zero-start is applied. `--continue-run` restores the saved horizon/exploration,
original curve and gate baseline; phase origin remains **182** and the update-**192**
gate still runs after three resumed updates. The original 150-update phase budget
leaves 143 updates (through 332, subject to the unchanged gate), not 150 more.
