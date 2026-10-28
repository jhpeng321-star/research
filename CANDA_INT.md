# S-INTEGRATE — candidate-A integrated launch runtime (2026-10-29 06:00 dispatch)

Worktree `E:/agent-work/c4-candA`, branch `c4/candA-int` off `c4/c4prep` (f5bbeec0).
CPU only; zero SUMO, zero GPU, no push. Every step committed separately.

## Deliverable 1 — integrated branch + CPU suites

Merges (in order, conflicts resolved keeping every default-OFF flag OFF):
`c4/auditfix` 13ee37f7 (phase_resume --new-phase refusing silent TOML/checkpoint overrides,
checkpoint_every, hash-bound new-WM authority, inactive-config warnings, --real-credit
no-op rejection, direct-credit grad norm) → `c4/polspeed` f6dea57b (FGA exact env flags;
kept auditfix new-phase checks + polspeed bench-starts hook) → `c4/polspeed2` 48e14203
(N4 parallel futures; NINE-row exact path — six-row preset present but NOT used) →
`c4/gated300` a5953fb9 (gated long credit, CCT-1) → `c4/cct2` b60e12dd (CCT-2 tool,
gain_only mode, WM-hash authority files, WM2/WM3 incumbent H_G=300 receipts) →
`c4/exploit` f0a26a26 (exploit_monitor, exploit_disable_direct_credit; ensemble/anchor OFF)
→ `c4/execparity2` 6b7e0d24 (always-commanded mode-2 law + shared `sumo_idm` mirror +
e3final deploy fixes: residual-aware BranchController, deploy observer, --population
textbook requirement). e3final deploy fixes verified delivered via execparity2
(`traffic_shepherd/c4/branching.py` blob-identical to c4/e3final).

**c4/execparity3 slot: OPEN.** Its tip still equals execparity2 (6b7e0d24), no
EXECPARITY3.md, last measured parity FAIL (0.267858 > 0.160312). The authority template
records `leader_rule.status = OPEN_SLOT`; GO condition (1) requires MAIN's passing mode-2
parity receipt regardless.

Integration repairs (committed): parallel-futures workers inherit the source-tree
PYTHONPATH; ownership.py execution-law imports routed through the sanctioned
`c4/wms_glue` single sim→wms edge; the execparity2 model→control lazy edge documented in
test_import_graph EXCEPTIONS (gated300 tmp checkpoints under C4_ALLOWED_ROOTS).

**CPU suite counts** (miniconda torch 2.14.1+cpu, Windows):
- New integration suite `tests/unit/test_canda_integration.py`: **11 passed** (flag-default
  parity bitwise; executor/mirror shared-law binding; mode-2 sole writer; candA config
  monitors-on resolution; stale-secant direct credit disabled+flagged; --new-phase refusal
  + checkpoint_every=1; nine-row N4 parallel == serial bitwise on CPU [forward on/off];
  nine-row reference receipt 158/158 leaves @ 350.756 s; authority build/verify + mismatch
  refusals incl. learned-mix).
- Merged-branch suites: auditfix 23 passed; polspeed2/gated300/CCT-1/CCT-2/exploit 58
  passed + 3 CUDA-only skips; execparity (5 files) 50 passed.
- Full `tests/unit`: **2813 passed, 27 skipped, 38 failed** — the 38 are IDENTICAL to a
  pristine `c4/c4prep` baseline run in a clean worktree (Windows/pre-existing
  environmental; list archived). The integration adds zero new failures.
- `tests/unit/test_polspeed2_parallel.py`: 4 passed + 3 CUDA-only skips (run separately,
  spawns worker processes).

## Deliverable 2 — launch authority (`tools/candA/authority.py`)

`build` binds: source commit (+ clean-tree check; staged runtimes bind via SOURCE.json +
byte manifest), runtime manifest hashes of 21 execution-law/resume/monitor files, the FULL
candidate-A config chain hashes, C1 actor 64f70af4…, chosen WM sha + WM config sha +
learned-mix flag state (mix ON requires a hash-bound errsrc inference module AND a
`learned_mix_qualified` CCT-2 file — refuses otherwise, implementing ASTRA blocker 3),
CCT-1 (optional) + CCT-2 (H_G re-validated against the registered rule), execution-law
hashes (sumo_idm, alwayscmd, sole writer ownership, deploy), the execparity3 leader-rule
slot, and the mode-2 parity receipt (validated through `wms.control.execution.
verify_parity`). `verify` recomputes everything and **refuses on any mismatch**; it also
refuses when no passing parity receipt is bound (GO condition 1).

## Deliverable 3 — candidate-A config (`configs/c4_policy14_candA.toml`)

Resolves the registered candidate-A POLICY.toml content: full actor from C1, entrants ON,
deterministic CRN, q10/K8 pessimism, focus 0.6, nine-row weights 8.9/9.9/8.9/10.3/1.0,
dense 90 s horizon (curriculum [0,1], pads [32,288]), guarded real credit floor,
checkpoint_every=1, LONGMEM micro1/segment8/both offloads, anchor OFF (measured fallback),
monitors ON (`exploit_monitor`, `exploit_disable_direct_credit`; ensemble/STOP-conflict
OFF), `[long_credit] gain_only` bound to `authority/CCT2_{wm_sha256}.json` (placeholder
until the WM choice; missing file ⇒ dense 90 s only, mismatch ⇒ startup refusal).
GO-condition STOP rules: **implemented** — nonfinite paired/joint credit (ExploitStop),
>95 % any-clipping-or-zero admission (execution.authority_guard + verify_authority),
phase-10 signal failure (phase-relative gate), CRN/binding/CCT-2/authority mismatches
(fail-closed startup), new-phase resume override refusal, GPU cap. **Manual checks
(listed for MAIN)** — per-update coverage receipts (unique worlds/cells/horizons),
collection clock failures, same-executor optimism inspection (suspend adoption if
bootstrap lower95 % > 0), sampled memory watch beyond the allocator cap.

## Deliverable 4 — exact GPU commands (NOT run)

`tools/candA/COMMANDS.md`: step A authority build+verify (refuses without a passing
parity receipt), step B exact 2-update qualification (FGA + N4 parallel futures, nine-row,
`--resume` C1 `--new-phase --prep-dry-run-updates 2`, full registered flag set), step C
production 150-update launch (identical minus the dry flag; STOP-file only; C4A_RUNNING
only after ≥10 unique finite updates + evidence + signal gate, per S-C4LAUNCH).

## Deliverable 5 — design notes for MAIN to append to REGISTRATION.md

`tools/candA/REGISTRATION_APPEND.md`: A-LONGX executable STOP band (window = 1 update;
K8-q10..q90 band per gated row via exploit.joint_credit; cross-WM q10 gain-agreement ≥0.80
over WM2/WM3/non-chosen best; halt on 3 consecutive tripping windows; fork from
policy_000050.pt via --new-phase; H_long=300 EXPLORATORY requires the registered
`exploratory_unqualified` switch, not yet coded) and the frozen S-EVAL-A note (exposure
audit procedure for 977300-977323 with next-unused fallback; canary = native 977300 first,
900 sim-s charged inside the 86,400 cap + stored-977000 byte-identity recheck; arms
native/CTRL0/MAPPO/ours-A(+A-LONGX); 2000-draw world-cluster bootstrap seed 0).

## Boundary

Runtime is READY-TO-QUALIFY, not qualified: no GPU two-update receipt, no passing mode-2
parity receipt (execparity3 slot OPEN), WM not yet chosen by MAIN. Nothing dispatched.
