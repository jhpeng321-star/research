# T2C fix round for the current independent r1 review

Resume coordination: the original harness was observed active again after this
receipt was started (it committed lifecycle step `624b0815` and launched
`.t2c-r1-final-prep`). This resumed harness is leaving its remaining tests and
readiness edits to that active harness, and will inspect their final receipts
before reporting completion. Additional completed lifecycle evidence below may be
included in the final ready receipt; no overlapping suite was started here.

Input: `REVIEW_T2C_r1.md`, verdict BLOCK at `7b148d87`, and the complete shared
specification through MAIN 02:30. This supersedes the older, differently scoped
`T2C_FIX_R1.md`/`T2C_FIX_R2.md` receipts.

## Recovery

The resumed session preserved all prior commits, edits, test directories and the
replacement review. Recovered implementation commits:

- `c2a753ba`: blocker 5, opt-in public quote/budget/preparation compatibility.
- `21148f70`: blocker 1, observable realised team reward; WM teaching/credit OFF.
- `0360812b`: blocker 2, successful native all8 augmented-execution admission.
- `b500a3c9`: blocker 3, independent per-CAV contexts and isolated member masks.

The prior lifecycle pytest process (PID 23512) was allowed to finish. Its final
output was unavailable to the resumed harness, so it is not claimed as passing
evidence. No matching process remained before subsequent CPU checks. No dispatch
or live job was launched.

## Blocker 4 — smoke-to-full-run lifecycle

One approval/config/registration/root and one ledger cover both phases. Smoke uses
`smoke/` artifacts, `T2C_SMOKE.claim`, and `t2c_smoke_` episode/reference IDs. Full
training uses the root artifacts, `T2C_RUNTIME.claim`, and `t2c_` IDs, loading fresh
common-u0 models. An exclusive start claim serialises first admission. Full-run
admission requires a completed self-hashed matching smoke receipt if smoke was
attempted; interrupted/duplicate attempts remain refused.

Augmentation is generated/verified once, charged to its own approved allocation,
and reused through immutable validated root-level caches. Full training reloads
the existing ledger, preserving smoke charges and job identities. Both per-phase
and total budget ceilings apply. Smoke's 9,600 episode/reference sim-s are separate
from the common 143,280 augmentation allowance.

CPU check: requested Python, CUDA hidden, bytecode/cache disabled, OMP/MKL=1:

`python -m pytest -q --tb=short -p no:cacheprovider tools/coopmwm/test_t2c_lifecycle_cpu.py --basetemp=E:/agent-work/c4-rerun-tracks/.t2c-resume-lifecycle`

**3 passed**, 29.72 s, exit 0. Covers completed smoke then all three full update-1
arms, fresh u0 weights, immutable smoke checkpoints/caches, disjoint unique job
IDs, cumulative charges, duplicate refusal, interrupted smoke refusal and a
self-hashed receipt with a different approval binding. The inherited unknown
`cache_dir` warning results from disabling pytest's cache provider.
