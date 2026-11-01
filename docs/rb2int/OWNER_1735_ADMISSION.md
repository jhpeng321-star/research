# Owner17:35 GPU-priority intervention — 2026-11-01

## Applied17:40:51/56 +08

The frozen supervisors read slots4/gpuA and3/gpuR once at startup. They have no
live admission/reorder API. Changing RUN_PLAN.json would neither change those
in-memory values nor preserve its immutable hash. No source/recipe/plan was edited.

`tools/rb2int/drain_ops.py` verified each original boot/start/PID, exact command,
root and source hash, opened an exact pidfd, and paused **only** that owned CPU
dispatcher with SIGSTOP. No process-group or evaluate-child signal was sent.
New admission is temporarily **zero**, conservatively below the requested2/1.
Four existing gpuA and three gpuR evaluate children were allowed to finish.
No task was killed, retried or redispatched. Remote exclusive intent/application
receipts and local `OWNER_1735_{GPUA,GPUR}_DRAIN.json` record the action.

The first invocation on gpuA failed before any intent/signal because remote Python
lacks os.pidfd_open. The corrected fallback uses Linux x86_64 pidfd syscalls,
matching the existing INT retirement convention. No action was repeated.

## Continuation required

Do not blindly SIGCONT the frozen supervisors: that restores4/3 admission.
Prepare a claim-preserving capped continuation: wait for every existing evaluate
child to exit, validate all completed K8 claims, then retire only the drained own
dispatcher and acquire its exclusive lock before admitting previously unclaimed
jobs at2/gpuA and1/gpuR. Original source/plan/results/claims remain preserved;
this is an operational owner-priority amendment, not candidate reselection.
Any failed or incomplete existing claim must stop continuation rather than retry.

The existing sole collector25240 and cache-only publisher44988 remain owners.
Collection keys off hash-validated task COMPLETE.json, not supervisor EXIT.json;
completed draining children can still be collected while dispatchers are paused.
The interim3h/complete-reserved8 triggers remain in place. Existing09:00 dispatch
and09:40/09:55 collection/publication bounds need a later explicit amendment if
the reduced-resource panel is to continue beyond them; do not claim full delivery
from a partial cache.

Recovery read git status/log, common rules, morning report/HANDOFF/noon publication,
current INT receipts and all four current worker handoffs before action. Latest
remote snapshots show19/216complete world jobs and no failed jobs; sourceb2df7458
remains frozen.5guard/intervention sandbox tests PASS (no actual signals in tests).
