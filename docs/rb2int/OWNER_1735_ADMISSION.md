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

`tools/rb2int/capped_worker.py` now implements that explicit handover. A coordinator
lock prevents duplicate waiters. After the old children are all exited/complete,
it signals only the exact original paused CPU dispatcher (pidfd TERM then CONT),
waits for its actual exit, acquires SUPERVISOR.lock, and revalidates every prior
claim before any Popen. Prior claims never enter its pending queue. Missing
supervisor EXIT receipts are recovered from verified K8 markers with returncode
**null** (not invented0). Source/model/recipe/registration/plan bytes stay frozen.
New admission is hard-capped2/gpuA and1/gpuR, using the original resource guards,
5GiB/nice19/single-thread evaluate invocation and no automatic retries.

Remaining reserved FULL-v5 jobs move first; the rest retain their stable original
order. This amendment uses only seed/model metadata, not outcomes. The dispatcher
cutoff remains the original Nov2 09:00 for now. `capped_ops` stages only an exact
committed standalone coordinator, starts it once, and writes a distinct process
receipt; original supervisor/source/queue receipts remain immutable. The waiting
coordinator cannot dispatch while the original owns the dispatch lock.12CPU
guard/claim/priority checks PASS; Ruff PASS. Launch/status receipts follow.

The existing sole collector25240 and cache-only publisher44988 remain owners.
Collection keys off hash-validated task COMPLETE.json, not supervisor EXIT.json;
completed draining children can still be collected while dispatchers are paused.
The interim3h/complete-reserved8 triggers remain in place. Existing09:00 dispatch
and09:40/09:55 collection/publication bounds need a later explicit amendment if
the reduced-resource panel is to continue beyond them; do not claim full delivery
from a partial cache.

## Prospective later-delivery pass (prepared before its tasks)

The owner explicitly permits later final delivery. A separate standby late pass
is prepared with admission cutoff **Nov4 09:00+08**, collection/publication bound
**Nov4 09:55+08**, and immediate full delivery whenever all216new world jobs exist.
It waits for each current capped supervisor's **natural** clean cutoff terminal
AND actual exit, acquires both locks, validates every old claim, and admits only
unclaimed jobs at the same2/1 cap. No current process is restarted, no task
retried, and the original Nov2 cutoff/attrition receipt is preserved. A failure
or source STOP stops this bridge instead of silently selecting/retrying results.
The frozen model/recipe/panel/design/gate/CI families are unchanged.

`late_watch.py` is a standby local owner, NOT an additional active collector. It
takes collection AND publication only after both existing local owners naturally
write deadline-incomplete receipts and exit (exact Windows creation-time checks).
An already completed final causes no takeover; an unexpected STOP requires explicit
recovery. It carries forward prior interim timestamps/bytes, keeps3h/reserved8
triggers, and uses the same cache/scorers/final report assembly. Existing watched
analysis/collector/publisher sources are not modified. The new late pass and
standby PROCESS receipts must be read before any future dispatch.

Recovery read git status/log, common rules, morning report/HANDOFF/noon publication,
current INT receipts and all four current worker handoffs before action. Latest
remote snapshots show19/216complete world jobs and no failed jobs; sourceb2df7458
remains frozen.5guard/intervention sandbox tests PASS (no actual signals in tests).
