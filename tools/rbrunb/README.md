# S-RB-RUNB evaluation runner

`watch.py` reads both required handoffs every 600 seconds and consumes the
ordered manifest linked under `READY FOR EVAL`. Shared protocol:
`E:/agent-work/rb-shared/evalclaims/RUNB_PROTOCOL.md`.

The watcher runs in the existing Python 3.13 installation. It deploys only this
runner and the published EVAL scorer into a new RUNB directory on gpuR. INT
provides the immutable candidate runtime/entrypoint. No simulator or training
entrypoint is invoked. Only exact published RBEVAL CLI options are generated.

```powershell
py -3.13 -B tools/rbrunb/watch.py --once
py -3.13 -B tools/rbrunb/watch.py
```

Persistent local status/logs live in ignored `runs/rbrunb/`. The remote supervisor
is idempotent, serializes GPU admission, caps each worker at 5 GiB, uses up to
three shards and waits for capacity without holding the admission lock. It
preserves all prior runs, refuses automatic retry after worker failure, verifies
all 160 completion hashes, then runs the frozen B10000 scorer. The captured owner
deadline cooperatively stops only RUNB's own workers at task boundaries. `.done` requires successful full-panel scoring; diagnostics are
not qualification claims.

## Owner noon continuation (2026-11-01 02:11 +08)

`continue_noon.py` is a one-time, explicitly owner-authorized operation for the
unlaunched B2 waiter. It preserved the original claims, source bindings and launch
receipts, verified zero dispatch/tasks, and replaced only the exact owned waiting
supervisor under the shared GPU admission and launch locks (pidfd identity).
The old Windows watcher was closed using its verified process handle/incarnation.
It did not retry a failed job, repeat any inference or alter staged sources.

Current identities: Windows watcher35616; gpuR B2 supervisor762343. Shared receipt:
`E:/agent-work/rb-shared/evalclaims/RUNB_NOON_CONTINUATION.json`.
New claim intake freezes09:30; already claimed evaluation/supervision runs through
11:30. The watcher polls every600s and retains04/08 interim checkpoints. Explicit
timezone-aware arguments prevent the old07:00 deadline from being reused:

```powershell
py -3.13 -B tools/rbrunb/start.py --deadline 2026-11-01T11:30:00+08:00 --freeze-at 2026-11-01T09:30:00+08:00
```

That command is the recorded invocation shape; the watcher is already live.
