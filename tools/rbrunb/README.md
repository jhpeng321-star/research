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
all 160 completion hashes, then runs the frozen B10000 scorer. A 07:00 deadline
cooperatively stops only RUNB's own workers at task boundaries. It never sends
process signals. `.done` requires successful full-panel scoring; diagnostics are
not qualification claims.
