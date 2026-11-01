# Resource-accounting correction before baseline v4 bridge launch

The first v4 bridge admission stopped **before claim/Popen**, because the old
check used `memory.max-memory.current` as free RAM. A read-only resource audit
(`RESOURCE_GPUA_BRIDGE_MEMORY.json`) shows why this was misleading:

- cgroup limit96,636,764,160B, current93,104,488,448B;
- anonymous resident10,784,251,904B;
- file cache80,953,556,992B, including inactive file50,314,874,880B;
- host MemAvailable1,026,912,223,232B; dirty file3,960,832B, writeback0.

The container has roughly50GB of clean inactive file cache that Linux can reclaim.
Treating all of it as irreducible resident memory would unnecessarily block the
one-thread bridge while large archives remain cached.

Read-only admission now uses the minimum of host MemAvailable and cgroup
`limit-current+clean inactive_file`, bounded by the cgroup limit. It counts no
active-file or anonymous memory as reclaimable, subtracts dirty/writeback bytes,
and retains the8GiB headroom guard. Tests cover limits, dirty/resident memory and
snapshot bounds; admission/bridge generated-program gate6PASS/0.68s.

No process was signalled, no cache was dropped, and no cgroup/peer file changed.
Existing WM2 bridge909809 remains untouched; this changes only resource accounting
for the not-yet-launched v4 bridge. Model source/recipe and scientific RNG unchanged.
