# Clean3 qualification recovery: F6 completion spacing

The adopted b98d7f56 full CPU run completed: 1722 passed, 1 failed,
4 platform skips, 14 subtests. The sole failure preserved in
`_out/clean3_cpu_b98d7f56/pytest.log` measured actual subprocess starts
0.1199305 seconds apart against the unchanged >=0.14 assertion.

Root cause: the admission timestamp is durably recorded **before** fsync and
spawn. Unequal fsync latency can consume part of the interval before the first
actual attempt. Keep that original pre-attempt record and add a separately
durable completion timestamp under the same shared lock. Successors conservatively
wait from the later of the two. No interval or assertion is relaxed.

New deterministic regression injects a slow first fsync. Full F6 suite:
43 passed, 2 POSIX-only skips on Windows. G5 suite: 82 passed; new G1 collector:
7 passed. The latter two passed in the initial combined run, whose one F6
record-ABI failure was corrected by keeping completion in its own file;
the subsequent full F6 suite passed. Linux qualification still required.

G5 b903c6c2 is now merged additively. The full selection now also contains
G5, the native G1 collector, its affected bank suites, and the two G5 env
unit suites. The inherited actual-SUMO env test is explicitly deselected
before collection/execution; no local physics is authorized by CPU qualification.
