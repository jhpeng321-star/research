# Actual short-first dispatch

- Code938dfef8 committed before launch. Exact owned old schedulers confirmed
  absent; preserve all old claims and live workers. Prefix CPU parity gate PASS.
- Local detached dispatcher29400 launched2026-10-31~21:23. Durable receipt:
  _out/wmvalue_short_v1/LAUNCH.json SHA
  0db8c2dca2bcc5d9ff4b87db78cee8d024180c2964398aaf29cb89e5a8535cf1.
- First inventory:10old complete checkpoints harvested;9oldCPU workers live.
  New short CPU workers gpuR634293/634303/634315 alive at~37s,98..100% oneCPU.
  Three new plus five old gpuR CPU jobs respect cap8; four old gpuA respect cap4.
  No GPU allocation while peer compute PIDs remain; no duplicate old task key.
- s969432_t300_k0 still alive, no retry consumed. Never repeat it while alive
  or after its completed output exists. All old task outputs remain in original root.
- Same frozen real arrays SHAa61d1846 and V/D/E SHAef3e7bce. No scientific test
  has run yet. Monitor both dispatch roots; combined completed prefixes count.
- New runner invokes finalize_short only after all320futures/200cells exist.
  Publish FAIL immediately if neither B nor C beats EACH D/E on BOTH registered
  CI gates; otherwise retain NEEDS_A and complete the original A conjunction.
