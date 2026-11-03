# MAIN23:18 gpuA archive extension — qualified and registered

- Added an additive SHA-bound archive route to the existing mover. gpuR's old
  namespace and every receipt remain; gpuA is the new active destination after
  an identity-proven idle/no-child handover of the single source mover.
- New control:`/home/ubuntu/coopmwm20261103_rolling2318_gpua_v4`.
- gpuA destination:`/root/cpuq_archive_20261103/rolling2318_gpua_v4`.
- gpuA floor8GiB before/after each complete batch; cpuQ maintenance target12GiB.
  Source eligibility, age, open-FD/reference guards, per-file SHA/inode/mtime
  checks and proof-before-unlink are unchanged. No scientific source changes.
- Nine storage safety tests PASS locally and on cpuQ Linux. Three additional
  local tests compile the actual drain/bridge/route programs without signalling,
  transferring or launching a worker.
- Registered amendmentSHA
  `52a168369200db757f69edde38b69bc84f2be6240de5125b8bacb07e81a8c8d2`.
  Registration is append-only in the canonical registry and new remote control.
- Strict known-host cpuQ->gpuA route PASS; free29,199,044,608bytes. Its existing
  rented-box SSH identity was loaded into the existing local agent (no key
  copied to another host). Initial missing-agent-identity failure and a transport
  timeout caused no source removal or worker handover.

At this checkpoint the prior mover is still retained; handover/launch is next,
only after proving the old owner idle with no transfer children. No new SUMO,
inference, simulator charge or V2V dispatch.
