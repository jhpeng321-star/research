# Four GPU slots restored without restarting a worker

Both new secondary spools were already READY with eight exact startup gates.
Their read-only V2 admission monitors had exited before any admission because
`tools.coopmwm.hotadd` was absent from the frozen import path.

Registered repair invokes the original 26-CPU-qualified gate helper by its
verified file SHA, checking expected boot/start/PID, eight exact gates, service
identity, search SHA and the original restricted endpoint on cpuQ.

- gpuA secondary926556/start899315942 admitted unix1793616987.584779.
- gpuR secondary420202/start4224036367 admitted unix1793616998.071469.
- Both original primary workers retained; no qualification or query replay.
- Zero new SUMO or inference calls. The existing broker discovers the additive
  READY/endpoint records; no duplicate monitor or search process was launched.

Receipts: `FAST1844_ADMISSION_REGISTRATION_gpu{A,R}.json`,
`FAST1844_ADMITTED_gpu{A,R}.json` and the two fresh GPU census receipts.
