# Exact registered continuation and one-use dispatch

Shared registration: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`,
entry `S-SELMAPPO MAIN02:08 CONTINUATION REGISTERED BEFORE DISPATCH (implementation0a771926; fresh-sessiond8a49b51)`.
Appended before final staging/dispatch, following MAIN02:08 exact schedule acknowledgement.

- Original design SHA256: `992c05601f03d1d9caad0421b041dbeaa05fa8621ae7e3020be42e3c9f5b41d0`.
- Exact fleet cap: **7,012,800 sim-s** = 43,200 historical + 7,728 future episodes
  x900 (6,955,200) +14,400 unstarted peer-owned evaluation reservation.
- Per-CAV budget here: zero; S-SELMAPPO2 independently owns it.
- Update1: 18000046..18000068 (23); update2: 18000069..18000091 (23);
  updates3..169: 18000092..18007773 (167x46). All original cell assignments retained.
- Skip all46 consumed production worlds and both consumed mechanics seeds.
- Disclose46 fewer gradient episodes and smaller first two batches; final169 only.
- Both ancestor ledgers retained byte-for-byte; 14 unclosed historical reservations,
  five known zero-physics failures; historical actual bounds30,600..37,800 retained.
- cpuQ final stage `build_9bb7869d8f590818` passed40/40CPU and real frozen checkpoint,
  WALL lifecycle, seed-alias, exact registry, old initial model/RNG parity gates.
- Full source binding `9bb7869d8f590818c46980a16a63be00f72e00d3491dec95f903649dfa7bf790`.
- Dispatched once, owncpuQ645235, exclusive `FLEET_HISTORY_CONTINUATION_ONCE.json`.
  Receipt: `DISPATCH_build_9bb7869d8f590818.json`; never redispatch.
- nice19, CUDA hidden, one thread, <=12workers. Every episode admission before
  precharge requires load+pending+1<=0.8cores, >=8GiB available RAM and>=2GiB disk.
- C1SWITCH2 owns all evaluation on16world PANEL_v2SHA
  `7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827`.
  Final checkpoint/config/source hashes will be supplied; local evaluation unstarted.

The shared file is concurrently maintained by MAIN/peers. This local receipt commits
our registration/dispatch evidence without committing unrelated shared-file edits.
