# MAIN12:02 append-only B-u6 lesson completion

The new 12:02 addendum explicitly authorizes measuring the five missing frozen
lesson entries after SHA restoration. It supersedes the earlier data-only
repair's no-replay instruction for these five unmeasured entries only.

Pinned source: qualified launchcoop_src7; runtime SHA
`06e88ecd7c71d274a054d7082bcd5d19e9e11e53523099b9ea9c0dadbc8247f2`.
Fixed B-u6 checkpoint SHA:
`bfe29812a3663a6dbc0d07458441512449f53a179e75ca7407e5d07f99d7f615`.
Original incomplete lesson file SHA:
`2aaf15795acd16fc4e0bc95c6d698bc797810d99394711d4369d50956f88ab6b`.

Exact quote: **3,480 sim-s** (720 + 600 + 600 + 900 + 660), phase
`v2v_lessons`, available 16,560 at the quote. Existing shared ledger and lock,
unchanged phase/global caps, and original qualified branch admission/precharge
are mandatory. No transfer or cap increase. Use existing deployed
`record_lesson` with seven exact existing rows adopted, unique attempt kind
`lesson_u006_main1202`, and additive output suffix `_MAIN1202_REPAIRED`.
The original lesson file is hash-pinned and preserved. One owned durable worker;
no retries or duplicate branches, no policy update. Physics/prefix failure is
reported, never imputed.

Before dispatch add an immutable input-protection binding listing frozen
lesson/isolated inputs and all present B/R replay_bank decision/snapshot files.
Verify both active movers exclude bank paths; their source scopes cannot move
replay_bank directories. Keep every referenced pending envelope, snapshot and
episode-result input. Zero-outcome selection; all five original failures are
completed regardless of their eventual grade.
