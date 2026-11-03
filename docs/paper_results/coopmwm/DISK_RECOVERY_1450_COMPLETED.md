# MAIN14:50 safe archive subset completed — 2026-11-03 15:16 UTC+8

Moved **288 completed interim observable tapes / 529,080,134 bytes** by rsync to
gpuR:`/root/cpuq_archive_20261103/interim008_observables/`. Every remote file
matched its per-file SHA256 and size before any source removal. Source traces
were re-audited against process FDs (privileged, not permission-skipping), public
binding/config/manifest references, and source hashes immediately before removal.
All RESULT files and bank/ownership/checkpoint artifacts remain on cpuQ.

- Manifest SHA256: `b69f1c5c86d9f2fedc5dd317cbd68de76bd5508b46c7121ee0e2bfb1d46ef86a`.
- Cleanup source SHA256: `a011cb65c2f0a34776cc4581e6d7900e88489e5a20a3371979b963fb2600acc4`.
- cpuQ free bytes: **1,994,756,096 → 2,524,499,968**, above unchanged2GiB guard.
- Simulation/inference/reservation: **zero**. No outcome payload was opened.
- Receipt: `ARCHIVE_MOVE_interim008_observables.json`; per-file manifest remains
  at `/home/ubuntu/coopmwm_archive_20261103_1450/MANIFEST.json`.

Earlier attempts are retained: Windows→POSIX staging-path failure, insufficient
FD-inspection permissions, and route authentication/interpreter failures. No source
removal occurred in those attempts. The existing local SSH agent was forwarded
for rsync; no key was copied or account/system setting changed. The SSH tool timed
out during transfer, so that running transfer was adopted, not duplicated. After
all288 destinations existed, an independent authenticated remote SHA verifier
issued the certificate; cleanup resumed from that certificate.

This is **not** the requested10GiB free-space target. Other candidate roots have
live readers or bank references; whole-root archival would break them. Further
movement needs per-file dependency qualification. Never lower the guard.

Six archive-safety CPU tests pass: source tampering, active readers, bad remote
counts, incomplete episodes, protected/live paths, interrupted cleanup and RESULT
preservation. No training/GPU owner was replaced.
