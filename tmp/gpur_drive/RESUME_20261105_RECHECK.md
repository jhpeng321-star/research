# S-GPURCOPY fresh-session connectivity checkpoint

Checkpoint: 2026-11-05T07:27:57Z. Recovered HEAD: `fe0cdad0`.
Repository: `E:/research-dense`; harness directory `E:/agent-work/_sol` is not
a git repository. Existing unrelated changes were preserved.

Read git status/log and `RESUME_20261105.md` / `DONE.txt` before attempting
remote work. Prior work remains on disk. No new coordinator, archive,
transfer, or verification job was dispatched.

## Current checks

- `ssh -o BatchMode=yes -o ConnectTimeout=12 cpuS "df -B1 /home/ubuntu/gpur_final"`
  failed: connection closed by `111.229.249.49:22`.
- `ssh -o BatchMode=yes -o ConnectTimeout=12 gpuR "df -B1 /root /root/autodl-tmp"`
  failed: connection closed by `172.19.0.13:53897` (locally resolved address).
- A verbose, single-attempt cpuS probe established TCP, then failed during
  SSH banner exchange, before authentication. No remote command executed.
- A single-attempt cpuS relay probe through cpuV timed out during banner
  exchange. No remote command executed.
- A local process query for Python/SSH/rsync command lines containing the
  gpuR/cpuS task names returned no matches. This does not establish remote
  coordinator liveness. No processes were terminated.

The requested first remote capacity check could not execute. Current cpuS
free space and preservation of the 30 GiB reserve are UNKNOWN. Remote writes
were therefore stopped. The cause of the SSH failures is not established.

## Preserved prior state; not freshly reverified

See `RESUME_20261105.md` for the full recovery receipt. In particular:

- The 33,433,600-byte broken stream was previously renamed to
  `cpuS:/home/ubuntu/gpur_final/autodl-tmp.tar.gz.partial_broken`.
- Existing coordinator: `/home/ubuntu/gpur_final/_ctl/gpur_preserve.py`;
  receipts: `/home/ubuntu/gpur_final/_ctl/S-GPURCOPY-20261105/`;
  log: `/home/ubuntu/gpur_final/_ctl/S-GPURCOPY-20261105.log`.
- Archive completion and source/destination SHA256 remain unverified.
- Manifest upload/equality remains unverified. Previously verified local
  manifest SHA256:
  `ba6a1d652dbbf175105f96f831162af82f86223bf7018c01de86f5944fa9c46e`.
- Earlier root-copy completeness/SHA256 remain unverified; the recovered
  earlier copy script excluded protected-name paths as well as other trees.

## Next safe action

Restore SSH connectivity. Run cpuS `df` first and stop if the reserve cannot
be maintained. Inspect existing remote receipts and live jobs before any
dispatch. Resume the existing work; do not recreate or duplicate it. Preserve
the broken partial and all research data. Protected-name files may only be
copied/hashed as opaque bytes. Verify both archive endpoints, manifest upload,
and the earlier root copy before authorizing release.

Release status: **gpuR safe to release: no**.

Concurrent work observed during checkpoint staging: another session added
`RESUME_20261105T0727Z.md`, committed as `28c55c6a`, recording the same access
blocker. That receipt and its work were preserved. This receipt adds the
independent verbose SSH and local-process observations; no further remote
probes or replacement dispatches follow from this session.
