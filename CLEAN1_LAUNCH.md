# Clean1 launch integration

MAIN 08:30 decision: plan augmentation is OFF for BOTH B and R. All prior
augmentation measurements and charges are retained as diagnostics. The live
banks are restored from the original read-only banks, excluding experimental
children from imitation and practice. Smoke requires zero augmented examples
in both arms; every other smoke, safety, budget and full-window check is retained.

Owner authority is the Nov 5 registration amendments: 01:39 cap **5,000,000 sim-s including smoke**, 02:28 real observable-neighbourhood reward redesign, 03:33 S-LAUNCH finishes T2C. S0 and its 960 parity sim-s are reused.

## Integrated production path

- `launch_preparation.prepare`: new own root/ledger, real-reader SHA closure through accepted P4, prospective TRAIN worlds/state/window registration, complete quote and approval binding.
- `launch_qualification smoke`: once-only native all8 augmentation, six full episodes and matched practice A references, actual component receipts and P3 replicated-test dry expansion. Models, update receipts and practice plans live in `smoke/`; accounting stays exclusively in the parent experiment.
- `launch_qualification gate`: final P1 u0 policies, the first three prospectively registered probe states for B/R, complete original windows and group/solo/base branches. Zero collisions and teleports are mandatory. Failed/incomplete work is never silently retried.
- `launch_training arm --arm A|B|R`: detached owner for each arm, immutable per-update base and B/R sidecars; paired A safety before B/R optimization; real full-rollout temporal credit and verified-plan bank CE. The prior monitor receipt gates the next update.
- `launch_training monitor`: the single accepted P3 monitor/diagnostic owner. Writes `LEARNING_MONITOR_uNNN.json`, runs u2/u4 probes, u6/u12/u24 replicated checks, scheduled lessons and the shared T1D paired evaluation. Current immutable checkpoints are used; look-ahead is preserved.

The canonical structured T2C configuration remains approval-bound. `launch_production` translates it to the narrow sibling APIs. Legacy/OFF ledger/preflight tests remain intact.

The real reward is P2's measured post-state speed deficit from direct sensing and causal t-1 packets. An observed CAV disappearance explicitly ends that observable agent lifetime with zero terminal occupancy reward; a missing frame without a measured transition still refuses. Raw physical vectors remain in update receipts. P3's integer safety counters conservatively round the inherited fractional vehicle-second braking counter upward; safety admission itself remains the exact existing composite comparison.

The base quote is 4,310,220 sim-s. Carry parity960 once; reserve another10,400 smoke sim-s (20,000 total smoke ceiling) and35,820 solely for the owner's one environment-void gate rerun. Complete upper bound **4,357,400 <=5,000,000**. Every actual dispatch is precharged atomically to the new parent ledger. Augmentation has its own143,280 allocation, within200,000; no WM score admits a child.

Coverage correction before further qualification: use the complete same-arm verified/effective training-bank reader, including imported B training sources. Imported parent snapshots are SHA-bound, registered and marked parent-only; they never enter practice directly. Only their native all8-admitted children can enter. The symmetric fixed slate is four candidates per arm, retaining the two already measured/rejected B candidates as its immutable prefix. The six remaining two-CAV candidates quote at most100,800 additional sim-s; with9,600 already spent the complete augmentation ceiling is110,400 <=143,280. No phase/global cap, radius, member limit, horizon or admission/launch criterion is relaxed.

## Verification checkpoint

Native-template correction: a nearest negative-box move label of `none` does not imply a verified native plan has zero offsets. Signed positive/oscillating source waveforms remain eligible for offline augmentation through fitted coordinates in the existing windows; online move vocabulary and all8 admission do not change. The bounded continuation retains all four measured candidates per arm and adds exactly two per arm, for six total each. 50,400 spent +67,200 conservative remaining =117,600 within the unchanged143,280 phase. The final runtime also publishes complete fsynced JSON receipts atomically for detached-arm barriers.

Windows full integration selection: **1090 passed, 1 platform skip, 14 subtests passed**, 430.38s. All required accepted component, launch/effective-imitation and V2V suites were included. The skipped symlink-ledger case is scheduled on cpuV before physics. Live smoke and safety gate remain mandatory; this document is not a launch receipt.

## Owner12:30 safety recovery

`launch_recovery` runs over the unchanged qualified CONFIG/smoke/gate and validates
a separate SHA-bound immutable recovery source manifest. Only `composite_health`
and `warm_health` change executable behaviour: R's full paired comparison is
reported, preserving `release=false` when statistically worse, but permits the
update; ANY R collision/teleport still pauses. A/B retain their original rules.
The exact original R u3 pause remains on disk and is SHA-released by the registered
owner amendment. Exact STOP/FAULT, original owner receipts and unoptimized R3
STARTED receipt are preserved in the recovery control before continuation.
Native completed-wave adoption reuses all R3 physics and practice references;
sealed A/B updates are adopted, never optimized again. One lock per detached role,
one P3 monitor, unchanged u4/u6 schedules. Local focused selection:216 passed,
1 Windows platform skip; Linux qualification is required before resume.

# Owner 14:34 B safety recovery

The registered owner decision `"1"` extends report-only paired TTC/critical-event
comparison to B. R retains its 12:30 amendment; A retains the full original rule.
Any B/R collision or teleport still pauses. `B_SAFETY_REPORT_ONLY_1434.json`
SHA-binds only the original `PAUSE_SAFETY_COMPOSITE_B_u009.json`; other pauses
remain active. The latest sealed safety authorization binds the immutable
recovery source and original launch config. Register the verbatim REGISTRATION
entry and exact clearance procedure in AUTHORITY before publishing that receipt.
Preserve completed episodes, checkpoints, original stops/owners and charges;
resume only after fresh no-live-owner checks. Reuse collected B u9 episodes
before its still-unperformed optimization, with no replay or recharge.

## F6: cpuV SSH connection discipline (all agents and runtime clients)

Use `tools.coopmwm.cpuv_ssh` for **every cpuV command, status read, staging copy,
and download**. All callers use the existing `cpuV` SSH alias, the same OS user,
and the same Linux/WSL client environment. Direct `ssh cpuV`, `scp cpuV:...`,
direct-address connections, alternate aliases, and per-agent retry/poll loops
bypass admission and must be replaced by this wrapper. Clean1 arm/monitor workers
already run locally on cpuV; monitoring should read a combined receipt per call,
not open an SSH connection per arm or per file.

The wrapper shares `/tmp/c4-cpuv-<uid>/master.sock` across all worktrees and agents
on that client. The directory must be a real, user-owned mode-0700 directory.
It creates one `ControlMaster=yes` connection with `ControlPersist=600` seconds.
A cross-process OS lock permits **one active command/transfer at a time** and
times out waiting callers after 60 seconds. Every potential network attempt
(master creation and command/transfer) is separated by **at least 2 seconds**;
failed attempts also consume the interval. There is one connection attempt per
SSH process, a 20-second connect timeout, and no wrapper retries. Concurrent
clients must share this directory; do not create separate pools by arm, run, or PID.

Each operation publishes a durable `active.json` reservation **before** starting
a local session supervisor. That supervisor starts the actual SSH/SCP process,
watches the wrapper's lifetime, and terminates/reaps the owned session on wrapper
cancellation, wrapper death, or deadline expiry. On POSIX, cancellation targets
the session's separate process group (including SCP's SSH child); the shared
persistent master is outside that group. The reservation clears only after
cleanup. A second caller checks it even if a dead wrapper has released its OS
lock, so killing the wrapper PID cannot admit an overlapping session. Streams
and rsync use this same session supervision. An rsync `rsh` wrapper additionally
watches its actual rsync parent: parent exit (including an outer transfer timeout)
requests session cleanup and the wrapper then exits. Owner checks also cover
gate/reservation/rate waits and master setup, rather than relying on pipe closure.

Each session checks the existing master under the lock and uses
`ControlMaster=no`, the same `ControlPath`, and `ProxyCommand=false`. If the master
dies between check and dispatch, the session fails instead of silently opening
another TCP connection. Upload/download and rsync use the same gate/socket.
Host-key verification stays strict; existing identity/hostname/port settings are
read from the client's existing SSH configuration. The wrapper does not modify
local SSH configuration or any remote configuration. **No sshd/MaxStartups change
is authorized by F6; any such change requires explicit owner approval.**

### Client setup and commands (for the later authorized operator)

Run from the staged checkout root with Linux/WSL Python and Linux/WSL OpenSSH,
where the existing `cpuV` alias, identity and known-host record are already
available. Native Windows OpenSSH does not implement the required multiplexing;
the wrapper refuses it before starting SSH. A Windows agent must run the entire
wrapper in the shared WSL environment, rather than call Windows `ssh.exe`.
For this worktree its WSL checkout path is `/mnt/e/agent-work/c4-fix-ssh`.

```bash
# Examples only; run only when remote operation is authorized.
python -B -m tools.coopmwm.cpuv_ssh run -- 'printf "cpuV mux ready\n"'
python -B -m tools.coopmwm.cpuv_ssh --timeout 60 run -- 'cat /path/to/CLEAN1_STATUS.md'
python -B -m tools.coopmwm.cpuv_ssh put ./release.tgz /absolute/staging/release.tgz
python -B -m tools.coopmwm.cpuv_ssh get ./status-copy.md /absolute/run/CLEAN1_STATUS.md
rsync -a -e 'python -B -m tools.coopmwm.cpuv_ssh --timeout 1800 rsh' ./release/ cpuV:/absolute/staging/
```

`run -- '...'` passes one remote-shell command unchanged and inherits binary-safe
stdin/stdout/stderr. When passed multiple words, the wrapper shell-quotes each
word. Both `put` and `get` take **local path first, absolute remote path second**.
The CLI defaults to a 180-second **end-to-end** timeout, including admission,
master checks/creation, rate spacing and the session. `command(..., timeout=...)`
also carries an absolute monotonic deadline across wrapper startup.
`rsync_shell(host, timeout=...)` does the same before starting the local rsync
owner. Both checker transfers pass their existing budgets (1800 seconds for files,
600 for the staged record), retaining the outer rsync timeout as well.
`--timeout 0` removes the session duration limit; an `rsh` wrapper still watches
its rsync parent's lifetime. The admission and connect bounds still apply.
Existing cpuV RPC callers forward their deadline to
the wrapper and wait for supervised cleanup instead of running a competing outer
timeout. Cleanup may extend return time beyond the execution deadline (normally
up to one second of termination grace plus process reaping). An external owner
can still terminate the wrapper; its supervisor independently cleans up before
releasing admission. If an outer timeout kills only rsync, the surviving rsh
wrapper detects owner death while polling (50 ms), requests cleanup, waits for
the supervisor to reap the session, and exits. The outer caller may return before
that cleanup completes; `active.json` continues excluding new sessions until
reaping is confirmed. There is no operation replay. This owner-death path and the
inner deadline both preserve the separate persistent master.
The Python API is `Transport.run(command, ...)` / `Transport.copy(local, remote,
upload=..., ...)`; `command(host, command)` provides a gated argv for existing
subprocess/Popen clients. The shared `tools.coopbandit.ops.remote` used by
`tools.coopmwm.ops.ssh` automatically routes cpuV through that argv; Python-script
callers must bind `PYTHON['cpuV']` to the qualified remote interpreter first.
The configurable RPC/rsync checker and host-parameterized stream bridges use the
same routing. Other hosts retain their existing command lines.

Transport failure is **not** evidence that a remote command did not execute.
Do not automatically replay launches, reservations or other mutations. Read the
existing authoritative receipt/owner first through a later gated call. A failed
command returns its exit status; wrapper/master/timeouts return 255 with a stderr
diagnostic. A later explicit call may recreate a dead master, under the same
lock/rate limit. Do not remove live sockets, force-close shared masters, or shorten
poll/retry intervals to recover from flapping. In idle periods the master expires
itself after 600 seconds.

If the supervisor itself is killed, or startup is interrupted after reservation
publication, `active.json` deliberately remains and future admission times out
with an operator-recovery diagnostic. **Do not delete it based on age or on the
wrapper PID being gone.** The authorized operator must inspect the local process
tree and confirm the supervisor and its entire owned SSH/SCP session group have
exited (terminate/reap only those owned processes if necessary), then remove only
that stale reservation while all clients are stopped. Preserve `last_attempt.json`
and the shared master/socket. Reconcile any remote mutation from its authoritative
receipt before making another request; recovery never replays the command.

This infrastructure is arm-independent: it does not inspect/change arm A policy,
reward, RNG, bank or training configuration, and B/R use the same transport rules.
F6 qualification includes real local-process wrapper cancellation, abrupt death,
RPC deadline/cleanup, transfer and stream lifetime, and lock contention. Rsync
regressions include an extra live owner above the rsh wrapper, owner termination
and kill, actual `subprocess.run` outer timeout, and the inner transfer deadline;
they assert exclusion through session reaping, wrapper exit, gate reuse and one
dispatch only. All network commands are replaced by local Python children.
Live Linux/WSL connection
qualification is still required before use.
