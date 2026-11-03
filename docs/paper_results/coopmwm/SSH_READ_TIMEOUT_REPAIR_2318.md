# Metadata transport correction — Nov3 23:18 UTC+8

Observed during a read-only cpuQ archive-manifest probe:

```text
Read from remote host 1.15.42.207: Connection timed out
client_loop: send disconnect: Connection reset
```

The existing explicit-diagnostic classifier recognised the second line but not
the first, so the established bounded transport recovery rejected this genuine
connection failure. Added only the exact OpenSSH remote-read timeout form to
the allowlist. The six-attempt/300s bound and identity-verification ordering are
unchanged. Executor/parity/code failures and bare subprocess timeouts remain
non-retryable.

Verification: `python -B -m unittest tools.coopmwm.test_warm_transport_cpu -v`
completed **10 tests PASS**, 0.008s, no SSH/GPU/SUMO. The new regression uses the
actual two-line diagnostic, requires verification before replay, and checks that
a traceback or unknown physics line prevents recovery.

One subsequent archive-manifest metadata read succeeded. The next probe hit a
bare subprocess.TimeoutExpired after35s and stopped, as designed. No physics,
worker restart, remote source deployment, deletion or budget change. The final
integrator must preserve this transport-only fix in the eventual merged pin;
it is independent of their learner/audit-fix files.
