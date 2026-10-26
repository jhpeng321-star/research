# Exact later resume sequence (not executed by S-RESUMEPREP)

Use the dedicated worktree `E:/agent-work/c4-resumeprep`. Wait until S-NATIVEINIT
has committed its completed branch. Keep the original arm and WAIT_OWNER receipts.

## 1. Merge, stage from gpuR, test

PowerShell, in the dedicated worktree:

```powershell
git merge --no-edit c4/nativeinit
python -m pytest -q -p no:cacheprovider --basetemp .pytest-resume-merged tests/unit/test_resumeprep.py
python tools/resumeprep/release.py --host gpuA --source-host gpuR
ssh gpuA 'cd /root/resumeprep_release/runtime && env PYTHONPATH=src:. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/bin/python -m pytest -q -p no:cacheprovider --basetemp /root/resumeprep_release/pytest tests/unit/test_resumeprep.py tests/unit/test_loop14.py tests/unit/test_fastloop.py tests/unit/test_main14.py tests/unit/test_nativeinit.py'
```

Resolve only actual merge conflicts, preserving both the scientific native-init
collection veto/barrier and the operational historical-ledger generation offset.
`release.py` uses a box-local regular-file/SHA-verified baseline and streams each
committed changed source file separately. No source/model/data archive is staged
on Windows; file contents never enter command arguments. No push is involved.

## 2. Snapshot inputs and collector source (later operator, not performed here)

From gpuR, using the owner-installed gpuA alias/key. Stage only the following
approved TRAIN artifacts. Source q-view symlinks materialize as regular bytes only
after lexical/resolved-root/restricted-token/evaluation-seed guards; every file SHA
is verified on gpuA. Existing WM datasets on gpuA were checked present. The two
policy q-view directories are missing and must be staged without substituting a
different corpus or touching another task's data. Preserve any partial attempt.

```bash
P=/root/miniconda3/bin/python
B=/root/resumeprep_probe/boxstage.py
$P "$B" --source-host local --source-root /root --dest-host gpuA --dest-root /root/resumeprep_inputs/core main14_loop/LOOP14_STATE.json main14_loop/seed_ledger.jsonl main14_loop/mainr16_base autodl-tmp/c4/v5/mainstart/mainr16/restart/CONFIG.json autodl-tmp/c4/v5/mainstart/mainr16/restart/REGISTRATION.json autodl-tmp/c4/v5/mainstart/mainr16/restart/gate_panel.json autodl-tmp/c4/v5/mainstart/mainr16/restart/horizon_gate_current.json autodl-tmp/c4/v5/policy14_stage/policy14_native_returns.npz
$P "$B" --source-host local --source-root /root/autodl-tmp/c4/v5/data --dest-host gpuA --dest-root /root/resumeprep_inputs/data --materialize-links v5_native_q genrand_plain_q
```

The native-init executor must be present on both CPUs before any smoke. From gpuA:

```bash
P=/root/miniconda3/bin/python
R=/root/resumeprep_release/runtime
$P "$R/tools/resumeprep/boxstage.py" --source-host local --source-root "$R" --dest-host cpuP --dest-root /home/ubuntu/c4/main14/resumeprep/runtime src traffic_shepherd configs scripts tools
$P "$R/tools/resumeprep/boxstage.py" --source-host local --source-root "$R" --dest-host cpuQ --dest-root /home/ubuntu/c4/main14/resumeprep/runtime src traffic_shepherd configs scripts tools
```

Then prepare the successor and repoint the two CPU `code_root` fields to those
new directories **before prearm**, using `control.py prepare` below. The prepared
configuration does this automatically. Existing CPU data/env/work roots stay
available; seven workers/box, one numerical thread each. Main collection retains
the S-NATIVEINIT four-world scientific barrier. The separately registered two-world
mechanics mini is explicit and does not authorize a four-world replacement.

## 3. Prearm and tiny policy, append bindings, smoke + WM probe

From gpuA:

```bash
P=/root/miniconda3/bin/python
R=/root/resumeprep_release/runtime
A=/root/resumeprep_release/arm
C="$R/tools/resumeprep/control.py"
$P "$C" prepare --old-arm /root/resumeprep_inputs/core/autodl-tmp/c4/v5/mainstart/mainr16/restart --snapshot-root /root/resumeprep_inputs/core --shared-gpu --arm "$A"
$P "$C" prearm --arm "$A"
$P "$C" mini-policy --arm "$A"
```

`prearm` is two CPU synthetic updates, with productive panel SHA checked separately.
`mini-policy` is ten real-panel native-init updates: the minimum existing gate
interval, versus the recommended thirty main updates. A rejected candidate stops
before seed reservation/charge. This stage writes `SMOKE_BINDINGS.json`.

Now, PowerShell in the dedicated worktree:

```powershell
python tools/resumeprep/register_bindings.py
ssh gpuA '/root/miniconda3/bin/python /root/resumeprep_release/runtime/tools/resumeprep/control.py mini --arm /root/resumeprep_release/arm'
```

The second command is the **future authorized SUMO execution boundary**. It checks
the exact appended bindings acknowledgment, reserves/charges 1,922 sim-s in the
successor's parent ledger before dispatch, runs only seeds968142/968143, converts
all four paired arms, then performs two WM-FT steps. Failed charges remain;
no automatic extra smoke SUMO is authorized. Preserve any failed mini directory.
Every configuration/source change invalidates MINI_PASS and requires a freshly
registered cohort; this single registration must never be reused.

## 4. Detached launch and verify

From gpuA:

```bash
$P "$C" launch --arm "$A"
sleep 3
$P "$C" verify --arm "$A"
cat "$A/PID.json" "$A/PREARM.json" "$A/MINI_PASS.json"
tail -F "$A/driver.log"
```

The child holds `driver.lock`; verify checks its exact PID/start tick, current
phase, seed/charge totals and WAIT_OWNER. Inspect the new policy's
`gate_panel_result.json` and `native_gate_result.json` before collection. The
successor starts **fresh cycle0/round16**, with no old policy adoption, retaining
all prior charges/seeds/cycle histories. Its normal outputs are under `$A/loop`.
Cooperative stop: `touch "$A/STOP_ARM" "$A/loop/STOP_DRIVER"`; signal no peer PIDs.

## Host/resource recommendation

The owner's20:34 plan supersedes the earlier lease discussion: **gpuA main+round17,
gpuR coop-off; gpuR rental extended**. Policy microprobe is9% faster onR, WM5%
faster onA; neither is a decisive production benchmark. Stage q-views/core from
gpuR directly; intermittent gpuA transport failures must preserve incomplete dirs.
The prepared shared configuration is short-horizon only: main policy/WM <=0.20
each, round17<=0.50. The guard requires48GB and available memory plus verified
owned allocations sufficient for the combined main cap and2GiB headroom; it never
signals foreign PIDs. Actual real mini-cycle memory/finite completion is mandatory.
For >=90s, use gpuA only after unchanged model-specific horizon gates pass and a
newly registered/memory-tested configuration: suspend large WM/policy overlap and
reserve about20GB for policy alongside round17's roughly24GB. Do not increase
fractions or reduce scientific workload in response to an OOM.
