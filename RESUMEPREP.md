# S-RESUMEPREP — prepared; original loop remains WAIT_OWNER

Branch `c4/resumeprep`, from `c4/mainstart` **1aee086e**. No changes to
`c4-nativeinit`, CPU hosts, live loop/state/config, or peer processes. No new SUMO.
Smoke registration was appended to the shared register; nothing was dispatched.

## Restart mechanics

`loop14_resume.py` provides one durable operational fix/retry **per authorized
continuation**, shared across setup, foreground phases and overlapped policy.
Receipts preserve the original failure, repair, attempt and charges; a second
failure or scientific refusal reaches WAIT_OWNER. Official retries use distinct
outputs and preserve verified worlds/charges. GPU busy waits consume no retry.

Overlap starts policy k+1 during WM-FT k from an immutable accepted-WM snapshot
and the just-completed conversion's returns/credit. Panel/horizon bytes and WM,
policy, returns and credit hashes are recorded before training. Only the main
thread writes loop state. Promotion retains the actual training WM even if the
candidate WM is accepted. Concurrent GPU guards admit only this driver's live,
exact PID/start-tick children; no foreign process is signalled.

`control.py` creates a new cycle0/round16 root, archives histories, clears old
policy adoption, and retains all historical seeds/87,399 charges. Historical
ledger generation offsets prevent those charges being mistaken for a new-cycle
allowance. Main excludes the smoke's two reserved seeds.

## Mini-cycle and staging

Every source/configuration requires a matching hash-bound MINI_PASS before launch:
two CPU synthetic prearm updates; ten real-panel native-init updates; registered
mechanics smoke; complete conversion; two WM steps. Ten is the existing panel
gate interval; the main uses S-NATIVEINIT's recommended thirty updates. Rejection
blocks dispatch. Exact candidate/source/config/native-gate hashes are appended
before the separately acknowledged execution boundary. The two-world mechanics
transaction does not replace or certify the main's four-world efficacy barrier.

`boxstage.py` validates regular members, traversal and SHA manifests. Baseline
staging occurs on boxes; committed changed files stream separately, never as
base64/oversized Windows command arguments. Initial full checkout hit disk-full;
sparse checkout recovered. Large laptop-relayed transfer attempts were preserved;
the verified box-local baseline staged1,072 files successfully.

## Measured host choice

Same round16 SHA5b256941…, identical policy/FT config hashes and seeds, bounded
synthetic inputs: policy10s median after warmup **gpuR7.836s / gpuA8.616s**
(three updates); WM120s/B4/K96/P24 **60.774s /57.771s** (one measured step after
warmup). Peak reserved **1.90GiB**; observed owned gpuR process2,406MiB, below6GB.
These are microprobes, not production-data throughput. Raw receipts are committed;
failed invocation/input probes are retained remotely. Both exact final PIDs exited.

The newer20:34 owner plan selects **gpuA for main + round17**, gpuR for coop-off.
There is no decisive microprobe speed winner;48GB and future>=90s capacity favor A.
Prepared short-horizon sharing caps main policy/WM at0.20 each, round17 at0.50,
with an explicit free-memory guard. A new real mini-cycle must prove capacity.
Long horizons require a newly registered configuration and serialized large phases.

## Registration and launch

Registered **S-C0DIAG-SMOKE-1**: rechecked seeds968142/968143,1900vph,5%/40%,
warmup300,start360,native/policy300s,extrasOFF: **1,922sim-s**. Retain210,000 total;
charge before dispatch. Timeout amendment sets driver/registered WM budget **4.5h**
**and** trainer `curriculum.max_hours=4.5`, preserving40 steps/data/checks; exact
training-safe diagnostic flags only.

Exact **merge → tests → box staging → prearm → bindings → smoke → launch → verify**
commands: `tools/resumeprep/RUNBOOK.md`. S-NATIVEINIT must land first; its gates remain
authoritative. Verification: `tools/resumeprep/receipts/VERIFIED.json`.
