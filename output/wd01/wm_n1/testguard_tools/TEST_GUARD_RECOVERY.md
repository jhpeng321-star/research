# WM-N1-RECOVERY-01 — synthetic guard, version 3 source draft

Status: **not runtime-accepted**. This revision was edited and reviewed as source
only. No Python imports, pytest, launcher, simulation, training, outcome reads,
D: access, or scientific data inventory were performed during this repair.
Independent coordinator source acceptance is required before synthetic checks.
This is the existing guard-recovery task, not a new scientific experiment.

The earlier 555b212 and 0911f48 versions remain in Git history. Existing logs and
receipts are untouched. Only the five guard source/test/report files changed.

## Coordinator manifest and invocation

Use the existing rl_driving interpreter to invoke `run_guarded.py` with:

```
--allowlist E:/tmp/review/REVIEWED.json
--allowlist-sha256 <SHA256 of the reviewed JSON bytes>
--sandbox-root E:/<existing dedicated synthetic sandbox parent>
--label <letters_digits_underscore_hyphen>
--pytest-arg=-q
```

There is no arbitrary `--allow-path`, alternate `--python`, plugin, positional
pytest selection, or arbitrary pytest argument surface. Optional pytest extras
are only `-q`, `-v`, and `--disable-warnings`. Optional `--extra-root` adds denials.
The JSON schema is:

```json
{
  "version": 3,
  "nodes": ["E:/path/to/test_reviewed.py::test_synthetic"],
  "files": {
    "E:/path/to/test_reviewed.py": {"role": "code", "sha256": "<64 hex>"},
    "E:/path/to/imported_module.py": {"role": "code", "sha256": "<64 hex>"},
    "E:/path/to/reviewed_config.json": {"role": "config", "sha256": "<64 hex>"}
  },
  "pythonpath": ["E:/path/to/reviewed/source/root"],
  "helpers": {}
}
```

Every selected node must name an absolute Python source path and an exact test
function, optionally with class/parameter suffix. External recovery tests are
supported; they are not restricted to this tool's own tests directory. Every
selected/imported project source and config must be reviewed and pinned in
`files`. `pythonpath` directories need reviewed code descendants and only enable
module discovery; they do not grant data reads. The three guard sources and the
exact running interpreter are additionally pinned by the launcher. Put the guard
sources in the reviewed manifest too when an independent fixed revision pin is
required rather than launcher-time identity recording.

Config is an explicit coordinator trust decision, not a data escape flag. Allowed
config extensions are JSON, TOML, INI, CFG, YAML and Markdown (including reviewed
registration documents). Data roles and parquet/npy/
jsonl/CSV disguised as config are rejected before hashing. The coordinator must
review each JSON config's intended use; this tool cannot infer whether arbitrary
JSON content is scientifically sealed. All protected path rules override even an
explicit reviewed config entry.

## Boundaries and startup

- All D: spellings are denied lexically before resolution, including extended
  Windows prefixes, then checked again after resolution.
- Registered repo/worktree `runs` and `archive` trees and the real CAL split
  basename are denied before allowances. `test_cal_split` modules are refused.
- All other research data outside the fresh sandbox is denied by default.
- Reads of exact pinned code/config/manifest/interpreter files are permitted.
  Installed Python runtime prefixes permit only Python/native code extensions and
  specific package metadata filenames. No broad runtime JSON/data access exists.
- Directory metadata discovery is permitted for reviewed-source ancestors and
  runtime directories. It does not grant file content access.
- Writes occur only inside the fresh E: sandbox. Delete, unlink, rename,
  truncate, move, rmtree and cleanup operations are refused throughout the child.
  Raw descriptor writes are part of the trusted-code scope noted below.
- New symlink/hardlink destinations are allowed only inside the same fresh
  sandbox as their target/source. Relative symlink targets resolve against the
  destination parent. Existing destinations, protected targets and all links
  outside the sandbox are refused; no rename or removal is enabled.
- TEMP/TMP/TMPDIR and pytest's never-existing basetemp are inside that sandbox.
  Python bytecode writes, pytest cache, external plugin autoload, inherited
  PYTEST_ADDOPTS and repository conftest collection are disabled.
  Fixed `--capture=sys` avoids pytest descriptor capture opening Windows `nul`;
  fixed `--log-file <fresh-sandbox>/pytest.log` directs the builtin logging
  plugin to a new sandbox file instead of its default null device. Neither
  setting adds a filesystem exception. Installed package metadata basenames are
  compared case-insensitively and must be direct children of `.dist-info`.
- `sitecustomize` attests the startup PID after installation/selfcheck. Failures
  exit 87 using `os._exit`; CPython cannot swallow them and continue the test.
- Preflight and actual pytest bootstrap require the existing `sys.modules`
  guard and sitecustomize marker. Neither imports a missing guard nor repairs it.
  Policy SHA, full policy contents, guard/interpreter/allowlist/source/config
  hashes and PID are checked before importing pytest or project code.
- Pytest argv must exactly equal the pinned policy argv. Each started run has
  fresh started/finished or interrupted receipts plus logs inside its sandbox;
  no prior receipt is truncated or rewritten.

## Loader and process coverage

Audit hooks use event-specific path operand positions: an open mode such as `r`
is never misclassified as a filename. Loader wrappers check only each documented
path argument, not arbitrary string columns or array values. Lists/tuples recurse
and excessive nesting fails closed. Anonymous native handles and unsupported
filesystem objects are refused. Python filename-bearing IO handles and memory
buffers are supported; native handles created by guarded factories carry checked
path provenance.

Required imported entry points are verified by live attribute identity, and
preflight imports all required modules before claiming complete coverage:

- NumPy load/fromfile/save/savez/savez_compressed; memmap remains a subclass so
  calls and isinstance checks continue to work.
- Pandas readers; actual DataFrame/Series serialization methods rather than
  nonexistent module-level to_csv/to_parquet functions.
- PyArrow CSV, parquet, feather, dataset and IPC functions; ParquetFile,
  ParquetWriter, OSFile, memory_map/create_memory_map and IPC reader factories.
  Native immutable classes are not mutated in place. OSFile uses a factory with
  instance compatibility. LocalFileSystem uses PyFileSystem/FileSystemHandler
  to check all delegated paths, including C++ filesystem callbacks. Destructive
  native filesystem methods refuse. Only the guarded local filesystem is
  accepted as an explicit filesystem argument.
- subprocess.Popen rejects shell/string commands, executable/preexec overrides,
  alternate interpreters and unreviewed helpers. Positional options are bound to
  the actual Popen signature before validation. Child cwd passes the metadata
  policy before process creation, including lexical D: rejection. Critical inherited environment
  values are immutable and case-insensitive duplicate keys are rejected.
  Interpreter escape flags are refused. Multiprocessing spawn executable access
  is also checked, preserving the same interpreter and inherited guard env for
  trusted spawned workers, including DataLoader lineage.

Explicitly reviewed CPU helpers may appear as exact executable SHA pairs in
`helpers`; these are trusted programs, not automatically sandboxed processes.

This remains **Python interception over trusted reviewed code, not an OS
sandbox**. Raw descriptors, direct ctypes/syscalls, direct low-level process
creation, mutation of guard internals, and unrelated/unwrapped native entry
points are outside the contract. Installed runtime code is trusted. The focused
native fixture tests must actually pass before acceptance; merely failing closed
on an immutable native patch is not a functional success.

## Draft synthetic verification

The 17 exact functions in `tests/test_guard.py` cover attestation and actual
native entry points; harmless positive IO; denied fake outcome roots; D: lexical
strings without disk access; protected-before-allow precedence; directory-only
metadata; pandas columns versus paths; memmap class behavior; PyArrow native
filesystem and IPC; unsupported handles/deep lists; inherited child startup;
environment-removal/alternate-interpreter rejection; missing startup reference
before pytest import; external exact nodes; changed synthetic source/config SHA;
no arbitrary data allowance/plugin surface; fresh sandbox and denied cleanup;
fresh synthetic aliases with protected-path precedence.

Fixtures are harmless synthetic files inside the fresh sandbox. No test opens a
real protected path. Missing-attestation behavior is simulated in a child whose
real guard remains installed by removing its sys.modules reference; it does not
launch a child with a stripped guard environment. No TemporaryDirectory cleanup,
junction creation, deletion, or old basetemp reuse is present. The alias test
creates only fresh sandbox hardlinks/symlinks; the positive symlink portion is
reported as skipped if Windows explicitly denies symlink privilege.

Coordinator runtime attempt against dc80e40: preflight exit 0; pytest child exit
1 before collection because descriptor capture attempted to open Windows `nul`.
Seventeen nodes were selected; zero tests were collected or executed. The
preserved receipt is
`E:/tmp/wmn1_guard_runs_20261002/guard_v3_accept_20261002_084644_608238_11428/finished.receipt.json`.
Coordinator attempt 2 against c4e809c: preflight passed; pytest exited 3 before
collection because the builtin logging plugin opened its default Windows null
device. This was a child startup failure, not a preflight failure. Its preserved
log is
`E:/tmp/wmn1_guard_runs_20261002/guard_v3_accept2_20261002_085006_018543_20004/guarded_run.log`.
This source-only follow-up adds the explicit fresh-sandbox log path and has not
been run. Root owns the next guarded allocation, receipts, ledger and acceptance.
