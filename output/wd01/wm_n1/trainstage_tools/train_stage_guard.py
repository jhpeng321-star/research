"""Fail-closed TRAIN-stage interception, version 1; not an OS sandbox.

Separate strict lineage cloned from the accepted synthetic-test guard
(``testguard_tools/wm_testguard.py``). Differences are only narrowings:

* Three frozen stages (``metadata-plan``, ``train-sidecars``, ``cpu40``);
  the policy is stage-specific and immutable once installed.
* Default-deny for every real read: only exact, raw-SHA-pinned registered
  files are readable.  Registered ``data`` files must sit under the
  canonical ``D:/rd-staging/wmn1_feat/pack_v3`` or
  ``D:/rd-staging/wmn1_feat/shards_v3`` roots, under an approved fresh-E
  input root, or inside the fresh sandbox; every other ``D:`` path,
  including all of ``D:/research-dense``, is denied before resolution.
* Every deletion, truncation and cleanup call is denied.  Exactly one
  rename is permitted: a fresh ``.tmp`` receipt file renamed to a NEW
  ``TRAIN_ISOLATION_RECEIPT.json`` inside the sandbox during the
  ``train-sidecars`` stage (the accepted ``label_isolation`` atomic
  receipt write).  Existing destinations, symlinks and hardlinks are
  refused.
* Registered source files are verified to be regular, single-link,
  reparse-free files before hashing; ancestry reparse points are refused.
* Content names carrying HOLDOUT/CALIBRATION/CAL/chunk-5/J4B/INC1/SEALED
  markers are refused at registration and denied on access.

Runtime code under the installed Python environment is trusted only by
narrow file-type membership, exactly like the accepted guard.  This is an
in-process audit-hook policy, NOT an OS sandbox; see LIMITATIONS.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import io
import json
import ntpath
import os
import re
import stat
import sys
import time

GUARD_EXIT_CODE = 87
DENY_MARKER = "[wm-trainstage]"
SPEC_VERSION = 1
POLICY_SPEC_ENV = "WM_TRAINSTAGE_POLICY_SPEC_PATH"
POLICY_SHA_ENV = "WM_TRAINSTAGE_POLICY_SHA256"
REQUIRE_GUARD_ENV = "WM_TRAINSTAGE_REQUIRE_GUARD"
POLICY_FILE_NAME = "child_spec.json"
SANDBOX_PARENT = "E:/tmp/wmn1_train_runs_20261002"
FROZEN_STAGES = ("metadata-plan", "train-sidecars", "cpu40")
CANONICAL_DATA_ROOTS = ("D:/rd-staging/wmn1_feat/pack_v3",
                        "D:/rd-staging/wmn1_feat/shards_v3")
RECEIPT_DST_NAME = "train_isolation_receipt.json"
PLAN_RECEIPT_NAME = "PLAN_RECEIPT.json"
CPU40_THREADS = 4
PACK_ARRAYS = ("veh_step", "veh_mask", "plan", "wall", "wall_mask",
               "counter", "accel_next", "target_mask", "lane_next",
               "arrivals", "veh_kin", "hdv_base", "cav_plan_ch",
               "event_tok", "event_mask")
IDENTITY_FILES = ("TRAIN_IDENTITY_RECEIPT.json", "identities.npy",
                  "global_positions.npy", "vehicle_map.json")
SIDECAR_FILES = ("TRAIN_ISOLATION_RECEIPT.json", "train_outcomes.parquet")
SHARD_COMMON_FILES = ("DONE.json", "slot_vid.json", "slot_index.npy")
SHARD_OUTCOME_FILE = "outcomes.parquet"
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
LIMITATIONS = (
    "in-process CPython audit-hook policy; NOT an OS sandbox, container or "
    "process-level isolation",
    "native code that opens files without raising a CPython audit event "
    "(ctypes, CRT handles inside extensions, JIT, subprocesses started "
    "outside Python) is outside interception scope",
    "numpy/pandas/pyarrow wrappers cover reviewed public entry points of the "
    "installed versions; an unwrapped native entry point may bypass Python "
    "audit events, so coverage must be re-reviewed on version changes",
    "in-memory BytesIO/StringIO are accepted; other anonymous native handles "
    "are refused by reviewed wrappers, not by an OS boundary",
    "the rename exception is narrow but is a real rename: the destination "
    "must be a new non-existent TRAIN_ISOLATION_RECEIPT.json inside the "
    "fresh sandbox; the source must be a single-link regular .tmp file",
    "guard integrity depends on the env-var policy lineage surviving into "
    "every spawned worker; interpreters that skip sitecustomize (e.g. -S, "
    "-I, -E) are refused at subprocess launch and fail closed in children",
)
_TOOL_DIR = os.path.normcase(os.path.realpath(os.path.dirname(__file__)))
_GUARD_FILES = ("sitecustomize.py", "train_stage_guard.py", "run_train_stage.py")
_RUNTIME_EXTS = {".py", ".pyc", ".pyd", ".pyi", ".dll", ".so", ".dylib"}
_METADATA_FILES = {"metadata", "wheel", "entry_points.txt", "top_level.txt",
                   "record", "installer", "pkg-info"}
_ROLES = ("code", "config", "data")
_CODE_EXTS = {".py"}
_CONFIG_EXTS = {".json", ".toml", ".ini", ".cfg", ".yaml", ".yml", ".md"}
_CRITICAL_ENV = (REQUIRE_GUARD_ENV, POLICY_SPEC_ENV, POLICY_SHA_ENV,
                 "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
                 "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS",
                 "WD_FORBID_D", "TMP", "TEMP", "TMPDIR", "PYTHONHOME",
                 "PYTHONSTARTUP", "WM_N1_CANONICAL_PACK",
                 "WM_N1_TRAIN_IDENTITIES", "WM_N1_TRAIN_OUTCOMES") + _THREAD_ENV
_DESTRUCTIVE = {"os.remove", "os.unlink", "os.rmdir", "os.renames",
                "os.truncate", "shutil.rmtree", "shutil.move"}
_AUDITED_PATH_INDEX = {"open": (0,), "os.listdir": (0,), "os.scandir": (0,),
                       "os.mkdir": (0,), "shutil.copyfile": (0, 1),
                       "shutil.copytree": (0, 1)}
_MKSTEMP_TMP = re.compile(r"tmp[a-z0-9_]{4,64}\.tmp")
_INSTALLED = False
_POLICY = None
_ATTESTATION = None
_DENY_LOG_FD = -1
_ENV_SNAPSHOT = {}
_LIVE_WRAPPERS = {}
_SAFE_HANDLES = {}
_GUARDED_FS = None


class GuardInstallError(RuntimeError):
    pass


def _as_text(value):
    if isinstance(value, bytes):
        return os.fsdecode(value)
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    return None


def _strip_extended_prefix(text):
    text = text.replace("/", "\\")
    if text.upper().startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    if text.startswith("\\\\?\\") or text.startswith("\\??\\"):
        return text[4:]
    return text


def _lexical_drive(text):
    return ntpath.splitdrive(_strip_extended_prefix(text))[0].lower()


def _protected_word(text):
    norm = re.sub(r"[\s_.\-]+", "", str(text)).lower()
    return any(word in norm for word in
               ("holdout", "calibration", "calsplit", "chunks5", "chunk5",
                "j4b", "inc1", "sealed")) or norm == "cal"


def _lexical_deny_reason(text):
    """Every D: path is lexically denied; registered canonical data files
    are re-admitted only by exact string membership in the policy tables
    (see _Policy.reason)."""
    if _lexical_drive(text) == "d:":
        return "D: drive denied outside exact registered canonical files"
    return None


def _lexnorm(text):
    return os.path.normcase(os.path.abspath(_strip_extended_prefix(text)))


def _norm(path):
    text = _as_text(path)
    if text is None or not text:
        raise GuardInstallError("invalid or empty path")
    return os.path.normcase(os.path.realpath(text))


def _is_under(path, prefix):
    return path == prefix or path.startswith(prefix + os.sep)


def _runtime_library_path(path, prefix):
    if not _is_under(path, prefix) or path == prefix:
        return False
    relative = os.path.relpath(path, prefix).replace("/", "\\").lower()
    return (relative.startswith("lib\\") or relative == "lib" or
            relative.startswith("dlls\\") or relative == "dlls" or
            relative.startswith("library\\bin\\") or
            relative == "library\\bin" or
            relative.startswith("scripts\\") or relative == "scripts")


def _canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lstat_clean(path, what):
    """Regular existing node with no reparse point; directories allowed."""
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise GuardInstallError(what + " is a symlink: " + str(path))
    if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise GuardInstallError(what + " carries a reparse point: " + str(path))
    return info


def _regular_file(path, what):
    info = _lstat_clean(path, what)
    if not stat.S_ISREG(info.st_mode):
        raise GuardInstallError(what + " is not a regular file: " + str(path))
    return info


def _regular_single_link(path, what):
    info = _regular_file(path, what)
    if info.st_nlink != 1:
        raise GuardInstallError(what + " has multiple hard links: " + str(path))
    return info


def _ancestry_clean(path, what):
    current = os.path.normcase(os.path.abspath(path))
    while True:
        if os.path.lexists(current):
            _lstat_clean(current, what)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def _read_pinned_spec():
    """Read only the fixed sandbox policy file; authenticate bytes first."""
    raw_path = os.environ.get(POLICY_SPEC_ENV)
    expected = os.environ.get(POLICY_SHA_ENV)
    if not isinstance(raw_path, str) or not os.path.isabs(raw_path):
        raise GuardInstallError("absolute child policy path required")
    if _lexical_drive(raw_path) != "e:" or \
            os.path.basename(raw_path).lower() != POLICY_FILE_NAME:
        raise GuardInstallError("child policy must be the fixed E: leaf")
    path = _lexnorm(raw_path)
    if (path != _norm(raw_path) or
            os.path.dirname(os.path.dirname(path)) !=
            _lexnorm(SANDBOX_PARENT)):
        raise GuardInstallError("child policy outside reserved fresh sandbox")
    _sha_pin(expected, "raw child policy")
    _regular_single_link(path, "child policy")
    _ancestry_clean(os.path.dirname(path), "child policy ancestry")
    with open(path, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise GuardInstallError("raw child policy SHA mismatch")
    try:
        spec = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise GuardInstallError("invalid authenticated child policy JSON") from exc
    if not isinstance(spec, dict) or _lexnorm(spec.get("sandbox", "")) != \
            os.path.dirname(path):
        raise GuardInstallError("child policy sandbox/path mismatch")
    return spec, path, expected


def _sha_pin(value, what):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise GuardInstallError("missing exact SHA256 pin: " + what)
    return value


def _relative_train_shard(value):
    if not isinstance(value, str) or "\\" in value:
        raise GuardInstallError("invalid selected TRAIN shard")
    parts = value.split("/")
    allowed_sources = {"chunk1", "chunk2", "chunk3", "chunk4",
                       "chunk1_fix27", "truearms_chunk2",
                       "truearms_chunk3", "walls_pilot_chunk4"}
    if (len(parts) != 3 or any(not p or p in (".", "..") or
                               re.search(r"[:\x00-\x1f]", p) or
                               _protected_word(p) for p in parts) or
            parts[0] not in allowed_sources):
        raise GuardInstallError("non-TRAIN or unsafe selected shard: " + value)
    return parts


def _stage_data_paths(policy):
    """Validate the complete registered data surface before any data hash.

    The launcher separately re-derives the metadata plan, binds its digest,
    and hashes this exact set once per registered stage.  This child checks
    the immutable declaration and all source/runtime pins, without repeatedly
    hashing the large arrays in every worker.
    """
    inputs = policy.spec.get("stage_inputs")
    if not isinstance(inputs, dict):
        raise GuardInstallError("stage_inputs mapping required")
    common = {"pack", "synthetic_test_mode", "synthetic_root"}
    stage_keys = {
        "metadata-plan": {"plan_out"},
        "train-sidecars": {"plan_receipt", "shards_root",
                           "selected_shards", "positive_shards",
                           "identity_out", "sidecar_out"},
        "cpu40": {"plan_receipt", "identity_dir", "sidecar_dir",
                  "cpu_out", "num_workers"},
    }[policy.stage]
    if set(inputs) - common - stage_keys:
        raise GuardInstallError("foreign-stage input key")
    if inputs.get("synthetic_test_mode", False) not in (True, False) or \
            type(inputs.get("synthetic_test_mode", False)) is not bool:
        raise GuardInstallError("synthetic_test_mode must be boolean")
    if policy.stage == "cpu40" and inputs.get("num_workers") not in (0, 2):
        raise GuardInstallError("cpu40 worker count is frozen to 0 or 2")
    pack_raw = inputs.get("pack")
    if not isinstance(pack_raw, str) or not os.path.isabs(pack_raw):
        raise GuardInstallError("absolute pack path required")
    pack = _lexnorm(pack_raw)
    canonical_pack = _lexnorm(CANONICAL_DATA_ROOTS[0])
    canonical_shards = _lexnorm(CANONICAL_DATA_ROOTS[1])
    synthetic = inputs.get("synthetic_test_mode") is True
    synthetic_root = inputs.get("synthetic_root")
    if synthetic:
        if (not isinstance(synthetic_root, str) or
                _lexical_drive(synthetic_root) != "e:"):
            raise GuardInstallError("synthetic mode needs explicit E: fixture root")
        fixture = _lexnorm(synthetic_root)
        if (not _is_under(pack, fixture) or pack == fixture or
                any(_is_under(fixture, p) or _is_under(p, fixture)
                    for p in (*policy.roots, *policy.env_prefixes,
                              policy.sandbox, canonical_pack,
                              canonical_shards))):
            raise GuardInstallError("synthetic fixture root overlaps another root")
        policy.synthetic_root = fixture
    else:
        if synthetic_root is not None or pack != canonical_pack:
            raise GuardInstallError("production pack must be canonical pack_v3")
        policy.synthetic_root = None
    expected = {os.path.join(pack, "manifest.json"),
                os.path.join(pack, "INDEX.json")}
    stage = policy.stage
    if stage != "metadata-plan":
        _sha_pin(policy.spec.get("selection_sha256"), "TRAIN selection")
        plan = inputs.get("plan_receipt")
        if not isinstance(plan, str) or os.path.basename(plan).lower() != PLAN_RECEIPT_NAME.lower():
            raise GuardInstallError("exact PLAN_RECEIPT.json required")
        plan_n = _lexnorm(plan)
        if _lexical_drive(plan_n) != "e:" or _is_under(plan_n, policy.sandbox):
            raise GuardInstallError("plan receipt must be prior E: output")
        expected.add(plan_n)
    elif policy.spec.get("selection_sha256") is not None:
        raise GuardInstallError("metadata plan cannot claim selected TRAIN data")
    if stage == "train-sidecars":
        shards_raw = inputs.get("shards_root")
        if not isinstance(shards_raw, str) or not os.path.isabs(shards_raw):
            raise GuardInstallError("absolute shard root required")
        shards = _lexnorm(shards_raw)
        if synthetic:
            if not _is_under(shards, policy.synthetic_root) or shards == policy.synthetic_root:
                raise GuardInstallError("synthetic shards outside fixture root")
        elif shards != canonical_shards:
            raise GuardInstallError("production shard root must be canonical shards_v3")
        selected = inputs.get("selected_shards")
        if not isinstance(selected, list) or not selected or \
                any(not isinstance(x, str) for x in selected) or \
                len(selected) != len(set(selected)):
            raise GuardInstallError("unique selected TRAIN shard list required")
        positive = inputs.get("positive_shards")
        if not isinstance(positive, list) or any(not isinstance(x, str) for x in positive):
            raise GuardInstallError("ordered positive TRAIN outcome subset required")
        positive_set = set(positive)
        if (len(positive) != len(positive_set) or
                positive != [x for x in selected if x in positive_set]):
            raise GuardInstallError("ordered positive TRAIN outcome subset required")
        for rel in selected:
            parts = _relative_train_shard(rel)
            shard = os.path.join(shards, *parts)
            expected.update(os.path.join(shard, name) for name in SHARD_COMMON_FILES)
        for rel in positive:
            shard = os.path.join(shards, *_relative_train_shard(rel))
            expected.add(os.path.join(shard, SHARD_OUTCOME_FILE))
    elif (inputs.get("shards_root") is not None or
          inputs.get("selected_shards") is not None or
          inputs.get("positive_shards") is not None):
        raise GuardInstallError("this stage cannot register shard files")
    if stage == "cpu40":
        expected.update(os.path.join(pack, name + ".npy") for name in PACK_ARRAYS)
        for key, names in (("identity_dir", IDENTITY_FILES),
                           ("sidecar_dir", SIDECAR_FILES)):
            root = inputs.get(key)
            if not isinstance(root, str) or _lexical_drive(root) != "e:":
                raise GuardInstallError("prior E: " + key + " required")
            root_n = _lexnorm(root)
            if _is_under(root_n, policy.sandbox):
                raise GuardInstallError("prior sidecar cannot be current sandbox")
            expected.update(os.path.join(root_n, name) for name in names)
    elif inputs.get("identity_dir") is not None or inputs.get("sidecar_dir") is not None:
        raise GuardInstallError("sidecar inputs belong only to cpu40")
    output_keys = {"metadata-plan": ("plan_out",),
                   "train-sidecars": ("identity_out", "sidecar_out"),
                   "cpu40": ("cpu_out",)}[stage]
    for key in output_keys:
        value = inputs.get(key)
        if not isinstance(value, str) or not _is_under(_lexnorm(value), policy.sandbox) \
                or _lexnorm(value) == policy.sandbox:
            raise GuardInstallError("fresh sandbox output required: " + key)
    forbidden_outputs = {"plan_out", "identity_out", "sidecar_out", "cpu_out"} - set(output_keys)
    if any(inputs.get(key) is not None for key in forbidden_outputs):
        raise GuardInstallError("foreign-stage output registered")
    return {_lexnorm(path) for path in expected}


class _Policy:
    def __init__(self, spec, spec_path):
        if spec.get("version") != SPEC_VERSION:
            raise GuardInstallError("unsupported policy version")
        self.stage = spec.get("stage")
        if self.stage not in FROZEN_STAGES:
            raise GuardInstallError("unknown frozen stage")
        self.spec = json.loads(json.dumps(spec))
        self.spec_path = spec_path
        self.sandbox = _norm(spec["sandbox"])
        self.sandbox_tmp = _norm(spec["sandbox_tmp"])
        self.roots = tuple(_norm(p) for p in spec["roots"])
        e_roots = spec.get("e_input_roots", ())
        if not isinstance(e_roots, (list, tuple)) or any(
                not isinstance(p, str) or _lexical_drive(p) != "e:"
                for p in e_roots):
            raise GuardInstallError("E: input roots only")
        self.e_input_roots = tuple(_norm(p) for p in e_roots)
        trusted_runtime = _lexnorm(
            "C:/Users/jpeng/Miniconda3/envs/rl_driving")
        if _norm(sys.prefix) != trusted_runtime:
            raise GuardInstallError("rl_driving is the only trusted runtime")
        self.env_prefixes = (trusted_runtime,)
        raw_files = spec.get("files")
        if not isinstance(raw_files, dict) or not raw_files:
            raise GuardInstallError("externally pinned files mapping required")
        if any(not isinstance(p, str) or not os.path.isabs(p) or
               ".." in re.split(r"[\\/]+", p) for p in raw_files):
            raise GuardInstallError("registered paths must be absolute and direct")
        self.files = {_lexnorm(p): dict(value, raw=p)
                      for p, value in raw_files.items()}
        if len(self.files) != len(raw_files):
            raise GuardInstallError("duplicate registered path alias")
        self.interpreter = _norm(spec["interpreter"]["path"])
        self.helpers = {_norm(p): sha for p, sha in spec.get("helpers", {}).items()}
        self.threads = spec.get("threads")
        self.canonical_roots = tuple(_lexnorm(p) for p in CANONICAL_DATA_ROOTS)
        self.forbidden_subroots = set()
        self.deny_subroots = tuple(os.path.join(root, leaf) for root in self.roots
                                   for leaf in ("runs", "archive"))
        self.metadata_dirs = set()
        for path in self.files:
            directory = os.path.dirname(path)
            while directory and directory not in self.metadata_dirs:
                self.metadata_dirs.add(directory)
                if directory in (*self.roots, *self.e_input_roots,
                                 *self.canonical_roots):
                    break
                if not any(_is_under(directory, root)
                           for root in (*self.roots, *self.e_input_roots,
                                        *self.canonical_roots)):
                    break
                parent = os.path.dirname(directory)
                if parent == directory:
                    break
                directory = parent
        if _lexical_drive(self.sandbox) != "e:":
            raise GuardInstallError("sandbox must be on E:")
        if os.path.dirname(self.spec_path) != self.sandbox or \
                os.path.basename(self.spec_path).lower() != POLICY_FILE_NAME:
            raise GuardInstallError("sandbox policy file mismatch")
        if not _is_under(self.sandbox_tmp, self.sandbox) \
                or self.sandbox_tmp == self.sandbox:
            raise GuardInstallError("temporary directory must be a strict "
                                    "sandbox descendant")
        if any(_is_under(self.sandbox, p) or _is_under(p, self.sandbox)
               for p in (*self.roots, *self.e_input_roots, _TOOL_DIR,
                         *self.env_prefixes, *self.canonical_roots)):
            raise GuardInstallError("sandbox overlaps source/input/runtime roots")
        if self.interpreter != _norm(sys.executable):
            raise GuardInstallError("different Python interpreter")
        if type(self.threads) is not int or self.threads != CPU40_THREADS:
            raise GuardInstallError("CPU thread pin must be exactly four")
        if self.helpers:
            raise GuardInstallError("external helpers are outside TRAIN stages")
        expected_data = _stage_data_paths(self)
        actual_data = {p for p, record in self.files.items()
                       if record.get("role") == "data"}
        if actual_data != expected_data:
            raise GuardInstallError("stage data registrations differ from exact "
                                    "required set: missing=%r extra=%r" %
                                    (sorted(expected_data - actual_data),
                                     sorted(actual_data - expected_data)))
        verified = spec.get("verified_data")
        if not isinstance(verified, dict):
            raise GuardInstallError("stage data verification attestation required")
        verified_n = {_lexnorm(p): sha for p, sha in verified.items()}
        if len(verified_n) != len(verified) or set(verified_n) != actual_data:
            raise GuardInstallError("verified data set differs from stage data")
        for path in actual_data:
            if (_sha_pin(verified_n[path], path) !=
                    _sha_pin(self.files[path].get("sha256"), path)):
                raise GuardInstallError("verified data SHA disagrees with pin: " + path)
        self.verified_data = verified_n
        for path, record in self.files.items():
            role = record.get("role")
            if role not in _ROLES:
                raise GuardInstallError("unsupported registered role")
            if os.path.basename(path).lower() == "cal_split_a4.json":
                raise GuardInstallError("CAL split is denied in every role")
            protected_parts = [part for part in re.split(r"[\\/]+", record["raw"])
                               if _protected_word(part)]
            allowed_cal_source = self._is_exact_cal_source(path, record)
            if protected_parts and not (allowed_cal_source and
                                        protected_parts == ["cal_split.py"]):
                raise GuardInstallError("protected registered path refused: " + path)
            suffix = os.path.splitext(path)[1].lower()
            if role == "code" and suffix not in _CODE_EXTS:
                raise GuardInstallError("reviewed code must be Python source")
            if role == "config" and suffix not in _CONFIG_EXTS:
                raise GuardInstallError("unsupported config file type")
            if role != "data" and not any(_is_under(path, root) for root in self.roots):
                raise GuardInstallError("source/config outside pinned project roots: "
                                        + path)
        self.forbidden_subroots.add(
            os.path.join(self.sandbox_tmp, "__wm_guard_selftest_forbidden__"))

    def immutable_fingerprint(self):
        return _canonical_sha256(self.spec)

    def _registered_lexical(self, text):
        return self.files.get(_lexnorm(text))

    def _is_exact_cal_source(self, path, record=None):
        record = self.files.get(path) if record is None else record
        return (record is not None and record.get("role") == "code" and
                isinstance(record.get("sha256"), str) and
                re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None and
                any(path == os.path.join(root, "output", "wd01", "wm_n1",
                                         "evalh", "cal_split.py")
                    for root in self.roots))

    def protected_reason(self, raw):
        text = _as_text(raw)
        if text is None or not text:
            return "unsupported or empty path"
        lexical = _lexical_deny_reason(text)
        if lexical and self._registered_lexical(text) is None:
            return lexical
        try:
            resolved = _norm(text)
            lexical_path = _lexnorm(text)
        except (OSError, ValueError, GuardInstallError):
            return "unresolvable path"
        if lexical_path != resolved:
            return "path alias, symlink or reparse point refused"
        if _lexical_deny_reason(resolved) and resolved not in self.files:
            return "resolved non-registered D: path denied"
        for candidate in (lexical_path, resolved):
            if any(_protected_word(part)
                   for part in re.split(r"[\\/]+", candidate)) and \
                    not self._is_exact_cal_source(candidate):
                return "protected content name"
            if any(_is_under(candidate, prefix)
                   for prefix in (*self.deny_subroots, *self.forbidden_subroots)):
                return "protected root"
            if "test_cal_split" in os.path.basename(candidate).lower():
                return "prohibited real-split test"
            if os.path.basename(candidate).lower() == "cal_split_a4.json" \
                    and not _is_under(candidate, self.sandbox):
                return "real CAL split denied in every mode"
        return None

    def reason(self, raw, access="read"):
        reason = self.protected_reason(raw)
        if reason:
            return reason
        text = _as_text(raw)
        n = _norm(text)
        if _is_under(n, self.sandbox):
            if access == "write" and os.path.lexists(n):
                return "sandbox writes require a new destination"
            return None
        if n == self.spec_path and access == "read":
            return None
        if access == "write":
            return "writes require the fresh sandbox"
        if access == "metadata":
            if n in self.metadata_dirs or any(
                    _runtime_library_path(n, p) for p in self.env_prefixes):
                if os.path.isdir(n):
                    return None
            return "directory discovery outside registered/runtime/sandbox roots"
        if _lexical_drive(text) == "d:" and self._registered_lexical(text) is None:
            return "non-registered D: path denied before resolution"
        if not _protected_word(os.path.basename(text)) \
                and not any(_protected_word(part)
                            for part in re.split(r"[\\/]+", text)) \
                and n in self.files and self.files[n]["role"] == "data" \
                and access == "read":
            return None
        if n in self.files and self.files[n]["role"] in ("code", "config"):
            return None
        if n == self.interpreter or n in self.helpers:
            return None
        if any(_runtime_library_path(n, p) for p in self.env_prefixes):
            if os.path.splitext(n)[1].lower() in _RUNTIME_EXTS:
                return None
            if os.path.basename(n).casefold() in _METADATA_FILES and \
                    os.path.basename(os.path.dirname(n)).casefold().endswith(".dist-info"):
                return None
        if any(_protected_word(part) for part in re.split(r"[\\/]+", text)):
            return "protected content name (HOLDOUT/CAL/chunk-5/J4B/INC1/sealed)"
        return "data denied by default outside the fresh sandbox"

    def check_rename_receipt(self, src, dst):
        """The single permitted rename; returns True only when the move is
        exactly a fresh .tmp receipt file becoming a NEW
        TRAIN_ISOLATION_RECEIPT.json inside the sandbox."""
        if self.stage != "train-sidecars":
            return False
        src_text, dst_text = _as_text(src), _as_text(dst)
        if not src_text or not dst_text:
            return False
        if _lexical_deny_reason(src_text) or _lexical_deny_reason(dst_text):
            return False
        src_lex, dst_lex = _lexnorm(src_text), _lexnorm(dst_text)
        if not _is_under(dst_lex, self.sandbox):
            return False
        if os.path.basename(dst_lex).lower() != RECEIPT_DST_NAME:
            return False
        if os.path.lexists(dst_text):
            return False
        if os.path.dirname(src_lex) != os.path.dirname(dst_lex):
            return False
        if not _MKSTEMP_TMP.fullmatch(os.path.basename(src_lex)):
            return False
        try:
            info = os.lstat(src_text)
        except OSError:
            return False
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return False
        if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            return False
        try:
            if src_lex != _norm(src_text):
                return False
            if not _is_under(src_lex, self.sandbox):
                return False
            parent = os.path.dirname(dst_lex)
            if parent != _norm(os.path.dirname(dst_text)) or not _is_under(parent, self.sandbox):
                return False
            _ancestry_clean(src_text, "receipt temporary file")
            _ancestry_clean(os.path.dirname(dst_text), "receipt destination ancestry")
        except (OSError, ValueError, GuardInstallError):
            return False
        return True

    def verify_pins(self):
        expected_guards = {os.path.join(_TOOL_DIR, name) for name in _GUARD_FILES}
        if not expected_guards.issubset(self.files):
            raise GuardInstallError("guard source pins missing")
        for path, record in self.files.items():
            if record["role"] == "data":
                _regular_single_link(path, "registered data file")
                _ancestry_clean(os.path.dirname(path), "registered data ancestry")
            else:
                _regular_single_link(path, "registered source file")
                _ancestry_clean(os.path.dirname(path), "registered source ancestry")
            if not isinstance(record.get("sha256"), str) \
                    or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                raise GuardInstallError("invalid registered SHA256: " + path)
            # Large feature arrays and TRAIN shard parquets were hashed by the
            # launcher once, after the stage/selection gate, and attested in
            # this immutable policy.  Rehashing 49 GB at every worker startup
            # would reopen all those files before any useful child work.
            light_names = {"manifest.json", "INDEX.json", PLAN_RECEIPT_NAME,
                           *IDENTITY_FILES, "TRAIN_ISOLATION_RECEIPT.json"}
            light_data = (record["role"] == "data" and
                          os.path.basename(path).lower() in
                          {name.lower() for name in light_names})
            if (record["role"] != "data" or light_data) and \
                    _sha256_file(path) != record["sha256"]:
                raise GuardInstallError("SHA256 mismatch for registered file: "
                                        + path)
        for path, expected in ((self.interpreter,
                                self.spec["interpreter"]["sha256"]),):
            # Conda legitimately hard-links python.exe across environments.
            # The executable still needs the exact pinned path, raw SHA and
            # reparse-clean regular-file/ancestry checks.
            _regular_file(path, "interpreter")
            _ancestry_clean(os.path.dirname(path), "interpreter ancestry")
            if not isinstance(expected, str) or len(expected) != 64 \
                    or _sha256_file(path) != expected:
                raise GuardInstallError("SHA256 mismatch for interpreter")
        for path, sha in self.helpers.items():
            _regular_single_link(path, "helper")
            if _sha256_file(path) != sha:
                raise GuardInstallError("SHA256 mismatch for helper: " + path)


def _deny(event, path, reason):
    if _DENY_LOG_FD >= 0:
        try:
            if os.fstat(_DENY_LOG_FD).st_size < 5 * 1024 * 1024:
                os.write(_DENY_LOG_FD, ("DENY %s %r: %s\n"
                                        % (event, path, reason)).encode("utf-8",
                                                                        "replace"))
        except OSError:
            pass
    raise PermissionError("%s %s %r: %s" % (DENY_MARKER, event, path, reason))


def _hook(event, args):
    if _POLICY is None:
        return
    if event == "os.mkdir":
        if any(fd not in (None, -1) for fd in args[2:]):
            _deny(event, args[0], "directory-descriptor mkdir refused")
        raw = args[0]
        reason = _POLICY.reason(raw, "metadata")
        if reason:
            _deny(event, raw, reason)
        # pathlib's mkdir(exist_ok=True) issues this harmless call for the
        # already-created output directory.  The OS still raises EEXIST;
        # pathlib handles it.  Permit only a real, clean sandbox directory.
        path = _norm(raw)
        if _is_under(path, _POLICY.sandbox) and os.path.isdir(path):
            _ancestry_clean(path, "existing sandbox directory")
            return
    if event in {"os.rename", "os.replace"}:
        if (all(fd in (None, -1) for fd in args[2:]) and
                _POLICY.check_rename_receipt(args[0], args[1])):
            return
        _deny(event, args[:2], "all renames are denied except the single new "
                               "TRAIN_ISOLATION_RECEIPT.json receipt move")
    if event == "os.renames":
        _deny(event, args[:2], "renames is denied outright")
    if event in _DESTRUCTIVE:
        _deny(event, args[:2], "deletion, truncation and cleanup are disabled")
    if event in {"os.symlink", "os.link"}:
        _deny(event, args[:2], "links are disabled")
    if event in {"os.putenv", "os.unsetenv"}:
        key = _as_text(args[0])
        if key and key.upper() in _CRITICAL_ENV:
            _deny(event, key, "guard lineage environment is immutable")
        return
    indices = _AUDITED_PATH_INDEX.get(event, ())
    for index in indices:
        raw = args[index]
        if isinstance(raw, int):
            continue
        if raw is None and event in {"os.listdir", "os.scandir"}:
            raw = os.getcwd()
        access = "read"
        if event in {"os.listdir", "os.scandir"}:
            access = "metadata"
        elif event == "os.mkdir" or (event.startswith("shutil.copy") and index == 1):
            access = "write"
        elif event == "open":
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                    (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR |
                                                         os.O_CREAT | os.O_TRUNC |
                                                         os.O_APPEND)):
                access = "write"
        reason = _POLICY.reason(raw, access)
        if reason:
            _deny(event, raw, reason)


def _check_arg(value, depth=0, access="read", allow_none=False):
    if depth > 8:
        _deny("loader", "nested paths", "path nesting exceeds supported depth")
    if value is None and allow_none:
        return
    text = _as_text(value)
    if text is not None:
        if _POLICY is None:
            raise GuardInstallError("loader used before guard installation")
        reason = _POLICY.reason(text, access)
        if reason:
            _deny("loader", text, reason)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_arg(item, depth + 1, access=access)
        return
    if isinstance(value, (io.BytesIO, io.StringIO)):
        return
    known = _SAFE_HANDLES.get(id(value))
    if known is not None and known[0] is value:
        if access == "write" and known[2] != "write":
            _deny("loader", type(value).__name__, "read handle cannot be written")
        if access == "write" and getattr(value, "closed", False):
            _deny("loader", type(value).__name__, "closed write handle")
        if access == "write" and _is_under(_norm(known[1]), _POLICY.sandbox):
            _check_arg(known[1], access="read")
        else:
            _check_arg(known[1], access=access)
        return
    if isinstance(value, io.IOBase) and _as_text(getattr(value, "name", None)) is not None:
        if access == "write":
            _deny("loader", value.name, "unregistered writable handle")
        _check_arg(value.name, access="read")
        if not _same_regular_open_file(value, value.name):
            _deny("loader", value.name, "named handle differs from pinned file")
        return
    _deny("loader", type(value).__name__,
          "unsupported file handle, filesystem or path object")


def check_paths(value):
    _check_arg(value)


def _same_regular_open_file(handle, path):
    """A named stream must identify the actual single-link regular file."""
    try:
        info = _regular_single_link(path, "named stream")
        _ancestry_clean(os.path.dirname(path), "named stream ancestry")
        opened = os.fstat(handle.fileno())
    except (OSError, ValueError, TypeError, GuardInstallError):
        return False
    return (stat.S_ISREG(opened.st_mode) and info.st_ino != 0 and
            opened.st_ino == info.st_ino and opened.st_dev == info.st_dev)


def _pandas_parquet_write_handle(path, frame):
    """Prove pandas just opened this fresh sandbox target for write_table."""
    if type(path) is not str or _POLICY is None or not _is_under(_norm(path), _POLICY.sandbox):
        return None
    module = sys.modules.get("pandas.io.parquet")
    impl = getattr(module, "PyArrowImpl", None)
    method = getattr(impl, "write", None)
    if frame is None or frame.f_code is not getattr(method, "__code__", None):
        return None
    local = frame.f_locals
    handles = local.get("handles")
    handle = getattr(handles, "handle", None)
    if (type(handle) is not io.BufferedWriter or handle.closed or
            not handle.writable() or local.get("path_or_handle") != path or
            local.get("filesystem") is not None or
            not any(item is handle for item in getattr(handles, "created_handles", ()))):
        return None
    name = _as_text(getattr(handle, "name", None))
    if (name is None or _norm(name) != _norm(path) or
            _POLICY.reason(path, "read") is not None or
            _POLICY.reason(path, "write") != "sandbox writes require a new destination" or
            not _same_regular_open_file(handle, path)):
        return None
    try:
        if os.fstat(handle.fileno()).st_size != 0:
            return None
    except OSError:
        return None
    return handle


def _identity_numpy_write_handle(value, frame):
    """Prove the pinned identity builder's live, exclusive output stream."""
    if _POLICY is None or _POLICY.stage != "train-sidecars" or \
            type(value) is not io.BufferedWriter or value.closed or \
            not value.writable() or value.mode != "xb":
        return None
    module = sys.modules.get("train.identity_sidecar")
    builder = getattr(module, "build_identity_sidecar", None)
    if frame is None or frame.f_code is not getattr(builder, "__code__", None):
        return None
    local = frame.f_locals
    name = local.get("name")
    out = local.get("out")
    if (local.get("stream") is not value or
            name not in ("identities.npy", "global_positions.npy") or
            name not in (getattr(module, "IDS_NAME", None),
                         getattr(module, "POSITIONS_NAME", None)) or
            not isinstance(out, os.PathLike)):
        return None
    expected_out = _POLICY.spec["stage_inputs"]["identity_out"]
    path = _as_text(getattr(value, "name", None))
    if (path is None or _norm(out) != _norm(expected_out) or
            _norm(path) != _norm(os.path.join(expected_out, name)) or
            _POLICY.reason(path, "read") is not None or
            _POLICY.reason(path, "write") != "sandbox writes require a new destination" or
            not _same_regular_open_file(value, path)):
        return None
    try:
        if os.fstat(value.fileno()).st_size != 0:
            return None
    except OSError:
        return None
    return path


def _remember_handle(handle, path, access="read"):
    if handle is not None:
        _SAFE_HANDLES[id(handle)] = (handle, path, access)
    return handle


def _wrap(fn, label, index, keys, access="read", allow_none=False, remember=False):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if label == "pyarrow.parquet.write_table":
            target = kwargs.get("where", args[index] if len(args) > index else None)
            handle = _pandas_parquet_write_handle(target, sys._getframe(1))
            if handle is not None:
                _remember_handle(handle, target, access="write")
                if "where" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["where"] = handle
                else:
                    args = (*args[:index], handle, *args[index + 1:])
        elif label == "numpy.save":
            stream = kwargs.get("file", args[index] if len(args) > index else None)
            path = _identity_numpy_write_handle(stream, sys._getframe(1))
            if path is not None:
                _remember_handle(stream, path, access="write")
        value = args[index] if len(args) > index else None
        supplied = len(args) > index
        for key in keys:
            if key in kwargs:
                value, supplied = kwargs[key], True
                _check_arg(value, access=access, allow_none=allow_none)
        if len(args) > index or not supplied:
            _check_arg(value, access=access, allow_none=allow_none)
        filesystem = kwargs.get("filesystem")
        if filesystem is not None and (_GUARDED_FS is None
                                       or not isinstance(filesystem, _GUARDED_FS)):
            _deny("loader", label, "only guarded local filesystem is supported")
        result = fn(*args, **kwargs)
        return _remember_handle(result, value, access=access) if remember else result
    wrapped._wm_trainstage_wrapped = True
    wrapped._wm_trainstage_label = label
    return wrapped


def _set_wrapper(owner, name, wrapped, label):
    try:
        setattr(owner, name, wrapped)
    except (TypeError, AttributeError) as exc:
        raise GuardInstallError("cannot wrap " + label) from exc
    _LIVE_WRAPPERS[label] = (owner, name, wrapped)


def _patch(owner, name, index=0, keys=("path",), access="read",
           allow_none=False, remember=False, label=None):
    label = label or owner.__name__ + "." + name
    original = getattr(owner, name, None)
    if original is None:
        raise GuardInstallError("missing required entry point " + label)
    if label in _LIVE_WRAPPERS:
        return
    _set_wrapper(owner, name,
                 _wrap(original, label, index, keys, access, allow_none, remember),
                 label)


def _factory_class(owner, name, index=0, keys=("source",), access="read",
                   remember=False):
    original = getattr(owner, name, None)
    label = owner.__name__ + "." + name
    if original is None:
        raise GuardInstallError("missing native entry point " + label)
    guarded_call = _wrap(original, label, index, keys, access, remember=remember)

    class FactoryMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, original)

    class GuardedFactory(metaclass=FactoryMeta):
        _wm_trainstage_wrapped = True
        def __new__(cls, *args, **kwargs):
            return guarded_call(*args, **kwargs)
    GuardedFactory.__name__ = name
    _set_wrapper(owner, name, GuardedFactory, label)


def _patch_numpy(np):
    original_load = np.load

    @functools.wraps(original_load)
    def guarded_load(file, *args, **kwargs):
        mmap_mode = kwargs.get("mmap_mode", args[0] if args else None)
        if mmap_mode not in (None, "r", "c"):
            _deny("numpy.load", file, "writable native map refused")
        _check_arg(file)
        return original_load(file, *args, **kwargs)
    _set_wrapper(np, "load", guarded_load, "numpy.load")
    _patch(np, "fromfile", keys=("file",))
    for name in ("save", "savez", "savez_compressed"):
        _patch(np, name, keys=("file",), access="write")
    original = np.memmap

    class GuardedMemmap(original):
        _wm_trainstage_wrapped = True
        def __new__(cls, filename, dtype="uint8", mode="r+", *args, **kwargs):
            _check_arg(filename, access="read" if mode in {"r", "c"} else "write")
            return original.__new__(cls, filename, dtype, mode, *args, **kwargs)
    GuardedMemmap.__name__ = "memmap"
    _set_wrapper(np, "memmap", GuardedMemmap, "numpy.memmap")
    for module in tuple(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("numpy.") \
                and getattr(module, "memmap", None) is original:
            setattr(module, "memmap", GuardedMemmap)


def _patch_pandas(pd):
    for name in ("read_csv", "read_table", "read_fwf", "read_json",
                 "read_pickle", "read_feather", "read_parquet", "read_hdf",
                 "read_sas", "read_spss", "read_stata"):
        _patch(pd, name, keys=("filepath_or_buffer", "path_or_buf", "path",
                               "filepath"))
    _patch(pd, "to_pickle", 1, ("filepath_or_buffer", "path"), "write")
    for cls in (pd.DataFrame, pd.Series):
        for name in ("to_csv", "to_json", "to_pickle", "to_hdf"):
            _patch(cls, name, 1, ("path_or_buf", "path", "filepath_or_buffer"),
                   "write", name in {"to_csv", "to_json"})
    for name in ("to_parquet", "to_feather", "to_stata"):
        _patch(pd.DataFrame, name, 1, ("path", "fname", "path_or_buf"), "write",
               False)


def _patch_arrow_fs(fs):
    global _GUARDED_FS
    native_local = fs.LocalFileSystem
    native_pyfs = fs.PyFileSystem
    native_base = fs.FileSystem
    # These installed Cython extensions cimport pyarrow._fs.FileSystem,
    # directly or through _dataset.pxd / _dataset_parquet.pxd.  Bind the
    # reviewed closure while the native type still lives in pyarrow._fs,
    # before replacing direct Python-level factory aliases.  The audit hook
    # is already active here; these are trusted runtime modules.
    for name in ("pyarrow._dataset", "pyarrow._dataset_parquet",
                 "pyarrow._dataset_orc", "pyarrow._parquet_encryption",
                 "pyarrow._dataset_parquet_encryption"):
        importlib.import_module(name)

    class GuardedHandler(fs.FileSystemHandler):
        def __init__(self, *args, **kwargs):
            self._native = native_local(*args, **kwargs)
        def get_type_name(self):
            return "wm-trainstage-local"
        def normalize_path(self, path):
            _check_arg(path)
            return self._native.normalize_path(path)
        def get_file_info(self, paths):
            _check_arg(paths)
            return self._native.get_file_info(paths)
        def get_file_info_selector(self, selector):
            _check_arg(selector.base_dir)
            return self._native.get_file_info(selector)
        def create_dir(self, path, recursive):
            _check_arg(path, access="write")
            return self._native.create_dir(path, recursive=recursive)
        def delete_dir(self, path):
            _deny("filesystem", path, "deletion disabled")
        def delete_dir_contents(self, path, missing_dir_ok=False):
            _deny("filesystem", path, "deletion disabled")
        def delete_root_dir_contents(self):
            _deny("filesystem", "root", "deletion disabled")
        def delete_file(self, path):
            _deny("filesystem", path, "deletion disabled")
        def move(self, src, dest):
            _deny("filesystem", (src, dest), "rename disabled")
        def copy_file(self, src, dest):
            _check_arg(src)
            _check_arg(dest, access="write")
            return self._native.copy_file(src, dest)
        def open_input_stream(self, path):
            _check_arg(path)
            return _remember_handle(self._native.open_input_stream(path), path)
        def open_input_file(self, path):
            _check_arg(path)
            return _remember_handle(self._native.open_input_file(path), path)
        def open_output_stream(self, path, metadata):
            _check_arg(path, access="write")
            return _remember_handle(self._native.open_output_stream(path,
                                                                    metadata=metadata),
                                    path, access="write")
        def open_append_stream(self, path, metadata):
            _check_arg(path, access="write")
            return _remember_handle(self._native.open_append_stream(path,
                                                                    metadata=metadata),
                                    path, access="write")

    class GuardedLocalFileSystem(native_pyfs):
        _wm_trainstage_wrapped = True
        def __init__(self, *args, **kwargs):
            super().__init__(GuardedHandler(*args, **kwargs))
        @classmethod
        def from_uri(cls, uri):
            _deny("pyarrow.fs", uri, "filesystem URI factory refused")
        _from_uri = from_uri
        _native_from_uri = from_uri
        _fsspec_from_uri = from_uri
    _GUARDED_FS = GuardedLocalFileSystem
    _set_wrapper(fs, "LocalFileSystem", GuardedLocalFileSystem,
                 "pyarrow.fs.LocalFileSystem")

    class FileSystemMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, native_base)
        def __subclasscheck__(cls, value):
            return issubclass(value, native_base)

    class GuardedFileSystem(metaclass=FileSystemMeta):
        _wm_trainstage_wrapped = True
        def __new__(cls, *args, **kwargs):
            _deny("pyarrow.fs", "FileSystem", "native filesystem factory refused")
        @classmethod
        def from_uri(cls, uri):
            _deny("pyarrow.fs", uri, "filesystem URI factory refused")
        _from_uri = from_uri
        _native_from_uri = from_uri
        _fsspec_from_uri = from_uri
    _set_wrapper(fs, "FileSystem", GuardedFileSystem,
                 "pyarrow.fs.FileSystem")

    def deny_factory(name):
        class DeniedFactory:
            _wm_trainstage_wrapped = True
            def __new__(cls, *args, **kwargs):
                _deny("pyarrow.fs", name, "unsupported filesystem factory refused")
            @classmethod
            def from_uri(cls, uri):
                _deny("pyarrow.fs", uri, "filesystem URI factory refused")
        DeniedFactory.__name__ = name
        return DeniedFactory

    denied = {}
    for name in ("PyFileSystem", "SubTreeFileSystem", "_MockFileSystem",
                 "S3FileSystem", "GcsFileSystem", "AzureFileSystem",
                 "HadoopFileSystem"):
        if vars(fs).get(name) is not None:
            denied[name] = deny_factory(name)
            _set_wrapper(fs, name, denied[name], "pyarrow.fs." + name)
    for name in ("copy_files", "_copy_files", "_copy_files_selector"):
        if vars(fs).get(name) is not None:
            def refuse_copy(*args, **kwargs):
                _deny("pyarrow.fs", "copy_files", "native copy factory refused")
            _set_wrapper(fs, name, refuse_copy, "pyarrow.fs." + name)
    # pyarrow.fs re-exports native Cython classes from these modules.  Remove
    # the same public constructor escape from direct module imports too.
    for module_name, mapping in (
            ("pyarrow._fs", {"FileSystem": GuardedFileSystem,
                              "LocalFileSystem": GuardedLocalFileSystem,
                              "PyFileSystem": denied.get("PyFileSystem"),
                              "SubTreeFileSystem": denied.get("SubTreeFileSystem"),
                              "_MockFileSystem": denied.get("_MockFileSystem")}),
            ("pyarrow._s3fs", {"S3FileSystem": denied.get("S3FileSystem")}),
            ("pyarrow._gcsfs", {"GcsFileSystem": denied.get("GcsFileSystem")}),
            ("pyarrow._azurefs", {"AzureFileSystem": denied.get("AzureFileSystem")}),
            ("pyarrow._hdfs", {"HadoopFileSystem": denied.get("HadoopFileSystem")})):
        module = sys.modules.get(module_name)
        if module is not None:
            for name, wrapper in mapping.items():
                if wrapper is not None and vars(module).get(name) is not None:
                    _set_wrapper(module, name, wrapper, module_name + "." + name)


_ARROW_FUNCS = {
    "pyarrow.csv": (("read_csv", 0, ("input_file",), "read"),
                    ("open_csv", 0, ("input_file",), "read"),
                    ("write_csv", 1, ("output_file",), "write")),
    "pyarrow.parquet": (("read_table", 0, ("source",), "read"),
                        ("read_pandas", 0, ("source",), "read"),
                        ("write_table", 1, ("where",), "write")),
    "pyarrow.feather": (("read_table", 0, ("source",), "read"),
                        ("read_feather", 0, ("source",), "read"),
                        ("write_feather", 1, ("dest",), "write")),
    "pyarrow.ipc": (("open_file", 0, ("source",), "read"),
                    ("open_stream", 0, ("source",), "read"),
                    ("new_file", 0, ("sink",), "write"),
                    ("new_stream", 0, ("sink",), "write")),
    "pyarrow.dataset": (("dataset", 0, ("source",), "read"),),
}
_TARGET_MODULES = ("numpy", "pandas", "pyarrow", "pyarrow.fs", *_ARROW_FUNCS,
                   "multiprocessing.spawn")
_PATCHED_MODULES = set()


def _patch_module(name):
    if name in _PATCHED_MODULES:
        return
    module = sys.modules[name]
    if name == "numpy":
        _patch_numpy(module)
    elif name == "pandas":
        _patch_pandas(module)
    elif name == "pyarrow":
        original_osfile = module.OSFile

        class OSFileMeta(type):
            def __instancecheck__(cls, value):
                return isinstance(value, original_osfile)

        class GuardedOSFile(metaclass=OSFileMeta):
            _wm_trainstage_wrapped = True

            def __new__(cls, path, mode="w", *args, **kwargs):
                access = "read" if mode in {"r", "rb"} else "write"
                _check_arg(path, access=access)
                return _remember_handle(original_osfile(path, mode, *args,
                                                        **kwargs), path,
                                        access=access)
        _set_wrapper(module, "OSFile", GuardedOSFile, "pyarrow.OSFile")
        original_map = module.memory_map

        @functools.wraps(original_map)
        def guarded_map(path, mode="r", *args, **kwargs):
            _check_arg(path, access="read" if mode == "r" else "write")
            return _remember_handle(original_map(path, mode, *args, **kwargs),
                                    path, access="read" if mode == "r" else "write")
        _set_wrapper(module, "memory_map", guarded_map, "pyarrow.memory_map")
        _patch(module, "create_memory_map", keys=("path",), access="write",
               remember=True)
    elif name == "pyarrow.fs":
        _patch_arrow_fs(module)
    elif name == "multiprocessing.spawn":
        original_get = module.get_executable
        original_set = module.set_executable

        def get_executable():
            value = original_get()
            if _norm(value) != _POLICY.interpreter:
                _deny("multiprocessing", value, "different interpreter refused")
            for key, expected in _ENV_SNAPSHOT.items():
                if os.environ.get(key) != expected:
                    _deny("multiprocessing", key, "guard environment changed")
            return value

        def set_executable(value):
            if _norm(value) != _POLICY.interpreter:
                _deny("multiprocessing", value, "different interpreter refused")
            return original_set(value)
        _set_wrapper(module, "get_executable", get_executable,
                     name + ".get_executable")
        _set_wrapper(module, "set_executable", set_executable,
                     name + ".set_executable")
    else:
        for fname, index, keys, access in _ARROW_FUNCS[name]:
            _patch(module, fname, index, keys, access)
        if name == "pyarrow.parquet":
            _patch(module.ParquetFile, "__init__", 1, ("source",),
                   label="pyarrow.parquet.ParquetFile.__init__")
            _factory_class(module, "ParquetWriter", keys=("where",), access="write")
        elif name == "pyarrow.ipc":
            for cls in ("RecordBatchFileReader", "RecordBatchStreamReader"):
                _factory_class(module, cls)
    _PATCHED_MODULES.add(name)


def ensure_required_wrappers():
    for name in _TARGET_MODULES:
        importlib.import_module(name)
        _patch_module(name)
    if not wrappers_active(require_all=True):
        raise GuardInstallError("required native/loader coverage is incomplete")


def wrappers_active(require_all=False):
    if not _INSTALLED or not _LIVE_WRAPPERS:
        return False
    if require_all and set(_TARGET_MODULES) != _PATCHED_MODULES:
        return False
    return all(getattr(owner, name, None) is expected
               for owner, name, expected in _LIVE_WRAPPERS.values())


class _GuardLoader:
    def __init__(self, loader, name):
        self._loader, self._name = loader, name
    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create else None
    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch_module(self._name)
    def __getattr__(self, name):
        return getattr(self._loader, name)


class _GuardFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGET_MODULES:
            return None
        for finder in tuple(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is None:
                    raise GuardInstallError("required module has no loader: "
                                            + fullname)
                spec.loader = _GuardLoader(spec.loader, fullname)
                return spec
        return None


def _check_subprocess_args(args, kwargs):
    if kwargs.get("shell") or kwargs.get("executable") is not None \
            or kwargs.get("preexec_fn") is not None:
        _deny("subprocess", "options", "shell/executable/preexec overrides refused")
    cwd = kwargs.get("cwd")
    cwd = os.getcwd() if cwd is None else cwd
    reason = _POLICY.reason(cwd, "metadata")
    if reason:
        _deny("subprocess", cwd, reason)
    if not isinstance(args, (list, tuple)) or not args \
            or not all(_as_text(a) is not None for a in args):
        _deny("subprocess", "argv", "only explicit argv sequences are supported")
    exe = _norm(args[0])
    if exe != _POLICY.interpreter and exe not in _POLICY.helpers:
        _deny("subprocess", args[0], "only the identical interpreter or a pinned "
                                     "helper is permitted")
    env = kwargs.get("env")
    env = os.environ if env is None else env
    folded = {}
    for key, value in env.items():
        upper = key.upper()
        if upper in folded:
            _deny("subprocess", key, "duplicate case-insensitive environment key")
        folded[upper] = value
    for key, expected in _ENV_SNAPSHOT.items():
        if folded.get(key) != expected:
            _deny("subprocess", key,
                  "inherited guard environment cannot be changed or removed")
    if exe == _POLICY.interpreter:
        skip_warning_value = False
        for token in args[1:]:
            if skip_warning_value:
                skip_warning_value = False
                continue
            if token in ("-c", "-m", "--") or not token.startswith("-"):
                break
            if token == "-W":
                skip_warning_value = True
                continue
            if token.startswith("-W"):
                continue
            if token not in {"-B", "-u", "-s", "-O", "-OO", "-q"}:
                _deny("subprocess", token, "interpreter escape flag refused")
        assert_startup()
    elif _sha256_file(exe) != _POLICY.helpers[exe]:
        _deny("subprocess", exe, "helper SHA mismatch")


def _install_subprocess_guard():
    import inspect
    import subprocess
    original = subprocess.Popen.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def guarded_init(self, args, *positional, **kwargs):
        bound = signature.bind(self, args, *positional, **kwargs)
        _check_subprocess_args(args, bound.arguments)
        return original(self, args, *positional, **kwargs)
    _set_wrapper(subprocess.Popen, "__init__", guarded_init,
                 "subprocess.Popen.__init__")
    for name in ("system", "popen"):
        def refuse(*args, **kwargs):
            _deny("process", "shell", "shell-style process launch refused")
        _set_wrapper(os, name, refuse, "os." + name)


def register_forbidden_root(path):
    n = _norm(path)
    if _POLICY is None or not _is_under(n, _POLICY.sandbox):
        raise ValueError("only synthetic sandbox fixtures can be registered")
    _POLICY.forbidden_subroots.add(n)


def registered_sha(path):
    """Raw SHA256 of the exact registered file, or None when unregistered."""
    if _POLICY is None:
        raise GuardInstallError("guard not installed")
    text = _as_text(path)
    record = _POLICY.files.get(_lexnorm(text)) if text else None
    return None if record is None else record["sha256"]


def is_protected(path):
    return _POLICY.reason(path) if _POLICY else "guard not installed"


def is_active():
    return _INSTALLED


def startup_attestation():
    return dict(_ATTESTATION) if _ATTESTATION else None


def assert_startup(strict_pid=True):
    site = sys.modules.get("sitecustomize")
    if not _INSTALLED or _POLICY is None or _ATTESTATION is None or site is None:
        raise GuardInstallError("sitecustomize startup attestation missing; "
                                "no repair")
    if getattr(site, "WM_TRAINSTAGE_BOOT_PID", None) != os.getpid():
        raise GuardInstallError("sitecustomize did not attest this process")
    if strict_pid and _ATTESTATION["pid"] != os.getpid():
        raise GuardInstallError("startup PID mismatch")
    if _POLICY.immutable_fingerprint() != _ATTESTATION["policy_fingerprint"]:
        raise GuardInstallError("policy fingerprint changed")
    for key, expected in _ENV_SNAPSHOT.items():
        if os.environ.get(key) != expected:
            raise GuardInstallError("guard environment changed: " + key)
    spec, path, raw_sha = _read_pinned_spec()
    if (path != _POLICY.spec_path or raw_sha != _ATTESTATION["policy_sha"] or
            _canonical_sha256(spec) != _ATTESTATION["policy_fingerprint"]):
        raise GuardInstallError("child policy file changed")
    _POLICY.verify_pins()
    return dict(_ATTESTATION)


def _seed_tempfile_directory(policy):
    expected = policy.spec["sandbox_tmp"]
    if any(os.environ.get(key) != expected for key in ("TMP", "TEMP", "TMPDIR")):
        raise GuardInstallError("temporary directory environment must match "
                                "policy exactly")
    for key, resolved in (("sandbox", policy.sandbox),
                          ("sandbox_tmp", policy.sandbox_tmp)):
        raw = policy.spec[key]
        if not os.path.isabs(raw) or _lexnorm(raw) != resolved:
            raise GuardInstallError("temporary directory cannot use a resolved "
                                    "path alias")
    directory = policy.sandbox_tmp
    if directory == policy.sandbox or not _is_under(directory, policy.sandbox):
        raise GuardInstallError("temporary directory must be a strict sandbox "
                                "descendant")
    current = directory
    while True:
        info = os.lstat(current)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise GuardInstallError("temporary directory ancestry must contain "
                                    "only directories")
        if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise GuardInstallError("temporary directory ancestry contains a "
                                    "reparse point")
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    import tempfile
    tempfile.tempdir = directory


def install():
    global _INSTALLED, _POLICY, _ATTESTATION, _DENY_LOG_FD, _ENV_SNAPSHOT
    if _INSTALLED or os.environ.get(REQUIRE_GUARD_ENV) != "1":
        return
    spec, spec_path, raw_sha = _read_pinned_spec()
    _POLICY = _Policy(spec, spec_path)
    _POLICY.verify_pins()
    if os.environ.get("PYTHONPATH") != os.pathsep.join(spec["pythonpath"]):
        raise GuardInstallError("PYTHONPATH differs from pinned policy")
    for key in ("PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "WD_FORBID_D"):
        if os.environ.get(key) != "1":
            raise GuardInstallError("required startup flag missing: " + key)
    if os.environ.get("PYTEST_ADDOPTS", "") or os.environ.get("PYTHONHOME") \
            or os.environ.get("PYTHONSTARTUP"):
        raise GuardInstallError("unsafe inherited Python/pytest options")
    if any(_norm(os.environ.get(key, "")) != _POLICY.sandbox_tmp
           for key in ("TMP", "TEMP", "TMPDIR")):
        raise GuardInstallError("temporary directory environment mismatch")
    for key in _THREAD_ENV:
        if os.environ.get(key) != str(_POLICY.threads):
            raise GuardInstallError("thread environment not pinned to %d: %s"
                                    % (_POLICY.threads, key))
    inputs = spec["stage_inputs"]
    expected_stage_env = {
        "WM_N1_CANONICAL_PACK": inputs["pack"] if _POLICY.stage == "cpu40" else None,
        "WM_N1_TRAIN_IDENTITIES": inputs["identity_dir"] if _POLICY.stage == "cpu40" else None,
        "WM_N1_TRAIN_OUTCOMES": inputs["sidecar_dir"] if _POLICY.stage == "cpu40" else None,
    }
    for key, expected in expected_stage_env.items():
        if os.environ.get(key) != expected:
            raise GuardInstallError("stage environment differs from policy: " + key)
    _seed_tempfile_directory(_POLICY)
    _ENV_SNAPSHOT = {key: os.environ.get(key) for key in _CRITICAL_ENV}
    sys.dont_write_bytecode = True
    sys.addaudithook(_hook)
    deny_log = os.path.join(_POLICY.sandbox,
                            "guard_denies_%d_%d.log" % (os.getpid(), time.time_ns()))
    _DENY_LOG_FD = os.open(deny_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    sys.meta_path.insert(0, _GuardFinder())
    _install_subprocess_guard()
    _ATTESTATION = {"pid": os.getpid(),
                    "policy_sha": raw_sha,
                    "policy_fingerprint": _POLICY.immutable_fingerprint(),
                    "spec_path": spec_path,
                    "guard_shas": {n: _POLICY.files[os.path.join(_TOOL_DIR, n)]["sha256"]
                                   for n in _GUARD_FILES},
                    "deny_log": deny_log, "sandbox": _POLICY.sandbox,
                    "stage": _POLICY.stage}
    _INSTALLED = True
    for name in _TARGET_MODULES:
        if name in sys.modules:
            _patch_module(name)


def selfcheck():
    if not _INSTALLED:
        raise GuardInstallError("guard installation incomplete")
    probe = os.path.join(_POLICY.sandbox_tmp,
                         "__wm_guard_selftest_forbidden__", "probe.bin")
    try:
        with open(probe, "rb"):
            pass
    except PermissionError as exc:
        if DENY_MARKER in str(exc):
            return
        raise GuardInstallError("wrong denial source") from exc
    raise GuardInstallError("audit hook failed to deny synthetic probe")


def install_and_selfcheck():
    install()
    if os.environ.get(REQUIRE_GUARD_ENV) == "1":
        selfcheck()
