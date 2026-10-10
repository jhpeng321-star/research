"""Attested synthetic-test interception, version 3; not an OS sandbox.

Only the fresh sandbox may contain data. Reviewed project code/config files
are exact SHA-pinned reads. Installed-runtime code is trusted by narrow type.
Native IO is checked at real path parameters; unsupported handles are refused.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import io
import json
import ntpath
import os
import stat
import sys
import time

GUARD_EXIT_CODE = 87
DENY_MARKER = "[wm-testguard]"
SPEC_VERSION = 3
POLICY_SPEC_ENV = "WM_TESTGUARD_POLICY_SPEC"
POLICY_SHA_ENV = "WM_TESTGUARD_POLICY_SHA256"
REQUIRE_GUARD_ENV = "WM_TESTGUARD_REQUIRE_GUARD"
_TOOL_DIR = os.path.normcase(os.path.realpath(os.path.dirname(__file__)))
_GUARD_FILES = ("sitecustomize.py", "wm_testguard.py", "run_guarded.py")
_RUNTIME_EXTS = {".py", ".pyc", ".pyd", ".pyi", ".dll", ".so", ".dylib"}
_METADATA_FILES = {"metadata", "wheel", "entry_points.txt", "top_level.txt", "record", "installer", "pkg-info"}
_CRITICAL_ENV = (REQUIRE_GUARD_ENV, POLICY_SPEC_ENV, POLICY_SHA_ENV,
                 "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
                 "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS", "WD_FORBID_D",
                 "TMP", "TEMP", "TMPDIR", "PYTHONHOME", "PYTHONSTARTUP")
_DESTRUCTIVE = {"os.remove", "os.rmdir", "os.rename", "os.replace", "os.truncate",
                "shutil.move", "shutil.rmtree"}
_AUDITED_PATH_INDEX = {"open": (0,), "os.listdir": (0,), "os.scandir": (0,),
                       "os.mkdir": (0,), "shutil.copyfile": (0, 1),
                       "shutil.copytree": (0, 1)}
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


def _lexical_deny_reason(text):
    if ntpath.splitdrive(_strip_extended_prefix(text))[0].lower() == "d:":
        return "D: drive denied before filesystem resolution"
    return None


def _norm(path):
    text = _as_text(path)
    if text is None or not text or _lexical_deny_reason(text):
        raise GuardInstallError("invalid or forbidden path (D: is never resolved)")
    return os.path.normcase(os.path.realpath(text))


def _is_under(path, prefix):
    return path == prefix or path.startswith(prefix + os.sep)


def _canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Policy:
    def __init__(self, spec):
        if spec.get("version") != SPEC_VERSION:
            raise GuardInstallError("unsupported policy version")
        self.spec = json.loads(json.dumps(spec))
        self.sandbox = _norm(spec["sandbox"])
        self.sandbox_tmp = _norm(spec["sandbox_tmp"])
        self.roots = tuple(_norm(p) for p in spec["roots"])
        self.env_prefixes = tuple(_norm(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix})
        self.forbidden_subroots = set()
        self.files = {_norm(p): value for p, value in spec["files"].items()}
        self.allowlist = _norm(spec["allowlist_path"])
        self.interpreter = _norm(spec["interpreter"]["path"])
        self.helpers = {_norm(p): sha for p, sha in spec.get("helpers", {}).items()}
        self.deny_subroots = tuple(os.path.join(root, leaf) for root in self.roots for leaf in ("runs", "archive"))
        self.metadata_dirs = set()
        for path in (*self.files, self.allowlist):
            directory = os.path.dirname(path)
            while directory and directory not in self.metadata_dirs:
                self.metadata_dirs.add(directory)
                parent = os.path.dirname(directory)
                if parent == directory:
                    break
                directory = parent
        if ntpath.splitdrive(self.sandbox)[0].lower() != "e:":
            raise GuardInstallError("sandbox must be on E:")
        if not _is_under(self.sandbox_tmp, self.sandbox):
            raise GuardInstallError("temporary directory outside sandbox")
        if any(_is_under(self.sandbox, p) or _is_under(p, self.sandbox)
               for p in (*self.roots, _TOOL_DIR, *self.env_prefixes)):
            raise GuardInstallError("sandbox overlaps source/runtime roots")
        if self.interpreter != _norm(sys.executable):
            raise GuardInstallError("different Python interpreter")
        self.forbidden_subroots.add(os.path.join(self.sandbox_tmp, "__wm_guard_selftest_forbidden__"))

    def immutable_fingerprint(self):
        return _canonical_sha256(self.spec)

    def protected_reason(self, raw):
        text = _as_text(raw)
        if text is None or not text:
            return "unsupported or empty path"
        lexical = _lexical_deny_reason(text)
        if lexical:
            return lexical
        try:
            resolved = _norm(text)
            lexical_path = os.path.normcase(os.path.abspath(text))
        except (OSError, ValueError, GuardInstallError):
            return "unresolvable path"
        if _lexical_deny_reason(resolved):
            return "resolved D: drive denied"
        for candidate in (lexical_path, resolved):
            if any(_is_under(candidate, prefix) for prefix in (*self.deny_subroots, *self.forbidden_subroots)):
                return "protected root"
            basename = os.path.basename(candidate).lower()
            if "test_cal_split" in basename:
                return "prohibited real-split test"
            if basename == "cal_split_a4.json" and not _is_under(candidate, self.sandbox):
                return "real CAL split denied in every mode"
        return None

    def reason(self, raw, access="read"):
        reason = self.protected_reason(raw)
        if reason:
            return reason
        n = _norm(raw)
        if _is_under(n, self.sandbox):
            return None
        if access == "write":
            return "writes require fresh sandbox"
        if access == "metadata":
            if n in self.metadata_dirs or any(_is_under(n, p) for p in self.env_prefixes):
                if os.path.isdir(n):
                    return None
            return "directory discovery outside source/runtime/sandbox"
        if n in self.files or n == self.allowlist or n == self.interpreter or n in self.helpers:
            return None
        if any(_is_under(n, p) for p in self.env_prefixes):
            if os.path.splitext(n)[1].lower() in _RUNTIME_EXTS:
                return None
            if os.path.basename(n).casefold() in _METADATA_FILES and os.path.basename(os.path.dirname(n)).casefold().endswith(".dist-info"):
                return None
        return "data denied by default outside fresh sandbox"

    def verify_pins(self):
        expected_guards = {os.path.join(_TOOL_DIR, name) for name in _GUARD_FILES}
        if not expected_guards.issubset(self.files):
            raise GuardInstallError("guard source pins missing")
        for path, record in self.files.items():
            if record.get("role") not in {"code", "config"}:
                raise GuardInstallError("unsupported reviewed-file role")
            suffix = os.path.splitext(path)[1].lower()
            if record["role"] == "code" and suffix != ".py":
                raise GuardInstallError("reviewed code must be Python source")
            if record["role"] == "config" and suffix not in {".json", ".toml", ".ini", ".cfg", ".yaml", ".yml", ".md"}:
                raise GuardInstallError("unsupported config file type")
        pairs = [(path, rec["sha256"]) for path, rec in self.files.items()]
        pairs += [(self.allowlist, self.spec["allowlist_sha256"]),
                  (self.interpreter, self.spec["interpreter"]["sha256"])]
        pairs += list(self.helpers.items())
        for path, expected in pairs:
            reason = self.protected_reason(path)
            if reason:
                raise GuardInstallError("protected manifest entry refused before hashing: " + reason)
            if not isinstance(expected, str) or len(expected) != 64 or _sha256_file(path) != expected:
                raise GuardInstallError("SHA256 mismatch for reviewed file: " + path)
        for node in self.spec["nodes"]:
            path, separator, name = node.partition("::")
            if not separator or not name or self.files.get(_norm(path), {}).get("role") != "code":
                raise GuardInstallError("test node is not exact and source-pinned")


def _deny(event, path, reason):
    if _DENY_LOG_FD >= 0:
        try:
            if os.fstat(_DENY_LOG_FD).st_size < 5 * 1024 * 1024:
                os.write(_DENY_LOG_FD, ("DENY %s %r: %s\n" % (event, path, reason)).encode("utf-8", "replace"))
        except OSError:
            pass
    raise PermissionError("%s %s %r: %s" % (DENY_MARKER, event, path, reason))


def _hook(event, args):
    if _POLICY is None:
        return
    if event in _DESTRUCTIVE:
        _deny(event, args[:2], "deletion, rename and cleanup are disabled")
    if event in {"os.symlink", "os.link"}:
        source, destination = args[:2]
        if any(fd not in (None, -1) for fd in args[2:]):
            _deny(event, destination, "directory-descriptor links unsupported")
        source_text, destination_text = _as_text(source), _as_text(destination)
        if source_text is None or destination_text is None:
            _deny(event, destination, "only named synthetic links supported")
        if _lexical_deny_reason(source_text) or _lexical_deny_reason(destination_text):
            _deny(event, destination, "D: link refused before resolution")
        # Relative symlink targets are relative to the link's parent, not cwd.
        target = source_text
        if event == "os.symlink" and not os.path.isabs(target):
            target = os.path.join(os.path.dirname(os.path.abspath(destination_text)), target)
        for path, access in ((target, "read"), (destination_text, "write")):
            reason = _POLICY.reason(path, access)
            if reason or not _is_under(_norm(path), _POLICY.sandbox):
                _deny(event, path, reason or "links require same fresh sandbox")
        if os.path.lexists(destination_text):
            _deny(event, destination_text, "link destination must be new")
        return
    if event in {"os.putenv", "os.unsetenv"}:
        key = _as_text(args[0])
        if key and key.upper() in _CRITICAL_ENV:
            _deny(event, key, "guard lineage environment is immutable")
        return
    indices = _AUDITED_PATH_INDEX.get(event, ())
    for index in indices:
        raw = args[index]
        # Existing raw descriptors are outside this trusted-code interception scope.
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
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
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
        _check_arg(known[1], access=access)
        return
    # Python file handles carry an auditable filename; anonymous/native handles do not.
    if isinstance(value, io.IOBase) and _as_text(getattr(value, "name", None)) is not None:
        _check_arg(value.name, access=access)
        return
    _deny("loader", type(value).__name__, "unsupported file handle, filesystem or path object")


def check_paths(value):
    _check_arg(value)


def _remember_handle(handle, path):
    _SAFE_HANDLES[id(handle)] = (handle, path)
    return handle


def _wrap(fn, label, index, keys, access="read", allow_none=False, remember=False):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        value = args[index] if len(args) > index else None
        supplied = len(args) > index
        for key in keys:
            if key in kwargs:
                value, supplied = kwargs[key], True
                _check_arg(value, access=access, allow_none=allow_none)
        if len(args) > index or not supplied:
            _check_arg(value, access=access, allow_none=allow_none)
        filesystem = kwargs.get("filesystem")
        if filesystem is not None and (_GUARDED_FS is None or not isinstance(filesystem, _GUARDED_FS)):
            _deny("loader", label, "only guarded local filesystem is supported")
        result = fn(*args, **kwargs)
        return _remember_handle(result, value) if remember else result
    wrapped._wm_testguard_wrapped = True
    wrapped._wm_testguard_label = label
    return wrapped


def _set_wrapper(owner, name, wrapped, label):
    try:
        setattr(owner, name, wrapped)
    except (TypeError, AttributeError) as exc:
        raise GuardInstallError("cannot wrap " + label) from exc
    _LIVE_WRAPPERS[label] = (owner, name, wrapped)


def _patch(owner, name, index=0, keys=("path",), access="read", allow_none=False, remember=False, label=None):
    label = label or owner.__name__ + "." + name
    original = getattr(owner, name, None)
    if original is None:
        raise GuardInstallError("missing required entry point " + label)
    if label in _LIVE_WRAPPERS:
        return
    _set_wrapper(owner, name, _wrap(original, label, index, keys, access, allow_none, remember), label)


def _factory_class(owner, name, index=0, keys=("source",), access="read", remember=False):
    original = getattr(owner, name, None)
    label = owner.__name__ + "." + name
    if original is None:
        raise GuardInstallError("missing native entry point " + label)
    guarded_call = _wrap(original, label, index, keys, access, remember=remember)
    class FactoryMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, original)
    class GuardedFactory(metaclass=FactoryMeta):
        _wm_testguard_wrapped = True
        def __new__(cls, *args, **kwargs):
            return guarded_call(*args, **kwargs)
    GuardedFactory.__name__ = name
    _set_wrapper(owner, name, GuardedFactory, label)


def _patch_numpy(np):
    for name in ("load", "fromfile"):
        _patch(np, name, keys=("file",))
    for name in ("save", "savez", "savez_compressed"):
        _patch(np, name, keys=("file",), access="write")
    original = np.memmap
    class GuardedMemmap(original):
        _wm_testguard_wrapped = True
        def __new__(cls, filename, dtype="uint8", mode="r+", *args, **kwargs):
            _check_arg(filename, access="read" if mode in {"r", "c"} else "write")
            return original.__new__(cls, filename, dtype, mode, *args, **kwargs)
    GuardedMemmap.__name__ = "memmap"
    _set_wrapper(np, "memmap", GuardedMemmap, "numpy.memmap")
    for module in tuple(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("numpy.") and getattr(module, "memmap", None) is original:
            setattr(module, "memmap", GuardedMemmap)


def _patch_pandas(pd):
    for name in ("read_csv", "read_table", "read_fwf", "read_json", "read_pickle", "read_feather", "read_parquet", "read_hdf", "read_sas", "read_spss", "read_stata"):
        _patch(pd, name, keys=("filepath_or_buffer", "path_or_buf", "path", "filepath"))
    _patch(pd, "to_pickle", 1, ("filepath_or_buffer", "path"), "write")
    for cls in (pd.DataFrame, pd.Series):
        for name in ("to_csv", "to_json", "to_pickle", "to_hdf"):
            _patch(cls, name, 1, ("path_or_buf", "path", "filepath_or_buffer"), "write", name in {"to_csv", "to_json"})
    for name in ("to_parquet", "to_feather", "to_stata"):
        _patch(pd.DataFrame, name, 1, ("path", "fname", "path_or_buf"), "write", name == "to_parquet")


def _patch_arrow_fs(fs):
    global _GUARDED_FS
    native_local = fs.LocalFileSystem
    class GuardedHandler(fs.FileSystemHandler):
        def __init__(self, *args, **kwargs):
            self._native = native_local(*args, **kwargs)
        def get_type_name(self):
            return "wm-guarded-local"
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
            return _remember_handle(self._native.open_output_stream(path, metadata=metadata), path)
        def open_append_stream(self, path, metadata):
            _check_arg(path, access="write")
            return _remember_handle(self._native.open_append_stream(path, metadata=metadata), path)
    class GuardedLocalFileSystem(fs.PyFileSystem):
        _wm_testguard_wrapped = True
        def __init__(self, *args, **kwargs):
            super().__init__(GuardedHandler(*args, **kwargs))
    _GUARDED_FS = GuardedLocalFileSystem
    _set_wrapper(fs, "LocalFileSystem", GuardedLocalFileSystem, "pyarrow.fs.LocalFileSystem")


_ARROW_FUNCS = {
    "pyarrow.csv": (("read_csv", 0, ("input_file",), "read"), ("open_csv", 0, ("input_file",), "read"), ("write_csv", 1, ("output_file",), "write")),
    "pyarrow.parquet": (("read_table", 0, ("source",), "read"), ("read_pandas", 0, ("source",), "read"), ("write_table", 1, ("where",), "write")),
    "pyarrow.feather": (("read_table", 0, ("source",), "read"), ("read_feather", 0, ("source",), "read"), ("write_feather", 1, ("dest",), "write")),
    "pyarrow.ipc": (("open_file", 0, ("source",), "read"), ("open_stream", 0, ("source",), "read"), ("new_file", 0, ("sink",), "write"), ("new_stream", 0, ("sink",), "write")),
    "pyarrow.dataset": (("dataset", 0, ("source",), "read"),),
}
_TARGET_MODULES = ("numpy", "pandas", "pyarrow", "pyarrow.fs", *_ARROW_FUNCS, "multiprocessing.spawn")
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
        # OSFile / memory_map are treated as potentially writable regardless of mode.
        _factory_class(module, "OSFile", keys=("path",), access="write", remember=True)
        _patch(module, "memory_map", keys=("path",), access="write", remember=True)
        _patch(module, "create_memory_map", keys=("path",), access="write", remember=True)
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
        _set_wrapper(module, "get_executable", get_executable, name + ".get_executable")
        _set_wrapper(module, "set_executable", set_executable, name + ".set_executable")
    else:
        for fname, index, keys, access in _ARROW_FUNCS[name]:
            _patch(module, fname, index, keys, access)
        if name == "pyarrow.parquet":
            _patch(module.ParquetFile, "__init__", 1, ("source",), label="pyarrow.parquet.ParquetFile.__init__")
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
    return all(getattr(owner, name, None) is expected for owner, name, expected in _LIVE_WRAPPERS.values())


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
                    raise GuardInstallError("required module has no loader: " + fullname)
                spec.loader = _GuardLoader(spec.loader, fullname)
                return spec
        return None


def _check_subprocess_args(args, kwargs):
    if kwargs.get("shell") or kwargs.get("executable") is not None or kwargs.get("preexec_fn") is not None:
        _deny("subprocess", "options", "shell/executable/preexec overrides refused")
    cwd = kwargs.get("cwd")
    cwd = os.getcwd() if cwd is None else cwd
    reason = _POLICY.reason(cwd, "metadata")
    if reason:
        _deny("subprocess", cwd, reason)
    if not isinstance(args, (list, tuple)) or not args or not all(_as_text(a) is not None for a in args):
        _deny("subprocess", "argv", "only explicit argv sequences are supported")
    exe = _norm(args[0])
    if exe != _POLICY.interpreter and exe not in _POLICY.helpers:
        _deny("subprocess", args[0], "only identical interpreter or pinned CPU helper permitted")
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
            _deny("subprocess", key, "inherited guard environment cannot be changed or removed")
    if exe == _POLICY.interpreter:
        # Check interpreter options only, never literal contents of -c source.
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
        _deny("subprocess", exe, "CPU helper SHA mismatch")


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
    _set_wrapper(subprocess.Popen, "__init__", guarded_init, "subprocess.Popen.__init__")
    for name in ("system", "popen"):
        def refuse(*args, **kwargs):
            _deny("process", "shell", "shell-style process launch refused")
        _set_wrapper(os, name, refuse, "os." + name)


def register_forbidden_root(path):
    n = _norm(path)
    if _POLICY is None or not _is_under(n, _POLICY.sandbox):
        raise ValueError("only synthetic sandbox fixtures can be registered")
    _POLICY.forbidden_subroots.add(n)


def is_protected(path):
    return _POLICY.reason(path) if _POLICY else "guard not installed"


def is_active():
    return _INSTALLED


def startup_attestation():
    return dict(_ATTESTATION) if _ATTESTATION else None


def assert_startup(strict_pid=True):
    site = sys.modules.get("sitecustomize")
    if not _INSTALLED or _POLICY is None or _ATTESTATION is None or site is None:
        raise GuardInstallError("sitecustomize startup attestation missing; no repair")
    if getattr(site, "WM_TESTGUARD_BOOT_PID", None) != os.getpid():
        raise GuardInstallError("sitecustomize did not attest this process")
    if strict_pid and _ATTESTATION["pid"] != os.getpid():
        raise GuardInstallError("startup PID mismatch")
    if _POLICY.immutable_fingerprint() != _ATTESTATION["policy_sha"]:
        raise GuardInstallError("policy fingerprint changed")
    if os.environ.get(POLICY_SHA_ENV) != _ATTESTATION["policy_sha"]:
        raise GuardInstallError("policy environment fingerprint changed")
    if _canonical_sha256(json.loads(os.environ[POLICY_SPEC_ENV])) != _ATTESTATION["policy_sha"]:
        raise GuardInstallError("policy environment contents changed")
    _POLICY.verify_pins()
    return dict(_ATTESTATION)


def assert_pytest_argv(argv):
    if argv != _POLICY.spec["pytest_args"]:
        raise GuardInstallError("pytest arguments do not match pinned policy")


def _seed_tempfile_directory(policy):
    """Use the precreated sandbox tmp without tempfile's create/unlink probe."""
    expected = policy.spec["sandbox_tmp"]
    if any(os.environ.get(key) != expected for key in ("TMP", "TEMP", "TMPDIR")):
        raise GuardInstallError("temporary directory environment must match policy exactly")
    for key, resolved in (("sandbox", policy.sandbox), ("sandbox_tmp", policy.sandbox_tmp)):
        raw = policy.spec[key]
        if not os.path.isabs(raw) or os.path.normcase(os.path.abspath(raw)) != resolved:
            raise GuardInstallError("temporary directory cannot use a resolved path alias")
    directory = policy.sandbox_tmp
    if directory == policy.sandbox or not _is_under(directory, policy.sandbox):
        raise GuardInstallError("temporary directory must be a strict sandbox descendant")
    if policy.reason(directory, "write"):
        raise GuardInstallError("temporary directory is protected")
    # Check every existing ancestor too: a junction above the run must not turn
    # a lexical sandbox path into a route to another filesystem tree.
    current = directory
    while True:
        metadata = os.lstat(current)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise GuardInstallError("temporary directory ancestry must contain only directories")
        if getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise GuardInstallError("temporary directory ancestry contains a reparse point")
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
    spec = json.loads(os.environ[POLICY_SPEC_ENV])
    if _canonical_sha256(spec) != os.environ.get(POLICY_SHA_ENV):
        raise GuardInstallError("policy SHA mismatch")
    _POLICY = _Policy(spec)
    _POLICY.verify_pins()
    if os.environ.get("PYTHONPATH") != os.pathsep.join(spec["pythonpath"]):
        raise GuardInstallError("PYTHONPATH differs from pinned policy")
    for key in ("PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "WD_FORBID_D"):
        if os.environ.get(key) != "1":
            raise GuardInstallError("required startup flag missing: " + key)
    if os.environ.get("PYTEST_ADDOPTS", "") or os.environ.get("PYTHONHOME") or os.environ.get("PYTHONSTARTUP"):
        raise GuardInstallError("unsafe inherited Python/pytest options")
    if any(_norm(os.environ.get(key, "")) != _POLICY.sandbox_tmp for key in ("TMP", "TEMP", "TMPDIR")):
        raise GuardInstallError("temporary directory environment mismatch")
    _seed_tempfile_directory(_POLICY)
    _ENV_SNAPSHOT = {key: os.environ.get(key) for key in _CRITICAL_ENV}
    sys.dont_write_bytecode = True
    sys.addaudithook(_hook)
    deny_log = os.path.join(_POLICY.sandbox, "guard_denies_%d_%d.log" % (os.getpid(), time.time_ns()))
    _DENY_LOG_FD = os.open(deny_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    sys.meta_path.insert(0, _GuardFinder())
    _install_subprocess_guard()
    _ATTESTATION = {"pid": os.getpid(), "policy_sha": _POLICY.immutable_fingerprint(),
                    "guard_shas": {n: _POLICY.files[os.path.join(_TOOL_DIR, n)]["sha256"] for n in _GUARD_FILES},
                    "deny_log": deny_log, "sandbox": _POLICY.sandbox}
    _INSTALLED = True
    for name in _TARGET_MODULES:
        if name in sys.modules:
            _patch_module(name)


def selfcheck():
    if not _INSTALLED:
        raise GuardInstallError("guard installation incomplete")
    probe = os.path.join(_POLICY.sandbox_tmp, "__wm_guard_selftest_forbidden__", "probe.bin")
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
