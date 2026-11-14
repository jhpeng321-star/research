"""Shared helpers for the CORPUS piece (no I/O policy lives here).

Pure utility layer used by every ``corpus`` module: path roots, atomic JSON,
sha256, run-key formatting, dataframe records, a Windows below-normal priority
setter, and the :class:`OpenedAudit` recorder that lets a driver prove it never
touched a sealed or forbidden file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- roots -----
# output/wd01/wm_n1/corpus/_util.py -> parents: corpus, wm_n1, wd01, output...
WM_N1_ROOT = Path(__file__).resolve().parents[1]
WD01_ROOT = WM_N1_ROOT.parent
WORKSPACE_ROOT = WD01_ROOT.parent.parent

DEFAULT_WORKERS = 4
MAX_WORKERS = 6


def ensure_wm_n1_on_path() -> None:
    """Make ``features`` / ``model`` / ``evalh`` importable (idempotent)."""
    for p in (str(WM_N1_ROOT), str(WD01_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


# --------------------------------------------------------------- hashing ----
def sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ json ----
def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(obj, indent=1, default=_json_default)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------- run keys ---
def run_key(chunk, world, arm) -> str:
    """``<first component>/<world>__<arm>``; pass the BATCH as ``chunk`` so
    cross-batch runs (e.g. chunk2 vs truearms_chunk2) never share a key."""
    return f"{chunk}/{world}__{arm}"


def arm_suffix(run_name, arm=None) -> str:
    """Exact run-dir arm suffix: the ``<arm>`` of ``<world>__<arm>``.

    Replicated plan arms (``fixed`` -> run dirs ``__fixed1``/``__fixed2``)
    must key shards by this suffix; the plan-level arm is only the fallback
    for run names without ``__``.
    """
    name = str(run_name)
    if "__" in name:
        return name.split("__", 1)[1]
    return str(arm) if arm is not None else name


def split_run_key(key: str):
    chunk, tail = str(key).split("/", 1)
    world, arm = tail.rsplit("__", 1)
    return chunk, world, arm


def records_from_dataframe(df) -> list:
    """Plain list of dicts with json-safe scalars (no pandas/numpy objects)."""
    out = []
    for rec in df.to_dict(orient="records"):
        out.append({k: _json_default(v) if not isinstance(
            v, (str, int, float, bool, type(None))) else v
            for k, v in rec.items()})
    return out


def sanitize(obj):
    """Recursively coerce a record subtree to JSON-safe python values."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


# -------------------------------------------------------------- priority ----
def lower_priority_below_normal() -> bool:
    """Set the current Windows process to BELOW_NORMAL priority (no-op else).

    Returns True when the priority was changed.  Prefers psutil; the ctypes
    fallback declares the proper ``HANDLE`` restype for ``GetCurrentProcess``
    (the default ``c_int`` truncates the 64-bit pseudo-handle and makes
    ``SetPriorityClass`` fail with ``GetLastError: 6`` on 64-bit Python).
    """
    if os.name != "nt":
        return False
    try:
        import psutil

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return True
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        return False
    try:
        import ctypes

        below_normal = 0x00004000
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p,
                                              ctypes.c_uint32)
        kernel32.SetPriorityClass.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        return bool(kernel32.SetPriorityClass(handle, below_normal))
    except Exception:  # noqa: BLE001
        return False


def current_priority_class() -> int:
    """Read the current Windows priority class (0 when unavailable)."""
    if os.name != "nt":
        return 0
    try:
        import psutil

        return int(psutil.Process().nice())
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        return 0
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetPriorityClass.argtypes = (ctypes.c_void_p,)
        kernel32.GetPriorityClass.restype = ctypes.c_uint32
        return int(kernel32.GetPriorityClass(kernel32.GetCurrentProcess()))
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------- D: output override -----
def activate_d_output_override(root) -> str:
    """Activate the owner-approved D: output override for this process.

    Registers ``root`` as the single permitted D: path prefix with
    ``forbid_d_hook`` (which refuses D:/research-dense and any other D:
    path outright), installs the forbid-D audit hook, and pins the policy
    into the environment so spawned worker processes re-apply it.  Raises
    ``ValueError`` for any root that is not an approved D: output root.
    """
    ensure_wm_n1_on_path()
    import forbid_d_hook

    allowed = forbid_d_hook.allow_root(root)
    forbid_d_hook.install()
    os.environ["WD_FORBID_D"] = "1"
    os.environ["WD_D_ALLOWED_ROOT"] = allowed
    return allowed


def apply_env_d_policy() -> None:
    """Re-apply the env-carried D: override in this process (workers)."""
    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        return
    root = os.environ.get("WD_D_ALLOWED_ROOT", "").strip()
    if not root:
        return
    ensure_wm_n1_on_path()
    import forbid_d_hook

    forbid_d_hook.allow_root(root)
    forbid_d_hook.install()


def refuse_unapproved_d_out(out_root) -> None:
    """Refuse an output root on D: without the registered override."""
    try:
        import forbid_d_hook
    except ImportError:                        # pragma: no cover
        return
    if forbid_d_hook.is_d_path(out_root):
        forbid_d_hook.assert_not_d(out_root, "output root")


# ---------------------------------------------------------- opened audit ----
_FORBIDDEN_DEFAULT = ("dispatcher_rows", "insertions", "evaluate_summary",
                       "suffix_summary")


class OpenedAudit:
    """Audit-hook recorder of every ``open`` event during a bounded block.

    The driver records the paths each worker opened and then asserts the set is
    clean of forbidden files and sealed-world tokens.  ``paths`` is only
    meaningful between :meth:`start` and :meth:`stop`.
    """

    _installed = False
    paths: list = []
    active = False

    @classmethod
    def _hook(cls, event, args):
        if cls.active and event == "open":
            try:
                cls.paths.append(str(args[0]))
            except Exception:  # noqa: BLE001
                pass

    @classmethod
    def reset(cls) -> None:
        cls.paths = []

    @classmethod
    def start(cls) -> None:
        if not cls._installed:
            sys.addaudithook(cls._hook)
            cls._installed = True
        cls.active = True

    @classmethod
    def stop(cls) -> None:
        cls.active = False

    @classmethod
    def opened_under(cls, root) -> list:
        root = str(Path(root))
        return [p for p in cls.paths if str(p).startswith(root)]

    @classmethod
    def assert_clean(cls, forbidden=_FORBIDDEN_DEFAULT, sealed_tokens=()) -> None:
        """Raise :class:`PermissionError` if a forbidden path was opened."""
        tokens = {str(t) for t in sealed_tokens if t}
        for p in cls.paths:
            low = str(p).lower()
            for bad in forbidden:
                if bad.lower() in low:
                    raise PermissionError(
                        f"opened-path audit: forbidden file {bad!r} in {p!r}")
            for tok in tokens:
                if tok in str(p):
                    raise PermissionError(
                        f"opened-path audit: sealed token {tok!r} in {p!r}")
