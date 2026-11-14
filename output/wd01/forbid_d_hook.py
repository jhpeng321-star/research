"""Forbid-D audit hook: fail hard on any filesystem access under drive D:.

Owner context: D:/research-dense, D:/rd-wt and D:/research-dense-backups
were junctions to E: and will be deleted.  WD-01 live code must never
touch D: again.  install() registers a sys.addaudithook that raises
ForbiddenDriveDError for the file-system audit events ('open',
'os.listdir', 'os.scandir', 'os.mkdir', 'os.rename', 'os.remove',
'shutil.copyfile', ...) whenever an argument resolves to drive D:
(after os.fspath; \\?\D: extended-length forms included).

Some readers bypass Python-level audit events (pyarrow's C++ file IO),
so install() also wraps numpy.load, pandas.read_parquet,
DataFrame.to_parquet and pyarrow.parquet.read_table/write_table with the
same drive check.  Wrapping is best-effort per library and idempotent.

Owner-approved override (FEAT1): ``allow_root()`` registers ONE D: output
root (e.g. ``D:/rd-staging/wmn1_feat``) whose paths are exempt from the
hook.  Every other D: path stays blocked, and the protected roots
(:data:`PROTECTED_D_ROOTS`, never writable) are refused as override roots.

Activation: set WD_FORBID_D=1 -- output/wd01/conftest.py installs the
hook for every wd01 pytest suite.  Self-test:  python forbid_d_hook.py
"""
from __future__ import annotations

import functools
import os
import sys

WD_FORBID_D_ENV = "WD_FORBID_D"


class ForbiddenDriveDError(RuntimeError):
    """A filesystem access tried to resolve to drive D:."""


_EVENTS = frozenset({
    "open",
    "os.listdir", "os.scandir",
    "os.mkdir", "os.rename", "os.replace", "os.remove", "os.rmdir",
    "os.truncate", "os.chmod", "os.chown", "os.link", "os.symlink",
    "shutil.copyfile", "shutil.copymode", "shutil.copystat",
    "shutil.copytree",
})


def _core(s: str) -> str:
    """Strip the \\?\ extended-length prefix, if present."""
    if s.startswith("\\\\?\\UNC\\"):          # UNC form: not a drive path
        return ""
    if s.startswith("\\\\?\\"):
        return s[4:]
    return s


def is_d_path(arg) -> bool:
    """True iff arg is a path-like whose string form resolves to drive D:."""
    if isinstance(arg, int):                 # open(fd, ...): not a path
        return False
    try:
        s = os.fspath(arg)
    except TypeError:
        return False
    if not isinstance(s, str):
        return False
    core = _core(s)
    return len(core) >= 2 and core[1] == ":" and core[0] in "Dd"


# ------------------------------------------------- approved D: override ----
# Roots that may NEVER be written, not even via the override (retired
# junction targets / frozen evidence trees).
PROTECTED_D_ROOTS = ("D:/research-dense", "D:/research-dense-backups")

_ALLOWED_ROOTS: list[str] = []


def norm_d(path) -> str:
    """Canonical comparison form of a path (forward slashes, abs, D: upper)."""
    try:
        s = os.fspath(path)
    except TypeError:
        s = str(path)
    s = _core(str(s)).replace("\\", "/")
    drive, rest = (s[:2], s[2:]) if len(s) >= 2 and s[1] == ":" else ("", s)
    if drive:
        drive = drive[0].upper() + ":"
    joined = os.path.normpath(drive + rest).replace("\\", "/")
    return joined


def _under(child: str, root: str) -> bool:
    return child == root or child.startswith(root.rstrip("/") + "/")


def allow_root(root) -> str:
    """Register the ONE owner-approved D: output root (idempotent).

    Refuses anything that is not a D: path, any protected root (or a path
    inside one), and any root that would CONTAIN a protected root (e.g.
    ``D:/``).  Returns the normalized registered root.
    """
    n = norm_d(root)
    if len(n) < 2 or n[1] != ":" or not n.startswith("D"):
        raise ValueError(
            f"allow-d-output-root: {str(root)!r} is not a D: path")
    for prot in PROTECTED_D_ROOTS:
        p = norm_d(prot)
        if _under(n, p) or _under(p, n):
            raise ValueError(
                f"allow-d-output-root: {str(root)!r} overlaps the protected "
                f"root {prot!r} (never writable)")
    if n not in _ALLOWED_ROOTS:
        _ALLOWED_ROOTS.append(n)
    return n


def allowed_roots() -> tuple:
    return tuple(_ALLOWED_ROOTS)


def reset_allowed_roots() -> None:
    """Drop every registered override root (test helper)."""
    _ALLOWED_ROOTS.clear()


def _exempt(arg) -> bool:
    if not _ALLOWED_ROOTS:
        return False
    try:
        s = os.fspath(arg)
    except TypeError:
        return False
    n = norm_d(s)
    return any(_under(n, r) for r in _ALLOWED_ROOTS)


def assert_not_d(arg, event: str = "call") -> None:
    if is_d_path(arg) and not _exempt(arg):
        raise ForbiddenDriveDError(
            f"forbid-D: {event} on drive-D path {os.fspath(arg)!r} "
            "(D: junctions are retired; use wdpaths roots / remap())")


def _hook(event, args):
    if event not in _EVENTS:
        return
    for a in args:
        if is_d_path(a) and not _exempt(a):
            raise ForbiddenDriveDError(
                f"forbid-D: audit event {event!r} got drive-D path "
                f"{os.fspath(a)!r} (D: junctions are retired; use wdpaths "
                "roots / remap())")


_installed = False


def install() -> bool:
    """Register the audit hook + library wrappers (idempotent).

    Returns True iff this call installed anything new.
    """
    global _installed
    if _installed:
        return False
    _wrap_libraries()
    sys.addaudithook(_hook)
    _installed = True
    return True


def installed() -> bool:
    return _installed


# ----------------------------------------------------- library wrappers ----

def _wrap(module, name, *, argpos=0, kwarg=None):
    fn = getattr(module, name, None)
    if fn is None or getattr(fn, "_wd_forbid_d", False):
        return

    @functools.wraps(fn)
    def wrapped(*a, **k):
        target = k.get(kwarg) if kwarg else None
        if target is None and len(a) > argpos:
            target = a[argpos]
        if target is not None and not isinstance(target, int):
            assert_not_d(target, f"{module.__name__}.{name}")
        return fn(*a, **k)

    wrapped._wd_forbid_d = True
    setattr(module, name, wrapped)


def _wrap_method(cls, name):
    fn = getattr(cls, name, None)
    if fn is None or getattr(fn, "_wd_forbid_d", False):
        return

    @functools.wraps(fn)
    def wrapped(self, *a, **k):
        for x in (a[:1] or (None,)):
            if not isinstance(x, int):
                assert_not_d(x, f"{cls.__name__}.{name}")
        return fn(self, *a, **k)

    wrapped._wd_forbid_d = True
    setattr(cls, name, wrapped)


def _wrap_libraries() -> None:
    try:
        import numpy as np
        _wrap(np, "load")
    except ImportError:
        pass
    try:
        import pandas as pd
        _wrap(pd, "read_parquet")
        _wrap_method(pd.DataFrame, "to_parquet")
    except ImportError:
        pass
    try:
        import pyarrow.parquet as pq
        _wrap(pq, "read_table")
        _wrap(pq, "write_table")
    except ImportError:
        pass


# ------------------------------------------------------------- self-test ----

def _expect_forbidden(fn, what: str) -> None:
    try:
        fn()
    except ForbiddenDriveDError as e:
        print(f"PASS  fires on {what}: {e}")
        return
    except Exception as e:                    # noqa: BLE001
        raise SystemExit(f"FAIL  {what}: wrong error "
                         f"{type(e).__name__}: {e}")
    raise SystemExit(f"FAIL  {what}: hook did not fire")


def selftest() -> int:
    install()
    print(f"PASS  install() (WD_FORBID_D env="
          f"{os.environ.get(WD_FORBID_D_ENV, '')!r})")

    _expect_forbidden(lambda: open("D:/research-dense/x", "rb"),
                      "open('D:/research-dense/x')")
    _expect_forbidden(lambda: open(r"\\?\D:\research-dense\x", "rb"),
                      r"open(r'\\?\D:\research-dense\x')")
    _expect_forbidden(lambda: os.listdir("D:/"), "os.listdir('D:/')")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        epath = os.path.join(td, "wdpaths_selftest.txt")
        with open(epath, "w", encoding="utf-8") as fh:
            fh.write("ok")
        with open(epath, "r", encoding="utf-8") as fh:
            assert fh.read() == "ok"
        os.listdir(td)
        print(f"PASS  no fire for E:/temp paths ({epath})")

        import numpy as np
        npy = os.path.join(td, "a.npy")
        np.save(npy, np.arange(4))
        assert int(np.load(npy).sum()) == 6
        print("PASS  numpy.load on E:/ works under the hook")
        _expect_forbidden(lambda: np.load("D:/research-dense/a.npy"),
                          "numpy.load('D:/research-dense/a.npy')")

        try:
            import pandas as pd
            df = pd.DataFrame({"x": [1, 2]})
            pq = os.path.join(td, "a.parquet")
            df.to_parquet(pq)
            assert len(pd.read_parquet(pq)) == 2
            print("PASS  pandas parquet round-trip on E:/ works")
        except ImportError:
            print("SKIP  pandas not importable")

    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
