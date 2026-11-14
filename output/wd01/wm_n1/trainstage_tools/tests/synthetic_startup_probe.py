"""Guarded synthetic metadata-stage startup and inherited child attestation.

Imported only when an externally pinned synthetic metadata policy requests it.
The probe reads no corpus or shard bytes and starts no training step.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_CHILD_PROBE = (
    "import json, os, sys\n"
    "import train_stage_guard as guard\n"
    "a = guard.assert_startup(strict_pid=True)\n"
    "guard.ensure_required_wrappers()\n"
    "if not guard.wrappers_active(require_all=True): raise RuntimeError('wrappers missing')\n"
    "print(json.dumps({'pid':os.getpid(),'policy_sha':a['policy_sha'],"
    "'policy_fingerprint':a['policy_fingerprint'],'wrappers':True},sort_keys=True))\n"
)


def probe_inherited_startup():
    guard = sys.modules.get("train_stage_guard")
    if guard is None:
        raise RuntimeError("startup guard missing")
    parent = guard.assert_startup(strict_pid=True)
    guard.ensure_required_wrappers()
    if not guard.wrappers_active(require_all=True):
        raise RuntimeError("parent wrappers missing")
    child = subprocess.run([sys.executable, "-c", _CHILD_PROBE],
                           cwd=parent["sandbox"], env=os.environ.copy(),
                           capture_output=True, text=True, timeout=120,
                           check=False)
    if child.returncode:
        raise RuntimeError("inherited guarded child refused: " + child.stderr[-1000:])
    lines = child.stdout.splitlines()
    if len(lines) != 1:
        raise RuntimeError("inherited guarded child emitted unexpected output")
    result = json.loads(lines[0])
    if (result.get("pid") == os.getpid() or result.get("pid") is None
            or result.get("policy_sha") != parent["policy_sha"]
            or result.get("policy_fingerprint") != parent["policy_fingerprint"]
            or result.get("wrappers") is not True):
        raise RuntimeError("inherited child PID/spec/wrapper attestation failed")
    native = _probe_native_sandbox(parent["sandbox"], guard)
    return {"parent_pid": os.getpid(), "child_pid": result["pid"],
            "policy_sha": result["policy_sha"],
            "policy_fingerprint": result["policy_fingerprint"],
            "wrappers": True, "native_probe": native}


def _probe_native_sandbox(sandbox, guard):
    """Exercise guarded public native APIs only on new, tiny synthetic files."""
    import numpy as np
    import pandas as pd
    import pyarrow.fs as arrow_fs
    import pyarrow.parquet as parquet

    guard.ensure_required_wrappers()
    if not guard.wrappers_active(require_all=True):
        raise RuntimeError("native wrappers missing before probe")
    forbidden = "D:/research-dense/__wm_probe__/outcomes.parquet"
    for factory in (arrow_fs.FileSystem.from_uri,
                    arrow_fs.LocalFileSystem.from_uri):
        try:
            factory(forbidden)
        except PermissionError as exc:
            if guard.DENY_MARKER not in str(exc):
                raise RuntimeError("URI factory refused without guard marker") from exc
        else:
            raise RuntimeError("URI factory bypassed guard")

    probe_dir = Path(sandbox) / "native_probe"
    probe_dir.mkdir(exist_ok=False)
    parquet_path = probe_dir / "two_rows.parquet"
    expected = pd.DataFrame({"case_id": [1, 2], "value": [3, 5]})
    expected.to_parquet(str(parquet_path), engine="pyarrow", index=False)
    arrow_table = parquet.read_table(str(parquet_path))
    if arrow_table.num_rows != 2 or arrow_table.column("value").to_pylist() != [3, 5]:
        raise RuntimeError("Arrow sandbox parquet roundtrip failed")
    if not pd.read_parquet(str(parquet_path), engine="pyarrow").equals(expected):
        raise RuntimeError("pandas sandbox parquet roundtrip failed")

    npy_path = probe_dir / "read_only.npy"
    np.save(str(npy_path), np.asarray([7, 11], dtype=np.int64))
    mapped = np.load(str(npy_path), mmap_mode="r", allow_pickle=False)
    if (not isinstance(mapped, np.memmap) or mapped.flags.writeable
            or mapped.tolist() != [7, 11]):
        raise RuntimeError("NumPy sandbox read-only mmap failed")
    return {"uri_factories_denied": 2, "parquet_rows": 2,
            "read_only_mmap": True, "directory": str(probe_dir)}
