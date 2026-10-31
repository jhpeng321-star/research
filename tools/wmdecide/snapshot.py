"""Read-only collection of registered evaluator JSON outputs into a NEW directory.

No remote source/runtime files, inference, dispatch, process control or retries.
The locally authored inventory code is sent over SSH; downloaded data are never
executed. Every byte is hash-bound before being written into the empty snapshot.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zlib

from .rbeval_adapter import safe_path, sha256
from .report import write_json

ROOT = "/root/rbeval20261031"
PANEL = re.compile(r"(?:int/WM-PHYS-FULL-(?:v1|v3-gpu)|runa/WM-PHYS-(?:A2|C2)|runb/WM-PHYS-B2)")
TASK = re.compile(r"w9694(?:2[0-9]|3[0-9])_k[0-7]\.json")


def registered_root(host, root):
    if host not in ("gpuA", "gpuR") or (root != ROOT and
            (not root.startswith(ROOT+"/") or not PANEL.fullmatch(root[len(ROOT)+1:]))):
        raise ValueError("Only explicitly registered evaluator output roots are admitted")
    expected_host = "gpuR" if root == ROOT or root.startswith(ROOT+"/runb/") else "gpuA"
    if host != expected_host:
        raise ValueError("Root/host must match the registered producer")
    return root


def unpack(payload, out):
    """Validate the complete transfer before creating any snapshot files."""
    out = safe_path(out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError("Snapshot must be NEW under an existing parent")
    registered_root(payload["host"], payload["root"])
    checked = {}
    prefix = "wm2_shard" if payload["root"] == ROOT else "shard"
    for name, entry in payload["files"].items():
        path = PurePosixPath(name)
        if (name != path.as_posix() or "\\" in name or len(path.parts) != 2
                or not re.fullmatch(prefix+r"[0-3]", path.parts[0])
                or (path.name not in ("IDENTITY.json", "COMPLETE.json") and not TASK.fullmatch(path.name))):
            raise ValueError("Unexpected snapshot member path")
        raw = base64.b64decode(entry["data"], validate=True)
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Snapshot transfer checksum mismatch")
        json.loads(raw)  # Data only: reject truncated/non-JSON output.
        checked[name] = raw
    if not checked:
        raise ValueError("No registered output JSON files available")
    out.mkdir()
    for name, raw in checked.items():
        target = out / name
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as f:
            f.write(raw)
    receipt = {k: v for k, v in payload.items() if k != "files"}
    receipt.update(schema="wmdecide-snapshot/1", collection="read_only_no_inference_or_remote_writes",
                   collector_sha256=sha256(Path(__file__)),
                   files={name: payload["files"][name]["sha256"] for name in checked})
    write_json(out / "SNAPSHOT.json", receipt)
    return receipt


def snapshot(host, root, out):
    root = registered_root(host, root)
    out = safe_path(out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError("Snapshot must be NEW under an existing parent")
    # Only the named output JSON files are opened, never recursively glob sources.
    code = f'''
from pathlib import Path
import base64,hashlib,json,os,re,time,zlib
root=Path({root!r}); files={{}}
prefix={'wm2_shard' if root == ROOT else 'shard'!r}
for i in range(4):
    shard = root / (prefix + str(i))
    if not shard.is_dir():
        continue
    paths = sorted(shard.glob('w*_k*.json')) + [shard/'IDENTITY.json', shard/'COMPLETE.json']
    for path in paths:
        if not path.is_file():
            continue
        if path.name not in ('IDENTITY.json', 'COMPLETE.json') and not re.fullmatch({TASK.pattern!r}, path.name):
            raise ValueError('unexpected output filename')
        resolved = str(path.resolve()).lower()
        if any(word in resolved for word in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype')):
            raise ValueError('prohibited resolved path')
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('output escaped registered root')
        raw = path.read_bytes()
        files[shard.name+'/'+path.name] = dict(sha256=hashlib.sha256(raw).hexdigest(), data=base64.b64encode(raw).decode())
payload=dict(host={host!r},root=str(root),remote_unix=time.time(),load=os.getloadavg(),files=files)
print(base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode())
'''
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                             "/root/miniconda3/bin/python -B -"], input=code,
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    payload = json.loads(zlib.decompress(base64.b64decode(result.stdout.strip(), validate=True)))
    if payload["host"] != host or payload["root"] != root:
        raise ValueError("Unexpected remote snapshot identity")
    payload["collected_utc"] = datetime.now(timezone.utc).isoformat()
    return unpack(payload, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("gpuA", "gpuR"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    receipt = snapshot(args.host, args.root, args.out)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, sort_keys=True))
    print(json.dumps({"files": len(receipt["files"]), "tasks": sum(
        TASK.fullmatch(PurePosixPath(p).name) is not None for p in receipt["files"])}))


if __name__ == "__main__":
    main()
