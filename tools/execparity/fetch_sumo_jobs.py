"""Read-only fetch of this task's own SUMO job outputs from cpuP/cpuQ.

Fetches only the registered job files this task created under
/home/ubuntu/execparity/always0020/jobs/. No traversal, no writes, no peer data.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

FILES = ["result.json", "START.json", "CTRL0_IDENTITY.json",
         "execution_log.jsonl.gz", "deploy/episode/episode.npz"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, choices=("cpuP", "cpuQ"))
    ap.add_argument("--jobs", required=True, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    names = [f"execparity/always0020/jobs/{job}/{name}"
             for job in args.jobs for name in FILES]
    code = ("import sys,tarfile,pathlib\n"
            "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:\n"
            f" for n in {names!r}:\n"
            "  p=pathlib.Path('/home/ubuntu')/n\n"
            "  if p.is_file() and not p.is_symlink():\n"
            "   t.add(p,arcname=n,recursive=False)\n"
            "  else:\n"
            "   assert n.endswith(('CTRL0_IDENTITY.json','execution_log.jsonl.gz')),n\n")
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                             args.host, "/home/ubuntu/miniconda3/bin/python -B -"],
                            input=code.encode(), capture_output=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    manifest = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as archive:
        for member in archive:
            assert member.name in names and member.isfile()
            data = archive.extractfile(member).read()
            target = args.out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == data
            else:
                target.write_bytes(data)
            manifest[member.name] = hashlib.sha256(data).hexdigest()
    (args.out / f"FETCH_{args.host}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"files": len(manifest), "compressed_bytes": len(result.stdout)}))


if __name__ == "__main__":
    main()
