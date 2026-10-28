"""Read-only cpuP fetch of explicitly named observable E3-final episode tapes.

No remote directory traversal, writes, simulator, scenario or assignment reads.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.out.parent.is_dir()
    names = [f"e3final/main/e3final/shard0/jobs/s{seed}_{arm}/deploy/episode/episode.npz"
             for seed in range(977000, 977020, 2) for arm in ("native", "ours", "mappo")]
    code = ("import sys,tarfile,pathlib\n"
            "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:\n"
            f" for n in {names!r}:\n"
            "  p=pathlib.Path('/home/ubuntu')/n\n"
            "  assert p.is_file() and not p.is_symlink(),n\n"
            "  t.add(p,arcname=n,recursive=False)\n")
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "cpuP",
                             "/home/ubuntu/miniconda3/bin/python -B -"],
                            input=code.encode(), capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    args.out.mkdir(exist_ok=True)
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
    assert set(manifest) == set(names)
    (args.out / "RECORDING_HASHES.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"episodes": len(manifest), "compressed_bytes": len(result.stdout)}))


if __name__ == "__main__":
    main()
