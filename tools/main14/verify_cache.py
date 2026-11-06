"""Verify this arm's completed-cycle converted cache against its retained cpuX original."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    if not root.is_relative_to(Path("/home/ubuntu/c4/main14")):
        raise ValueError("not this arm's cpuX cache")
    expected = json.load(sys.stdin)
    for rel, checksum in expected.items():
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or file_hash(target) != checksum:
            raise ValueError(f"archive mismatch: {rel}")
    print(json.dumps({"verified": len(expected), "root": str(root)}))


if __name__ == "__main__":
    main()
