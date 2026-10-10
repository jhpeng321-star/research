"""Archive the committed HEAD code/chain trees and verify bytes without extraction.

Use HEAD^{tree} with a fixed mtime so a documentation-only commit cannot change
the tar hashes through Git's commit-ID PAX header. Every member is checked against
its HEAD blob. Outputs are workspace-local, fresh, and streamed one file at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MTIME = "2026-10-09T00:00:00Z"
CODE = "output/wd01/wm_n1"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, paths: list[str]) -> dict:
    expected = {}
    for line in git("ls-tree", "-r", "HEAD", "--", *paths).decode("utf-8").splitlines():
        header, name = line.split("\t", 1)
        mode, kind, oid = header.split()
        if kind != "blob" or mode == "120000":
            raise ValueError(f"non-regular source member: {name}")
        expected[name] = oid
    seen, shells = set(), []
    with tarfile.open(path, "r|") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"unexpected/non-regular/duplicate member: {member.name}")
            stream = archive.extractfile(member)
            h = hashlib.sha1(f"blob {member.size}\0".encode("ascii"))
            shell = member.name.endswith(".sh")
            for block in iter(lambda: stream.read(1 << 20), b""):
                if shell and b"\r" in block:
                    raise ValueError(f"CR in archived shell script: {member.name}")
                h.update(block)
            if h.hexdigest() != expected[member.name]:
                raise ValueError(f"archive differs from HEAD blob: {member.name}")
            seen.add(member.name)
            if shell:
                shells.append(member.name)
    if seen != set(expected):
        raise ValueError("archive is missing HEAD members")
    return {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size,
            "files": len(seen), "shell_files_no_cr": sorted(shells),
            "head_blob_identity": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    out.relative_to(ROOT)
    if not out.parent.is_dir():
        parser.error("archive output parent must exist inside workspace")
    shells = sorted(name for name in git("ls-tree", "-r", "--name-only", "HEAD").decode(
        "utf-8").splitlines() if name.endswith(".sh"))
    specs = [("wm_n1_code.tar", [CODE]), ("main2_chains.tar", shells)]
    if not args.verify:
        out.mkdir(exist_ok=False)
        for name, paths in specs:
            command = ["git", "-c", "core.autocrlf=false", "-c", "tar.umask=0022",
                       "archive", "--format=tar",
                       f"--mtime={MTIME}", f"--output={out / name}", "HEAD^{tree}", *paths]
            subprocess.run(command, cwd=ROOT, check=True)
    results = [verify(out / name, paths) for name, paths in specs]
    receipt = {"schema": "wms_main2_archives/1",
               "verified_against_head": git("rev-parse", "HEAD").decode().strip(),
               "code_tree": git("rev-parse", f"HEAD:{CODE}").decode().strip(),
               "source": "HEAD^{tree}", "fixed_mtime": MTIME, "archives": results}
    if not args.verify:
        (out / "ARCHIVES.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        (out / "SHA256SUMS").write_text("".join(
            f"{row['sha256']}  {row['path']}\n" for row in results), encoding="ascii")
    else:
        stored = json.loads((out / "ARCHIVES.json").read_text(encoding="utf-8"))
        if stored["archives"] != results or stored["code_tree"] != receipt["code_tree"]:
            raise ValueError("archive receipt does not match final HEAD code")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
