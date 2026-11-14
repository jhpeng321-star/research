"""FEAT1 v2c step 4b: local verification of the upload_v2c bundle.

1. Re-hash every part file and compare with ``SHA256SUMS`` and
   ``PARTS.json`` (bytes + sha256), including part count and sizes.
2. Check the integrity anchors (pack SHA256SUMS sha256, upload manifest
   sha256).
3. Streaming check without writing the tar to disk: concatenate the part
   files in order through ``zstd -dc`` and stream-list the GNU tar;
   every member must match the pack file of the same name byte-for-byte
   (payload sha256 vs the pack's ``INDEX.json`` ``files`` entry), the
   member count must equal the pack file count minus ``SHA256SUMS``.

WD_FORBID_D=1 required; never touches upload_v2 / pack_v2.

Usage (cwd output/wd01/wm_n1):
  python -X utf8 feat1v2c_tools/upload_verify_v2c.py \
      --pack D:/rd-staging/wmn1_feat/pack_v2c \
      --out D:/rd-staging/wmn1_feat/upload_v2c \
      --manifest D:/rd-staging/wmn1_feat/upload_manifest_v2c.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

BLOCK = 1 << 20

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(BLOCK), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args(argv)

    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    pack = Path(a.pack)
    out = Path(a.out)
    manifest_path = Path(a.manifest)
    pj = json.loads((out / "PARTS.json").read_text(encoding="utf-8"))
    sums = {}
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        sums[name] = sha

    errors = []
    if pj["schema"] != "wm_n1_upload_parts/1":
        errors.append(f"PARTS.json schema {pj['schema']!r}")
    if sorted(sums) != sorted(p["name"] for p in pj["parts"]):
        errors.append("SHA256SUMS vs PARTS.json part-name mismatch")
    if len(pj["parts"]) < 1:
        errors.append("no parts recorded")

    for p in pj["parts"]:
        fp = out / p["name"]
        if not fp.is_file():
            errors.append(f"missing part {p['name']}")
            continue
        if fp.stat().st_size != p["bytes"]:
            errors.append(f"{p['name']}: size {fp.stat().st_size} != "
                          f"{p['bytes']}")
        sha = _sha(fp)
        if sha != p["sha256"]:
            errors.append(f"{p['name']}: sha mismatch vs PARTS.json")
        if sums.get(p["name"]) not in (None, sha):
            errors.append(f"{p['name']}: sha mismatch vs SHA256SUMS")

    if _sha(pack / "SHA256SUMS") != pj["pack_sha256sums_sha256"]:
        errors.append("pack SHA256SUMS anchor mismatch")
    if _sha(manifest_path) != pj["upload_manifest"]["sha256"]:
        errors.append("upload_manifest anchor mismatch")

    expected = sorted(p.name for p in pack.iterdir() if p.is_file()
                      and p.name != "SHA256SUMS")
    if pj["pack_files"]["count"] != len(expected):
        errors.append("PARTS.json pack_files.count mismatch")
    if pj["pack_files"]["bytes"] != sum((pack / n).stat().st_size
                                        for n in expected):
        errors.append("PARTS.json pack_files.bytes mismatch")
    index = json.loads((pack / "INDEX.json").read_text(encoding="utf-8"))
    file_shas = index.get("files", {})

    members_seen = []

    def pump_stdin(proc) -> None:
        try:
            for name in sorted(sums):
                with open(out / name, "rb") as fh:
                    while True:
                        chunk = fh.read(BLOCK)
                        if not chunk:
                            break
                        proc.stdin.write(chunk)
        finally:
            proc.stdin.close()

    z = subprocess.Popen(["zstd", "-dc", "-q", "-"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    t = threading.Thread(target=pump_stdin, args=(z,), daemon=True)
    t.start()
    total_member_bytes = 0
    with tarfile.open(fileobj=z.stdout, mode="r|") as tf:
        for ti in tf:
            members_seen.append(ti.name)
            fp = pack / ti.name
            if not fp.is_file():
                errors.append(f"member {ti.name}: no such pack file")
                continue
            if ti.size != fp.stat().st_size:
                errors.append(f"member {ti.name}: size {ti.size} != "
                              f"{fp.stat().st_size}")
            want = file_shas.get(ti.name)
            h = hashlib.sha256()
            n = 0
            ef = tf.extractfile(ti)
            while True:
                chunk = ef.read(BLOCK)
                if not chunk:
                    break
                n += len(chunk)
                h.update(chunk)
            total_member_bytes += n
            if n != ti.size:
                errors.append(f"member {ti.name}: short read {n} != {ti.size}")
            if want is not None and h.hexdigest() != want:
                errors.append(f"member {ti.name}: payload sha mismatch vs "
                              f"pack INDEX.json")
    t.join()
    rc = z.wait()
    if rc != 0:
        errors.append(f"zstd -dc exited {rc}")
    if sorted(members_seen) != expected:
        errors.append(f"member set mismatch: seen {len(members_seen)}, "
                      f"expected {len(expected)}")
    if total_member_bytes != pj["pack_files"]["bytes"]:
        errors.append("total member bytes != PARTS.json pack_files.bytes")

    ok = not errors
    result = dict(
        ok=ok, parts=len(pj["parts"]),
        part_bytes_total=sum(p["bytes"] for p in pj["parts"]),
        members=len(members_seen), expected_members=len(expected),
        member_bytes_total=total_member_bytes,
        pack_member_bytes=pj["pack_files"]["bytes"],
        errors=errors)
    print(json.dumps(result, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
