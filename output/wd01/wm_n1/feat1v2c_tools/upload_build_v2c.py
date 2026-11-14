"""FEAT1 v2c step 4: deterministic upload bundle builder (upload_v2c).

Replicates the frozen upload_v2 bundle contract for a NEW pack:

* GNU-format tar, members sorted by name (pack dir minus ``SHA256SUMS``),
  member mtime=0, uid=gid=0, mode 0644, no user/group names;
* streamed through zstd CLI (level 6, -T6) and split into fixed
  ``--part-size`` (default 2147483648) byte parts;
* ``SHA256SUMS`` (per-part sha256), ``PARTS.json`` (machine manifest,
  schema ``wm_n1_upload_parts/1``), ``UNPACK.md`` (reassemble guide).

Never reads labels; never touches upload_v2 / pack_v2; refuses to write
into an existing output dir.  WD_FORBID_D=1 required.

Usage (cwd output/wd01/wm_n1):
  python -X utf8 feat1v2c_tools/upload_build_v2c.py \
      --pack D:/rd-staging/wmn1_feat/pack_v2c \
      --out D:/rd-staging/wmn1_feat/upload_v2c \
      --manifest D:/rd-staging/wmn1_feat/upload_manifest_v2c.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import threading
from datetime import datetime, timezone
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


def _zstd_version() -> str:
    out = subprocess.run(["zstd", "--version"], capture_output=True,
                         text=True, check=True)
    m = re.search(r"v[0-9][0-9.]*,?", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"cannot parse zstd version: {out.stdout!r}")
    return m.group(0)


class _Splitter:
    """Splits one byte stream into fixed-size sha256-tracked parts."""

    def __init__(self, out_dir: Path, arcname: str, part_size: int):
        self.out_dir = out_dir
        self.arcname = arcname
        self.part_size = part_size
        self.parts: list[dict] = []
        self.total = 0
        self._idx = 0
        self._fh = None
        self._sha = None
        self._n = 0
        self._name = None

    def _rotate(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self.parts.append(dict(name=self._name, bytes=self._n,
                                   sha256=self._sha.hexdigest()))
            print(f"  part {self._name}: {self._n} bytes "
                  f"sha={self._sha.hexdigest()[:16]}...", flush=True)
        self._fh = None

    def _next_part(self) -> None:
        self._name = f"{self.arcname}.part{self._idx:03d}"
        self._fh = open(self.out_dir / self._name, "wb")
        self._sha = hashlib.sha256()
        self._n = 0
        self._idx += 1

    def pump(self, src) -> None:
        while True:
            if self._fh is None:
                self._next_part()
            chunk = src.read(min(BLOCK, self.part_size - self._n))
            if not chunk:
                break
            self._fh.write(chunk)
            self._sha.update(chunk)
            self._n += len(chunk)
            self.total += len(chunk)
            if self._n >= self.part_size:
                self._rotate()
        self._rotate()

    def pump_stdin(self, proc) -> None:
        """pump() for a subprocess stdout pipe (until EOF)."""
        self.pump(proc.stdout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--part-size", type=int, default=2147483648)
    ap.add_argument("--zstd-level", type=int, default=6)
    ap.add_argument("--zstd-threads", type=int, default=6)
    a = ap.parse_args(argv)

    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    pack = Path(a.pack)
    out = Path(a.out)
    manifest_path = Path(a.manifest)
    assert pack.is_dir(), f"pack missing: {pack}"
    assert manifest_path.is_file(), f"manifest missing: {manifest_path}"
    assert not out.exists(), f"refusing: output dir already exists: {out}"

    names = sorted(p.name for p in pack.iterdir() if p.is_file())
    assert "SHA256SUMS" in names, "pack missing SHA256SUMS"
    members = [n for n in names if n != "SHA256SUMS"]
    n_npy = sum(1 for n in members if n.endswith(".npy"))
    assert n_npy == 15 and len(members) == 18, (
        f"unexpected pack layout: {len(members)} members, {n_npy} npy")
    for extra in ("INDEX.json", "manifest.json", "outcomes.parquet"):
        assert extra in members, f"pack missing {extra}"
    member_bytes = sum((pack / n).stat().st_size for n in members)

    packs_sums_sha = _sha(pack / "SHA256SUMS")
    manifest_sha = _sha(manifest_path)

    out.mkdir(parents=True)
    version = _zstd_version()
    arcname = f"{pack.name}.tar.zst"
    z = subprocess.Popen(
        ["zstd", f"-{a.zstd_level}", f"-T{a.zstd_threads}", "-q", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    splitter = _Splitter(out, arcname, a.part_size)
    pump_thread = threading.Thread(target=splitter.pump_stdin, args=(z,),
                                   daemon=True)
    pump_thread.start()

    with tarfile.open(fileobj=z.stdin, mode="w|",
                      format=tarfile.GNU_FORMAT) as tf:
        for n in members:
            fp = pack / n
            ti = tarfile.TarInfo(name=n)
            ti.size = fp.stat().st_size
            ti.mtime = 0
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            ti.mode = 0o644
            ti.type = tarfile.REGTYPE
            with open(fp, "rb") as fh:
                tf.addfile(ti, fh)
            print(f"  tar member {n} ({ti.size} bytes)", flush=True)
    z.stdin.close()
    pump_thread.join()
    rc = z.wait()
    assert rc == 0, f"zstd exited {rc}"
    assert len(splitter.parts) >= 1, "no parts produced"

    sums_path = out / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{p['sha256']}  {p['name']}\n" for p in splitter.parts),
        encoding="utf-8")

    parts_json = dict(
        schema="wm_n1_upload_parts/1",
        created_utc=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        source_pack=str(pack.resolve()),
        pack_files=dict(count=len(members), bytes=member_bytes),
        pack_sha256sums_sha256=packs_sums_sha,
        upload_manifest=dict(path=str(manifest_path.resolve()),
                             sha256=manifest_sha),
        archive=dict(
            name=arcname, tar_format="gnu", member_order="sorted-by-name",
            member_mtime=0, member_uid=0, member_gid=0, member_mode="0644",
            zstd=dict(version=version, level=a.zstd_level,
                      threads=a.zstd_threads)),
        part_size_bytes=a.part_size,
        parts=splitter.parts)
    (out / "PARTS.json").write_text(json.dumps(parts_json, indent=1),
                                    encoding="utf-8")

    unpack = f"""# upload_v2c - resumable compressed upload of pack_v2c

Deterministic archive: GNU tar (members sorted by name, mtime=0, uid=gid=0,
mode 0644) compressed with zstd {version} level {a.zstd_level} (-T{a.zstd_threads}),
split into {len(splitter.parts)} parts of ~{a.part_size} bytes.

Source: {pack.resolve()} ({len(members)} files, {member_bytes} bytes)
Built (UTC): {parts_json['created_utc']}

## Files in this bundle
- {len(splitter.parts)} parts: {arcname}.part000 .. {splitter.parts[-1]['name']} (~2 GiB each)
- `SHA256SUMS` - sha256 of every part (per-part resumable verification)
- `PARTS.json` - machine-readable manifest (sizes, hashes, source hashes)

## Reassemble + verify + extract (Linux box)
```bash
# 1) verify each part independently; re-download ONLY parts that fail
sha256sum -c SHA256SUMS

# 2) concatenate parts in numeric order
cat {arcname}.part* > {arcname}

# 3) extract (any tar with zstd support)
tar --zstd -xf {arcname}
#    or: zstd -dc {arcname} | tar -xf -

# 4) verify the extracted corpus pack
python -m corpus verify --pack {pack.name}
```

## Resumable re-download
`sha256sum -c SHA256SUMS` names every mismatching part; fetch only those,
re-run the check, then re-concatenate. Parts are independent fixed-size
slices of one zstd frame - the concatenation order is fixed.

## Integrity anchors
- pack_v2c SHA256SUMS sha256: `{packs_sums_sha}`
- upload_manifest (upload_manifest_v2c.json) sha256: `{manifest_sha}`
- decompressed member bytes: {member_bytes}; {len(members)} members
"""
    (out / "UNPACK.md").write_text(unpack, encoding="utf-8")

    summary = dict(
        out=str(out), parts=len(splitter.parts),
        part_bytes=[p["bytes"] for p in splitter.parts],
        member_count=len(members), member_bytes=member_bytes,
        tar_stream_bytes=splitter.total,
        sha256sums=str(sums_path), parts_json=str(out / "PARTS.json"),
        unpack=str(out / "UNPACK.md"), zstd_version=version)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
