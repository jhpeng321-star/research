"""Build and verify the resumable compressed upload bundle for pack_v2.

Pipeline (build):
    sorted pack_v2 files -> deterministic GNU tar (mtime=0, uid=gid=0,
    uname=gname="", mode 0644) -> zstd CLI (-T6) -> ~2 GiB parts
    pack_v2.tar.zst.partNNN -> SHA256SUMS + PARTS.json + UNPACK.md

Pipeline (verify):
    re-hash all parts -> streaming decompress (zstd -d) -> stream tar ->
    per-member name/size/sha256 check against pack_v2/SHA256SUMS.

Writes only under the upload output dir. Refuses to overwrite any file.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

CHUNK = 8 * 1024 * 1024
DEFAULT_PART_SIZE = 2 * 1024**3
ZSTD = r"C:\Program Files\MSYS2\mingw64\bin\zstd.exe"
ZSTD_THREADS = 6
ZSTD_LEVEL = 6


def set_below_normal() -> None:
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_sums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sha, name = line.split(None, 1)
        out[name.strip().lstrip("*")] = sha.strip().lower()
    return out


def zstd_version() -> str:
    txt = subprocess.run(
        [ZSTD, "--version"], capture_output=True, text=True, check=True
    ).stdout
    for tok in txt.split():
        tok = tok.strip(",()[]")
        if tok.startswith("v") and "." in tok:
            return tok
    return txt.strip()


def activate_d_override(root: Path) -> str | None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from corpus._util import activate_d_output_override

        return activate_d_output_override(str(root))
    except Exception as exc:  # pragma: no cover - hook env differences
        os.environ["WD_FORBID_D"] = "1"
        os.environ["WD_D_ALLOWED_ROOT"] = str(root)
        print(f"[warn] forbid_d_hook unavailable ({exc}); env-only policy set")
        return None


def find_upload_manifest(pack: Path, base: Path) -> Path:
    for cand in (pack / "upload_manifest.json", base / "upload_manifest.json"):
        if cand.is_file():
            return cand
    raise SystemExit(
        "upload_manifest.json not found (looked in pack_v2/ and wmn1_feat/); "
        "pack not final yet - refusing to build"
    )


def load_members(pack: Path, sums: dict[str, str]) -> list[tuple[str, Path, int]]:
    """Tar members are exactly the files listed in pack SHA256SUMS.

    Bundle-level artifacts living in the pack dir (SHA256SUMS itself,
    upload_manifest.json) are not pack members and are not listed there.
    """
    by_name = {p.name: p for p in pack.iterdir() if p.is_file()}
    missing = sorted(set(sums) - set(by_name))
    if missing:
        raise SystemExit(f"SHA256SUMS lists absent files: {missing}")
    return [(name, by_name[name], by_name[name].stat().st_size) for name in sorted(sums)]


def preverify_pack(members, sums) -> None:
    print(f"[preverify] hashing {len(members)} pack files ...")
    for name, path, size in members:
        got = sha256_file(path)
        if got != sums[name]:
            raise SystemExit(f"pack integrity FAIL {name}: {got} != {sums[name]}")
        print(f"[preverify] ok {name} ({size} B)")


class PartWriter:
    def __init__(self, out_dir: Path, part_size: int,
                 archive: str = "pack_v2.tar.zst") -> None:
        self.out_dir = out_dir
        self.part_size = part_size
        self.archive = archive          # e.g. pack_v3.tar.zst (pack dir name)
        self.index = -1
        self.f = None
        self.sha = None
        self.written = 0
        self.parts: list[dict] = []

    def _rotate(self) -> None:
        if self.f is not None:
            self.f.close()
            self.parts.append(
                {
                    "name": f"{self.archive}.part{self.index:03d}",
                    "bytes": self.written,
                    "sha256": self.sha.hexdigest(),
                }
            )
            print(f"[part] {self.parts[-1]['name']} done: {self.written} B")
        self.index += 1
        name = f"{self.archive}.part{self.index:03d}"
        path = self.out_dir / name
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {path}")
        self.f = open(path, "xb")
        self.sha = hashlib.sha256()
        self.written = 0

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while len(view) > 0:
            if self.f is None or self.written >= self.part_size:
                self._rotate()
            take = min(len(view), self.part_size - self.written)
            piece = bytes(view[:take])
            self.f.write(piece)
            self.sha.update(piece)
            self.written += take
            view = view[take:]

    def close(self) -> None:
        if self.f is not None and self.written > 0:
            self.f.close()
            self.parts.append(
                {
                    "name": f"{self.archive}.part{self.index:03d}",
                    "bytes": self.written,
                    "sha256": self.sha.hexdigest(),
                }
            )
            print(f"[part] {self.parts[-1]['name']} done: {self.written} B")
        elif self.f is not None:
            self.f.close()
        self.f = None


def cmd_build(args: argparse.Namespace) -> None:
    pack = Path(args.pack).resolve()
    base = pack.parent
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to build into non-empty {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = f"{pack.name}.tar.zst"    # pack_v2 -> pack_v2.tar.zst (as before)

    manifest_path = find_upload_manifest(pack, base)
    sums_path = pack / "SHA256SUMS"
    sums = parse_sums(sums_path)
    members = load_members(pack, sums)
    preverify_pack(members, sums)

    t0 = time.monotonic()
    proc = subprocess.Popen(
        [ZSTD, f"-T{ZSTD_THREADS}", f"-{ZSTD_LEVEL}", "--no-progress", "-q", "-c", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    print(f"[zstd] {zstd_version()} level={ZSTD_LEVEL} threads={ZSTD_THREADS} pid={proc.pid}")

    stats = {"tar_bytes": 0}
    writer_exc: list[BaseException] = []

    def tar_writer() -> None:
        try:
            with tarfile.open(
                fileobj=proc.stdin, mode="w|", format=tarfile.GNU_FORMAT, bufsize=10240
            ) as tf:
                for name, path, size in members:
                    ti = tarfile.TarInfo(name)
                    ti.size = size
                    ti.mtime = 0
                    ti.mode = 0o644
                    ti.uid = 0
                    ti.gid = 0
                    ti.uname = ""
                    ti.gname = ""
                    ti.type = tarfile.REGTYPE
                    with open(path, "rb", buffering=0) as f:
                        tf.addfile(ti, f)
                    stats["tar_bytes"] += size
                    print(f"[tar] +{name} ({size} B) t+{time.monotonic() - t0:.1f}s")
        except BaseException as exc:  # noqa: BLE03
            writer_exc.append(exc)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    thread = threading.Thread(target=tar_writer, daemon=True)
    thread.start()

    pw = PartWriter(out_dir, args.part_size, archive)
    total_zst = 0
    last = t0
    try:
        while True:
            chunk = proc.stdout.read(CHUNK)
            if not chunk:
                break
            pw.write(chunk)
            total_zst += len(chunk)
            now = time.monotonic()
            if now - last >= 60:
                last = now
                rate = total_zst / max(now - t0, 1e-6) / 1e6
                print(f"[zstd-out] {total_zst} B in {now - t0:.0f}s ({rate:.0f} MB/s)")
        pw.close()
    except BaseException:
        proc.kill()
        raise
    finally:
        thread.join(timeout=30)
        if pw.f is not None:
            pw.f.close()
    proc.stdout.close()
    rc = proc.wait()
    if writer_exc:
        raise SystemExit(f"tar writer failed: {writer_exc[0]!r}")
    if rc != 0:
        raise SystemExit(f"zstd exited rc={rc}")

    pack_bytes = sum(size for _, _, size in members)
    pack_sums_sha = sha256_file(sums_path)
    manifest_sha = sha256_file(manifest_path)
    wall = time.monotonic() - t0

    sha_path = out_dir / "SHA256SUMS"
    parts_path = out_dir / "PARTS.json"
    unpack_path = out_dir / "UNPACK.md"
    for p in (sha_path, parts_path, unpack_path):
        if p.exists():
            raise SystemExit(f"refusing to overwrite existing {p}")

    lines = []
    for p in pw.parts:
        lines.append(f"{p['sha256']}  {p['name']}")
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    part_names = [p["name"] for p in pw.parts]
    cat_cmd = " ".join(part_names)
    pack_label = pack.name                     # pack_v2 / pack_v3 / ...
    unpack = f"""# upload - resumable compressed upload of {pack_label}

Deterministic archive: GNU tar (members sorted by name, mtime=0, uid=gid=0,
mode 0644) compressed with zstd {zstd_version()} level {ZSTD_LEVEL} (-T{ZSTD_THREADS}),
split into {len(part_names)} parts of ~{args.part_size} bytes.

Source: {pack.as_posix()} ({len(members)} files, {pack_bytes} bytes)
Built (UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

## Files in this bundle
- {len(part_names)} parts: {part_names[0]} .. {part_names[-1]} (~{args.part_size // (1 << 30)} GiB each)
- `SHA256SUMS` - sha256 of every part (per-part resumable verification)
- `PARTS.json` - machine-readable manifest (sizes, hashes, source hashes)

## Reassemble + verify + extract (Linux box)
```bash
# 1) verify each part independently; re-download ONLY parts that fail
sha256sum -c SHA256SUMS

# 2) concatenate parts in numeric order
cat {cat_cmd} > {archive}
#    (equivalent: cat {archive}.part* > {archive})

# 3) extract (any tar with zstd support)
tar --zstd -xf {archive}
#    or: zstd -dc {archive} | tar -xf -

# 4) verify the extracted corpus pack
python -m corpus verify --pack {pack_label}
```

## Resumable re-download
`sha256sum -c SHA256SUMS` names every mismatching part; fetch only those,
re-run the check, then re-concatenate. Parts are independent fixed-size
slices of one zstd frame - the concatenation order is fixed.

## Integrity anchors
- {pack_label} SHA256SUMS sha256: `{pack_sums_sha}`
- upload_manifest ({manifest_path.name}) sha256: `{manifest_sha}`
- decompressed tar size: {stats["tar_bytes"]} bytes; {len(members)} members
"""
    unpack_path.write_text(unpack, encoding="utf-8", newline="\n")

    parts_doc = {
        "schema": "wm_n1_upload_parts/1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_pack": pack.as_posix(),
        "pack_files": {"count": len(members), "bytes": pack_bytes},
        "pack_sha256sums_sha256": pack_sums_sha,
        "upload_manifest": {
            "path": manifest_path.as_posix(),
            "sha256": manifest_sha,
        },
        "archive": {
            "name": archive,
            "tar_format": "gnu",
            "member_order": "sorted-by-name",
            "member_mtime": 0,
            "member_uid": 0,
            "member_gid": 0,
            "member_mode": "0644",
            "zstd": {
                "version": zstd_version(),
                "level": ZSTD_LEVEL,
                "threads": ZSTD_THREADS,
            },
        },
        "part_size_bytes": args.part_size,
        "parts": pw.parts,
        "parts_total_bytes": total_zst,
        "decompressed_tar_bytes": stats["tar_bytes"],
        "compressed_ratio_vs_pack": round(total_zst / pack_bytes, 6),
        "build_wall_seconds": round(wall, 1),
    }
    parts_path.write_text(
        json.dumps(parts_doc, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    print(
        f"[done] parts={len(part_names)} compressed={total_zst} B "
        f"ratio={total_zst / pack_bytes:.4f} wall={wall:.1f}s"
    )


class ConcatReader:
    """Exact-fill reader over an ordered list of part files."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.i = 0
        self.f = None
        self.pos = 0
        self.total = 0

    def _next_file(self) -> bool:
        if self.f is not None:
            self.f.close()
            self.f = None
        if self.i >= len(self.paths):
            return False
        self.f = open(self.paths[self.i], "rb", buffering=0)
        self.i += 1
        return True

    def read(self, n: int) -> bytes:
        chunks = []
        while n > 0:
            if self.f is None and not self._next_file():
                break
            b = self.f.read(n)
            if not b:
                self.f.close()
                self.f = None
                continue
            chunks.append(b)
            n -= len(b)
            self.pos += len(b)
            self.total += len(b)
        return b"".join(chunks)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


def cmd_verify(args: argparse.Namespace) -> None:
    pack = Path(args.pack).resolve()
    out_dir = Path(args.out).resolve()
    sums = parse_sums(pack / "SHA256SUMS")
    members = load_members(pack, sums)
    parts_doc = json.loads((out_dir / "PARTS.json").read_text(encoding="utf-8"))
    part_paths = [out_dir / p["name"] for p in parts_doc["parts"]]
    for p in part_paths:
        if not p.is_file():
            raise SystemExit(f"missing part {p}")

    t0 = time.monotonic()
    print(f"[verify] re-hashing {len(part_paths)} parts ...")
    for doc, path in zip(parts_doc["parts"], part_paths):
        got = sha256_file(path)
        if got != doc["sha256"]:
            raise SystemExit(f"part hash FAIL {path.name}: {got} != {doc['sha256']}")
        if path.stat().st_size != doc["bytes"]:
            raise SystemExit(f"part size FAIL {path.name}")
        print(f"[verify] part ok {path.name}")
    sha_lines = parse_sums(out_dir / "SHA256SUMS")
    for doc in parts_doc["parts"]:
        if sha_lines.get(doc["name"]) != doc["sha256"]:
            raise SystemExit(f"SHA256SUMS/PARTS.json disagree for {doc['name']}")

    reader = ConcatReader(part_paths)
    proc = subprocess.Popen(
        [ZSTD, "-d", f"-T{ZSTD_THREADS}", "--no-progress", "-q", "-c", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    def pump() -> None:
        try:
            while True:
                b = reader.read(CHUNK)
                if not b:
                    break
                proc.stdin.write(b)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()

    seen: dict[str, str] = {}
    tally = {"tar_bytes": 0}
    stream_err: list[BaseException] = []
    expected_sizes = {name: size for name, _, size in members}

    def stream_check() -> None:
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
                for m in tf:
                    if m.name not in sums:
                        raise SystemExit(f"unexpected tar member {m.name}")
                    if m.size != expected_sizes[m.name]:
                        raise SystemExit(f"size mismatch {m.name}: {m.size}")
                    h = hashlib.sha256()
                    left = m.size
                    f = tf.extractfile(m)
                    while left > 0:
                        b = f.read(min(CHUNK, left))
                        if not b:
                            raise SystemExit(f"truncated member {m.name}")
                        h.update(b)
                        left -= len(b)
                    if h.hexdigest() != sums[m.name]:
                        raise SystemExit(f"content sha FAIL {m.name}")
                    seen[m.name] = h.hexdigest()
                    tally["tar_bytes"] += m.size
                    print(f"[verify] member ok {m.name} t+{time.monotonic() - t0:.1f}s")
        except BaseException as exc:  # noqa: BLE03
            stream_err.append(exc)
        finally:
            try:
                if not stream_err:
                    while proc.stdout.read(CHUNK):
                        pass
                proc.stdout.close()
            except Exception:
                pass

    sthread = threading.Thread(target=stream_check, daemon=True)
    sthread.start()
    sthread.join(timeout=None)
    thread.join(timeout=30)
    rc = proc.wait()
    reader.close()
    if stream_err:
        raise SystemExit(f"stream verify failed: {stream_err[0]!r}")
    if rc != 0:
        raise SystemExit(f"zstd -d exited rc={rc}")
    if set(seen) != set(sums):
        raise SystemExit(
            f"member set mismatch: missing={sorted(set(sums) - set(seen))} "
            f"extra={sorted(set(seen) - set(sums))}"
        )
    if tally["tar_bytes"] != parts_doc["decompressed_tar_bytes"]:
        raise SystemExit(
            f"tar byte total mismatch: {tally['tar_bytes']} != {parts_doc['decompressed_tar_bytes']}"
        )
    wall = time.monotonic() - t0
    print(
        f"[done] {len(seen)}/{len(sums)} members byte-verified, tar={tally['tar_bytes']} B, "
        f"parts={parts_doc['parts_total_bytes']} B, wall={wall:.1f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--pack", default=r"D:\rd-staging\wmn1_feat\pack_v2")
    b.add_argument("--out", default=r"D:\rd-staging\wmn1_feat\upload_v2")
    b.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    b.add_argument("--d-override", action="store_true")

    v = sub.add_parser("verify")
    v.add_argument("--pack", default=r"D:\rd-staging\wmn1_feat\pack_v2")
    v.add_argument("--out", default=r"D:\rd-staging\wmn1_feat\upload_v2")
    v.add_argument("--d-override", action="store_true")

    args = ap.parse_args()
    set_below_normal()
    if getattr(args, "d_override", False):
        allowed = activate_d_override(Path(args.out).parent)
        print(f"[d-policy] allowed root: {allowed or os.environ.get('WD_D_ALLOWED_ROOT')}")
    if args.cmd == "build":
        cmd_build(args)
    else:
        cmd_verify(args)


if __name__ == "__main__":
    main()
