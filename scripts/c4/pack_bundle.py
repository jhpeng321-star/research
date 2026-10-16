#!/usr/bin/env python
"""Build the C4 transfer bundle (code + TRAIN-only data subsets + checkpoints).

Output directory (nothing is written with --dry-run):
    c4_code.tar            code, configs, SUMO assets, tests (shell files normalised to LF);
                           members are repo-relative (extract into the box repo root)
    c4_data_<group>.tar    one tar per data group in scripts/c4/bundle_manifest.json
    c4_ckpt.tar            checkpoints
    BUNDLE_MANIFEST.json   every member with size + sha256, tar sha256, git identity, filters
    SHA256SUMS             sha256 of every tar + manifest (LF)

Pack groups are *row-range* archives: the big ``.npy`` arrays keep their full shape and
global indexing (so INDEX/manifest/sidecar positions stay valid) but only the tick /
decision rows of the selected TRAIN runs are stored; ``unpack_bundle.py`` recreates sparse
full-size arrays on the box and fills those rows.  Rows of every other run (CALIBRATION
items, designed-controller windows, eval worlds) never leave this machine.

Hard filter (case-insensitive): holdout, heldout, cal, calibration, cal_split*, sealed,
chunk5 / chunk-5, j4b, inc1 in any bundle path -> member dropped; after building, every
tar is re-opened and every member name re-checked (self-check); a hit aborts the build.
Run metadata inside the copied INDEX/manifest JSON still lists non-TRAIN runs by name
(positions must stay valid) but no array bytes of those runs are included.

Standard library + numpy only.  Reads source data, never modifies it.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
DEFAULT_DATA_ROOT = Path("data_cache")
RESTRICTED = re.compile(
    r"(?:^|[^a-z0-9])(cal(?:ibration)?|cal_split\w*|holdout|heldout|sealed|chunk[-_]?5|j4b|inc1)"
    r"(?:$|[^a-z0-9])", re.IGNORECASE)
CHUNK = 1 << 24
RANGES_NAME = "C4_RANGES.json"
PART_DIR = "__ranges__"


def restricted(text: str) -> str | None:
    m = RESTRICTED.search(str(text).replace("\\", "/"))
    return m.group(1).lower() if m else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return str(n)


# ----------------------------------------------------------------------------- tar helpers
class Dropped(list):
    """Collects (path, reason) for everything the filter removed."""


class RangeReader(io.RawIOBase):
    """File-like object streaming selected rows of an npy array (hashes while reading)."""

    def __init__(self, path: Path, data_offset: int, row_bytes: int, ranges: list[list[int]]):
        self._f = open(path, "rb")
        self._off, self._rb = data_offset, row_bytes
        self._ranges = ranges
        self._ri = 0
        self._pos = 0  # byte position inside the current range
        self.sha = hashlib.sha256()
        self.size = sum((b - a) for a, b in ranges) * row_bytes

    def readable(self):
        return True

    def readinto(self, buf):
        view = memoryview(buf)
        n = 0
        while n < len(view) and self._ri < len(self._ranges):
            a, b = self._ranges[self._ri]
            span = (b - a) * self._rb
            if self._pos == 0:
                self._f.seek(self._off + a * self._rb)
            take = min(len(view) - n, span - self._pos)
            got = self._f.readinto(view[n:n + take])
            if not got:
                raise IOError("short read in range reader")
            n += got
            self._pos += got
            if self._pos >= span:
                self._ri += 1
                self._pos = 0
        self.sha.update(view[:n])
        return n

    def close(self):
        self._f.close()
        super().close()


def tarinfo(name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.mtime = size, mode, 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    return ti


class Bundle:
    """One output tar plus the manifest rows of its members."""

    def __init__(self, path: Path, dry: bool, dropped: Dropped, allow: set[str] | None = None):
        self.path, self.dry, self.dropped = path, dry, dropped
        self.allow = allow or set()   # exact member names exempt from the token filter (code only)
        self.rows: list[dict] = []
        self.tar = None if dry else tarfile.open(path, "w", format=tarfile.PAX_FORMAT)
        self.seen: set[str] = set()

    def _check(self, name: str, src: str) -> bool:
        tok = None if name in self.allow else (restricted(name) or restricted(src))
        if tok:
            self.dropped.append((src, f"restricted token '{tok}'"))
            return False
        if name in self.seen:
            raise SystemExit(f"duplicate bundle member {name}")
        self.seen.add(name)
        return True

    def add_bytes(self, name: str, data: bytes, src: str = "<generated>", mode: int = 0o644,
                  extra: dict | None = None) -> None:
        if not self._check(name, src):
            return
        row = {"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "src": src, **(extra or {})}
        self.rows.append(row)
        if self.tar is not None:
            self.tar.addfile(tarinfo(name, len(data), mode), io.BytesIO(data))

    def add_file(self, name: str, path: Path, mode: int = 0o644, lf: bool = False) -> None:
        if not self._check(name, str(path)):
            return
        size = path.stat().st_size
        if lf:
            data = path.read_bytes()
            norm = data.replace(b"\r\n", b"\n")
            if b"\r" in norm:
                raise SystemExit(f"{path}: stray CR bytes remain after LF normalisation")
            row = {"name": name, "size": len(norm), "sha256": hashlib.sha256(norm).hexdigest(),
                   "src": str(path), "lf_normalised": norm != data}
            self.rows.append(row)
            if self.tar is not None:
                self.tar.addfile(tarinfo(name, len(norm), mode), io.BytesIO(norm))
            return
        row = {"name": name, "size": size, "sha256": None if self.dry else sha256_file(path),
               "src": str(path)}
        self.rows.append(row)
        if self.tar is not None:
            with open(path, "rb") as f:
                self.tar.addfile(tarinfo(name, size, mode), f)

    def add_ranges(self, name: str, reader: RangeReader | None, size: int, src: str,
                   extra: dict) -> dict:
        if not self._check(name, src):
            return {}
        row = {"name": name, "size": size, "sha256": None, "src": src, **extra}
        self.rows.append(row)
        if self.tar is not None and reader is not None:
            self.tar.addfile(tarinfo(name, size), reader)
            row["sha256"] = reader.sha.hexdigest()
        return row

    def close(self) -> str | None:
        if self.tar is not None:
            self.tar.close()
            return sha256_file(self.path)
        return None


# ----------------------------------------------------------------------------- selection
def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


EXTERNAL_ROOTS: dict[str, Path] = {}


def resolve_src(data_root: Path, rel: str) -> Path:
    """``rel`` under ``data_root``; ``@name/...`` resolves under EXTERNAL_ROOTS[name] (manifest
    ``external_roots`` or ``--external-root name=PATH``), for inputs that live outside data_cache
    (e.g. the generated C4 window lists in the git-ignored runs/ tree)."""
    if rel.startswith("@"):
        name, _, rest = rel[1:].partition("/")
        if name not in EXTERNAL_ROOTS:
            raise SystemExit(f"unknown external root @{name}; known: {sorted(EXTERNAL_ROOTS)}")
        return EXTERNAL_ROOTS[name] / rest if rest else EXTERNAL_ROOTS[name]
    return data_root / rel


def window_ok(w: dict, flt: str | None, policies: set[str]) -> bool:
    if flt is None:
        return True
    kind, _, arg = flt.partition(":")
    if kind == "policy":
        return str(w.get("policy")) in policies
    if kind == "arm":
        return str(w.get("arm")) == arg
    raise SystemExit(f"unknown filter {flt}")


def select_runs(group: dict, data_root: Path, policies: set[str]) -> tuple[list[str], dict]:
    keys: list[str] = []
    stats = {"windows_total": 0, "windows_kept": 0}
    for rel in group["lists"]:
        doc = load_json(resolve_src(data_root, rel))
        for w in doc["windows"]:
            if w.get("pack") not in (None, group["pack_name"]):
                continue
            stats["windows_total"] += 1
            if str(w.get("split", "TRAIN")).upper() != "TRAIN":
                raise SystemExit(f"window {w.get('key')} in {rel} is not TRAIN")
            if window_ok(w, group.get("filter"), policies):
                keys.append(w["key"])
                stats["windows_kept"] += 1
    return sorted(set(keys)), stats


def merge(ranges: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def npy_info(path: Path) -> dict:
    """Header geometry of an .npy without loading data."""
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.flags.f_contiguous and arr.ndim > 1 and not arr.flags.c_contiguous:
        raise SystemExit(f"{path}: Fortran-ordered arrays are not supported")
    off = int(arr.offset)
    with open(path, "rb") as f:
        header = f.read(off)
    row_bytes = int(np.prod(arr.shape[1:], dtype=np.int64)) * arr.dtype.itemsize if arr.ndim > 1 \
        else arr.dtype.itemsize
    info = {"shape": list(arr.shape), "dtype": str(arr.dtype), "offset": off,
            "row_bytes": int(row_bytes), "header_hex": header.hex(),
            "total_bytes": int(path.stat().st_size)}
    del arr
    return info


# ----------------------------------------------------------------------------- group builders
def build_pack(name: str, group: dict, data_root: Path, out: Path, dry: bool, policies: set[str],
               dropped: Dropped, report: dict) -> Bundle:
    src_dir = resolve_src(data_root, group["src"])
    dest = group["dest"]
    index = load_json(src_dir / "INDEX.json")
    runs = {r["key"]: r for r in index["runs"]}
    keys, wstats = select_runs(group, data_root, policies)
    missing = [k for k in keys if k not in runs]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} selected run keys missing from INDEX, e.g. {missing[:3]}")
    bad = [k for k in keys if str(runs[k].get("split", "")).upper() != "TRAIN"]
    if bad:
        raise SystemExit(f"{name}: selected runs not TRAIN: {bad[:3]}")
    tick_r = merge([(runs[k]["tick_start"], runs[k]["tick_end"]) for k in keys])
    dec_r = merge([(runs[k]["dec_start"], runs[k]["dec_end"]) for k in keys])
    n_ticks, n_dec = int(index["n_ticks"]), int(index["n_decision"])
    items = load_json(src_dir / "manifest.json")["items"]
    n_items = len(items)
    # item-axis arrays (e.g. controller_timeline): rows of manifest items whose tick lies in a selected run
    starts = sorted((runs[k]["tick_start"], runs[k]["tick_end"]) for k in keys)
    item_r = merge([(i, i + 1) for i, it in enumerate(items)
                    if any(a <= it["tick"] < b for a, b in starts)])
    bundle = Bundle(out / f"c4_data_{name}.tar", dry, dropped)
    ranges_doc = {"schema": "c4_pack_ranges/1", "pack": name, "dest": dest, "n_ticks": n_ticks,
                  "n_decision": n_dec, "selected_runs": len(keys), "arrays": {}}
    sel_ticks = sum(b - a for a, b in tick_r)
    sel_dec = sum(b - a for a, b in dec_r)
    arrays_bytes = 0
    for npy in sorted(src_dir.glob("*.npy")):
        info = npy_info(npy)
        rows = info["shape"][0]
        hits = [a for a, n in (("tick", n_ticks), ("dec", n_dec), ("item", n_items)) if rows == n]
        if len(hits) != 1:
            raise SystemExit(f"{npy}: first axis {rows} matches {hits or 'none'} of n_ticks {n_ticks}, "
                             f"n_decision {n_dec}, n_items {n_items}")
        axis = hits[0]
        ranges = {"tick": tick_r, "dec": dec_r, "item": item_r}[axis]
        size = sum(b - a for a, b in ranges) * info["row_bytes"]
        arrays_bytes += size
        member = f"{dest}/{PART_DIR}/{npy.stem}.bin"
        reader = None if dry else RangeReader(npy, info["offset"], info["row_bytes"], ranges)
        row = bundle.add_ranges(member, reader, size, str(npy), {"axis": axis, "kind": "ranges"})
        if reader is not None:
            reader.close()
        if row:
            ranges_doc["arrays"][npy.name] = {
                "axis": axis, "ranges": ranges, "part": member, "part_sha256": row.get("sha256"),
                "part_bytes": size, **info}
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and f.suffix != ".npy":
            bundle.add_file(f"{dest}/{f.name}", f)
    bundle.add_bytes(f"{dest}/{RANGES_NAME}",
                     (json.dumps(ranges_doc, indent=1, sort_keys=True) + "\n").encode(),
                     "<generated>")
    report[name] = {"kind": "pack", "windows": wstats, "runs": len(keys),
                    "ticks_selected": sel_ticks, "ticks_total": n_ticks,
                    "dec_selected": sel_dec, "dec_total": n_dec,
                    "array_bytes": arrays_bytes}
    return bundle


def build_plain(name: str, group: dict, data_root: Path, out: Path, dry: bool,
                dropped: Dropped, report: dict) -> Bundle:
    bundle = Bundle(out / f"c4_data_{name}.tar", dry, dropped)
    base = resolve_src(data_root, group["src"])
    dest = group["dest"]
    for rel in group.get("files", []):
        bundle.add_file(f"{dest}/{rel}".replace("//", "/"), base / rel)
    for d in group.get("dirs", []):
        root = base / d
        for f in sorted(root.rglob("*")):
            if f.is_file():
                bundle.add_file(f"{dest}/{d}/{f.relative_to(root).as_posix()}", f)
    report[name] = {"kind": "plain"}
    return bundle


def build_lists(name: str, group: dict, data_root: Path, out: Path, dry: bool, policies: set[str],
                dropped: Dropped, report: dict) -> Bundle:
    bundle = Bundle(out / f"c4_data_{name}.tar", dry, dropped)
    info = {}
    for out_name, spec in group["sources"].items():
        doc = load_json(resolve_src(data_root, spec["from"]))
        kept = [w for w in doc["windows"] if window_ok(w, spec["filter"], policies)]
        new = dict(doc)
        new["windows"] = kept
        new["n_windows"] = len(kept)
        new["c4_derived_from"] = spec["from"]
        new["c4_filter"] = spec["filter"] if not spec["filter"].startswith("policy") \
            else f"policy in {sorted(policies)}"
        if "counts" in doc:
            for key in ("by_pack", "by_policy", "by_batch"):
                cnt: dict = {}
                field = {"by_pack": "pack", "by_policy": "policy", "by_batch": "batch"}[key]
                for w in kept:
                    cnt[str(w.get(field))] = cnt.get(str(w.get(field)), 0) + 1
                new["counts"] = {**new.get("counts", {}), key: cnt}
            new["counts"]["total"] = len(kept)
            new["counts"]["n_worlds"] = len({w.get("world") for w in kept})
        bundle.add_bytes(f"{group['dest']}/{out_name}",
                         (json.dumps(new, indent=1, sort_keys=True) + "\n").encode(),
                         spec["from"])
        info[out_name] = {"windows_in": len(doc["windows"]), "windows_out": len(kept)}
    report[name] = {"kind": "derived_lists", "files": info}
    return bundle


def build_code(manifest: dict, repo: Path, out: Path, dry: bool, dropped: Dropped,
               report: dict) -> Bundle:
    spec = manifest["code"]
    allow = set(manifest.get("code_allowlist", []))
    bundle = Bundle(out / "c4_code.tar", dry, dropped, allow)
    excl = spec["exclude_globs"]
    lf_suffix = tuple(spec["text_lf_suffixes"])

    def excluded(rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel, g.removeprefix("**/"))
                   for g in excl)

    def add(rel: str, path: Path) -> None:
        if excluded(rel):
            return
        is_sh = rel.endswith(".sh")
        bundle.add_file(rel, path, mode=0o755 if is_sh else 0o644,
                        lf=rel.endswith(lf_suffix))

    for rel in spec.get("root_files", []) + spec.get("files", []):
        p = repo / rel
        if p.is_file():
            add(rel, p)
        else:
            print(f"warning: code file missing: {rel}", file=sys.stderr)
    for d in spec["dirs"]:
        root = repo / d
        if not root.is_dir():
            print(f"warning: code dir missing: {d}", file=sys.stderr)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                add(p.relative_to(repo).as_posix(), p)
    report["code"] = {"kind": "code"}
    return bundle


def build_ckpts(manifest: dict, data_root: Path, out: Path, dry: bool, dropped: Dropped,
                report: dict) -> Bundle:
    bundle = Bundle(out / "c4_ckpt.tar", dry, dropped)
    for ck in manifest["checkpoints"]:
        src = Path(ck["src"])
        if not src.is_absolute():
            src = data_root / src
        if not src.is_file():
            if ck.get("optional"):
                print(f"note: optional checkpoint companion missing, skipped: {src}", file=sys.stderr)
                continue
            raise SystemExit(f"checkpoint missing: {src}")
        bundle.add_file(ck["dest"], src)
        if ck.get("expect_sha256_prefix") and not dry:
            got = bundle.rows[-1]["sha256"]
            if got and not got.startswith(ck["expect_sha256_prefix"]):
                raise SystemExit(f"{src}: sha256 {got} != expected prefix {ck['expect_sha256_prefix']}")
        elif ck.get("expect_sha256_prefix") and dry:
            got = sha256_file(src)
            if not got.startswith(ck["expect_sha256_prefix"]):
                raise SystemExit(f"{src}: sha256 {got} != expected prefix {ck['expect_sha256_prefix']}")
    report["checkpoints"] = {"kind": "ckpt"}
    return bundle


# ----------------------------------------------------------------------------- self-check
def self_check(tars: list[Path], allow: set[str]) -> list[str]:
    bad = []
    for t in tars:
        with tarfile.open(t, "r") as tf:
            for m in tf:
                tok = None if (t.name == "c4_code.tar" and m.name in allow) else restricted(m.name)
                if tok:
                    bad.append(f"{t.name}: {m.name} ({tok})")
                if m.name.endswith(".sh"):
                    data = tf.extractfile(m).read()
                    if b"\r" in data:
                        bad.append(f"{t.name}: {m.name} has CR bytes")
    return bad


def git_identity(repo: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([l for l in run("status", "--porcelain").splitlines() if l.strip()])}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="output directory (must not exist or be empty)")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--manifest", type=Path, default=HERE / "bundle_manifest.json")
    ap.add_argument("--groups", default="all",
                    help="comma list of manifest groups: code,ckpt,c4_wm_lists,lpwo_v3,pack_v3,"
                         "pack_dspeed,pack_switching_train,sidecars,eval_states (default all but eval_states)")
    ap.add_argument("--external-root", action="append", default=[], metavar="NAME=PATH",
                    help="override/add an @NAME root used by manifest paths (default: manifest external_roots)")
    ap.add_argument("--policies", default=None,
                    help="window policies to keep for lpwo packs (default from the manifest)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list files/sizes and range statistics; write nothing")
    ap.add_argument("--list-files", action="store_true", help="with --dry-run: print every member")
    args = ap.parse_args(argv)

    manifest = load_json(args.manifest)
    policies = set((args.policies or ",".join(manifest["default_policies"])).split(","))
    for name, path in manifest.get("external_roots", {}).items():
        EXTERNAL_ROOTS[name] = Path(path)
    for item in args.external_root:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"--external-root expects NAME=PATH, got {item!r}")
        EXTERNAL_ROOTS[name] = Path(path)
    all_groups = ["code", "ckpt", *manifest["groups"].keys()]
    wanted = all_groups if args.groups == "all" else args.groups.split(",")
    for g in wanted:
        if g not in all_groups and g not in manifest.get("optional_groups", {}):
            raise SystemExit(f"unknown group {g}; known: {all_groups + list(manifest['optional_groups'])}")
    if not args.dry_run:
        if args.out is None:
            raise SystemExit("--out is required unless --dry-run")
        if args.out.exists() and any(args.out.iterdir()):
            raise SystemExit(f"{args.out} is not empty; choose a fresh directory (never overwrite)")
        args.out.mkdir(parents=True, exist_ok=True)
    out = args.out or Path(".")

    dropped = Dropped()
    report: dict = {}
    bundles: list[Bundle] = []
    groups = dict(manifest["groups"])
    groups.update({k: v for k, v in manifest.get("optional_groups", {}).items() if "kind" in v})
    for g in wanted:
        if g == "code":
            bundles.append(build_code(manifest, args.repo, out, args.dry_run, dropped, report))
        elif g == "ckpt":
            bundles.append(build_ckpts(manifest, args.data_root, out, args.dry_run, dropped, report))
        else:
            spec = groups[g]
            kind = spec["kind"]
            if kind == "pack":
                bundles.append(build_pack(g, spec, args.data_root, out, args.dry_run, policies,
                                          dropped, report))
            elif kind == "plain":
                bundles.append(build_plain(g, spec, args.data_root, out, args.dry_run, dropped, report))
            elif kind == "derived_lists":
                bundles.append(build_lists(g, spec, args.data_root, out, args.dry_run, policies,
                                           dropped, report))
            else:
                raise SystemExit(f"unknown kind {kind}")

    # ---- report
    total = 0
    print(f"policies kept: {sorted(policies)}   repo: {args.repo}   data root: {args.data_root}")
    for b in bundles:
        size = sum(r["size"] for r in b.rows)
        total += size
        print(f"{b.path.name:34s} {len(b.rows):6d} members {human(size):>10s}")
        if args.dry_run and args.list_files:
            for r in b.rows:
                print(f"    {human(r['size']):>10s}  {r['name']}")
    for g, info in report.items():
        if info.get("kind") == "pack":
            w = info["windows"]
            print(f"  [{g}] windows {w['windows_kept']}/{w['windows_total']} -> {info['runs']} runs; "
                  f"ticks {info['ticks_selected']}/{info['ticks_total']} "
                  f"({100.0 * info['ticks_selected'] / max(info['ticks_total'], 1):.1f}%), "
                  f"decision rows {info['dec_selected']}/{info['dec_total']}, "
                  f"array bytes {human(info['array_bytes'])}")
        elif info.get("kind") == "derived_lists":
            for n, v in info["files"].items():
                print(f"  [{g}] {n}: {v['windows_in']} -> {v['windows_out']} windows")
    print(f"TOTAL {human(total)}")
    if dropped:
        print(f"filter dropped {len(dropped)} path(s):")
        for p, why in dropped[:40]:
            print(f"    {p}  [{why}]")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    # ---- finalise
    shas = {}
    for b in bundles:
        shas[b.path.name] = b.close()
    bad = self_check([b.path for b in bundles],
                     set(manifest.get("code_allowlist", [])))
    if bad:
        for line in bad:
            print("SELF-CHECK FAILURE:", line, file=sys.stderr)
        return 3
    doc = {"schema": "c4_bundle/1", "git": git_identity(args.repo), "policies": sorted(policies),
           "groups": wanted, "report": report,
           "dropped_by_filter": [{"path": p, "reason": r} for p, r in dropped],
           "tars": {b.path.name: {"sha256": shas[b.path.name], "bytes": b.path.stat().st_size,
                                  "members": b.rows} for b in bundles}}
    (out / "BUNDLE_MANIFEST.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                                              encoding="utf-8", newline="\n")
    for boot in ("box_setup.sh", "unpack_bundle.py"):    # bootstrap files next to the tars (LF)
        data = (HERE / boot).read_bytes().replace(bytes([13, 10]), bytes([10]))
        (out / boot).write_bytes(data)
    lines = [f"{shas[n]}  {n}" for n in sorted(shas)]
    lines += [f"{sha256_file(out / b)}  {b}" for b in ("box_setup.sh", "unpack_bundle.py")]
    lines.append(f"{sha256_file(out / 'BUNDLE_MANIFEST.json')}  BUNDLE_MANIFEST.json")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"bundle written to {out}; self-check passed ({sum(len(b.rows) for b in bundles)} members)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
