#!/usr/bin/env python3
"""Build ``tools/data_registry.json`` from the Google Drive upload receipts.

Offline and deterministic: it only reads local files (no network, no Drive access).  Live
Drive checks are separate (``--verify-drive``: read-only ``rclone lsjson --hash``).

Receipts are read from ``data_ops/receipts/`` in this repository (``--receipts``):

* ``dataops/receipts/*.json``  (``dataops_receipt/1``): one ``tar.zst`` per uploaded unit.
* ``driveops/receipts/*.json`` (``driveops_receipt/1``): per-file Drive directories whose
  per-file MD5 lists are ``driveops/receipts/md5/<name>.md5``.
* ``wtclean/receipts/*.json``: worktree backups; only the ``tar.zst`` method is registered
  (as ``wt_<worktree>``); per-file mirrors carry no recorded MD5 manifest and are counted.
* ``wtclean/receipts_mirror_keeplocal.json``: ``rclone check`` proofs of the five hot-data mirrors.

Three static tables below carry what no receipt in ``data_ops/`` provides (legacy 2026-10-06
archives, the five hot-data mirrors' per-file MD5s, the multi-part ``pack_v3``); all were
re-measured against Drive with ``--verify-drive``.

Member-manifest annotations (``covered_by``, ``restricted_member_names``) need the member
manifests, which are not in ``data_ops/``: pass ``--dataops-root <dir containing
manifests/members>`` to recompute them; otherwise they are carried over from the existing
registry file (entries are matched by name).

Fetch model (``tools/fetch_data.py``): ``rclone copy <drive_path> <cache>/<local_path>``
then every file is verified against ``md5_manifest``.  For a ``tar.zst`` unit the fetched
and verified object is the archive itself (``md5_manifest = {<archive>: <md5>}``,
``size_bytes`` = archive bytes); it is NOT unpacked by the tool.  Extra fields describe
the archive (``format``, ``unpacked_bytes``, ``unpacked_files``) and its provenance
(``original_path``, ``class``, ``receipt``, ``covered_by``).  Units whose id, name or path
contain a restricted token (holdout, cal_split, chunk5, j4b, inc1, sealed and the
``wms.core.guard`` tokens) are skipped and only counted.

Usage::

    python tools/build_drive_registry.py            # write tools/data_registry.json
    python tools/build_drive_registry.py --check    # exit 1 if the file would change
    python tools/build_drive_registry.py --check --verify-drive --rclone <rclone.exe>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(__file__).with_name("data_registry.json")
DEFAULT_RECEIPTS = ROOT / "data_ops" / "receipts"
DRIVE_ROOT = "gdrive:research-archive/"

# Names/paths containing any of these (case-insensitive substrings) are never registered.
SENSITIVE_SUBSTRINGS = ("holdout", "cal_split", "chunk5", "chunk-5", "j4b", "inc1", "sealed")
# Mirrors src/wms/core/guard.py (tests cross-check them); anything the repo guard would
# refuse to resolve is not registered either.
_GUARD = re.compile(
    r"(?:^|[^a-z0-9])(cal(?:ibration)?|cal_split_a4|holdout|sealed|chunk[-_]?5|j4b|inc1)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Personal administrative documents swept up by the C: drive upload; not datasets.
EXCLUDE_PERSONAL = frozenset({"c__ResearchProposalForm.pdf", "c__ResearchProposalForm_Jinghan_filled.pdf"})

# Logical names requested by the research code (receipt-derived name -> logical name).
ALIASES = {
    "runs_campaign_stage1_chunk3": "campaign_stage1_chunk3",
    "runs_campaign_stage1_chunk4": "campaign_stage1_chunk4",
}

# Legacy archives uploaded 2026-10-06 (their receipts, dataops/_out/drive_receipts, name no Drive
# path and are not part of data_ops/) and later moved inside Drive (MOVE_WITHIN_DRIVE.json).
# name, Drive path below research-archive/, archive bytes, archive md5, unpacked bytes, files, source
LEGACY = (
    ("raw_chunk1_p40", "repo-evidence/runs/raw_and_superseded/raw_chunk1_p40.tar.zst",
     478235839, "0ec9a20b2ce64bc3c804299c7575aceb", 4888455748, 294,
     "D:/rd-staging/wmn1_feat/raw_chunk1_p40"),
    ("upload_v2", "data/packs/archived_training_packs_v2/upload_v2.tar.zst",
     6861296150, "086de86e3261a204dd506e61132d2128", 6861792222, 7,
     "D:/rd-staging/wmn1_feat/upload_v2"),
    ("upload_v2c", "data/packs/archived_training_packs_v2/upload_v2c.tar.zst",
     6912006074, "fddb5f5d18bdf65e8687e6a9dd6ef7f1", 6912442062, 7,
     "D:/rd-staging/wmn1_feat/upload_v2c"),
    ("shards_v2__chunk1", "data/packs/derived_shards_v2/Chunks/shards_v2__chunk1.tar.zst",
     1248877236, "84dcd41fa2c5a7980acf0cca7cd01e44", 6570238182, 9200,
     "D:/rd-staging/wmn1_feat/shards_v2/chunk1"),
    ("shards_v2__chunk1_fix27", "data/packs/derived_shards_v2/Chunks/shards_v2__chunk1_fix27.tar.zst",
     227353339, "a0160c9dc9982a316dc5dc9f2f22d1d2", 1174914770, 1764,
     "D:/rd-staging/wmn1_feat/shards_v2/chunk1_fix27"),
    ("shards_v2__chunk2", "data/packs/derived_shards_v2/Chunks/shards_v2__chunk2.tar.zst",
     1294650964, "6be1b3c4046581ff72c8bfd804791897", 6739818719, 9576,
     "D:/rd-staging/wmn1_feat/shards_v2/chunk2"),
    ("shards_v2__index", "data/packs/derived_shards_v2/Index/shards_v2__index.tar.zst",
     1807572, "92de89e4be63407177d3c5aad92b123d", 7969005, 5,
     "D:/rd-staging/wmn1_feat/shards_v2"),
    ("worldmodel_dataset_v1", "datasets/exports/worldmodel_dataset_v1.tar.zst",
     126582513, "50feb2eb5d7361b679bcf140e3c1ea37", 6300268222, 85,
     "E:/research-dense/output/worldmodel_dataset_v1"),
    ("worldmodel_dataset_v2", "datasets/exports/worldmodel_dataset_v2.tar.zst",
     467378157, "080223dd66e3a9add3de776114be59ed", 24074324179, 180,
     "E:/research-dense/output/worldmodel_dataset_v2"),
    ("regenerated_v1", "repo-evidence/runs/validation_and_regeneration/regenerated_v1.tar.zst",
     610934547, "a46c5b02bb4facea06f9b28961f09920", 3925939914, 864,
     "E:/research-dense/runs/regenerated_v1"),
    ("regenerated_v1_retry1", "repo-evidence/runs/validation_and_regeneration/regenerated_v1_retry1.tar.zst",
     53018657, "89563e57ba169fba5b8cf4d6585daded", 339012684, 80,
     "E:/research-dense/runs/regenerated_v1_retry1"),
    ("fresh_validation_v1_r3", "repo-evidence/runs/validation_and_regeneration/fresh_validation_v1_r3.tar.zst",
     631029279, "1ea7292e3416d51dbad2bf1bd12c8f61", 4006412985, 800,
     "E:/research-dense/runs/fresh_validation_v1_r3"),
    ("campaign_pilot_v1_r3", "repo-evidence/runs/campaigns/campaign_pilot_v1_r3.tar.zst",
     7653985, "8053045d7792b61c0e37b49106ebad4d", 62163329, 17,
     "E:/research-dense/runs/campaign_pilot_v1_r3"),
    ("campaign_pilot_v1_r4", "repo-evidence/runs/campaigns/campaign_pilot_v1_r4.tar.zst",
     99901810, "6d0c02f89f1360d96085cc061b042959", 822120519, 162,
     "E:/research-dense/runs/campaign_pilot_v1_r4"),
    ("campaign_rep_pilot_v1", "repo-evidence/runs/campaigns/campaign_rep_pilot_v1.tar.zst",
     15694368, "c54aa4b6adab5365e7b766afa62630d1", 156909241, 30,
     "E:/research-dense/runs/campaign_rep_pilot_v1"),
    ("campaign_stage1_chunk1", "repo-evidence/runs/campaigns/campaign_stage1_chunk1.tar.zst",
     3248528559, "b606e14d60b533d885d9012602ea7af2", 48765241173, 4149,
     "E:/research-dense/runs/campaign_stage1_chunk1"),
    ("campaign_stage1_chunk1_fix27", "repo-evidence/runs/campaigns/campaign_stage1_chunk1_fix27.tar.zst",
     857247225, "fd44441051071ca81b3db44964ff19f5", 26328460040, 785,
     "E:/research-dense/runs/campaign_stage1_chunk1_fix27"),
    ("campaign_stage1_verify27", "repo-evidence/runs/campaigns/campaign_stage1_verify27.tar.zst",
     203214659, "113b8f4aa04f32148c5658514069daa8", 2661519442, 197,
     "E:/research-dense/runs/campaign_stage1_verify27"),
)
# Per-file hot-data mirrors on Drive (rclone check --checksum passed; listed in
# wtclean/receipts_mirror_keeplocal.json), registered under the configs/data_registry.toml names.
# name, Drive dir below research-archive/, manifest id, original path, {relpath: (md5, size)}
MIRRORS = (
    ("sidecars_packv3_v2", "worktrees/ws-packv3fix/_out/sidecars_packv3_v2", "sidecars_packv3_v2",
     "E:/rd-wt/ws-packv3fix/_out/sidecars_packv3_v2",
     {
         "BUILD_RECEIPT.json": ("848c01f64059757e4fbed495d857beba", 3393),
         "identities/TRAIN_IDENTITY_RECEIPT.json": ("97f992ba7806b3d683cedf8840e67ca0", 2469213),
         "identities/global_positions.npy": ("e40cd9277199525887f6ddce5d9a30b5", 17352),
         "identities/identities.npy": ("f70930f3b94113448fe86033fa34e02e", 3307136),
         "identities/vehicle_map.json": ("62a91282cfc4fa20ec25422d364bbfa4", 1059559),
         "outcomes/SHA256SUMS": ("c84a52e35dedd11f45c6dc5f6ba3572c", 283),
         "outcomes/TRAIN_ISOLATION_RECEIPT.json": ("729bf488f2a8b78117a883341436c8a7", 4781295),
         "outcomes/base_train_outcomes.parquet": ("3d97ea4755c147673077ceae45daf48a", 4913167),
         "outcomes/overlay_outcomes.parquet": ("7cf3feb2028ec766aefb0cf4596548b8", 4570023),
     }),
    ("identities_dspeed", "worktrees/a9/_out/identities_dspeed", "identities_dspeed",
     "E:/rd-wt/a9/_out/identities_dspeed",
     {
         "BUILD_STARTED.json": ("93fdc88da0d380d6caca46a59435f518", 270633),
         "TRAIN_IDENTITY_RECEIPT.json": ("7cf4b0c2b03b56d02dee939105b0e4dd", 712092),
         "global_positions.npy": ("1df7f3bfe71987fb5900ed956195ff9b", 5280),
         "identities.npy": ("24b4c82bc45c6554898abb388c03834f", 989312),
         "vehicle_map.json": ("af0af48c17229a9a17d620815f9a8212", 757706),
     }),
    ("identities_damping", "worktrees/damping-snap/_out/identities_damping", "identities_damping",
     "E:/rd-wt/damping-snap/_out/identities_damping",
     {
         "BUILD_STARTED.json": ("3e8b7ca91eb00139b26d8af7decf377e", 111484),
         "TRAIN_IDENTITY_RECEIPT.json": ("8942a51c0300bb7395c85b6d46b0c927", 293374),
         "global_positions.npy": ("4425bdb214aac51935e934ac9f7340ff", 2232),
         "identities.npy": ("952df831aa90ebc483a899cbea9f2107", 404096),
         "vehicle_map.json": ("68cb43ddcfb7ada2f2cb3afdd2f4ee85", 977970),
     }),
    ("overlay_damping", "worktrees/damping-snap/_out/labels_overlay_damping", "labels_overlay_damping",
     "E:/rd-wt/damping-snap/_out/labels_overlay_damping",
     {
         "SHA256SUMS": ("3307924ea371edd38b346551001ed7cd", 188),
         "TRAIN_ISOLATION_RECEIPT.json": ("1bbfab8e8e0462a4df0aba4f6626c06f", 101762),
         "overlay_outcomes.parquet": ("68fed4eceab99db0b72092e48b53c991", 1166780),
     }),
    ("overlay_dspeed", "worktrees/dspeed-data/_out/labels_overlay_dspeed", "labels_overlay_dspeed",
     "E:/rd-wt/dspeed-data/_out/labels_overlay_dspeed",
     {
         "SHA256SUMS": ("6ba1f37bcbef2942203ab8fd24f07094", 188),
         "TRAIN_ISOLATION_RECEIPT.json": ("3d18c9093ae0f07b7a042994c522ef45", 447551),
         "overlay_outcomes.parquet": ("0646206f70120c5e8748df4bc0a3f2d0", 2138367),
     }),
)


# pack_v3: four 2 GiB-split tar.zst parts plus their checksum/unpack files, uploaded
# 2026-10-01 and moved to this folder on 2026-10-09 (MOVE_WITHIN_DRIVE.json).  MD5/size
# measured with `rclone lsjson --hash` on 2026-10-10.
PACK_V3_DRIVE = "data/packs/training_pack_v3_k192"
PACK_V3_FILES = {
    "PARTS.json": (1417, "4484f944a0cb6d0cf3ab290b2b15596c"),
    "SHA256SUMS": (364, "9bd6fc15ba4b2a35f741cbecae432365"),
    "UNPACK.md": (1578, "4866497f6d4ed8029aaf879331051e66"),
    "pack_v3.tar.zst.part000": (2147483648, "d54c846f62228556a438e10efd22ebc8"),
    "pack_v3.tar.zst.part001": (2147483648, "fd5ba83e0f2bffa41e71b4af9cf5751d"),
    "pack_v3.tar.zst.part002": (2147483648, "0222e1bcdb0ac4fa46c82fea52fcea76"),
    "pack_v3.tar.zst.part003": (1030427407, "2530747a5e44c7d9b86b87b255ec8a13"),
    "upload_manifest_v3.json": (107606, "633d9ae036b429338183099a2e8f8ae0"),
}


_PREFILTER = ("cal", "holdout", "sealed", "chunk", "j4b", "inc1")  # cheap superset of _GUARD


def sensitive_kind(*texts: object) -> str | None:
    """'substring' (coordinator list), 'guard' (repo guard only) or None."""
    blob = " ".join(str(t) for t in texts if t is not None).replace("\\", "/")
    low = blob.lower()
    if any(token in low for token in SENSITIVE_SUBSTRINGS):
        return "substring"
    if any(token in low for token in _PREFILTER) and _GUARD.search(blob):
        return "guard"
    return None


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def norm_path(path: object) -> object:
    if isinstance(path, list):
        return [norm_path(p) for p in path]
    return str(path).replace("\\", "/")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_md5_lines(path: Path) -> dict[str, tuple[str, int]]:
    """``md5<TAB or 2 spaces>size<TAB or 2 spaces>relpath`` lines -> {relpath: (md5, size)}."""
    out: dict[str, tuple[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("	", 2)
        if len(parts) == 3 and len(parts[0]) == 32 and parts[1].isdigit():
            md5, size, rel = parts
        else:
            match = re.match(r"^([0-9a-fA-F]{32})\s+(\d+)\s+(.+)$", line)
            if not match:
                raise ValueError(f"{path}: unparsable line {line!r}")
            md5, size, rel = match.groups()
        out[rel] = (md5.lower(), int(size))
    return out


class Builder:
    def __init__(self, receipts: Path = DEFAULT_RECEIPTS, dataops_root: Path | None = None,
                 previous: dict | None = None) -> None:
        self.receipts = receipts
        self.dataops_root = dataops_root  # holds manifests/members/*.md5 (optional)
        self.previous = previous or {}  # existing registry "datasets" (annotation carry-over)
        self.datasets: dict[str, dict] = {}
        self.skipped: Counter[str] = Counter()

    def add(self, name: str, entry: dict) -> None:
        if not NAME_RE.fullmatch(name):
            raise ValueError(f"invalid dataset name {name!r}")
        if name in self.datasets:
            raise ValueError(f"duplicate dataset name {name!r}")
        head = {"drive_path": entry.pop("drive_path"), "size_bytes": entry.pop("size_bytes"),
                "local_path": f"data_cache/{name}", "md5_manifest": entry.pop("md5_manifest")}
        self.datasets[name] = {**head, **entry}

    def skip(self, why: str, source: str = "") -> None:
        self.skipped[f"{source}:{why}" if source else why] += 1

    # -- dataops: one tar.zst per unit ---------------------------------------------------
    def dataops(self) -> None:
        folder = self.receipts / "dataops" / "receipts"
        for path in sorted(folder.glob("*.json")):
            rec = read_json(path)
            if rec.get("schema") != "dataops_receipt/1":
                continue
            if rec.get("check") != "passed":
                self.skip("check_not_passed", "dataops")
                continue
            reason = sensitive_kind(rec["id"], rec["name"], rec["local"], rec["drive"], path.name)
            if reason:
                self.skip(f"sensitive_{reason}", "dataops")
                continue
            base = path.name[: -len(".json")]
            if base in EXCLUDE_PERSONAL:
                self.skip("personal_document", "dataops")
                continue
            name = ALIASES.get(base, base)
            drive = rec["drive"]
            if not drive.startswith(DRIVE_ROOT):
                raise ValueError(f"{path.name}: drive outside research-archive: {drive}")
            self.add(name, {
                "drive_path": drive,
                "size_bytes": rec["archive_bytes"],
                "md5_manifest": {PurePosixPath(drive).name: rec["archive_md5"]},
                "format": "tar.zst",
                "unpacked_bytes": rec["bytes"],
                "unpacked_files": rec["files"],
                "original_path": norm_path(rec["local"]),
                "class": rec["class"],
                "kind": rec["kind"],
                "receipt": {"source": "dataops", "id": rec["id"], "file": path.name,
                            "check": rec["check"], "timestamp": rec["timestamp"],
                            "member_manifest": rec["member_manifest"]},
            })

    # -- driveops: per-file Drive directories ----------------------------------------------
    def driveops(self) -> None:
        folder = self.receipts / "driveops" / "receipts"
        for path in sorted(folder.glob("*.json")):
            rec = read_json(path)
            if rec.get("schema") != "driveops_receipt/1":
                continue
            if rec.get("check") != "passed":
                self.skip("check_not_passed", "driveops")
                continue
            members = read_md5_lines(folder / "md5" / f"{rec['name']}.md5")
            reason = sensitive_kind(rec["name"], rec["local"], rec["drive"], path.name,
                                    *members)
            if reason:
                self.skip(f"sensitive_{reason}", "driveops")
                continue
            total = sum(size for _, size in members.values())
            if total != rec["bytes"] or len(members) != rec["files"]:
                raise ValueError(f"{path.name}: md5 list ({len(members)} files, {total} B) "
                                 f"!= receipt ({rec['files']} files, {rec['bytes']} B)")
            self.add(rec["name"], {
                "drive_path": rec["drive"],
                "size_bytes": total,
                "md5_manifest": {rel: md5 for rel, (md5, _) in sorted(members.items())},
                "format": "files",
                "unpacked_bytes": total,
                "unpacked_files": len(members),
                "original_path": norm_path(rec["local"]),
                "class": rec["class"],
                "kind": rec["kind"],
                "receipt": {"source": "driveops", "id": rec["name"], "file": path.name,
                            "check": rec["check"], "timestamp": rec["timestamp"]},
            })

    # -- wtclean: worktree backups (tar.zst method only) -----------------------------------
    def wtclean(self) -> None:
        folder = self.receipts / "wtclean" / "receipts"
        for path in sorted(folder.glob("*.json")):
            rec = read_json(path)
            if rec.get("method") != "tar.zst":
                self.skip("perfile_mirror_not_registered", "wtclean")
                continue
            if rec.get("status") != "ok":
                self.skip("status_not_ok", "wtclean")
                continue
            arch = rec["archive"]
            if not all(arch["verified"].get(k) for k in ("remote_md5", "remote_size")):
                self.skip("archive_not_verified", "wtclean")
                continue
            reason = sensitive_kind(rec["name"], rec["path"], arch["remote"], path.name,
                                    rec.get("branch"))
            if reason:
                self.skip(f"sensitive_{reason}", "wtclean")
                continue
            self.add("wt_" + sanitize(rec["name"]), {
                "drive_path": arch["remote"],
                "size_bytes": arch["size"],
                "md5_manifest": {PurePosixPath(arch["remote"]).name: arch["md5"]},
                "format": "tar.zst",
                "unpacked_bytes": arch["member_bytes"],
                "unpacked_files": arch["member_count"],
                "original_path": norm_path(rec["path"]),
                "class": rec["class"],
                "kind": "worktree",
                "branch": rec.get("branch"),
                "receipt": {"source": "wtclean", "id": rec["name"], "file": path.name,
                            "check": "passed", "timestamp": rec["timestamp"],
                            "manifest_remote": arch["manifest_remote"],
                            "manifest_md5": arch["manifest_md5"]},
            })

    # -- legacy archives (receipts without a Drive path) -----------------------------------
    def legacy(self) -> None:
        for name, rel, size, md5, unpacked, files, source in LEGACY:
            reason = sensitive_kind(name, source, rel)
            if reason:
                self.skip(f"sensitive_{reason}", "legacy")
                continue
            self.add(name, {
                "drive_path": DRIVE_ROOT + rel,
                "size_bytes": size,
                "md5_manifest": {PurePosixPath(rel).name: md5},
                "format": "tar.zst",
                "unpacked_bytes": unpacked,
                "unpacked_files": files,
                "original_path": source,
                "class": "ARCHIVE",
                "kind": "dir",
                "receipt": {"source": "dataops/_out/drive_receipts", "id": name,
                            "file": f"{name}.tar.zst.upload.json",
                            "check": "passed (2026-10-06 upload; md5 and size re-measured on "
                                     "Drive 2026-10-10)"},
            })

    # -- per-file hot-data mirrors -----------------------------------------------------------
    def mirrors(self) -> None:
        proofs = {p["dest"]: p for p in read_json(
            self.receipts / "wtclean" / "receipts_mirror_keeplocal.json")}
        for name, rel, manifest_id, original, members in MIRRORS:
            proof = proofs.get(DRIVE_ROOT + rel)
            if not proof or proof["copy_rc"] != 0 or proof["check_rc"] != 0:
                raise ValueError(f"{name}: no passing rclone check proof for {rel}")
            reason = sensitive_kind(name, rel, original, *members)
            if reason:
                self.skip(f"sensitive_{reason}", "mirrors")
                continue
            total = sum(size for _, size in members.values())
            self.add(name, {
                "drive_path": DRIVE_ROOT + rel,
                "size_bytes": total,
                "md5_manifest": {r: m for r, (m, _) in sorted(members.items())},
                "format": "files",
                "unpacked_bytes": total,
                "unpacked_files": len(members),
                "original_path": original,
                "class": "KEEP-ACTIVE",
                "kind": "dir",
                "receipt": {"source": "wtclean/receipts_mirror_keeplocal.json",
                            "id": manifest_id, "file": f"dataops/manifests/{manifest_id}.md5",
                            "check": "passed (rclone check --checksum; listing re-measured "
                                     "on Drive 2026-10-10)"},
            })

    # -- pack_v3 (multi-part) ------------------------------------------------------------------
    def pack_v3(self) -> None:
        total = sum(size for size, _ in PACK_V3_FILES.values())
        self.add("pack_v3", {
            "drive_path": DRIVE_ROOT + PACK_V3_DRIVE,
            "size_bytes": total,
            "md5_manifest": {rel: md5 for rel, (_, md5) in sorted(PACK_V3_FILES.items())},
            "format": "tar.zst.parts",
            "parts": sorted(r for r in PACK_V3_FILES if ".tar.zst.part" in r),
            "unpack": "cat the .partNNN files in order | zstd -dc | tar -x; see UNPACK.md "
                      "and SHA256SUMS in the fetched directory",
            "original_path": "D:/rd-staging/wmn1_feat/pack_v3",
            "note": "uploaded 2026-10-01 (upload_manifest_v3.json on Drive); that this equals the "
                    "current local pack_v3 is not verified here - compare SHA256SUMS after unpacking",
            "class": "KEEP-ACTIVE",
            "kind": "dir",
            "receipt": {"source": "driveops/MOVE_WITHIN_DRIVE.json", "id": "training_pack_v3_k192",
                        "file": "(no receipt; listing measured with rclone lsjson --hash 2026-10-10)",
                        "check": "md5/size measured on Drive"},
        })

    # -- member-manifest annotations: overlap between units and restricted member names --------
    def annotate_members(self) -> None:
        """Add ``covered_by`` (registered parent archives that contain every member of a unit,
        same relpath and MD5) and ``restricted_member_names`` (how many archive members have a
        restricted token in their path; names only, nothing is opened).  Without
        ``dataops_root`` the two fields are carried over from the previous registry file."""
        if self.dataops_root is None:
            for name, entry in self.datasets.items():
                old = self.previous.get(name, {})
                if old.get("covered_by"):
                    entry["covered_by"] = [p for p in old["covered_by"] if p in self.datasets]
                    if not entry["covered_by"]:
                        del entry["covered_by"]
                if old.get("restricted_member_names"):
                    entry["restricted_member_names"] = old["restricted_member_names"]
            return
        cache: dict[str, dict[str, str]] = {}

        def members(entry: dict) -> dict[str, str] | None:
            rel = entry["receipt"].get("member_manifest")
            if entry["receipt"]["source"] != "dataops" or not rel:
                return None
            if rel not in cache:
                path = self.dataops_root / rel
                cache[rel] = ({r: m for r, (m, _) in read_md5_lines(path).items()}
                              if path.is_file() else {})
            return cache[rel] or None

        sources = [(n, e) for n, e in self.datasets.items() if e["receipt"]["source"] == "dataops"]
        for name, entry in sources:
            own = entry["original_path"].lower().rstrip("/")
            child = members(entry)
            if not child:
                continue
            # one scan of the joined names first: almost no archive has a flagged member
            flagged = (sum(1 for rel in child if sensitive_kind(rel))
                       if sensitive_kind(*child) else 0)
            if flagged:
                entry["restricted_member_names"] = flagged
            found = []
            for pname, parent in sources:
                if pname == name or parent["kind"] != "dir":
                    continue
                root = parent["original_path"].lower().rstrip("/")
                if not own.startswith(root + "/"):
                    continue
                pm = members(parent)
                if not pm:
                    continue
                prefix = entry["original_path"][len(root) + 1:].strip("/")
                if entry["kind"] == "file":
                    ok = pm.get(prefix) == next(iter(child.values()))
                else:
                    ok = all(pm.get(f"{prefix}/{rel}") == md5 for rel, md5 in child.items())
                if ok:
                    found.append(pname)
            if found:
                entry["covered_by"] = sorted(found)

    def build(self) -> dict:
        self.dataops()
        self.driveops()
        self.wtclean()
        self.legacy()
        self.mirrors()
        self.pack_v3()
        self.annotate_members()
        datasets = {name: self.datasets[name] for name in sorted(self.datasets)}
        skipped = ", ".join(f"{key}={n}" for key, n in sorted(self.skipped.items()))
        return {
            "schema_version": 1,
            "description": (
                "Verified, fetchable data objects on gdrive:research-archive/. Generated by "
                "tools/build_drive_registry.py from the Drive upload receipts "
                f"({len(datasets)} entries; counted but not registered among the receipts in data_ops/: "
                f"{skipped}). Most entries are tar.zst "
                "archives: the object fetch_data.py downloads and verifies is the archive "
                "itself (md5_manifest has one member), not its unpacked tree; "
                "unpacked_bytes/unpacked_files describe the content. Others are per-file "
                "directories or multi-part packs. WMS_DATA_<NAME> and "
                "configs/data_registry.toml take precedence over this file."),
            "datasets": datasets,
        }


def render(registry: dict) -> str:
    return json.dumps(registry, indent=1, ensure_ascii=True) + "\n"


# -- optional live check against Google Drive (read-only: rclone lsjson only) -----------------
RECURSIVE_ROOTS = ("repo-evidence", "data/staging", "code", "datasets",
                   "data/packs/training_pack_v3_k192", "data/packs/archived_training_packs_v2",
                   "data/packs/derived_shards_v2")


def _lsjson(rclone: str, rel: str, recursive: bool) -> dict[str, tuple[int, str]]:
    import subprocess

    cmd = [rclone, "lsjson", "--files-only", "--hash", "--fast-list", DRIVE_ROOT + rel]
    if recursive:
        cmd.insert(2, "-R")
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {done.stderr.strip()[-300:]}")
    return {f"{rel}/{e['Path']}": (e["Size"], (e.get("Hashes") or {}).get("md5", ""))
            for e in json.loads(done.stdout)}


def verify_drive(registry: dict, rclone: str, sample: int | None = None) -> list[str]:
    """Compare every entry (or an evenly spaced sample) with Drive; return problem strings."""
    datasets = registry["datasets"]
    names = sorted(datasets)
    if sample:
        step = max(1, len(names) // sample)
        names = names[::step][:sample]
    files: dict[str, tuple[int, str]] = {}
    needed: dict[str, bool] = {}  # rel dir -> recursive?
    for name in names:
        rel = datasets[name]["drive_path"][len(DRIVE_ROOT):]
        if any(rel == r or rel.startswith(r + "/") for r in RECURSIVE_ROOTS):
            continue
        multi = len(datasets[name]["md5_manifest"]) > 1 or datasets[name]["format"] == "files"
        needed[rel if multi else str(PurePosixPath(rel).parent)] = multi
    roots = RECURSIVE_ROOTS
    if sample:  # keep a sampled run cheap: scan only the roots the sample touches
        roots = tuple(r for r in RECURSIVE_ROOTS if any(
            datasets[n]["drive_path"][len(DRIVE_ROOT):].startswith(r + "/") for n in names))
    for root in roots:
        files.update(_lsjson(rclone, root, True))
    for rel, recursive in sorted(needed.items()):
        files.update(_lsjson(rclone, rel, recursive))
    problems: list[str] = []
    for name in names:
        entry = datasets[name]
        rel = entry["drive_path"][len(DRIVE_ROOT):]
        manifest = entry["md5_manifest"]
        if rel in files:  # a single fetched file
            size, md5 = files[rel]
            found = {PurePosixPath(rel).name: md5}
            total = size
        else:
            prefix = rel + "/"
            under = {k[len(prefix):]: v for k, v in files.items() if k.startswith(prefix)}
            found = {k: v[1] for k, v in under.items()}
            total = sum(v[0] for v in under.values())
        if found != manifest:
            missing = sorted(set(manifest) - set(found))
            extra = sorted(set(found) - set(manifest))
            wrong = sorted(k for k in set(found) & set(manifest) if found[k] != manifest[k])
            problems.append(f"{name}: manifest differs from Drive (missing={len(missing)}, "
                            f"extra={len(extra)}, md5={len(wrong)})")
        elif total != entry["size_bytes"]:
            problems.append(f"{name}: size {total} on Drive != registry {entry['size_bytes']}")
    print(f"verified {len(names)} entries against Drive: {len(problems)} problem(s)")
    return problems


def finish(args: argparse.Namespace, registry: dict) -> int:
    if not args.verify_drive:
        return 0
    problems = verify_drive(registry, args.rclone, args.sample)
    for line in problems:
        print("PROBLEM", line)
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS,
                        help="folder holding dataops/ driveops/ wtclean/ receipts (default: data_ops/receipts)")
    parser.add_argument("--dataops-root", type=Path, default=None,
                        help="folder holding manifests/members/*.md5; recomputes covered_by and "
                             "restricted_member_names (default: carry them over from --out)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 on a diff")
    parser.add_argument("--verify-drive", action="store_true",
                        help="after writing/checking, compare entries with Drive (rclone lsjson --hash)")
    parser.add_argument("--sample", type=int, default=None, help="with --verify-drive: only N entries")
    parser.add_argument("--rclone", default="rclone")
    args = parser.parse_args(argv)
    previous = {}
    if args.out.is_file():
        previous = json.loads(args.out.read_text(encoding="utf-8")).get("datasets", {})
    builder = Builder(args.receipts, args.dataops_root, previous)
    text = render(builder.build())
    spec = importlib.util.spec_from_file_location("fetch_data_tool", Path(__file__).with_name("fetch_data.py"))
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    tmp = args.out.with_name(args.out.name + ".validate.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        tool._load_registry(tmp)  # the exact validation the fetch tool applies
    finally:
        tmp.unlink()
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        if current != text:
            print("data_registry.json is out of date", file=sys.stderr)
            return 1
        print("data_registry.json is up to date")
        return finish(args, json.loads(text))
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(builder.datasets)} entries, skipped {dict(builder.skipped)}")
    return finish(args, json.loads(text))


if __name__ == "__main__":
    raise SystemExit(main())
