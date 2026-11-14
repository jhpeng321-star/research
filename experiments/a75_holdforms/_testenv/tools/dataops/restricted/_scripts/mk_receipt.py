"""Write one receipt JSON for a sealed-archive backup. Counts and hashes only; never names."""
import json
import sys

(out, label, target, drive_path, arch_bytes, md5, sha256, in_bytes, n_files, n_entries,
 members, tar_rc, drive_md5, drive_bytes, started, finished, zstd_level, split_note) = sys.argv[1:19]

rec = {
    "label": label,
    "target_path": target,
    "drive_path": drive_path,
    "archive_bytes": int(arch_bytes),
    "md5": md5,
    "sha256": sha256,
    "input_bytes": int(in_bytes),
    "file_count": int(n_files),
    "entry_count_source": int(n_entries),
    "member_count_in_archive": int(members),
    "tar_exit_code": int(tar_rc),
    "drive_md5": drive_md5,
    "drive_bytes": int(drive_bytes),
    "verified_md5_equal": md5 == drive_md5,
    "verified_size_equal": int(arch_bytes) == int(drive_bytes),
    "member_count_equals_source_entries": int(members) == int(n_entries),
    "compression": "zstd level " + zstd_level + " (-T0)",
    "format": "tar (GNU, no -v) piped to zstd; restore: zstd -dc <file> | tar -xf - -C <dest_dir>",
    "note": split_note,
    "started_utc": started,
    "finished_utc": finished,
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=2)
    fh.write("\n")
