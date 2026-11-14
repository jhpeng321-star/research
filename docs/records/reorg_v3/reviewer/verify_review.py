"""Static-only verification of review artifacts and supplied metadata.

Does not import or execute any reorganisation script, contact Drive, inspect
external sources, or open datasets. Each JSON is processed separately.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GIB = 1024**3


def load(name):
    with (ROOT / "_in" / name).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def main():
    move_map = load("MOVE_MAP.json")
    moves = move_map["moves"]
    assert len(moves) == move_map["count"] == 83
    destinations = [m["new"].replace("\\", "/").lower() for m in moves]
    assert len(set(destinations)) == 83
    worktrees = [m for m in moves if m["method"].startswith("git worktree")]
    assert len(worktrees) == 60
    wt_bytes = sum(m["size_bytes"] for m in worktrees)
    imports = sum(m["size_bytes"] or 0 for m in moves if m["phase"] == "5")
    inventory = load("inventory_E.json")
    tmp = inventory["research_dense_tmp_subfolders"]
    tmp_bytes = sum(item["bytes"] for item in tmp if item["proposed_class"] == "DELETE-CANDIDATE")
    scouts = [item for item in inventory["worktrees"] if item["proposed_class"] == "DELETE-CANDIDATE"]
    scout_bytes = sum(item["bytes"] for item in scouts)
    reorg = next(item for item in scouts if item["name"] == "reorg3")
    assert reorg["processes"] and not reorg["live"]
    verify_bytes = next(item["bytes"] for item in tmp if item["name"] == "drive_verify_20261006")
    assert (wt_bytes, tmp_bytes, scout_bytes, verify_bytes) == (
        39288490760, 404401113, 96560755, 2178100
    )
    assert imports == 108794445033
    optimistic = 6.30 + 0.916 + (tmp_bytes - verify_bytes + scout_bytes - reorg["bytes"]) / GIB
    print(f"metadata: 83 moves, 60 worktrees ({wt_bytes/GIB:.9f} GiB)")
    print(f"local candidates: tmp={tmp_bytes}, scouts={scout_bytes} bytes")
    print(f"imports={imports/GIB:.9f} GiB; baseline shortfall={imports/GIB-6.30:.9f} GiB")
    print(f"conditional local-cleanup ceiling={optimistic:.9f} GiB (not deletion authorization)")
    conditional_archive = optimistic + (wt_bytes + 563055757) / GIB
    print(f"counterfactual archive ceiling={conditional_archive:.9f} GiB")
    del inventory
    for name in ("inventory_D.json", "inventory_CG.json", "path_refs.json"):
        parsed = load(name)
        assert isinstance(parsed, dict)
        del parsed
    for name in ("verify_review.py", "fixed/backup_receipt.py"):
        ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
    shell = (ROOT / "fixed/drive_offload.sh").read_bytes()
    assert b"\r" not in shell, "Bash copy must have LF line endings"
    original_shell = (ROOT / "_in/scripts/drive_offload.sh").read_bytes()
    assert b"\r" not in original_shell, "original Bash line-ending finding changed"
    assert b"backup_verified" in shell and b"--immutable" not in shell.split(b"rclone copy", 1)[1].split(b"; then", 1)[0]
    delete = (ROOT / "fixed/DELETE_AFTER_VERIFY.ps1").read_text(encoding="utf-8")
    assert "Remove-Item" not in delete and "if ($Execute)" in delete
    phase = (ROOT / "fixed/phase2_moves.ps1").read_text(encoding="utf-8")
    assert "ValidateSet('2a')" in phase and "$fs.Flush($true)" in phase
    assert "File]::Open(" not in phase and "robocopy" not in phase
    report = (ROOT / "REVIEW_V3.md").read_text(encoding="utf-8")
    assert "VERDICT: FAIL" in report
    assert f"{wt_bytes/GIB:.6f}" in report
    assert f"{optimistic:.6f}" in report
    assert f"{conditional_archive:.6f}" in report
    for path in sorted((ROOT / "_in").iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            print(f"input-sha256 {path.name} {digest}")
    for path in sorted((ROOT / "_in" / "scripts").iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            print(f"input-sha256 scripts/{path.name} {digest}")
    print("PASS: JSON, evidence arithmetic, Python AST, original/fixed Bash LF, conservative copy contracts")


if __name__ == "__main__":
    main()
