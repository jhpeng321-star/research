import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_MAP = ROOT / "archive" / "circles_era" / "PATH_MAP.json"


def _archived_modules():
    rows = json.loads(PATH_MAP.read_text(encoding="utf-8"))["files"]
    return {Path(r["old"]).stem for r in rows if r["old"].startswith("traffic_shepherd/")}


def test_active_modules_do_not_import_archived_code():
    archived = _archived_modules()
    offenders = []
    for path in sorted((ROOT / "traffic_shepherd").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                if name.split(".")[-1] in archived:
                    offenders.append(f"{path.name}:{node.lineno}: {name}")
    assert not offenders, offenders


def test_path_map_entries_exist_at_new_location():
    rows = json.loads(PATH_MAP.read_text(encoding="utf-8"))["files"]
    assert rows
    for row in rows:
        assert not (ROOT / row["old"]).exists(), row["old"]
        moved = ROOT / row["new"]
        size = row.get("current_bytes", row["bytes"])
        assert moved.is_file() and moved.stat().st_size == size, row["new"]
        digest = row.get("current_sha256", row["sha256"])
        assert hashlib.sha256(moved.read_bytes()).hexdigest() == digest, row["new"]
