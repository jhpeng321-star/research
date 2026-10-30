"""Allowlisted LF source bundle; never include research data or assignment files."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path("E:/Temp/opencode/mappospeed-code.tgz")
PREFIXES = ("src/", "traffic_shepherd/", "configs/", "output/")
EXTRA = {"pyproject.toml", "tests/__init__.py", "tests/unit/__init__.py",
         "tests/unit/test_c4_mappo.py", "tests/unit/test_c4_mappospeed.py",
         "tests/unit/test_c4_srcwait_mappo.py", "tests/unit/test_c4_accelcap.py",
         "tests/unit/test_c4_policy2.py", "tools/mappospeed/package.py"}
BANNED = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
          "hetdata_truth", "vtype", "assignment")


def main():
    assert TARGET.parent.is_dir()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names = set(names) | EXTRA
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "tools/mappospeed").iterdir()
                 if p.is_file())
    manifest = {}
    with tarfile.open(TARGET, "w:gz") as archive:
        for name in sorted(names):
            if not name or any(x in name.lower() for x in BANNED):
                continue
            if not (name in EXTRA or name.startswith(PREFIXES + ("tools/mappospeed/",))):
                continue
            if Path(name).suffix not in (".py", ".json", ".toml", ".sh", ".md"):
                continue
            path = ROOT / name
            if not path.is_file():
                continue
            data = path.read_bytes().replace(b"\r\n", b"\n")
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(manifest, indent=1).encode()
        info = tarfile.TarInfo("SOURCE_MANIFEST.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(json.dumps({"archive": str(TARGET), "files": len(manifest),
                      "sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
