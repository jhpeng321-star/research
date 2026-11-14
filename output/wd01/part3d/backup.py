"""Part 3d: backup trained weights to BACKUPS_ROOT/wd01/part3d/ (default
E:/research-dense-backups/wd01/part3d/) with SHA256SUMS verified."""
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

HERE = Path(__file__).resolve().parent
BACKUP = Path(wdpaths.BACKUPS_ROOT) / "wd01" / "part3d"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in sorted((HERE / "weights" / "run3d").glob("*.pt")):
        dst = BACKUP / f"run3d_{f.name}"
        shutil.copy2(f, dst)
        h = sha(dst)
        assert h == sha(f)
        lines.append(f"{h}  {dst.name}")
        print(f"backup {dst.name}: {h[:16]}... verified")
    (BACKUP / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    print("SHA256SUMS written")


if __name__ == "__main__":
    main()
