"""WD-01 part 3: backups + WEIGHTS_MANIFEST.json.

Copies weights/*.pt to BACKUPS_ROOT/wd01/part3/ (default
E:/research-dense-backups/wd01/part3/), writes
SHA256SUMS.txt there, verifies the copies, and writes
results/WEIGHTS_MANIFEST.json (sha256 of every trained weight file and
its backup, weights_only loadability check, model metadata)."""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

HERE = Path(__file__).resolve().parent
BACKUP = Path(wdpaths.BACKUPS_ROOT) / "wd01" / "part3" / "run3"
CKPT = Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" / \
    "worldmodel_v3_vehicle" / "vehicle_dynamics.pt"
CKPT_SHA256 = ("ded83907b8d218574d3f5359c332408f17167da055574afbe"
               "29eaecb59d4a8f5")
P2C_MANIFEST_SHA256 = "2439004d528a860c701bfb147afcda99ab3f864a29e6187be721db81cc82dd27"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    entries = {}
    for f in sorted((HERE / "weights" / "run3").glob("member*.pt")):
        sd = torch.load(f, map_location="cpu", weights_only=True)
        meta = sd.get("meta", {})
        ok = "model_state" in sd and len(sd["model_state"]) > 0
        entries[f.name] = dict(
            sha256=sha256_file(f),
            weights_only_loadable=bool(ok),
            meta=json.loads(json.dumps(meta, default=str)),
            bytes=f.stat().st_size)
    # pinned inputs
    entries["#pinned"] = dict(
        vehicle_dynamics_pt=str(CKPT), vehicle_dynamics_pt_sha256=sha256_file(
            CKPT),
        p2c_manifest_sha256=sha256_file(HERE.parent / "partA_data" /
                                        "MANIFEST.json"))
    assert (entries["#pinned"]["vehicle_dynamics_pt_sha256"]
            == CKPT_SHA256)
    assert (entries["#pinned"]["p2c_manifest_sha256"]
            == P2C_MANIFEST_SHA256)
    (HERE / "results" / "WEIGHTS_MANIFEST.json").write_text(
        json.dumps(dict(generated_utc=datetime.now(timezone.utc)
                        .isoformat(), weights=entries), indent=1),
        encoding="utf-8")
    print("WEIGHTS_MANIFEST.json written;",
          sum(1 for k in entries if not k.startswith("#")), "weights")

    # backups
    lines = []
    for name, e in entries.items():
        if name.startswith("#"):
            continue
        bname = "run3_" + name
        dst = BACKUP / bname
        shutil.copy2(HERE / "weights" / "run3" / name, dst)
        h = sha256_file(dst)
        assert h == e["sha256"], f"backup copy mismatch for {name}"
        lines.append(f"{h}  {bname}")
        print(f"backup {bname}: {h[:16]}... verified")
    (BACKUP / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    print("SHA256SUMS.txt written to", BACKUP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
