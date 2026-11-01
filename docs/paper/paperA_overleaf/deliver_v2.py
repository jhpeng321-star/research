"""Preserve the existing v1 and deliver the QA-checked v2 document only."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESKTOP = Path("C:/Users/jpeng/Desktop")
V1_SHA = "37e10a533eb788a1c9696bb90a62aafd53968f80fe48819ce9f27a5f8accb7c3"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preserve-v1", action="store_true")
    mode.add_argument("--deliver", action="store_true")
    args = parser.parse_args()
    assert DESKTOP.is_dir(), "Expected existing Desktop directory"
    v1 = DESKTOP / "PaperA_outline_v1.pdf"
    if args.preserve_v1:
        if not v1.exists():
            assert digest(ROOT / "main.pdf") == V1_SHA, "Original v1 PDF required"
            shutil.copyfile(ROOT / "main.pdf", v1)
        assert digest(v1) == V1_SHA, "Existing v1 differs; preserve and investigate"
        print(json.dumps({"v1": str(v1), "sha256": V1_SHA, "preserved": True}))
        return

    assert v1.is_file() and digest(v1) == V1_SHA, "v1 must remain byte-identical"
    qa = json.loads((ROOT / "BUILD_QA.json").read_text(encoding="utf-8"))
    assert qa["status"] == "PASS" and qa["version"].startswith("v2:")
    assert qa["pdf_pages_including_references"] <= 6
    for name, artifact in qa["artifacts"].items():
        assert digest(ROOT / name) == artifact["sha256"], f"Changed since QA: {name}"
    v2 = DESKTOP / "PaperA_outline_v2.pdf"
    shutil.copyfile(ROOT / "main.pdf", v2)
    assert digest(v2) == digest(ROOT / "main.pdf")
    assert digest(v1) == V1_SHA
    receipt = {
        "delivered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "version": "v2: WM-guided exploration, SUMO-graded",
        "pdf_pages_including_references": qa["pdf_pages_including_references"],
        "v1": {"path": str(v1), "sha256": digest(v1), "preserved": True},
        "v2": {"path": str(v2), "sha256": digest(v2), "matches_main_pdf": True},
        "build_qa_sha256": digest(ROOT / "BUILD_QA.json"),
    }
    (ROOT / "V2_DELIVERY.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
