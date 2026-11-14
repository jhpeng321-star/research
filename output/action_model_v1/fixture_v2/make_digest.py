"""Write REGISTRATION_DIGEST.json for the WM-43B package (stdlib only).

Digest binds the registration text, the seeds diff and the exposure audit.
Write-once: refuses to overwrite unless --force.  Zero simulation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from wm43_plan import DIGEST_PATH, REGISTRATION_PATH, SEEDS_DIFF_PATH, EXPOSURE_AUDIT_PATH  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    for path in (REGISTRATION_PATH, SEEDS_DIFF_PATH, EXPOSURE_AUDIT_PATH):
        if not path.is_file():
            print(f"missing {path}")
            return 2
    if DIGEST_PATH.exists() and "--force" not in sys.argv:
        print(f"refusing to overwrite {DIGEST_PATH}")
        return 2
    DIGEST_PATH.write_text(json.dumps({
        "schema": "wm43b_registration_digest/1",
        "registration_sha256": sha(REGISTRATION_PATH),
        "seeds_diff_sha256": sha(SEEDS_DIFF_PATH),
        "exposure_audit_sha256": sha(EXPOSURE_AUDIT_PATH),
        "status": "DRAFT digest - recompute after any registration edit",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DIGEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
