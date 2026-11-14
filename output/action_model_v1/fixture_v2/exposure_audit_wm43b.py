"""WM-43B seed exposure audit (read-only; zero simulation).

Purpose: prove the proposed FRESH block (default 920249-920273, the block
right after WM43_WALL_MECHANICS_SEEDS 920244-920248) has zero prior
registration, execution or exposure BEFORE the seeds diff is applied and
BEFORE any launch.  Registry absence alone is not proof a seed is fresh
(AGENTS.md), so the audit scans file CONTENTS of every tracked, untracked
and git-ignored file of each given repository root with `git grep`.

A hit counts as an IDENTITY-LIKE occurrence when the digits are not embedded
in a longer number: not preceded/followed by a digit, and not preceded by a
'.' (decimal fraction) or followed by '.<digit>' (decimal value).  Numeric
literal look-alikes (e.g. 920252.54 in a CSV) are excluded by the pattern.
Verdict FRESH requires zero identity-like hits outside this fixture dir.

Usage (run by the INTEGRATOR on the main tree; also on any live worktree):
  python output/action_model_v1/fixture_v2/exposure_audit_wm43b.py \
      --repo E:/research-dense --repo E:/rd-wt/wd01wallB [--write]
--write stores EXPOSURE_AUDIT_WM43B.json next to this script (write-once:
refuses to overwrite).  Then rebuild REGISTRATION_DIGEST.json with
make_digest.py.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from wm43_plan import SEEDS, SEEDS_BY_ARM  # noqa: E402

AUDIT_PATH = HERE / "EXPOSURE_AUDIT_WM43B.json"
EXCLUDED_PREFIXES = ("output/action_model_v1/fixture_v2/",)
#: tool caches that merely echo test ids (not seed exposure)
EXCLUDED_PARTS = (".pytest_cache/", "__pycache__/")


def _identity_like(line: str, token: str) -> bool:
    return re.search(rf"(?<![\d.]){token}(?![\d]|\.\d)", line) is not None


def scan_repo(repo: Path, seeds: tuple[int, ...]) -> dict:
    """Streaming scan; only IDENTITY-LIKE lines are requested from git
    (PCRE: digits not embedded in a longer number / decimal), so huge
    numeric CSVs cannot flood memory.  Long lines are truncated."""
    tokens = [str(s) for s in seeds]
    pattern = r"(?<![\d.])(" + "|".join(tokens) + r")(?!\d|\.\d)"
    cmd = ["git", "-C", str(repo), "grep", "-I", "-n", "--untracked",
           "--no-exclude-standard", "-P", pattern]
    identity, non_use = [], []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace") as proc:
        for raw in proc.stdout:
            raw = raw.rstrip("\n")
            path = raw.split(":", 1)[0].replace("\\", "/")
            if path.startswith(EXCLUDED_PREFIXES) or any(
                    part in path for part in EXCLUDED_PARTS):
                continue
            for token in tokens:
                if _identity_like(raw, token):
                    # the first identity of the block appears as the EXCLUSIVE
                    # end bound of the previous block's range(): declared
                    # non-use (same classification as the WM-43 audit)
                    bucket = (non_use if re.search(rf"range\(\d+,\s*{token}\)", raw)
                              else identity)
                    bucket.append({"seed": int(token), "hit": raw[:240]})
                    break
        err = proc.stderr.read()
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed in {repo}: {err[:300]}")
    return {"repo": str(repo), "identity_like_hits": identity,
            "declared_non_use_exclusive_end_bounds": non_use}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", action="append", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    reports = [scan_repo(Path(r), SEEDS) for r in args.repo]
    fresh = all(not r["identity_like_hits"] for r in reports)
    audit = {
        "schema": "wm43b_exposure_audit/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "chosen_block": {arm: list(seeds) for arm, seeds in SEEDS_BY_ARM.items()},
        "method": ("git grep -I -n --untracked --no-exclude-standard -P over each "
                   "repo root (tracked + untracked + git-ignored file contents); "
                   "identity-like = digits not embedded in a longer number/decimal; "
                   "this fixture directory excluded"),
        "excluded_by_registration_rule": [
            "WM43_WALL_MECHANICS_SEEDS 920244-920248 (consumed by WM-43 r7)",
            "every SEALED_SEEDS / FINAL_TEST_SEEDS / PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS member",
        ],
        "repos": reports,
        "verdict": ("FRESH: zero identity-like occurrences in any audited repo"
                    if fresh else "NOT FRESH: identity-like hits present - choose another block"),
        "fresh": fresh,
    }
    text = json.dumps(audit, indent=2) + "\n"
    if args.write:
        if AUDIT_PATH.exists():
            print(f"refusing to overwrite {AUDIT_PATH}")
            return 2
        AUDIT_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {AUDIT_PATH}")
    print(json.dumps({"fresh": fresh,
                      "identity_like_hits": [r["identity_like_hits"] for r in reports]},
                     indent=1))
    return 0 if fresh else 1


if __name__ == "__main__":
    sys.exit(main())
