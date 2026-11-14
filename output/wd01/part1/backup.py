"""Copy part1 deliverables to BACKUPS_ROOT/wd01/part1/ (default
E:/research-dense-backups/wd01/part1/) and write + verify SHA256SUMS.txt.
Also re-verifies the two pinned checkpoints."""
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402
import wd1lib as L  # noqa: E402

DEST = Path(wdpaths.BACKUPS_ROOT) / "wd01" / "part1"


def main():
    t0 = datetime.now(timezone.utc).isoformat()
    picks = []
    picks += sorted(L.HERE.glob("*.py"))
    picks += sorted(L.HERE.glob("*.md"))
    picks += sorted(L.HERE.glob("*.json"))
    picks += sorted((L.HERE / "tests").glob("*.py"))
    picks += sorted((L.HERE / "results").glob("*.json"))
    picks += sorted((L.HERE / "results").glob("*.csv"))
    picks += sorted((L.HERE / "identity" / "retrained").glob("*.pt"))
    picks = [p for p in picks if p.is_file()]
    assert picks, "nothing to back up"

    entries = []
    for p in picks:
        rel = p.relative_to(L.HERE).as_posix()
        dst = DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        s, d = L.sha256_file(p), L.sha256_file(dst)
        assert s == d, f"backup hash mismatch: {rel}"
        entries.append((rel, s, p.stat().st_size))

    lines = [f"{sha}  {rel}" for rel, sha, _ in entries]
    (DEST / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    # verify the receipt itself by re-reading and re-hashing all copies
    ok = 0
    for line in (DEST / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        assert L.sha256_file(DEST / rel) == sha, f"receipt mismatch {rel}"
        ok += 1

    pinned = dict(
        vehicle_dynamics_pt=L.sha256_file(L.SURV_CKPT),
        dynamics_ensemble_pt=L.sha256_file(L.ENSEMBLE),
        match=bool(L.sha256_file(L.SURV_CKPT) == L.SURV_CKPT_SHA256 and
                   L.sha256_file(L.ENSEMBLE) == L.ENSEMBLE_SHA256))
    rec = dict(
        backup_utc=t0, dest=str(DEST), n_files=len(entries),
        sha256sums_verified=ok == len(entries),
        pinned_check=pinned,
        files=[dict(path=rel, sha256=sha, bytes=n)
               for rel, sha, n in entries])
    L.dump_json(DEST / "backup_receipt.json", rec)
    print(f"backup ok: {len(entries)} files -> {DEST}; "
          f"SHA256SUMS verified {ok}/{len(entries)}; "
          f"pinned match: {pinned['match']}")
    return 0 if pinned["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
