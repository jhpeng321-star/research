"""S-BACKLOG step 2: re-export loop cycles 0-2 paired datasets with exact clocks.

Runs on the collection host that holds the raw branch tapes (cpuP or cpuQ).
Converts each paired-branch tape of the ORIGINAL gpuA dataset manifest through
the fixed converter (exact entry_backlog, reconciled to the tape's authoritative
pair-summary ledger, fail closed), cross-checks every tape summary sha and J
against the gpuA returns receipts, and writes a NEW dataset (originals are never
touched) whose episode set and manifest mirror the original exactly.

Usage:
  python reexport_pairs.py --cycle c02 --dataset pairs_c02.a1 \
      --manifest <original C4_EPISODES_MANIFEST.json> \
      --policy-receipt <returns_c02.a1.receipt.json> \
      --native-receipt <native_returns_c02.a1.receipt.json> \
      --work-root /home/ubuntu/c4/main14/restart20261026/work \
      --out /home/ubuntu/backlog/reexport/pairs_c02.a1.backlog.part
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wms.data.c4_episodes import convert_episode  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tape_dir(work_root: Path, entry: dict, cycle: str):
    """Branch tape of one manifest entry on THIS host, or None when its world lives elsewhere.

    The cycle directory may be a retry attempt (``c01.a4``): all ``<cycle>*`` siblings are
    searched and exactly one tape must match (fail closed otherwise)."""
    base = Path(entry["branch"]["base_episode"])   # .../work/<wXXXXXX>/C4-.../<cycle>/collect/episodes/policy/<job>
    tag = entry["arm"].rsplit("_", 1)[1]            # native | policy
    t_k = int(entry["branch"]["branch_t_s"])
    world = work_root / base.parents[5].name        # host-local rebase of the recorded absolute path
    if not world.is_dir():
        return None
    hits = sorted(world.glob(f"*/{cycle}*/collect/branches/policy/{base.name}/t{t_k:04d}_{tag}"))
    hits = [h for h in hits if (h / "episode.npz").is_file()]
    if not hits:
        return None                                  # tape lives on the other host (union is checked at merge)
    if len(hits) != 1:
        raise SystemExit(f"REFUSED: {len(hits)} tapes for {base.name} t{t_k} {tag}: {hits}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--policy-receipt", required=True)
    ap.add_argument("--native-receipt", required=True)
    ap.add_argument("--work-root", default="/home/ubuntu/c4/main14/restart20261026/work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds", default="",
                    help="comma-separated seeds this host holds (others are skipped)")
    args = ap.parse_args()
    only = {int(s) for s in args.worlds.split(",") if s.strip()}

    original = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    receipts = {}
    for kind, path in (("POLICY", args.policy_receipt), ("B", args.native_receipt)):
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in doc["entries"]:
            receipts[(item["pair_key"], kind)] = item
    out = Path(args.out)
    if out.exists():
        print(f"REFUSED: {out} exists (never overwritten)", file=sys.stderr)
        return 2
    work_root = Path(args.work_root)
    entries, j_checks, skipped = [], 0, 0
    for src_entry in original["episodes"]:
        if only and int(src_entry["seed"]) not in only:
            continue
        kind = src_entry["branch"]["kind"]
        tape = tape_dir(work_root, src_entry, args.cycle)
        if tape is None:
            skipped += 1                              # world held by the other collection host
            continue
        entry = convert_episode(tape, out)
        rec = entry["queue_arrays"].get("pair_cost_reconciliation")
        if rec is None:
            print(f"REFUSED: no reconciliation receipt for {tape}", file=sys.stderr)
            return 4
        want = receipts.get((entry["pair_key"], kind))
        if want is None:
            print(f"REFUSED: no returns-receipt entry for {entry['pair_key']}/{kind}",
                  file=sys.stderr)
            return 5
        if rec["summary_sha256"] != want["sha256"]:
            print(f"REFUSED: tape summary sha != returns receipt sha for "
                  f"{entry['pair_key']}/{kind}: {rec['summary_sha256']} vs {want['sha256']}",
                  file=sys.stderr)
            return 6
        if abs(rec["clock_costs_vs"] - float(want["J"])) > 1e-3 * max(1.0, abs(float(want["J"]))):
            print(f"REFUSED: reconciled J != returns J for {entry['pair_key']}/{kind}: "
                  f"{rec['clock_costs_vs']} vs {want['J']}", file=sys.stderr)
            return 7
        j_checks += 1
        entry["backlog_reexport"] = {
            "task": "S-BACKLOG", "dataset": args.dataset, "cycle": args.cycle,
            "tape": str(tape), "episode_sha256": sha256(tape / "episode.npz"),
            **{k: rec[k] for k in ("summary_sha256", "requests_sha256",
                                   "clock_costs_vs", "difference")}}
        entries.append(entry)
    manifest = {**{k: v for k, v in original.items() if k != "episodes"},
                "episodes": entries,
                "backlog_reexport": {"task": "S-BACKLOG", "source_dataset": args.dataset,
                                     "cycle": args.cycle, "n_episodes": len(entries),
                                     "j_receipt_checks": j_checks,
                                     "original_manifest_episodes": len(original["episodes"]),
                                     "worlds": sorted(only) if only else "all",
                                     "skipped_other_host_entries": skipped}}
    (out / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    sums = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.json":
            rel = p.relative_to(out).as_posix()
            sums[rel] = sha256(p)
    (out / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "episodes": len(entries), "j_checks": j_checks, "skipped": skipped,
                      "still_waiting_total": sum(e["queue_arrays"]["still_waiting_at_end"]
                                                 for e in entries)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
