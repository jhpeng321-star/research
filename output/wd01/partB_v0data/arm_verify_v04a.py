"""WD-01-PB-V04A section 2: arm verification (ZERO SUMO).

For every source of the v0.4a root (chunk-4 TRAIN/CAL arms are the
registered scope; the J3 truearms and the auxiliary chunk arms are
verified with the same rule), verify the arm's policy from its
switch_schedule.json plus the suffix_commands.jsonl owners
(kind == "command" rows only) via
``actions_v04.ActionTimeline.verify_arm``.  Any mismatch REFUSES the
source (it is recorded and excluded from the build; nothing is
relabelled).

HOLDOUT is never opened: specs come from v04a_sources.enumerate_specs
(plan split checked first); an audit hook records every opened path and
the close-out asserts none names a HOLDOUT world.  dispatcher_rows.jsonl
and insertions.jsonl are never opened.

Usage:
  python arm_verify_v04a.py --out <verification.json> [--chunks chunk4,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import wdpaths  # noqa: E402
import actions_v04 as A  # noqa: E402
import v04a_sources as S  # noqa: E402
import v0lib as V  # noqa: E402


def verify_spec(spec: S.SrcSpec) -> dict:
    """Verification record for one source (never raises on a mismatch;
    the record says refused=True with the reason)."""
    arm = Path(wdpaths.remap(spec.arm_dir))
    rec = dict(key=spec.key(), chunk=spec.chunk, world=spec.world,
               source=spec.source, split=spec.split, role=spec.role,
               policy=spec.policy, verify_kind=spec.verify_kind)
    ok, receipt, reason = V.receipt_status(arm)
    if not ok:
        rec.update(refused=True, reason=f"incomplete run: {reason}",
                   incomplete=True)
        return rec
    try:
        tl = A.ActionTimeline.from_run_dir(arm, schedule_authoritative=True)
        ev = tl.verify_arm(spec.verify_kind)
        rec.update(refused=False, evidence=ev,
                   sources={k: Path(v).name for k, v in tl.sources.items()})
    except A.ActionReconstructionError as e:
        rec.update(refused=True, reason=f"{type(e).__name__}: {e}")
    return rec


def _worker(spec_dict):
    spec = S.SrcSpec(**spec_dict)
    return verify_spec(spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default=",".join(S.ALL_CHUNKS))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    chunks = tuple(c for c in args.chunks.split(",") if c)
    S.OpenAudit.start()
    t0 = time.time()
    holdout: set = set()
    for c in chunks:
        holdout.update(S.chunk_worlds(c)[1])
    specs = S.enumerate_specs(chunks)
    assert not ({s.world for s in specs} & holdout)
    assert all(s.split in S.KEEP_SPLITS for s in specs)
    recs = [verify_spec(s) for s in specs]
    n_open = S.OpenAudit.assert_no_holdout(holdout)
    S.OpenAudit.stop()
    for p in S.OpenAudit.paths:
        low = p.replace("\\", "/")
        assert "dispatcher_rows" not in low and "insertions.jsonl" \
            not in low, p
    refused = [r for r in recs if r["refused"]]
    summary = dict(
        schema="wd01_pb_v04a_arm_verification/1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        rule=("switch_schedule.json + kind=='command' owners of "
              "suffix_commands.jsonl/telemetry.jsonl; any mismatch "
              "refuses the source"),
        chunks=list(chunks), n_sources=len(recs),
        n_verified=len(recs) - len(refused), n_refused=len(refused),
        refused=[dict(key=r["key"], reason=r["reason"]) for r in refused],
        holdout_worlds_not_opened=len(holdout),
        n_paths_opened_audited=n_open,
        dispatcher_rows_opened=False, insertions_jsonl_opened=False,
        wall_s=round(time.time() - t0, 1),
        by_kind={k: sum(1 for r in recs if r["verify_kind"] == k
                        and not r["refused"])
                 for k in A.ARM_KINDS},
        records=recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1),
                              encoding="utf-8")
    print(f"arm verification: {summary['n_verified']}/{len(recs)} verified, "
          f"{len(refused)} refused; wall {summary['wall_s']} s")
    for r in refused[:20]:
        print("  REFUSED", r["key"], r["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
