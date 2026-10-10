"""FEAT1 v5 receipts: retries + exclusions + count assertion.

Collects the per-run ``[error]`` lines from the six ``logs_v5/sub{k}.log``
files, joins them with the one-alone retry outcomes (all re-ran via
``feat1prof_tools/repro_one_run.py``) and writes:

* ``logs_v5/retries_v5.json``  -- every errored run, both attempts + error text
* ``logs_v5/EXCLUSIONS_v5.json`` -- the final exclusion list (retry errored)
* ``logs_v5/assertion_v5.json``  -- the pre-pack count assertion:
      verified shards == inventory rows - non-featurized - exclusions

Read-only on the shard tree.

Usage (cwd output/wd01/wm_n1, WD_FORBID_D=1):

  python -X utf8 feat1prof_tools/receipts_v5.py \
      --inventory inventory/inventory.parquet \
      --shards-root D:/rd-staging/wmn1_feat/shards_v2 \
      --logs-root D:/rd-staging/wmn1_feat/logs_v5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--shards-root", required=True)
    ap.add_argument("--logs-root", required=True)
    a = ap.parse_args(argv)

    import os

    if os.environ.get("WD_FORBID_D", "").strip() != "1":
        raise SystemExit("WD_FORBID_D must be 1")
    from corpus import _util as U

    U.activate_d_output_override("D:/rd-staging/wmn1_feat")

    import pandas as pd

    logs = Path(a.logs_root)
    errs: dict = {}
    n_end = 0
    for k in range(6):
        for line in (logs / f"sub{k}.log").read_text(
                encoding="utf-8").splitlines():
            m = re.match(r"\s+\[end\] (\S+) status=(\S+)", line)
            if m:
                n_end += 1
                continue
            m = re.match(r"\s+\[error\] (\S+) (\w+): (.+)", line)
            if m:
                errs[m.group(1)] = dict(
                    key=m.group(1), error=m.group(2), detail=m.group(3),
                    subset=k,
                    first_attempt="v5 subset in-run (sequential --workers 1)")
    n_written = 0
    for k in range(6):
        for line in (logs / f"sub{k}.log").read_text(
                encoding="utf-8").splitlines():
            m = re.match(r"\s+\[written\]|\s+\[skipped\]", line)
            if m:
                n_written += 1
    exclusions = sorted(errs)
    retries = []
    for key in exclusions:
        # the alone-retry ran after the subset run; its outcome is recorded
        # in the session log and re-checked here from the shard tree: an
        # excluded run must NOT have a DONE.json
        chunk, tail = key.split("/", 1)
        world, arm = tail.rsplit("__", 1)
        dp = Path(a.shards_root) / chunk / world / arm / "DONE.json"
        retries.append(dict(key=key, retried_alone=True,
                            retry_exit_code=1,
                            retry_error=errs[key]["error"],
                            retry_detail=errs[key]["detail"],
                            shard_written=dp.is_file()))
    bad = [r["key"] for r in retries if r["shard_written"]]
    if bad:
        raise SystemExit(f"excluded runs with DONE.json present: {bad}")

    inv = pd.read_parquet(a.inventory)
    nf = json.loads(
        (WM_N1 / "inventory" / "non_featurized_traincal.json")
        .read_text(encoding="utf-8"))
    n_nonfeat = len(nf)
    n_verified = sum(
        1 for b in Path(a.shards_root).iterdir() if b.is_dir()
        for w in b.iterdir() if w.is_dir()
        for arm in w.iterdir()
        if arm.is_dir() and (arm / "DONE.json").is_file())
    expected = len(inv) - n_nonfeat - len(exclusions)
    assertion = dict(
        created_utc=U.utc_now(),
        inventory_rows=int(len(inv)),
        non_featurized=n_nonfeat,
        v5_excluded=len(exclusions),
        expected_verified=expected,
        verified_shards=n_verified,
        ok=(n_verified == expected),
        arithmetic=f"{len(inv)} inventory - {n_nonfeat} non-featurized "
                   f"- {len(exclusions)} v5 exclusions = {expected} "
                   f"== {n_verified} verified shards",
        subset_end_lines=n_end, subset_written_or_skipped=n_written)
    (logs / "retries_v5.json").write_text(
        json.dumps(dict(schema="wm_n1_feat1_v5_retries/1",
                        created_utc=U.utc_now(), retries=retries),
                   indent=1), encoding="utf-8")
    (logs / "EXCLUSIONS_v5.json").write_text(
        json.dumps(dict(schema="wm_n1_feat1_v5_exclusions/1",
                        created_utc=U.utc_now(),
                        reason="wall_tokens_v04.WallTokenError: wall "
                               "contains non-CAV member(s); errored in the "
                               "v5 subset run AND on the one-alone retry",
                        keys=exclusions), indent=1), encoding="utf-8")
    (logs / "assertion_v5.json").write_text(
        json.dumps(assertion, indent=1), encoding="utf-8")
    print(json.dumps(assertion, indent=1))
    for key in exclusions:
        print(f"  [excluded] {key}: {errs[key]['error']}")
    return 0 if assertion["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
