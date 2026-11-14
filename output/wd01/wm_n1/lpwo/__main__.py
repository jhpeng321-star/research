"""CLI: build the WM-S leave-PI-and-wall-out (LPWO) window lists (metadata only).

    python -m lpwo --lpo-dir _out/lpo --out _out/lpwo \
        --pack-v3-index D:/rd-staging/wmn1_feat/pack_v3/INDEX.json
    python -m lpwo --validate --lpo-dir _out/lpo --out _out/lpwo
"""
from __future__ import annotations

import argparse
import json
import sys

from .build_lpwo import RUNS_ROOT, build, validate
from .guard import DataGuard


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lpwo", description=__doc__)
    ap.add_argument("--lpo-dir", default="_out/lpo")
    ap.add_argument("--out", default="_out/lpwo")
    ap.add_argument("--pack-v3-index", default="D:/rd-staging/wmn1_feat/pack_v3/INDEX.json")
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--guard-spec", required=True)
    ap.add_argument("--guard-tool", required=True)
    ap.add_argument("--allowed-root", action="append", required=True)
    ap.add_argument("--extra-index", action="append", default=[], help="PACK=INDEX (repeatable)")
    args = ap.parse_args(argv)
    try:
        guard = DataGuard(args.guard_spec, args.guard_tool, args.allowed_root)
        extra = {}
        for entry in args.extra_index:
            name, sep, path = entry.partition("=")
            if not sep or not name or not path:
                raise ValueError("extra-index requires PACK=INDEX")
            extra[name] = path
        if args.validate:
            res = validate(args.out, args.lpo_dir, guard)
            print(json.dumps(res, indent=1))
            return 0 if res["ok"] else 1
        receipt = build(args.lpo_dir, args.out, pack_v3_index=args.pack_v3_index,
                        root=args.runs_root, guard=guard, extra_indexes=extra)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 2
    print(json.dumps({k: receipt[k] for k in (
        "n_train_before", "n_val_before", "n_train", "n_val", "n_removed_windows",
        "removed_by_reason", "train_by_policy_after", "val_by_policy_after",
        "n_excluded_run_keys")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
