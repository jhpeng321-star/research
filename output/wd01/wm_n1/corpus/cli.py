"""Corpus CLI.

    python -m corpus featurize-all --inventory INV.parquet --data-root D --out S
    python -m corpus pack --data-root S --out PACK [--workers 4]
    python -m corpus convert --in-dir PACK --out PACK2 [--workers 4]
    python -m corpus verify --pack PACK
    python -m corpus manifest --pack PACK --out MANIFEST.json

``--allow-d-output-root D:/rd-staging/wmn1_feat`` (owner-approved FEAT1
override) permits writing ONLY under that exact D: root; with it the
forbid-D guard is installed for the whole run and every other D: path
(in particular D:/research-dense) stays blocked.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import _util as U
from . import featurize_all as FA
from . import manifest as MF
from . import pack as PK
from . import verify as VF


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="corpus")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("featurize-all")
    f.add_argument("--inventory", required=True)
    f.add_argument("--data-root", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=U.DEFAULT_WORKERS)
    f.add_argument("--force", action="store_true")
    f.add_argument("--min-free-virtual-gb", dest="min_free_virtual_gb",
                   type=float, default=None)
    f.add_argument("--min-free-ram-gb", dest="min_free_ram_gb", type=float,
                   default=None)
    f.add_argument("--max-workers-override", dest="max_workers_override",
                   type=int, default=None,
                   help="explicit owner-approved override of the 6-worker "
                        "hard maximum (recorded in ALLOCATION.json); a run "
                        "above 8 workers drops batches to 8 while free "
                        "virtual < 8 GB or free RAM < 6 GB")
    f.add_argument("--idea-fast", dest="idea_fast", action="store_true",
                   help="byte-identical fast Idea A baseline (FEAT1 "
                        "profiling speed-up); recorded in ALLOCATION.json")
    f.add_argument("--format", dest="format", default=None,
                   choices=("v2", "v3"),
                   help="slot-width format: v2 keeps K_max=128 (frozen, "
                        "byte-reproducible); v3 carries K_max=192 "
                        "(amendment 5; recorded in ALLOCATION.json)")
    f.add_argument("--source-override", dest="source_override", default=None,
                   metavar="MAP.json",
                   help="run key -> physics_steps.jsonl map redirecting "
                        "selected runs' raw source (FEAT3 stage 1 archive "
                        "restore; count + path recorded in ALLOCATION.json)")
    k = sub.add_parser("pack")
    k.add_argument("--data-root", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--workers", type=int, default=U.DEFAULT_WORKERS)
    c = sub.add_parser("convert")
    c.add_argument("--in-dir", dest="in_dir", required=True)
    c.add_argument("--out", dest="out", required=True)
    c.add_argument("--workers", type=int, default=U.DEFAULT_WORKERS)
    v = sub.add_parser("verify")
    v.add_argument("--pack", required=True)
    v.add_argument("--allow-d-output-root", dest="allow_d_output_root",
                   default=None,
                   help="owner-approved override: permit D: reads ONLY "
                        "under this exact root (read-only verify)")
    m = sub.add_parser("manifest")
    m.add_argument("--pack", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--exclusions", default=None)
    m.add_argument("--tools-dir", dest="tools_dir", default=None)
    for s in (f, k, c, m):
        s.add_argument("--allow-d-output-root", dest="allow_d_output_root",
                       default=None,
                       help="owner-approved override: permit D: writes ONLY "
                            "under this exact root (never D:/research-dense)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "allow_d_output_root", None):
        U.activate_d_output_override(args.allow_d_output_root)
    if args.cmd == "featurize-all":
        rep = FA.featurize_all(
            args.inventory, args.data_root, args.out, workers=args.workers,
            force=args.force, min_free_virtual_gb=args.min_free_virtual_gb,
            min_free_ram_gb=args.min_free_ram_gb,
            max_workers_override=args.max_workers_override,
            idea_fast=args.idea_fast, format=args.format,
            source_override=args.source_override)
        print(json.dumps({k: rep[k] for k in
                          ("runs", "written", "skipped", "failed", "receipt")
                          if k in rep}, indent=1))
        return 0
    if args.cmd == "pack":
        man = PK.pack_shards(args.data_root, args.out, workers=args.workers)
        print(json.dumps({k: man[k] for k in
                          ("schema", "n_runs", "n_ticks", "n_decision",
                           "totals")}, indent=1))
        return 0
    if args.cmd == "convert":
        man = PK.convert(args.in_dir, args.out, workers=args.workers)
        print(json.dumps({k: man[k] for k in
                          ("schema", "n_runs", "totals")}, indent=1))
        return 0
    if args.cmd == "verify":
        rep = VF.verify_pack(args.pack)
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] else 1
    if args.cmd == "manifest":
        man = MF.build_upload_manifest(args.pack, exclusions_path=args.exclusions,
                                       tools_dir=args.tools_dir)
        sha = MF.write_manifest(man, args.out)
        print(json.dumps({"manifest": args.out, "sha256": sha,
                          "files": len(man["files"]),
                          "n_runs": man["n_runs"], "splits": man["splits"]},
                         indent=1))
        return 0
    raise SystemExit(f"unknown command {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
