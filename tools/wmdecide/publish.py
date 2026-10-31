"""Package verified credibility reports for the integrator, without rerunning scores."""

import argparse
import json
from pathlib import Path
import re

from .rbeval_adapter import safe_path, sha256
from .report import write_json


def publish(reports, out, *, matched=False):
    """Export compact, portable sections/figures and pooled numeric receipts.

    All source output hashes are checked before creating any destination. Raw
    row-level CSVs and full action/cell statistics stay in the immutable source
    reports; their manifest and audit digests bind this compact publication.
    """
    out = safe_path(out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError("Publication must be a NEW directory under an existing parent")
    reports = [safe_path(p).resolve() for p in reports]
    if not reports or len(set(reports)) != len(reports):
        raise ValueError("Distinct nonempty source report directories required")
    prepared = []
    for source in reports:
        manifest = json.loads((source / "FILES.json").read_text(encoding="utf-8"))
        required = {"CREDIBILITY.md", "credibility.json", "audit.json", "summary.csv",
                    "reliability.csv", "reliability_breakdown.svg", "reliability_preference.svg",
                    "reliability_selected_preference.svg", "decisions.csv", "events.csv",
                    "coverage.csv", "distributions.csv"}
        if not required <= set(manifest):
            raise ValueError("Source must be a complete wmdecide/2 credibility report")
        for name, digest in manifest.items():
            if Path(name).name != name or sha256(source / name) != digest:
                raise ValueError(f"Source report manifest mismatch: {source}/{name}")
        data = json.loads((source / "credibility.json").read_text(encoding="utf-8"))
        audit = json.loads((source / "audit.json").read_text(encoding="utf-8"))
        if data["provenance"] != audit or audit["schema"] != "wmdecide/2":
            raise ValueError("Report/audit provenance mismatch")
        if audit["status"] != "DIAGNOSTIC_ONLY" or not audit["audit"]["scored_worlds"]:
            raise ValueError("Publication requires scored complete K8 worlds")
        prepared.append((source, audit, data, manifest))
    if matched and len({tuple(p[1]["audit"]["scored_worlds"]) for p in prepared}) != 1:
        raise ValueError("Matched publication requires identical scored world sets")
    out.mkdir()
    receipt = dict(schema="wmdecide-publication/1", matched_worlds=matched,
                   publisher_sha256=sha256(Path(__file__)), reports=[])
    lines = ["# WM-PHYS credibility sections", "",
             "Complete K8 ensembles only; diagnostic evidence, not adoption qualification.", "",
             "Each linked section retains relative figure links inside its model directory. "
             "Full action/cell statistics and row-level outcomes remain in the hash-bound "
             "source reports recorded in `REPORTS.json`.", ""]
    for index, (source, audit, data, manifest) in enumerate(prepared):
        model = audit["model"]
        name = f"{index:02d}_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        dest = out / name
        dest.mkdir()
        for filename in ("CREDIBILITY.md", "reliability.csv", "reliability_breakdown.svg",
                         "reliability_preference.svg", "reliability_selected_preference.svg"):
            # Preserve the exact verified bytes, not a reformatted interpretation.
            if sha256(source / filename) != manifest[filename]:
                raise ValueError("Source changed during publication; preserve incomplete destination")
            with (dest / filename).open("xb") as f:
                f.write((source / filename).read_bytes())
        compact = dict(model=model, scored_worlds=audit["audit"]["scored_worlds"],
            summary=[r for r in data["summary"] if r["scope"] == "pooled"],
            source_report=str(source), source_manifest_sha256=sha256(source / "FILES.json"),
            source_audit_sha256=manifest["audit.json"],
            qualification=audit["qualification"], uncertainty=audit["intervals"])
        write_json(dest / "POOLED.json", compact)
        receipt["reports"].append(dict(directory=name, **compact))
        lines.append(f"- [{model}]({name}/CREDIBILITY.md): "
                     f"{len(compact['scored_worlds'])} scored worlds.")
    (out / "INDEX.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(out / "REPORTS.json", receipt)
    write_json(out / "FILES.json", {str(p.relative_to(out)).replace("\\", "/"): sha256(p)
                                   for p in sorted(out.rglob("*")) if p.is_file()})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matched", action="store_true", help="Require identical scored worlds")
    args = parser.parse_args()
    result = publish(args.reports, args.out, matched=args.matched)
    print(json.dumps({"models": [r["model"] for r in result["reports"]],
                      "matched_worlds": result["matched_worlds"]}))


if __name__ == "__main__":
    main()
