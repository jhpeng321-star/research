"""Independent, CPU-only artifact checks for the Paper B drafting package."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"


def csv_rows(name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO((DATA / name).read_text(encoding="utf-8-sig"))))


def main() -> None:
    checks = []
    for name in ["README.md", "OUTLINE.md", "CLAIMS.md", "FIGURES.md", "RELATED_WORK.md", "THREATS.md", "refs.bib"]:
        assert (ROOT / name).stat().st_size > 0, name
    checks.append("required deliverables nonempty")

    manifest = json.loads((DATA / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in manifest.items():
        blob = (DATA / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == info["sha256"], name
        assert len(blob) == info["bytes"], name
    checks.append("frozen source byte/hash integrity")

    # Compare exact source records, independently of the builder's selector loop.
    raw = {(r["panel"],r["metric"]):r for r in csv_rows("fair.source.csv")}
    for r in csv_rows("regret.csv"):
        original = raw[(r["panel"],r["metric"])]
        for field in ["estimate", "lo", "hi", "n_worlds", "n_states"]:
            assert r[field] == original[field], (r["selector"], field)
        assert float(r["lo"]) <= float(r["estimate"]) <= float(r["hi"])
    strict = csv_rows("regret.csv")
    assert math.isclose(float(strict[0]["estimate"]), 13.638898849487305)
    assert math.isclose(float(strict[1]["estimate"]), 22.050608545541763)
    checks.append("strict-history regret rows equal original CSV, no inferred paired CI")

    paired = csv_rows("regret_paired.csv")
    assert len(paired) == 1
    p = paired[0]
    original = raw[("paired_decisions/B_history_strict/60/team/13e_minus_bbtraj", "regret_oracle")]
    assert all(p[k] == original[k] for k in original)
    assert math.isclose(float(p["estimate"]), -8.411709696054459)
    assert float(p["hi"]) < 0
    checks.append("paired WM-minus-BB regret is an existing source contrast, not inferred from marginal intervals")

    raw = csv_rows("direction.source.csv")
    direction = csv_rows("direction.csv")
    assert len(raw) == len(direction)
    for a,b in zip(raw,direction):
        assert all(a[k] == b[k] for k in a)
    assert [r["verdict"] for r in direction] == ["PASS"]*4+["FAIL"]*2
    checks.append("direction values, CIs, thresholds and verdicts equal original export")

    regional = csv_rows("regional.csv")
    last = [r for r in regional if r["horizon_s"] == "300"]
    assert [float(r["covariance_share_pct"]) for r in last] == [-7.1,96.0,11.1]
    assert abs(sum(float(r["covariance_share_pct"]) for r in last)-100) < 1e-8
    assert [float(r["mae_veh_s"]) for r in last] == [871.3,1497.6,317.0]
    entry = [r for r in csv_rows("entry.csv") if r["horizon_s"] == "300"]
    assert [(r["component"],float(r["wm2_mae_veh_s"]),float(r["entry_mae_veh_s"])) for r in entry] == [
        ("Source",584.0,82.5), ("Upstream",1456.9,1393.0), ("Total J",1449.1,1536.2)]
    assert all(r["clock"] != last[0]["clock"] for r in entry)
    wall = [r for r in csv_rows("wall_teacher_forced.csv") if r["horizon_s"] == "60"]
    assert [float(r["median_m"]) for r in wall] == [8.71,25.41,40.07]
    assert all(r["n_states"] == "67" for r in wall)
    checks.append("reported regional/entry/wall cells cross-checked, clock conventions distinct")

    for name in ["direction.csv", "regret.csv", "regret_paired.csv", "regional.csv", "entry.csv", "wall_teacher_forced.csv", "architecture_edges.csv"]:
        for r in csv_rows(name):
            assert r["source_file"] and r["source_selector"], name
    checks.append("every derived numerical/schematic CSV row has source and selector")

    names = ["fig_direction", "fig_regret", "fig_regional", "fig_entry", "fig_wall_teacher_forced", "fig_architecture"]
    for name in names:
        with Image.open(FIG / f"{name}.png") as image:
            assert min(image.size) > 400
            assert image.info.get("Description"), name
            image.verify()
        pdf = (FIG / f"{name}.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-") and len(pdf)>1000, name
    checks.append("PNG decoding, source metadata, dimensions and vector PDF signatures")

    bib = (ROOT / "refs.bib").read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\{([^,]+),",bib)
    assert len(keys) == len(set(keys)) == 12
    # Basic brace/field checks; no claim of a full LaTeX/BibTeX compilation.
    depth = 0
    for char in bib:
        depth += (char == "{") - (char == "}")
        assert depth >= 0
    assert depth == 0
    entries = re.split(r"(?=@\w+\{)",bib)[1:]
    related = (ROOT / "RELATED_WORK.md").read_text(encoding="utf-8")
    for entry in entries:
        for field in ["author", "title", "year", "url"]:
            assert re.search(rf"\b{field}\s*=\s*\{{",entry)
        url = re.search(r"url\s*=\s*\{([^}]+)\}",entry)[1]
        assert url in related, url
    for key in keys:
        assert f"`{key}`" in related, key
    checks.append("verified bibliography pool: unique keys, balanced braces, required fields and documented verification URLs")

    claims = (ROOT / "CLAIMS.md").read_text(encoding="utf-8")
    assert all(status in claims for status in ["SUPPORTED", "PENDING", "CONTRADICTED"])
    for phrase in ["96.0%", "584.0", "13.64", "0.8895"]:
        line = next(line for line in claims.splitlines() if phrase in line)
        assert "<!-- Source" in line, phrase
    assert "[[PENDING:" in claims and "S-C1CLOSED" in claims and "S-WMCALIB" in claims
    checks.append("headline numerical claim comments and explicit agent-owned pending fields")

    receipt = {"status":"PASS", "checks":checks,
               "scope":"Package/source-selection/artwork QA only; no scientific requalification or typeset-page verification",
               "python":sys.version, "matplotlib":matplotlib.__version__,
               "source_manifest_sha256":hashlib.sha256((DATA/"SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
               "figure_count":len(names), "bib_entries":len(keys)}
    (ROOT / "VERIFICATION.json").write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
