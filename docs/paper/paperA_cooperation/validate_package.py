"""Offline integrity checks for this explicitly scoped drafting package."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess

from build_figures import ROOT, check, extract
from build_night_figures import check_night, extract_night
from build_morning_figures import check_morning, extract_morning
from build_noon_figures import check_noon, extract_noon
from build_student_figures import check_student, extract_student
from build_evening_figures import check_evening, extract_evening


def main():
    required = ["README.md", "OUTLINE.md", "CLAIMS.md", "FIGURES.md", "RELATED_WORK.md",
                "THREATS.md", "EVIDENCE_SOURCES.md", "REFRESH.md", "refs.bib", "reference_verification.csv"]
    docs = {name: (ROOT / name).read_text(encoding="utf-8") for name in required}
    for name, text in docs.items():
        if name.endswith(".md"):
            assert text.count("<!--") == text.count("-->"), name
            assert text.count("[[PENDING:") == len(re.findall(r"\[\[PENDING:[\s\S]*?\]\]", text)), name
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                path = (ROOT / target.split("#", 1)[0]).resolve()
                assert path.is_relative_to(ROOT), (name, target)
                assert path.exists(), (name, target)
    # The primary/backup hierarchy and prerequisite claims must survive refreshes.
    for claim in ("W-VALUE", "W-POLICY", "W-RISE", "W-WIN", "W-RATIO", "M-ARTIFACT",
                  "W-CALIB", "W-QUEUE", "W-DATA", "W-DIST", "N-FOLLOWUP",
                   "B-TRANSFER", "B-FLOOR", "B-SWITCH", "B-CLOSED",
                    "P-WIN", "P-INTERACT", "P-WALL", "P-TRANSFER", "N-BANDIT", "N-FREE",
                     "W-TEACH", "N-DISTILL", "W-CORE", "M-CREDIT", "W-GUIDE",
                     "W-NOISE", "W-RANK", "W-V5"):
        assert re.search(r"^### " + claim + r"\b", docs["CLAIMS.md"], re.MULTILINE), claim
    for name in ("OUTLINE.md", "FIGURES.md", "THREATS.md", "README.md"):
        assert "WMVALUE" in docs[name], name
        assert "backup" in docs[name].lower(), name
    # Verify the immutable report cut offline, without reading live producer inputs.
    refresh = json.loads((ROOT / "source_refresh.json").read_text(encoding="utf-8"))
    report_checks = []
    for key, entry in refresh["reports"].items():
        if not entry["exists"]:
            continue
        path = (ROOT / entry["snapshot"]).resolve()
        assert path.is_relative_to(ROOT / "source_snapshots"), key
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], key
        report_checks.append(key)
    bib = docs["refs.bib"]
    # Syntax sanity; final conference BibTeX/IEEEtran rendering remains owner's task.
    body = "\n".join(line for line in bib.splitlines() if not line.startswith("%"))
    assert body.count("{") == body.count("}")
    keys = re.findall(r"@\w+\{([^,]+),", body)
    assert len(keys) == len(set(keys))
    with (ROOT / "reference_verification.csv").open(encoding="utf-8", newline="") as handle:
        verified = list(csv.DictReader(handle))
    assert set(keys) == {r["key"] for r in verified}
    assert all(r["identity_status"] == "VERIFIED" for r in verified)
    assert all(r["verification_url"].startswith(("https://doi.org/", "https://arxiv.org/abs/"))
               for r in verified)
    for key in keys:
        assert f"`{key}`" in docs["RELATED_WORK.md"], key
    with (ROOT / "controller_contracts.csv").open(encoding="utf-8", newline="") as handle:
        controllers = list(csv.DictReader(handle))
    assert all(None not in row and all(value is not None for value in row.values()) for row in controllers)
    rows, manifest = extract()
    check(rows, manifest)
    night_rows, night_sources = extract_night()
    night_figures = check_night(night_rows, night_sources)
    morning_rows, morning_sources = extract_morning()
    morning_figures = check_morning(morning_rows, morning_sources)
    noon_rows, noon_source = extract_noon()
    noon_figures = check_noon(noon_rows, noon_source)
    student_rows, student_sources, student_worlds, _ = extract_student()
    student_figures = check_student(student_rows, student_sources, student_worlds)
    evening_rows, evening_source = extract_evening()
    evening_figures = check_evening(evening_rows, evening_source)
    # Check staged byte preservation as well as working-copy provenance.
    index_checks = []
    identities = [(entry["snapshot"], entry["snapshot_sha256"]) for entry in manifest.values()]
    figures = json.loads((ROOT / "figure_validation.json").read_text(encoding="utf-8"))
    identities.extend((name, entry["sha256"]) for name, entry in figures["artifacts"].items())
    identities.extend((name, entry["sha256"]) for name, entry in night_figures["artifacts"].items())
    identities.extend((name, entry["sha256"]) for name, entry in morning_figures["artifacts"].items())
    identities.extend((name, entry["sha256"]) for name, entry in noon_figures["artifacts"].items())
    identities.extend((name, entry["sha256"]) for name, entry in student_figures["artifacts"].items())
    identities.extend((name, entry["sha256"]) for name, entry in evening_figures["artifacts"].items())
    for name, expected in identities:
        relative = "docs/paper/paperA_cooperation/" + name
        command = subprocess.run(["git", "show", ":" + relative], cwd=ROOT,
                                 capture_output=True, check=False)
        if command.returncode == 0:
            assert hashlib.sha256(command.stdout).hexdigest() == expected, ("git index bytes", name)
            index_checks.append(name)
    result = {"status": "PASS", "required_documents": required,
              "verified_bibliography_entries": len(keys), "controller_contract_rows": len(controllers),
              "git_index_byte_checks": index_checks,
              "report_snapshot_checks": report_checks,
               "night_figure_source_cells": len(night_rows),
                "morning_figure_source_cells": len(morning_rows),
                "noon_figure_source_cells": len(noon_rows),
                 "student_figure_source_cells": len(student_rows),
                 "evening_figure_source_cells": len(evening_rows),
              "pending_insertions": {name: text.count("[[PENDING:") for name, text in docs.items()
                                     if name.endswith(".md")},
              "scope": "local links, provenance structure, bibliography identity coverage, CSV schema, plotted source values and hashes",
              "limitations": "Not a scientific efficacy check, exhaustive numerical prose proof, full BibTeX compile, or IEEE pagination check",
              "document_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in required}}
    (ROOT / "package_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
