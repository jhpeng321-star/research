"""Package-only QA: explicit report paths, verified citations, PDF and layout checks.

No experiment, simulator, recursive discovery, hidden input, or process control.
Generated receipts and renderings are build products, not scientific estimates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
INHERITED = ROOT.parent / "paperA_cooperation"
REPORTS = {
    "exact_planner": Path("E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md"),
    "opportunity_and_bandit": Path("E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md"),
    "wm_teacher_diagnostic": Path("E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md"),
    "negative_distillation": Path("E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md"),
    "earlier_learner_design": Path("E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md"),
    "wm_planner_status": Path("E:/agent-work/c4-wmplan/docs/paper_results/WMPLAN.md"),
    "cached_action_ranking": Path("E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md"),
    "owner_authority": Path("E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md"),
    "verified_bibliography": INHERITED / "refs.bib",
    "verification_receipt": INHERITED / "reference_verification.csv",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entries(text: str) -> dict[str, str]:
    """Balanced-brace entries, normalized only for layout and comments."""
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    result = {}
    for match in re.finditer(r"@\w+\s*\{\s*([^,]+),", text):
        depth, i = 1, match.end()
        while depth and i < len(text):
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        assert depth == 0, f"Unbalanced BibTeX: {match.group(1)}"
        result[match.group(1)] = re.sub(r"\s+", "", text[match.start():i])
    return result


def main() -> None:
    for path in [ROOT, *REPORTS.values()]:
        assert not re.search(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", str(path), re.I)
        assert path.exists(), str(path)
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    outline = (ROOT / "OUTLINE_IEEEIV.md").read_text(encoding="utf-8")
    bib = entries((ROOT / "refs.bib").read_text(encoding="utf-8"))
    inherited = entries((INHERITED / "refs.bib").read_text(encoding="utf-8"))
    with (INHERITED / "reference_verification.csv").open(encoding="utf-8", newline="") as handle:
        verified = {row["key"] for row in csv.DictReader(handle) if row["identity_status"] == "VERIFIED"}
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    assert cited <= set(bib) <= verified
    assert all(bib[key] == inherited[key] for key in bib), "Bibliographic fields changed"
    sections = re.findall(r"\\section\{([^}]+)\}", tex)
    assert sections == ["Introduction", "Related Work", "Problem Formulation", "Method",
                        "Experimental Setup", "Results", "Discussion and Limitations", "Conclusion"]
    assert len(re.findall(r"\\begin\{figure\}", tex)) == 4
    assert len(re.findall(r"\\begin\{table\}", tex)) == 2
    assert all(f"E{i}:" in tex for i in range(1, 6))
    assert "member-revert" in outline and "member-revert" in tex
    assert "same-budget" in outline and "same-budget" in tex
    log = (ROOT / "main.log").read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"Overfull \\[hv]box|undefined references|Citation .* undefined|^!", log, re.M)

    pdf = fitz.open(ROOT / "main.pdf")
    assert 1 <= len(pdf) <= 6, f"Page limit exceeded: {len(pdf)}"
    text = "\n".join(page.get_text() for page in pdf)
    compact = re.sub(r"\s+", "", text).upper()
    assert all(word in compact for word in ("REFERENCES", "PENDING", "CONCLUSION", "CITENEEDED"))
    assert "[?]" not in text
    page_audits = []
    renders = ROOT / "qa_render"
    renders.mkdir(exist_ok=True)
    thumbs = []
    for number, page in enumerate(pdf, 1):
        overflow = []
        red_spans = 0
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    box = fitz.Rect(span["bbox"])
                    if not page.rect.contains(box):
                        overflow.append(span["text"])
                    color = span["color"]
                    red_spans += (color >> 16 > 100 and (color & 0xFFFF) < 100)
        assert not overflow, f"Text outside page {number}: {overflow}"
        pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
        pix.save(renders / f"page_{number}.png")
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image.thumbnail((425, 550))
        thumbs.append(image)
        page_audits.append({"page": number, "words": len(page.get_text().split()),
                            "red_spans": red_spans, "outside_page": overflow})
    sheet = Image.new("RGB", (3 * 445, ((len(thumbs) + 2) // 3) * 585), "#dce1e7")
    draw = ImageDraw.Draw(sheet)
    for i, thumb in enumerate(thumbs):
        x, y = (i % 3) * 445 + 10, (i // 3) * 585 + 25
        sheet.paste(thumb, (x, y))
        draw.text((x, y - 18), f"Page {i + 1}", fill="black")
    sheet.save(renders / "contact_sheet.png")
    receipt = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Outline/document QA only; no scientific result recomputed",
        "status": "PASS",
        "pdf_pages_including_references": len(pdf),
        "page_limit": 6,
        "sections": sections,
        "verified_citation_keys": sorted(cited),
        "bibliographic_entries_match_inherited_verified_bytes_modulo_whitespace": True,
        "figure_placeholders": 4,
        "table_placeholders": 2,
        "tex_pending_markers": tex.count("\\pending{"),
        "tex_citation_needed_markers": tex.count("\\citeneeded{"),
        "source_comment_count": len(re.findall(r"(?m)^%\s*source:", tex)),
        "overfull_or_undefined_latex_warnings": False,
        "pages": page_audits,
        "artifacts": {name: {"sha256": digest(ROOT / name), "bytes": (ROOT / name).stat().st_size}
                      for name in ("main.tex", "refs.bib", "main.pdf", "OUTLINE_IEEEIV.md")},
        "source_reports_at_qa_cut": {name: {"path": str(path), "sha256": digest(path)}
                                     for name, path in REPORTS.items()},
        "visual_inspection": "Rendered all pages; inspect qa_render/contact_sheet.png and page images",
    }
    (ROOT / "BUILD_QA.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "pages": len(pdf), "citations": len(cited),
                      "page_audits": page_audits, "render": str(renders / "contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
