"""Build owner-review artifacts from the manuscript, without scientific execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
POOL = ROOT.parent / "paperB_c1_worldmodel"
DESKTOP = Path("C:/Users/jpeng/Desktop")
ASSETS = (
    "fig_architecture", "fig_wall_teacher_forced", "fig_direction",
    "fig_closed_loop", "fig_regret", "fig_switch_headroom",
)


def prepare() -> None:
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited.update(group.split(","))
    pool = (POOL / "refs.bib").read_text(encoding="utf-8")
    entries = re.findall(r"(?ms)^@\w+\{([^,]+),.*?^\}", pool)
    blocks = re.findall(r"(?ms)^@\w+\{[^,]+,.*?^\}", pool)
    assert cited <= set(entries), f"Unverified citation: {cited - set(entries)}"
    selected = [block for key, block in zip(entries, blocks) if key in cited]
    (ROOT / "refs.bib").write_text(
        "% Cited entries copied verbatim from the verified source pool.\n"
        "% Verification/scope: ../paperB_c1_worldmodel/RELATED_WORK.md\n\n"
        + "\n\n".join(selected) + "\n", encoding="utf-8",
    )
    assert ROOT.is_dir() and POOL.is_dir()
    (ROOT / "figures").mkdir(exist_ok=True)
    for name in ASSETS:
        for suffix in (".png", ".pdf"):
            source = POOL / "figures" / (name + suffix)
            assert source.is_file(), source
            shutil.copy2(source, ROOT / "figures" / source.name)
    print(f"Prepared {len(selected)} verified references and {2 * len(ASSETS)} optional artwork files.")


def unwrap(text: str, command: str, replacement: str = r"\1") -> str:
    pattern = r"\\" + command + r"\{([^{}]*)\}"
    while re.search(pattern, text):
        text = re.sub(pattern, replacement, text)
    return text


def readable(text: str) -> str:
    literals = re.findall(r"<!--.*?-->|\$\$.*?\$\$|\$[^$]*\$", text, flags=re.DOTALL)
    for index, literal in enumerate(literals):
        text = text.replace(literal, f"LITERALTOKEN{index}END", 1)
    text = text.replace("\\%", "%").replace("\\&", "&")
    for command in ("emph", "textit", "textbf", "mathrm", "mathbb", "text", "url"):
        text = unwrap(text, command)
    text = unwrap(text, "pending", r"[PENDING: \1]")
    text = re.sub(r"\\textcolor\{red\}\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\,", " ").replace("~", " ")
    text = text.replace("\\tfrac12", "1/2").replace("\\hbox", "")
    text = text.replace("\\newblock", " ").replace("\\BIBforeignlanguage{English}", "")
    text = text.replace("\\BIBentryALTinterwordspacing", "")
    text = text.replace("\\BIBentrySTDinterwordspacing", "")
    text = text.replace("\\relax", "").replace("\\mbox", "")
    text = text.replace("\\enquote", "").replace("\\IEEEauthorrefmark", "")
    text = text.replace("\\allowbreak", "").replace("\\penalty0", "")
    text = text.replace("\\hskip 0pt", "")
    text = text.replace("---", "—")
    text = re.sub(r"(?<!\[)--(?!\])", "–", text)
    text = text.replace("``", '"').replace("''", '"')
    for index, literal in enumerate(literals):
        text = text.replace(f"LITERALTOKEN{index}END", literal)
    return text


def markdown() -> str:
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    aux = (ROOT / "main.aux").read_text(encoding="utf-8")
    citations = dict(re.findall(r"\\bibcite\{([^}]+)\}\{([^}]+)\}", aux))
    title = re.search(r"\\title\{([^}]+)\}", tex).group(1)
    text = tex.split("\\maketitle", 1)[1].split("\\bibliographystyle", 1)[0]
    # Empirical source comments survive in the companion drafting text.
    text = re.sub(r"(?m) ?% Source: (.*)$", r"\n<!-- Source: \1 -->", text)
    text = re.sub(r"(?m)^%.*$", "", text)
    text = text.replace("\\begin{abstract}", "## Abstract\n")
    text = text.replace("\\end{abstract}", "")
    text = text.replace("\\begin{IEEEkeywords}", "## Index Terms\n")
    text = text.replace("\\end{IEEEkeywords}", "")
    section_index = 0
    roman = ("I", "II", "III", "IV", "V", "VI", "VII")

    def section(match: re.Match) -> str:
        nonlocal section_index
        heading = f"## {roman[section_index]}. {match.group(1)}"
        section_index += 1
        return heading

    text = re.sub(r"\\section\{([^}]+)\}", section, text)
    text = re.sub(r"\\section\*\{([^}]+)\}", r"## \1", text)
    text = re.sub(r"\\subsection\{([^}]+)\}", r"### \1", text)
    text = re.sub(
        r"\\cite\{([^}]+)\}",
        lambda m: "[" + ", ".join(citations[key] for key in m.group(1).split(",")) + "]",
        text,
    )
    text = text.replace("Table~\\ref{tab:pending}", "Table I")
    figure_index = 0

    def figure(match: re.Match) -> str:
        nonlocal figure_index
        figure_index += 1
        block = match.group(0)
        box = re.search(r"\\graphbox\{([^}]+)\}", block).group(1)
        caption = re.search(r"\\caption\{([^}]+)\}", block).group(1)
        return f"\n[FIGURE: {box}]\n\n**Figure {figure_index}.** {caption}\n"

    text = re.sub(r"(?s)\\begin\{figure\}.*?\\end\{figure\}", figure, text)

    def table(match: re.Match) -> str:
        block = match.group(0)
        caption = re.search(r"\\caption\{([^}]+)\}", block).group(1)
        return (
            f"\n**Table I.** {caption}\n\n"
            "Columns: selector; fleet/source cost; branch regret.\n\n"
            "- WM selector: [--]; [--].\n"
            "- BB-traj selector: [--]; [--].\n"
            "- Fleet PPO picker: [--]; [--].\n"
            "- Per-CAV MAPPO picker: [--]; [--].\n"
        )

    text = re.sub(r"(?s)\\begin\{table\}.*?\\end\{table\}", table, text)
    text = text.replace("\\begin{itemize}", "").replace("\\end{itemize}", "")
    text = text.replace("\\item ", "- ")
    text = text.replace("\\begin{equation}", "$$\n")
    text = text.replace("\\end{equation}", "\n$$")
    text = text.replace("\\begin{align}", "$$\n\\begin{aligned}")
    text = text.replace("\\end{align}", "\\end{aligned}\n$$")
    text = readable(text)
    # Reference text comes from the actual compiled BibTeX output, not new metadata.
    bbl = (ROOT / "main.bbl").read_text(encoding="utf-8")
    references = []
    for match in re.finditer(r"(?s)\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", bbl):
        key, body = match.groups()
        body = re.sub(r"(?m)%.*$", "", body)
        body = readable(body)
        body = body.replace(r'{\"a}', "ä")
        body = body.replace("{", "").replace("}", "")
        body = re.sub(r"\\BIBentry.*?\n", "", body)
        body = re.sub(r"\s+", " ", body).strip()
        references.append(f"[{citations[key]}] {body}")
    assert len(references) == len(citations), "Reference conversion incomplete"
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (
        f"# {title}\n\n"
        "Author names to be supplied\n\nAffiliations and contact details to be supplied\n\n"
        + text + "\n\n## References\n\n" + "\n\n".join(references) + "\n"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deliver() -> None:
    import pymupdf as fitz
    from pypdf import PdfReader

    pdf = ROOT / "main.pdf"
    reader = PdfReader(pdf)
    pages = len(reader.pages)
    assert 0 < pages <= 6, f"Owner inclusive page limit exceeded: {pages}"
    log = (ROOT / "main.log").read_text(encoding="utf-8", errors="replace")
    forbidden = (
        "There were undefined references", "There were undefined citations",
        "Rerun to get cross-references right", "LaTeX Error:", "Emergency stop",
    )
    assert not any(item in log for item in forbidden), "Unresolved build error/reference"
    assert not re.search(r"LaTeX Warning: (?:Citation|Reference).*undefined", log)
    text = markdown()
    assert "\\pending" not in text and "\\caption" not in text
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    assert text.count("<!-- Source:") == tex.count("% Source:")
    assert re.findall(r"<!-- Source: (.*?) -->", text) == re.findall(r"% Source: (.*)", tex)
    (ROOT / "DRAFT_TEXT.md").write_text(text, encoding="utf-8")
    assert DESKTOP.is_dir(), DESKTOP
    desktop_pdf = DESKTOP / "PaperB_draft_v1.pdf"
    shutil.copy2(pdf, desktop_pdf)
    assert sha256(pdf) == sha256(desktop_pdf)
    # Page renders are local review artifacts, not upload dependencies.
    (ROOT / "_review").mkdir(exist_ok=True)
    doc = fitz.open(pdf)
    for index, page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(ROOT / "_review" / f"page_{index + 1}.png")
    included = [ROOT / name for name in (
        "main.tex", "refs.bib", "main.pdf", "DRAFT_TEXT.md", "README_OVERLEAF.md",
        "DRAFT_RECEIPT.md", "REGISTRATION_RECEIPT.md", "build_delivery.py",
    )]
    included += sorted((ROOT / "figures").glob("*"))
    report = {
        "page_count_including_references": pages,
        "class": "IEEEtran conference",
        "compiler": "MiKTeX pdfLaTeX + BibTeX + pdfLaTeX + pdfLaTeX",
        "graphs": "intentionally empty per final owner clarification",
        "desktop_pdf": str(desktop_pdf),
        "desktop_matches_project_pdf": True,
        "overfull_boxes": re.findall(r"Overfull .*", log),
        "underfull_box_count": len(re.findall(r"Underfull ", log)),
        "unresolved_citations_or_references": False,
        "numerical_source_comments_preserved_exactly": text.count("<!-- Source:"),
        "page_size_points": [float(reader.pages[0].mediabox.width), float(reader.pages[0].mediabox.height)],
        "font_types": sorted({font[2] for page in doc for font in page.get_fonts()}),
        "bibliography_keys": re.findall(r"\\bibitem\{([^}]+)\}", (ROOT / "main.bbl").read_text(encoding="utf-8")),
        "page_text_characters": [len(p.extract_text() or "") for p in reader.pages],
        "files_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in included},
    }
    (ROOT / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    included.append(ROOT / "BUILD_REPORT.json")
    archive = ROOT / "paperB_overleaf.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in included:
            bundle.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for path in included:
            assert bundle.read(str(path.relative_to(ROOT)).replace("\\", "/")) == path.read_bytes()
    print(f"DELIVERED: {pages} pages including references; {len(included)} ZIP members; Desktop PDF hash verified.")
    print(f"Overfull boxes: {len(report['overfull_boxes'])}; unresolved references/citations: none.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "text", "deliver"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "text":
        (ROOT / "DRAFT_TEXT.md").write_text(markdown(), encoding="utf-8")
        print("Generated matching DRAFT_TEXT.md.")
    else:
        deliver()
