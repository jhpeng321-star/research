"""Render existing option-ranking CSV only; no forecasts, fitting or new intervals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from build_figures import ROOT, guarded

SOURCE = "E:/agent-work/c4-bdrank/docs/paper_results/bdrank/option_concordance.csv"
SNAPSHOT = "inputs/F7_option_concordance_source.csv"
MANIFEST = "inputs/F7_source_manifest.json"
STEM = "figures/F7_option_delay_ranking"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze():
    data = guarded(Path(SOURCE)).read_bytes()
    source = {"path": SOURCE, "sha256": sha(data), "snapshot": SNAPSHOT}
    for name, payload in ((SNAPSHOT, data),
                          (MANIFEST, (json.dumps(source, indent=2) + "\n").encode())):
        path = guarded(ROOT / name, output=True)
        if path.exists():
            assert path.read_bytes() == payload, "Immutable input differs"
        else:
            path.write_bytes(payload)


def extract_evening():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    data = guarded(ROOT / manifest["snapshot"]).read_bytes()
    assert sha(data) == manifest["sha256"]
    rows = []
    for line, row in enumerate(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))), 2):
        if (row["level"] == "pooled" and row["signal"] == "mean_delay"
                and row["target"] == "mean_delay" and row["region"] == "upstream"):
            rows.append({**row, "source_path": manifest["path"],
                         "source_sha256": manifest["sha256"], "source_line": line,
                         "transform": "identity; display concordance as percent"})
    assert {(r["model"], r["horizon_s"]) for r in rows} == {
        (m, h) for m in ("FULL-v4", "FULL-v5") for h in ("120", "300")}
    assert len(rows) == 4
    for row in rows:
        right, wrong = int(row["concordant"]), int(row["discordant"])
        assert right + wrong == 58 and row["worlds"] == "11"
        assert abs(float(row["concordance"]) - right / (right + wrong)) < 1e-14
        assert row["ci_low"] == row["ci_high"] == "NA"
        assert row["predicted_only_ties"] == "0"
    return rows, manifest


def render(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for model, offset, color in (("FULL-v4", -.18, "#226C91"),
                                  ("FULL-v5", .18, "#BF6737")):
        cells = [next(r for r in rows if r["model"] == model and r["horizon_s"] == h)
                 for h in ("120", "300")]
        values = [100 * float(r["concordance"]) for r in cells]
        bars = ax.bar([i + offset for i in range(2)], values, .34, label=model, color=color)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=4)
    ax.set(ylim=(0, 100), xticks=[0, 1], xticklabels=["H120", "H300"],
           ylabel="Available-option delay-pair concordance (%)",
           title="Cached option ranking: diagnostic, not learned efficacy")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.12, .035, "58 untied pairs / 11 contributing worlds per bar; no valid cell CIs.\n"
             "Roster-only, source-prior, cold-memory diagnostic; not C_G fidelity.", fontsize=9)
    fig.subplots_adjust(bottom=.25, top=.88, left=.12, right=.98)
    for suffix in ("png", "pdf"):
        fig.savefig(guarded(ROOT / f"{STEM}.{suffix}", output=True), dpi=300)
    plt.close(fig)
    with guarded(ROOT / f"{STEM}.csv", output=True).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_evening(rows, manifest):
    from PIL import Image
    with (ROOT / f"{STEM}.csv").open(encoding="utf-8", newline="") as handle:
        actual = list(csv.DictReader(handle))
    assert actual == [{k: str(v) for k, v in row.items()} for row in rows]
    with Image.open(ROOT / f"{STEM}.png") as image:
        image.verify()
    assert (ROOT / f"{STEM}.pdf").read_bytes().startswith(b"%PDF")
    names = [SNAPSHOT, MANIFEST] + [f"{STEM}.{ext}" for ext in ("csv", "png", "pdf")]
    return {"status": "PASS", "source_cells": len(rows), "source": manifest,
            "scope": "exact CSV cells, concordance count identity, missing CIs and image integrity; no new statistics",
            "artifacts": {name: {"sha256": sha((ROOT / name).read_bytes())} for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-inputs", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.freeze_inputs:
        freeze()
    rows, manifest = extract_evening()
    if not args.check:
        render(rows)
    result = check_evening(rows, manifest)
    (ROOT / "evening_figure_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
