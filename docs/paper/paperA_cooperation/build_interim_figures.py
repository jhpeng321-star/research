"""Copy published fixed-u8 contrasts; no experiment, inference or new intervals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re

from build_figures import ROOT, guarded

SOURCE = "E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM.md"
SNAPSHOT = "source_snapshots/coopmwm_report_625a173410be93db.md"
SOURCE_SHA = "625a173410be93db09dad75b50f85a8bd614f268f9a81c658446c2f925af799d"
STEM = "figures/F3I_fixed_u8_interim"
NUMBER = r"([+-]?\d+\.\d+)"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract_interim():
    data = guarded(ROOT / SNAPSHOT).read_bytes()
    assert sha(data) == SOURCE_SHA
    text = data.decode("utf-8")
    start = text.index("## Nov3 13:47:56 UTC+8")
    section = text[start:text.index("\n## ", start + 1)].replace("**", "")
    rows = []
    for comparator in ("MAPPO", "A", "R", "native"):
        pattern = (rf"- B-{comparator}: J600\s*{NUMBER}\[{NUMBER},{NUMBER}\][\s\S]*?J900\s*"
                   rf"{NUMBER}\[{NUMBER},{NUMBER}\]")
        match = re.search(pattern, section)
        assert match, comparator
        values = match.groups()
        for offset, horizon in ((0, 600), (3, 900)):
            mean, low, high = map(float, values[offset:offset + 3])
            assert low <= mean <= high
            rows.append({"contrast": f"B-{comparator}", "horizon_s": horizon,
                         "mean": mean, "ci_low": low, "ci_high": high,
                         "units": "veh-s/world", "scope": "all", "worlds": 96,
                         "checkpoint": "fixed u8; interim episode only",
                         "uncertainty": "95% paired whole-world bootstrap; 10000 draws; RNG20261101",
                         "source_path": SOURCE, "source_snapshot": SNAPSHOT,
                         "source_sha256": SOURCE_SHA,
                         "source_locator": f"Nov3 13:47:56; B-{comparator}; J{horizon}",
                         "transform": "identity from published report precision; negative better"})
    assert len(rows) == 8
    return rows


def render(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    for ax, horizon in zip(axes, (600, 900)):
        cells = [r for r in rows if r["horizon_s"] == horizon]
        for index, row in enumerate(cells):
            ax.errorbar(row["mean"], index,
                        xerr=[[row["mean"] - row["ci_low"]],
                              [row["ci_high"] - row["mean"]]],
                        fmt="o", capsize=3, color="#226C91")
        ax.axvline(0, color="#777777", lw=1, linestyle="--")
        ax.set(title=f"J{horizon}", xlabel="B minus comparator (veh-s/world)",
               yticks=range(4), yticklabels=[r["contrast"] for r in cells],
               xlim=(-4200, 900), ylim=(3.6, -.6))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("Fixed-u8 interim: MAPPO / A / R superiority unresolved", fontsize=11)
    fig.text(.08, .04, "96 paired worlds; published 95% world-bootstrap intervals. Negative = B better.\n"
             "Episode interim, not final branch tests or measured group cooperation.", fontsize=9)
    fig.subplots_adjust(left=.14, right=.98, bottom=.26, top=.81, wspace=.18)
    for suffix in ("png", "pdf"):
        fig.savefig(guarded(ROOT / f"{STEM}.{suffix}", output=True), dpi=300)
    plt.close(fig)
    with guarded(ROOT / f"{STEM}.csv", output=True).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_interim(rows):
    from PIL import Image
    with guarded(ROOT / f"{STEM}.csv").open(encoding="utf-8", newline="") as handle:
        actual = list(csv.DictReader(handle))
    assert actual == [{k: str(v) for k, v in row.items()} for row in rows]
    for row in rows:
        resolved = row["ci_high"] < 0
        assert resolved == (row["contrast"] == "B-native")
    with Image.open(guarded(ROOT / f"{STEM}.png")) as image:
        image.verify()
    assert guarded(ROOT / f"{STEM}.pdf").read_bytes().startswith(b"%PDF")
    names = [SNAPSHOT] + [f"{STEM}.{ext}" for ext in ("csv", "png", "pdf")]
    return {"status": "PASS", "source_cells": len(rows),
            "scope": "source-hash/contrast/CI identities and image integrity; no new statistics",
            "artifacts": {name: {"sha256": sha(guarded(ROOT / name).read_bytes())} for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = extract_interim()
    if not args.check:
        render(rows)
    result = check_interim(rows)
    guarded(ROOT / "interim_figure_validation.json", output=True).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
