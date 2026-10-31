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
    for name in ["README.md", "OUTLINE.md", "CLAIMS.md", "FIGURES.md", "RELATED_WORK.md", "THREATS.md", "refs.bib", "REFRESH.md", "SOURCE_REFRESH.json", "REGISTRATION_RECEIPT.md"]:
        assert (ROOT / name).stat().st_size > 0, name
    checks.append("required deliverables nonempty")

    manifest = json.loads((DATA / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in manifest.items():
        blob = (DATA / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == info["sha256"], name
        assert len(blob) == info["bytes"], name
    checks.append("frozen source byte/hash integrity")

    evening = json.loads((DATA / "EVENING_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in evening.items():
        blob = (DATA / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == info["sha256"], name
        assert len(blob) == info["bytes"], name
    checks.append("evening source snapshots retain exact producer bytes")

    overnight = json.loads((DATA / "OVERNIGHT_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in overnight.items():
        blob = (DATA / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == info["sha256"], name
        assert len(blob) == info["bytes"], name
    checks.append("overnight completed-report/export snapshots retain exact producer bytes")

    fields = ("model", "plant", "arm", "horizon_s", "metric")
    closed_raw = {tuple(r[k] for k in fields): r for r in csv_rows("closed.source.csv")}
    closed = csv_rows("closed_loop.csv")
    assert len(closed) == 28
    assert len({tuple(r[k] for k in fields) for r in closed}) == 28
    for row in closed:
        original = closed_raw[tuple(row[k] for k in fields)]
        assert all(row[k] == v for k, v in original.items())
        assert row["model"] == "wm2-minus-bbtraj" and row["plant"] == "common"
        assert row["arm"] in ("pi", "mappo") and row["metric"] in ("position", "J_ae")
        if row["metric"] == "position":
            expected_n = {10:40, 30:40, 60:40, 90:40, 120:33, 180:6 if row["arm"] == "pi" else 5, 300:0}
            assert int(row["n_worlds"]) == expected_n[int(row["horizon_s"])]
            if row["horizon_s"] == "300":
                assert row["status"] == "no_surviving_ids"
                assert all(row[k] == "" for k in ("estimate", "lo", "hi"))
            if row["horizon_s"] == "60":
                assert float(row["lo"]) < 0 < float(row["hi"])
                assert round(float(row["estimate"]), 2) == (-5.17 if row["arm"] == "pi" else -3.45)
        else:
            assert row["n_worlds"] == "40"
    checks.append("completed feedback rows equal producer CSV; pairing, inconclusive primary contrasts and survivor missingness preserved")

    fields = ("split", "target", "horizon_s", "clock", "arm")
    distribution_raw = {tuple(r[k] for k in fields): r for r in csv_rows("distribution.source.csv")}
    distribution = csv_rows("distribution.csv")
    assert len(distribution) == 6
    for row in distribution:
        assert row["split"] == "reserved8" and row["clock"] == "team" and row["arm"] == "pooled"
        original = distribution_raw[tuple(row[k] for k in fields)]
        assert all(row[k] == v for k, v in original.items())
    for target, expected_wm, expected_baseline, key in [
        ("absolute", [1522,7129,25523], [763,3584,8267], "cell_arm_climatology_crps"),
        ("effect", [390,2420,6906], [171,968,3286], "climatology_crps")]:
        selected = sorted([r for r in distribution if r["target"] == target], key=lambda r: int(r["horizon_s"]))
        assert [round(float(r["wm_crps"])) for r in selected] == expected_wm
        assert [round(float(r[key])) for r in selected] == expected_baseline
    late = next(r for r in distribution if r["target"] == "absolute" and r["horizon_s"] == "600")
    assert float(late["minmax_coverage"]) == .125 and float(late["below_all"]) == .875
    checks.append("distribution figures preserve separate absolute/effect baseline columns and previously exposed evaluation split")

    value = csv_rows("value_tail.csv")
    assert len(value) == 12
    for metric, expected in [
        ("sign_accuracy", [(47.25,38,56.25),(43.25,35.25,51.5),(58.75,47.75,67.75),(72.25,56,86)]),
        ("mae", [(8877.72,4590.41,14306.82),(10117.34,5832.41,14984.31),
                 (4460,3161,6183.67),(3038.93,1725.06,4769.29)])]:
        assert [tuple(float(r[k]) for k in ("estimate", "lo", "hi")) for r in value if r["metric"] == metric] == expected
    assert all(not r["method"].startswith("A:") for r in value)
    value_report = (DATA / "value.source.md").read_text(encoding="utf-8-sig")
    assert "**VERDICT: FAIL**" in value_report and "| A: WM to900 | NA | NA | NA |" in value_report
    checks.append("value method cells/CIs transcribed exactly; decisive necessary-condition FAIL retains long-WM A as NA")

    pi = csv_rows("pi_diagnosis.csv")
    assert [(int(r["horizon_s"]), float(r["estimate"]), float(r["lo"]), float(r["hi"])) for r in pi] == [
        (60,17.85,3.24,31.27), (120,-47.82,-95.20,-27.75),
        (60,-45.05,-52.73,-25.25), (120,-144.74,-250.11,-36.22),
        (60,-13.47,-26.92,-1.49)]
    assert len(set(r["panel"] for r in pi)) == 3
    checks.append("PI source contrasts checked; original, shifted and within-panel history effects stay separate")

    calib = csv_rows("calibration.csv")
    assert len(calib) == 18
    for split, expected in [("Grouped CV", [263,185,174,1719,999,865,6013,3298,3152]),
                            ("SPEEDHEAD transfer", [176,198,212,1399,1568,1783,4701,4073,4754])]:
        assert [float(r["mae_veh_s"]) for r in calib if r["split"] == split] == expected
    for row in calib:
        assert 0 <= float(row["sign_accuracy"]) <= 1
        assert -1 <= float(row["spearman"]) <= 1
    checks.append("all plotted calibration MAEs equal complete report cells; splits and selected variants retained")

    headroom = csv_rows("switch_headroom.csv")
    for h, expected in [(600,[22,17,19,29,3,0]), (60,[2,9,3,0,0,2]), (90,[2,7,1,3,1,2])]:
        selected = [r for r in headroom if int(r["horizon_s"]) == h]
        assert [int(r["count"]) for r in selected] == expected
        assert sum(int(r["count"]) for r in selected) == int(selected[0]["denominator"])
    # Check winner labels independently against all candidate costs, including ties.
    for source, clock, suffix in [("recorded_winners.source.csv", "clock_s", "team_delay_vs"),
                                  ("step0b.source.csv", "horizon_s", "cost_vs")]:
        for row in csv_rows(source):
            costs = {c:float(row[f"{c}_{suffix}"]) for c in ["native","ctrl0","mappo","pi","circles"]}
            best = min(costs.values())
            winners = {c for c,v in costs.items() if v-best <= 1e-6}
            actual = row["winner" if clock == "clock_s" else "winners"]
            assert set(actual.split("|")) == winners, (source,row["seed"],row[clock])
    checks.append("recorded/state winner counts and tie handling checked against original five-way costs")

    floor = csv_rows("floor.source.csv")
    for metric in ["position", "speed"]:
        for row in [r for r in floor if r["metric"] == metric]:
            if int(row["horizon_s"]) <= 120:
                assert float(row["estimate"]) == 0 and row["n_worlds"] == "6"
            if row["horizon_s"] == "300":
                assert all(row[k] == "" for k in ["estimate","lo","hi"])
                assert row["n_worlds"] == "0" and row["status"] == "no_surviving_ids"
    checks.append("pulse endpoints preserve zero short-clock medians and unavailable late survivors, never zero imputation")

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

    for name in ["direction.csv", "regret.csv", "regret_paired.csv", "regional.csv", "entry.csv", "wall_teacher_forced.csv", "architecture_edges.csv", "pi_diagnosis.csv", "calibration.csv", "switch_headroom.csv", "closed_loop.csv", "distribution.csv", "value_tail.csv"]:
        for r in csv_rows(name):
            assert r["source_file"] and r["source_selector"], name
    checks.append("every derived numerical/schematic CSV row has source and selector")

    names = ["fig_direction", "fig_regret", "fig_regional", "fig_entry", "fig_wall_teacher_forced", "fig_architecture", "fig_pi_diagnosis", "fig_calibration", "fig_switch_headroom", "fig_closed_loop", "fig_distribution", "fig_value_tail"]
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
    assert "[[PENDING:" in claims and "S-C1CLOSED" in claims and "S-C1SWITCH" in claims
    checks.append("headline numerical claim comments and explicit agent-owned pending fields")
    for phrase in ["+17.85", "−45.05", "263/1719/6013", "1362", "0.5193", "1.832371", "+508.3"]:
        line = next(line for line in claims.splitlines() if phrase in line)
        assert "<!-- Source" in line, phrase
    assert "Status: SUPPORTED for completed scoped comparison; superiority INCONCLUSIVE" in claims
    assert "completed verdict NOT USABLE" in claims
    for phrase in ["−5.17", "252.1303", "+5.102", "47.25%", "1522/7129/25523", "18/144"]:
        line = next(line for line in claims.splitlines() if phrase in line)
        assert "<!-- Source" in line, phrase
    assert "steps 20/30 were not scored" in claims and "A remains NA" in claims
    assert "scheduler/budget-selected" in claims and "C1-SWITCH2" in claims
    checks.append("new source-commented verdicts distinguish complete feedback, partial reference switching, failed value/repair and pending main switching")

    refresh = json.loads((ROOT / "SOURCE_REFRESH.json").read_text(encoding="utf-8"))
    assert refresh["sources"]["C1 gap report alternate documented delivery"]["status"] == "PRESENT"
    for label in ["C1 closed-loop authoritative report", "Final calibration"]:
        assert refresh["sources"][label]["status"] in ("PRESENT", "MISSING")
    registration = Path("E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md").read_text(encoding="utf-8-sig")
    own_lines = [line for line in registration.splitlines() if "S-PAPERB FIRST PACKAGE COMPLETE" in line]
    assert len(own_lines) == 1
    assert own_lines[0] in (ROOT / "REGISTRATION_RECEIPT.md").read_text(encoding="utf-8")
    checks.append("timestamped refresh pending statuses and append-once shared registration match committed receipt")

    receipt = {"status":"PASS", "checks":checks,
               "scope":"Package/source-selection/artwork QA only; no scientific requalification or typeset-page verification",
               "python":sys.version, "matplotlib":matplotlib.__version__,
               "source_manifest_sha256":hashlib.sha256((DATA/"SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
                "evening_manifest_sha256":hashlib.sha256((DATA/"EVENING_SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
                "overnight_manifest_sha256":hashlib.sha256((DATA/"OVERNIGHT_SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
               "figure_count":len(names), "bib_entries":len(keys)}
    (ROOT / "VERIFICATION.json").write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
