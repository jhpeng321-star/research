"""P4 end-to-end dry run: wms_eval predictions -> delay -> decision vs SUMO.

TRAIN only. Reads the frozen recorder runs through the read-only ``_ext``
junction, one trajectory file at a time, keeping memory bounded. Writes the
release under ``_out`` and drives ``python -m wms_decide`` once per plugin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_out"
sys.path.insert(0, str(ROOT))

import wms_decide  # noqa: E402

TAUS = (30.0, 60.0, 120.0, 300.0)
MODELS = ("keep_speed", "pure_physics")


def _resolve(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path


def build_release() -> Path:
    states = json.loads((OUT / "states_train.json").read_text(encoding="utf-8"))
    assert len(states) == 20, f"expected 20 TRAIN states, got {len(states)}"
    rows = []
    for record in states:
        assert record["split"] == "TRAIN", record["world"]
        base = _resolve(record["base_run_dir"])
        manifest_path = base / "saved_state" / "STATE_MANIFEST.json"
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        free = float(record["v0_mps"])
        t0 = float(record["t0_s"])
        truth = {}
        for arm in ("native", "pi"):
            path = _resolve(record["recorded_paths"][arm])
            truth[arm] = wms_decide.recorded_delay_seconds(
                path, t0_s=t0, taus_s=TAUS, free_speed_mps=free)
        rows.append({
            "world": record["world"], "split": "TRAIN", "source_sha256": digest,
            "t0_s": t0, "tau_s": list(TAUS), "taus_s": list(TAUS),
            "candidates": {"native": "native", "pi": "pi"},
            "state": record, "true": truth,
        })
    release = {"schema": "wms_t7_train_release/1", "split": "TRAIN", "decisions": rows}
    out = OUT / "wms_train_release.json"
    out.write_text(json.dumps(release, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"release": str(out), "n_rows": len(rows)}))
    return out


def decide(model: str, release: Path) -> dict:
    out = OUT / f"p4_{model}"
    if not out.exists():
        started = time.perf_counter()
        command = [sys.executable, "-m", "wms_decide", "--wms-eval-model", model,
                   "--release-spec", str(release), "--out", str(out)]
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"wms_decide {model} failed ({proc.returncode}): "
                             f"{proc.stdout}{proc.stderr}")
        print(json.dumps({"model": model, "elapsed_s": time.perf_counter() - started}))
    return json.loads((out / "DECISIONS.json").read_text(encoding="utf-8"))


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def report(models: tuple[str, ...]) -> None:
    results = {}
    for model in models:
        path = OUT / f"p4_{model}" / "DECISIONS.json"
        if path.is_file():
            results[model] = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "# P4 E2E dry run — TRAIN decision pipeline",
        "Status: COMPLETED (CPU inference; no SUMO/GPU/SSH).",
        "",
        "`_out/run_p4_e2e.py` streams 20 TRAIN states (HORIZON_ERROR meta.states)",
        "recorded SUMO runs from `_ext/campaign_chunk4` (READ-ONLY) into",
        "`_out/wms_train_release.json`, then runs `python -m wms_decide",
        "--wms-eval-model <spec>` to score each plugin's predicted states.",
        "Delay = sum(dt * clip(1 - v/v_free, 0, 1)) seconds; both arms share the",
        "same start state, seed and random draws. Recorded truth uses the same",
        "normalized delay on the recorded native/PI runs.",
        "",
        "| plugin | tau_s | sign_agr | rank_corr | regret_s | always_native | improve_s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model, result in results.items():
        summary = result.get("decision_summary", {})
        for tau in ("30.0", "60.0", "120.0", "300.0"):
            cell = summary.get(tau)
            if not cell:
                continue
            lines.append(
                f"| {model} | {tau} | {_fmt(cell['sign_agreement'])} | "
                f"{_fmt(cell['rank_correlation'])} | {_fmt(cell['regret_s'],1)} | "
                f"{_fmt(cell['always_native_regret_s'],1)} | "
                f"{_fmt(cell['regret_improvement_vs_native_s'],1)} |")
    lines += [
        "",
        "sign_agr = agreement of sign(predicted PI-native) with recorded;",
        "rank_corr = Spearman over the 20 states; regret is the recorded delay of",
        "the arm chosen from predicted differences; improve = always_native - regret",
        "(negative means always-native was better). Predictions are the physics /",
        "keep-speed baselines, not the trained WM-S checkpoint (P4 gate is the",
        "paired-difference decision, validated here end-to-end).",
        "Artifacts: `_out/p4_<plugin>/DECISIONS.json` (per-state + `decision_summary`).",
    ]
    (OUT / "P4_E2E.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "P4_E2E.md")}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="run_p4_e2e")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    models = tuple(args.models.split(","))
    if args.report_only:
        report(models)
        return 0
    release = OUT / "wms_train_release.json"
    if not release.is_file():
        release = build_release()
    else:
        print(json.dumps({"release": str(release), "reused": True}))
    if args.build_only:
        return 0
    for model in models:
        decide(model, release)
    report(models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
