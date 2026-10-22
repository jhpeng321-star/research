"""Paired TRAIN support/motion diagnosis, not a model-selection/audit fit."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from tools import merge_fit_stage2a as fitting  # noqa: E402
from tools import merge_fit_v2 as legacy  # noqa: E402
from tools import r9_exact as exact  # noqa: E402
from tools import r9_fit_exact as heads  # noqa: E402
from tools import r9_runtime as runtime  # noqa: E402
from tools import r9_support_audit as audit  # noqa: E402
from wms.data.c4_episodes import EpisodeArrays  # noqa: E402
from wms.model.merge_pivot import log_event_mass  # noqa: E402


def episode_job(job):
    ep, checkpoint, output = job
    torch.set_num_threads(1)
    frozen = exact.load_frozen(checkpoint)
    loaded = runtime.load_scene(ep["path"])
    cache = runtime.cached_prepare(frozen, loaded, ROOT / "results/r9_runtime_cache")
    model = runtime.runtime_model(frozen)
    # The authorizing/output row, rather than the input row, determines exposure.
    start = max(0., float(ep.get("warmup_s", 0)) - model.physics.dt_s)
    rows, counts = audit.audit_episode(model, EpisodeArrays(ep["path"]), ep, ep["source_name"],
                                        start_s=start, geometry_override=cache["geometry"])
    index = 0

    def proposal(**kwargs):
        nonlocal index
        value = cache["proposals"][index].numpy().copy()
        value[:, 1:4] = loaded["raw"][index, :, 1:4]
        index += 1
        return value

    sequences, cohort = legacy.build_dataset([ep], adapter=proposal,
                                            runtime_risk=cache["geometry"]["ramp"][0].numpy())
    heads.attach_episode(sequences, cache)
    for seq in sequences:
        entry, size = seq["entry_row"], len(seq["times"])
        seq["score_start"] = int((loaded["time_s"][entry + 1:entry + size + 1]
                                   < float(ep.get("warmup_s", 0))).sum())
    initial = {key: float(value.detach()) for key, value in frozen.core.params().items()}
    initial["sigma"] = float(frozen.core.alpha_scale())
    laws = {kind: fitting.make_model(kind, initial, frozen) for kind in ("P1-P3", "F1", "M0")}
    runtime_c = model.merge_pivot(cache["geometry"])
    law_counts, by_id = {kind: Counter() for kind in laws}, {}
    with torch.no_grad():
        for subset in legacy.sequence_batches(sequences, 32):
            batch = fitting.pack(subset)
            distributions = {kind: law.distribution(batch, 32) for kind, law in laws.items()}
            for i, seq in enumerate(subset):
                if not seq["event"]:
                    continue
                identity = (seq["vehicle"], seq["event_row"])
                physical = bool(seq["physical"][-1])
                by_id[identity] = dict(fitter_supported=physical)
                c = runtime_c[0, seq["entry_row"]:seq["event_row"], seq["slot"]]
                c = torch.where(torch.tensor(seq["lifecycle_active"]), c, -torch.inf)
                previous = c[:-1].amax() if len(c) > 1 else c.new_tensor(-torch.inf)
                logmass = log_event_mass(previous, c.amax(), model.merge_pivot.params()["tau"])
                by_id[identity]["runtime_strict_fullpath_mass"] = (
                    "zero" if torch.isneginf(logmass) else "positive")
                by_id[identity]["absorbed_fullpath"] = seq["absorbed_before_event"]
                for kind, dist in distributions.items():
                    value = dist["log_mass"][i, len(seq["times"]) - 1]
                    state = ("zero" if torch.isneginf(value) else "unidentified_history"
                             if torch.isnan(value) else "positive")
                    law_counts[kind][state] += 1
                    by_id[identity][kind + "_mass"] = state
    mismatch = []
    causes = Counter()
    for row in rows:
        identity = (row["vehicle_id"], int(row["event_id"].rsplit("/", 1)[-1]))
        law = by_id.get(identity, {})
        row.update(law)
        tick = identity[1] - 1
        row["raw_scene_unknown"] = bool((loaded["present"][tick]
                                          & (loaded["raw"][tick, :, 1:4].sum(-1) == 0)).any())
        if not row["is_cav"]:
            if not law or law["fitter_supported"] != (row["support_verdict"] == "supported"):
                mismatch.append(row["event_id"])
            if row["support_verdict"] == "vetoed":
                causes[json.loads(row["geometry_json"])["veto_cause"]] += 1
    counts["hdv_events_raw_scene_unknown"] = sum(r["raw_scene_unknown"] for r in rows if not r["is_cav"])
    counts["runtime_strict_fullpath_zero"] = sum(r.get("runtime_strict_fullpath_mass") == "zero"
                                                for r in rows if not r["is_cav"])
    counts["strict_numeric_mass_disagreements"] = sum(
        r.get("runtime_strict_fullpath_mass") != r.get("P1-P3_mass") for r in rows if not r["is_cav"])
    counts["strict_window_vs_fullpath_mass_disagreements"] = sum(
        r["zero_mass"] != (r.get("runtime_strict_fullpath_mass") == "zero")
        for r in rows if not r["is_cav"])
    report = dict(episode=ep["episode"], source=ep["source_name"], counts=dict(counts),
                  cohort=cohort, laws={k: dict(v) for k, v in law_counts.items()},
                  causes=dict(causes), mismatched_events=mismatch, metadata=cache["metadata"])
    directory = exact.guard(Path(output) / ep["source_name"] / ep["episode"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(audit.clean(report), indent=2), encoding="utf-8")
    (directory / "events.json").write_text(json.dumps(audit.clean(rows)), encoding="utf-8")
    print(json.dumps(dict(episode=ep["episode"], counts=dict(counts), causes=dict(causes),
                          laws=report["laws"], mismatches=len(mismatch))), flush=True)
    return rows, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--checkpoint", default=str(exact.DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default="results/r9_support")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args(argv)
    output = exact.guard(args.output)
    episodes, sources = legacy.discover(args.source)
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(ep, str(exact.guard(args.checkpoint)), str(output)) for ep in episodes]
    rows, reports = [], []
    if args.workers == 1:
        results = map(episode_job, jobs)
        for found, report in results:
            rows.extend(found)
            reports.append(report)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for found, report in pool.map(episode_job, jobs):
                rows.extend(found)
                reports.append(report)
    counts, causes = Counter(), Counter()
    laws = {kind: Counter() for kind in ("P1-P3", "F1", "M0")}
    for report in reports:
        counts.update(report["counts"])
        causes.update(report["causes"])
        for kind in laws:
            laws[kind].update(report["laws"][kind])
    summary = dict(episodes=len(reports), counts=dict(counts), causes=dict(causes),
                   laws={k: dict(v) for k, v in laws.items()},
                   mismatched_events=[e for r in reports for e in r["mismatched_events"]],
                   checkpoint=reports[0]["metadata"], sources=sources,
                   scope="all materialized TRAIN episodes; physical diagnostics only; no fit/selection or audit ledger consumption")
    (output / "summary.json").write_text(json.dumps(audit.clean(summary), indent=2), encoding="utf-8")
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(episodes=len(reports), counts=dict(counts), causes=dict(causes),
                          laws=summary["laws"], mismatches=len(summary["mismatched_events"]))))


if __name__ == "__main__":
    main()
