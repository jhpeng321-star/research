"""WD-01 part 2C build: Part A v0 NATIVE dataset at the 0.5 s physics step.

Zero simulation.  Streams the regenerated run logs read-only, hashes the
[60, 600] window of every run (arms of one traffic root are byte-identical
in-window), keeps ONE run per distinct window hash (traffic root), and
writes one compact npz per root plus MANIFEST.json and DATA_CARD.md.

Usage:  python build.py [--workers 8]
"""
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))         # output/wd01 (wdpaths)

import pdl as L  # noqa: E402
import wdpaths  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def worker(args):
    tag, run_dir = args
    h = L.window_hash(run_dir)
    out = L.extract_run(tag, run_dir)
    out["stats"]["window_sha256"] = h
    return out


def ttc_rate(s, v, lane):
    """Vehicle-steps with closing TTC < 3 / < 1.5 s, same-physical-lane
    leader (d-coordinate within lane; identical to s on one lane)."""
    lt3 = lt15 = 0
    for g in (-1, 0, 1, 2):
        m = lane == g
        if not m.any():
            continue
        idx = np.where(m)[0]
        order = idx[np.argsort(-s[idx], kind="stable")]  # front (large s) first
        for j in range(len(order) - 1):
            l_, f = order[j], order[j + 1]   # l_ ahead, f behind
            gap = s[l_] - s[f]
            closing = v[f] - v[l_]
            if gap > 0 and closing > 0:
                ttc = gap / closing
                lt3 += ttc < 3.0
                lt15 += ttc < 1.5
    return lt3, lt15


def main():
    workers = 8
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    workers = max(1, min(8, workers))

    ck = L.sha256_file(L.SURV_CKPT)
    assert ck == L.SURV_CKPT_SHA256, f"checkpoint sha256 {ck} != pinned"
    log(f"checkpoint guard ok ({ck[:16]}...)")

    run_map, _ = L.load_run_map()
    v3 = L.load_v3_splits()
    assert len(run_map) == 88
    mismatch = [t for t, _, s in run_map if v3.get(t) not in (None, s)]
    assert not mismatch, f"part1 map split != v3 split for {mismatch}"
    log("tag map ok: 88 runs, part1 split consistent with v3 manifest")

    log(f"extracting windows with {workers} workers ...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(worker, [(t, r) for t, r, _ in run_map],
                              chunksize=1))
    log(f"extracted {len(results)} runs")

    # ---------------- dedup by window content --------------------------
    by_hash = {}
    for out in results:
        by_hash.setdefault(out["stats"]["window_sha256"], []).append(out)
    log(f"distinct windows (roots): {len(by_hash)}")

    L.DATA.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema="wd01_partA_native_data/1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        zero_simulation=True,
        window_s=[L.WINDOW_T0, L.WINDOW_T1], dt_s=L.DT,
        checkpoint=L.SURV_CKPT, checkpoint_sha256=ck,
        physical_lanes={"0": "main_left", "1": "main_right", "2": "accel",
                        "-1": "junction/unmapped (in_junction=1)"},
        d_definition="d = OFFSET[lane] - s (merge-relative, cross-lane)",
        roots={})
    dedup_table = []
    conflicts = []
    for h, outs in sorted(by_hash.items(),
                          key=lambda kv: min(o["stats"]["tag"]
                                             for o in kv[1])):
        rep = min(o["stats"]["tag"] for o in outs)
        stems = sorted({L.tag_stem(o["stats"]["tag"]) for o in outs})
        splits = sorted({v3.get(o["stats"]["tag"], "?") for o in outs})
        if len(splits) > 1:
            conflicts.append(dict(window_sha256=h, tags=[o["stats"]["tag"]
                                                         for o in outs],
                                  splits=splits))
        root = L.group_name(stems, h)
        out = next(o for o in outs if o["stats"]["tag"] == rep)
        st = out["stats"]
        npz = out["npz"]
        path = L.DATA / f"root_{root}.npz"
        np.savez_compressed(path, **npz)
        sha = L.sha256_file(path)
        lane = npz["lane"]
        merge_lc = int(((npz["lc_from"] == 2) & (npz["lc_to"] == 1)).sum())
        bmask = npz["b_departed"] == 1
        n_bl = int((npz["b_lane"][bmask] == 0).sum())
        manifest["roots"][root] = dict(
            file=str(path), sha256=sha,
            window_sha256=h, representative_tag=rep,
            all_tags=sorted(o["stats"]["tag"] for o in outs),
            stems=stems, split=splits[0],
            n_ticks_present=st["n_ticks_present"],
            missing_ticks=st["missing_ticks"],
            n_vehicle_steps=st["n_vehicle_steps"],
            n_vehicles=st["n_vehicles"],
            n_cav_ids_observed=st["n_cav_ids"],
            manifest_cav_population=st["manifest_cav_population"],
            n_lane_changes=st["n_lane_changes"],
            n_accel_to_main_right=merge_lc,
            n_births_in_window=int(bmask.sum()),
            n_births_entrance_mainline=n_bl,
            n_births_entrance_ramp=int(bmask.sum()) - n_bl,
            n_exits_arrived=int((npz["e_arrived"] == 1).sum()),
            n_junction_steps=int((npz["in_junction"] == 1).sum()),
            resid_abs_p50_p95_p99_max=st["resid_abs_p50_p95_p99_max"],
            resid_bridge_abs_max_m=st["resid_bridge_abs_max_m"],
            n_bridge_pairs=st["n_bridge_pairs"],
            n_resid_pairs=st["n_resid"],
        )
        for o in outs:
            dedup_table.append(dict(tag=o["stats"]["tag"],
                                    root=root, split=v3.get(o["stats"]["tag"],
                                                            "?"),
                                    window_sha256=h,
                                    representative=(o["stats"]["tag"] == rep)))
        log(f"root {root:24s} split={splits[0]:7s} "
            f"runs={len(outs):2d} veh_steps={st['n_vehicle_steps']:7d} "
            f"sha={path.name} ({sha[:12]}...)")

    if conflicts:
        manifest["split_conflicts"] = conflicts
    manifest["dedup_table"] = dedup_table
    L.dump_json(HERE / "MANIFEST.json", manifest)
    L.dump_json(HERE / "results" / "dedup_table.json", dedup_table)
    log(f"MANIFEST.json written ({len(manifest['roots'])} roots); "
        f"conflicts={len(conflicts)}")

    # ---------------- DATA_CARD ----------------------------------------
    write_data_card(manifest)
    log(f"DATA_CARD.md written; total {time.time() - T0:.0f}s")
    return 0


def _root_rows(manifest):
    rows = []
    for name, r in sorted(manifest["roots"].items()):
        z = np.load(wdpaths.remap(r["file"]), allow_pickle=False)
        lane, v, a, s = z["lane"], z["v"].astype(np.float64), z["a"], z["s"]
        typ, cav = z["type_label"], z["is_cav"]
        veh_steps = len(lane)
        lt3, lt15 = 0, 0
        for tk in range(int(z["tick"].max()) + 1):
            m = z["tick"] == tk
            x3, x15 = ttc_rate(s[m], v[m], lane[m])
            lt3 += x3
            lt15 += x15
        hb = int((a < -3.0).sum())
        rows.append(dict(
            root=name, split=r["split"], vehicle_steps=veh_steps,
            vehicles=r["n_vehicles"],
            lc_total=r["n_lane_changes"],
            lc_01=int(((z["lc_from"] == 0) & (z["lc_to"] == 1)).sum()),
            lc_10=int(((z["lc_from"] == 1) & (z["lc_to"] == 0)).sum()),
            lc_21=int(((z["lc_from"] == 2) & (z["lc_to"] == 1)).sum()),
            merges_21=r["n_accel_to_main_right"],
            births=r["n_births_in_window"],
            births_mainline=r["n_births_entrance_mainline"],
            births_ramp=r["n_births_entrance_ramp"],
            exits_arrived=r["n_exits_arrived"],
            junction_steps=r["n_junction_steps"],
            hard_brake=hb,
            ttc_lt3=lt3, ttc_lt1p5=lt15,
            cav_steps=int(cav.sum()),
            type_agg=int((typ == 0).sum()), type_norm=int((typ == 1).sum()),
            type_caut=int((typ == 2).sum()), type_other=int((typ == -1).sum()),
            resid_max=r["resid_abs_p50_p95_p99_max"][3],
            resid_p99=r["resid_abs_p50_p95_p99_max"][2],
        ))
        z.close()
    return rows


def write_data_card(manifest):
    rows = _root_rows(manifest)
    with (HERE / "results" / "per_root_counts.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    def agg(rs):
        f = lambda k: sum(r[k] for r in rs)  # noqa: E731
        n = f("vehicle_steps")
        lc = f("lc_01") + f("lc_10") + f("lc_21")
        return dict(
            roots=len(rs), vehicle_steps=n, vehicles=f("vehicles"),
            lane_changes=lc,
            lc_left_to_right=f("lc_01"), lc_right_to_left=f("lc_10"),
            lc_accel_to_main=f("lc_21"),
            births=f("births"),
            births_mainline=f("births_mainline"),
            births_ramp=f("births_ramp"),
            exits_arrived=f("exits_arrived"),
            junction_steps=f("junction_steps"),
            hard_brake_steps=f("hard_brake"),
            hard_brake_per_1000=round(1000 * f("hard_brake") / max(n, 1), 3),
            ttc_lt3_steps=f("ttc_lt3"),
            ttc_lt3_per_1000=round(1000 * f("ttc_lt3") / max(n, 1), 3),
            ttc_lt1p5_steps=f("ttc_lt1p5"),
            ttc_lt1p5_per_1000=round(1000 * f("ttc_lt1p5") / max(n, 1), 3),
            cav_steps=f("cav_steps"),
            cav_frac=round(f("cav_steps") / max(n, 1), 4),
            type_aggressive=f("type_agg"), type_normal=f("type_norm"),
            type_cautious=f("type_caut"), type_unmapped=f("type_other"))

    card = dict(
        schema="wd01_partA_native_data_card/1",
        window_s=[60.0, 600.0], dt_s=0.5, n_ticks=1081,
        lane_ids={"0": "main_left", "1": "main_right", "2": "accel"},
        junction_rule="in_junction=1, lane=-1, s = junction boundary s "
                      "(internal lanes span zero physical s)",
        vType_note="type_label is SUPERVISION ONLY (ideal sensing does not "
                   "reveal vType); excluded from every input-feature "
                   "builder (enforced by test)",
        is_cav_note="derived from telemetry cav_report distinct ids (the "
                    "controller's polled fleet); coverage caveat: CAVs "
                    "alive less than one poll interval could be missed",
        overall=agg(rows),
        train=agg([r for r in rows if r["split"] == "train"]),
        holdout=agg([r for r in rows if r["split"] == "holdout"]),
        per_root=rows)
    L.dump_json(HERE / "results" / "data_card.json", card)

    o, tr, ho = card["overall"], card["train"], card["holdout"]

    def line(tag, d):
        return (f"| {tag} | {d['roots']} | {d['vehicle_steps']} | "
                f"{d['vehicles']} | {d['lane_changes']} | {d['births']} | "
                f"{d['exits_arrived']} | {d['hard_brake_per_1000']} | "
                f"{d['ttc_lt3_per_1000']} | {d['ttc_lt1p5_per_1000']} | "
                f"{d['cav_frac']} |")

    lines = [
        "# WD-01 Part A v0 NATIVE dataset - DATA_CARD", "",
        "Window [60, 600] s at the 0.5 s physics step on the WM-30 physical",
        "lanes (0 main_left, 1 main_right, 2 accel; -1 junction with",
        "in_junction=1).  Deduplicated to distinct traffic roots by window",
        "content hash; split by root (holdout iff the root's runs are",
        "holdout in the v3 build_tracks_manifest).  ZERO SIMULATION.", "",
        "vType (`type_label`: 0 aggressive, 1 normal, 2 cautious) is",
        "SUPERVISION ONLY - ideal sensing does not reveal vType - and is",
        "excluded from every input-feature builder (enforced by",
        "tests/test_inputs.py::test_vtype_not_in_features).",
        "`is_cav` is derived from telemetry cav_report distinct ids (the",
        "controller's polled fleet): CAVs alive shorter than one poll",
        "interval (~5 s) could in principle be missed; designated counts",
        "vs observed counts are in MANIFEST.json per root.", "",
        "## Split-level counts", "",
        "| split | roots | vehicle-steps | distinct vehicles | lane changes"
        " | births (in-window) | exits (arrived) | hard-brake /1000 |"
        " TTC<3s /1000 | TTC<1.5s /1000 | CAV step frac |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        line("train", tr), line("holdout", ho), line("overall", o), "",
        "Lane changes by (from -> to), overall: "
        f"{o['lc_left_to_right']} (0->1), {o['lc_right_to_left']} (1->0), "
        f"{o['lc_accel_to_main']} (2->1 accel merge).  Births by entrance: "
        f"{o['births_mainline']} mainline, {o['births_ramp']} ramp.",
        f"Junction vehicle-steps: {o['junction_steps']} "
        f"({100 * o['junction_steps'] / max(o['vehicle_steps'], 1):.3f}% "
        "of all vehicle-steps).", "",
        "## Per-root counts", "",
        "| root | split | veh-steps | vehicles | LC 0->1 | LC 1->0 |"
        " LC 2->1 | births | exits | junction | hard-brake | TTC<3 |"
        " resid max (m) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        " --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['root']} | {r['split']} | {r['vehicle_steps']} | "
            f"{r['vehicles']} | {r['lc_01']} | {r['lc_10']} | "
            f"{r['lc_21']} | {r['births']} | {r['exits_arrived']} | "
            f"{r['junction_steps']} | {r['hard_brake']} | {r['ttc_lt3']} | "
            f"{r['resid_max']:.3f} |")
    lines += ["", "s-advance residual (s[t+1]-s[t] - v*0.5, same-lane "
              "non-junction steps): per-root |resid| p99/max in the table "
              "above; full percentiles in results/per_root_counts.csv and "
              "MANIFEST.json.", "",
              "machine-readable copies: results/data_card.json, "
              "results/per_root_counts.csv, MANIFEST.json", ""]
    (HERE / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
