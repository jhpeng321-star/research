"""WD-01-PA2G-LC1 aggregation: every registered decomposition table from
the saved raw rows (tables.json + tables.md).  Pure offline pandas-free
aggregation; no run file is opened here (only saved lc1 artefacts and
the frozen eval_a2/eval_a2_post_lc1 raw rows)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LC = HERE / "results" / "lc1"
DT = 0.5
HS = (10, 30, 60, 120)
MERGE_S0, MERGE_S1 = 378.66, 454.10
CLASS_KEY = {(2, 1): "merge", (1, 0): "disc_left", (0, 1): "disc_right"}


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> dict:
    T: dict = {}

    # ---------- headline (post-LC1-fixes v2g) ----------
    post = json.loads((HERE / "results" / "eval_a2_post_lc1" /
                       "summary.json").read_text(encoding="utf-8"))
    T["headline_post_lc1"] = {
        k: {kk: v[kk] for kk in
            ("n", "lc_merge_pred", "lc_merge_true", "lc_disc_pred",
             "lc_disc_true")}
        for k, v in post["summary"]["per_model_h"].items()
        if k.startswith("v2g|")}

    # ---------- tracked cohorts per (world, w) ----------
    cohorts = {}
    cav_of = {}
    for r in read_jsonl(LC / "pred_states.jsonl"):
        cohorts[(r["world"], r["w"])] = r["vids"]
        for vid, rec in r["vids"].items():
            cav_of[(r["world"], vid)] = bool(rec[3])

    # ---------- events per side, H120 slice, tracked cohort ----------
    # index truth events per world for slicing
    te_by_world = defaultdict(list)
    for e in read_jsonl(LC / "truth_events.jsonl"):
        te_by_world[e["world"]].append(e)
    truth_ev = {}
    for (world, w), vids in cohorts.items():
        sel = []
        for e in te_by_world.get(world, []):
            if w < e["t"] <= w + 120 + 1e-9 and e["vid"] in vids:
                e = dict(e, w=w, is_cav=cav_of.get((world, e["vid"]),
                                                   False))
                sel.append(e)
        truth_ev[(world, w)] = sel
    pe_by_world = defaultdict(list)
    for e in read_jsonl(LC / "pred_events.jsonl"):
        pe_by_world[e["world"]].append(e)
    pe_by_world = defaultdict(list)
    for e in read_jsonl(LC / "pred_events.jsonl"):
        pe_by_world[e["world"]].append(e)
    pred_ev = {}
    for (world, w), vids in cohorts.items():
        sel = []
        for e in pe_by_world.get(world, []):
            if e.get("w") == w and w < e["t"] <= w + 120 + 1e-9 and \
                    e["vid"] in vids:
                sel.append(dict(e, w=w))
        pred_ev[(world, w)] = sel

    # T2 lane pair
    tt = Counter(CLASS_KEY.get((e["f"], e["t_"]), "other"))
    pt = Counter()
    for evs in pred_ev.values():
        pt.update(CLASS_KEY.get((e["f"], e["t_"]), "other")
                  for e in evs)
    tru = Counter()
    for evs in truth_ev.values():
        tru.update(CLASS_KEY.get((e["f"], e["t_"]), "other")
                   for e in evs)
    T["lane_pair_H120"] = dict(
        truth=dict(tru), pred=dict(pt),
        truth_unclassified=tt["other"],
        note="events in (w, w+120] over the tracked cohort, 576 windows")

    # T3 10 m s-bins
    def sbins(evs_iter, lo, hi, width=10.0):
        b = Counter()
        for e in evs_iter:
            if lo <= e["s"] < hi:
                b[int((e["s"] - lo) // width)] += 1
        return {f"{lo + k * width:.0f}-{lo + (k + 1) * width:.0f}": v
                for k, v in sorted(b.items())}

    merge_t = [e for evs in truth_ev.values() for e in evs
               if CLASS_KEY.get((e["f"], e["t_"])) == "merge"]
    merge_p = [e for evs in pred_ev.values() for e in evs
               if CLASS_KEY.get((e["f"], e["t_"])) == "merge"]
    disc_t = [e for evs in truth_ev.values() for e in evs
              if CLASS_KEY.get((e["f"], e["t_"])) != "merge"]
    disc_p = [e for evs in pred_ev.values() for e in evs
              if CLASS_KEY.get((e["f"], e["t_"])) != "merge"]
    T["merge_s10mbins"] = dict(
        truth=sbins(merge_t, MERGE_S0, MERGE_S1),
        pred=sbins(merge_p, MERGE_S0, MERGE_S1))
    T["disc_s100mbins"] = dict(
        truth=sbins(disc_t, 1845.67, 2871.11, 100.0),
        pred=sbins(disc_p, 1845.67, 2871.11, 100.0),
        truth_10m_full=sbins(disc_t, 1845.67, 2871.11, 10.0),
        pred_10m_full=sbins(disc_p, 1845.67, 2871.11, 10.0))

    # T4 regime
    def regime(evs_iter):
        c = Counter()
        for e in evs_iter:
            nn = e.get("n_near", 0)
            mv = e.get("near_mean_v")
            dcut = "n0" if nn == 0 else ("n1-2" if nn <= 2 else
                                         "n3-5" if nn <= 5 else "n6+")
            if mv is None or mv != mv:
                vcut = "v_undef"
            else:
                vcut = "v<10" if mv < 10 else ("v10-20" if mv < 20
                                               else "v>20")
            c[f"{dcut}|{vcut}"] += 1
        return dict(c)

    T["regime_H120"] = dict(truth=regime(disc_t), pred=regime(disc_p))

    # T5 CAV vs HDV (exposure from cohort composition)
    exp = Counter()
    for (world, w), vids in cohorts.items():
        n_cav = sum(1 for v in vids if cohorts[(world, w)][v][3])
        exp["cav_vs"] += n_cav * 240
        exp["hdv_vs"] += (len(vids) - n_cav) * 240

    def by_cav(evs_iter):
        c = Counter()
        for e in evs_iter:
            c["cav" if e["is_cav"] else "hdv"] += 1
        return c

    tc, pc = by_cav(disc_t + merge_t), by_cav(disc_p + merge_p)
    T["cav_vs_hdv_H120"] = dict(
        truth=tc, pred=pc,
        truth_per_1000_vs={k: round(1000.0 * tc[k] /
                                    max(exp["cav_vs" if k == "cav"
                                        else "hdv_vs"], 1), 3)
                           for k in tc},
        pred_per_1000_vs={k: round(1000.0 * pc[k] /
                                   max(exp["cav_vs" if k == "cav"
                                       else "hdv_vs"], 1), 3)
                          for k in pc},
        exposure_vs=dict(exp))

    # T6 time since window start
    def tbin(evs_iter):
        c = Counter()
        for e in evs_iter:
            dt = e["t"] - e["w"]
            c["0-10" if dt <= 10 else "10-30" if dt <= 30 else
              "30-60" if dt <= 60 else "60-120"] += 1
        return dict(c)

    T["time_since_start_H120"] = dict(truth=tbin(disc_t + merge_t),
                                      pred=tbin(disc_p + merge_p))

    # T7 repeats + change-back within 10 s
    def repeats(evs_iter):
        per = Counter()
        for e in evs_iter:
            per[(e["world"], e["w"], e["vid"])] += 1
        c = Counter()
        for n in per.values():
            c["0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3
              else "4-9" if n <= 9 else "10+"] += 1
        return c, per

    def changeback(evs_iter, per_veh):
        by = defaultdict(list)
        for e in evs_iter:
            by[(e["world"], e["w"], e["vid"])].append(e)
        n_cb = 0
        for key, evs in by.items():
            evs.sort(key=lambda e: e["t"])
            for i, e in enumerate(evs):
                for e2 in evs[i + 1:]:
                    if e2["t"] - e["t"] <= 10.0 + 1e-9 and \
                            (e["f"], e["t_"]) == (e2["t_"], e2["f"]):
                        n_cb += 1
                        break
                else:
                    continue
                break
        return n_cb, len(by)

    T["repeats_per_vehicle_H120"] = {}
    for name, evs_iter in (("truth", disc_t + merge_t),
                           ("pred", disc_p + merge_p)):
        c, per = repeats(evs_iter)
        n_cb, n_veh = changeback(evs_iter, per)
        T["repeats_per_vehicle_H120"][name] = dict(
            dist=c, vehicles_with_change=n_veh,
            changed_back_within_10s=n_cb,
            changeback_rate=round(n_cb / max(n_veh, 1), 3))

    # T8 merges: outcomes + dwell + position stats
    def merge_stats(rows, at_w_only=True):
        out = {"n": 0, "merged": 0, "stopped": 0, "never_H120": 0,
               "dwell_from_w": [], "dwell_from_edge_entry": []}
        worlds_seen = {k[0] for k in cohorts}
        for r in rows:
            if r["world"] not in worlds_seen:
                continue
            if at_w_only and not r.get("at_w", False):
                continue
            # comparable cohort: vehicles ON the merge edge (s >= 378.66)
            # at w -- the truth side counts 1487606179_0 rows only
            if at_w_only and r.get("s0", MERGE_S1) < MERGE_S0:
                continue
            out["n"] += 1
            if "merged_in" in r:
                mH = bool(r["merged_in"].get("H120", False))
            else:
                mH = r.get("merged_H120", False)
            out["merged"] += int(mH)
            out["never_H120"] += int(not mH)
            if r.get("stop_t") is not None and not mH:
                out["stopped"] += 1
            if r.get("merge_t") is not None and mH:
                out["dwell_from_w"].append(r["merge_t"] - r["w"])
                fe = r.get("first_edge_t")
                if fe is not None:
                    out["dwell_from_edge_entry"].append(
                        r["merge_t"] - max(fe, r["w"]))
        import statistics as st
        for k in ("dwell_from_w", "dwell_from_edge_entry"):
            v = out.pop(k)
            out[k + "_mean"] = round(st.mean(v), 2) if v else None
            out[k + "_median"] = round(st.median(v), 2) if v else None
        return out

    T["merge_outcomes_truth_at_w"] = merge_stats(
        read_jsonl(LC / "merge_outcomes_truth.jsonl"))
    T["merge_outcomes_pred_at_w"] = merge_stats(
        read_jsonl(LC / "pred_merge_outcomes.jsonl"))
    # merge position distribution stats
    import statistics as st2
    T["merge_position"] = {
        "truth": dict(n=len(merge_t),
                      mean_s=round(st2.mean([e["s"] for e in merge_t]), 2)
                      if merge_t else None,
                      below_415=round(sum(1 for e in merge_t
                                          if e["s"] < 415.0) /
                                      max(len(merge_t), 1), 3)),
        "pred": dict(n=len(merge_p),
                     mean_s=round(st2.mean([e["s"] for e in merge_p]), 2)
                     if merge_p else None,
                     below_415=round(sum(1 for e in merge_p
                                         if e["s"] < 415.0) /
                                     max(len(merge_p), 1), 3))}

    # event matching: pred disc/merge events vs truth events of the same
    # vehicle within +-5 s (per class pair)
    match = {"pred_matched_same_dir": 0, "pred_matched_opp_dir": 0,
             "pred_no_truth_event": 0, "truth_matched_same_dir": 0,
             "truth_matched_opp_dir": 0, "truth_no_pred_event": 0}
    for (world, w), pevs in pred_ev.items():
        tevs = truth_ev.get((world, w), [])
        for e in pevs:
            if CLASS_KEY.get((e["f"], e["t_"])) == "other":
                continue
            near = [x for x in tevs if x["vid"] == e["vid"] and
                    abs(x["t"] - e["t"]) <= 5.0 + 1e-9]
            if not near:
                match["pred_no_truth_event"] += 1
            elif any(x["f"] == e["f"] and x["t_"] == e["t_"]
                     for x in near):
                match["pred_matched_same_dir"] += 1
            else:
                match["pred_matched_opp_dir"] += 1
        for e in tevs:
            if CLASS_KEY.get((e["f"], e["t_"])) == "other":
                continue
            near = [x for x in pevs if x["vid"] == e["vid"] and
                    abs(x["t"] - e["t"]) <= 5.0 + 1e-9]
            if not near:
                match["truth_no_pred_event"] += 1
            elif any(x["f"] == e["f"] and x["t_"] == e["t_"]
                     for x in near):
                match["truth_matched_same_dir"] += 1
            else:
                match["truth_matched_opp_dir"] += 1
    T["event_matching_pm5s_H120"] = match

    # T9 worst windows (pred disc surplus vs truth, H120 slice)
    tw = {}
    for r in read_jsonl(LC / "truth_windows.jsonl"):
        if r["H"] == 120:
            tw[(r["world"], r["w"])] = r
    worst = []
    for r in read_jsonl(LC / "pred_window_rows.jsonl"):
        key = (r["world"], r["w"])
        if key not in tw or "H120" not in r["lc"]:
            continue
        pd = r["lc"]["H120"]["disc_left"] + r["lc"]["H120"]["disc_right"]
        td = tw[key]["disc_left"] + tw[key]["disc_right"]
        pm = r["lc"]["H120"]["merge"]
        tm = tw[key]["merge"]
        worst.append(dict(world=r["world"], w=r["w"], disc_pred=pd,
                          disc_true=td, disc_surplus=pd - td,
                          merge_pred=pm, merge_true=tm))
    worst.sort(key=lambda x: -x["disc_surplus"])
    T["worst_windows_H120"] = worst[:15]
    (LC / "worst_windows.json").write_text(
        json.dumps(worst[:15], indent=1), encoding="utf-8")

    # T11 rejection tallies
    tally = Counter()
    n_win = 0
    for r in read_jsonl(LC / "pred_rejections.jsonl"):
        n_win += 1
        for k, v in r["rejected_tally"].items():
            tally[k] += v
    T["rejection_tally"] = dict(totals=dict(tally),
        windows=n_win,
        note="per-window counts of REJECTED candidates by first failed "
             "check; accepted decisions are in decisions_accepted")

    # T10 ablations
    rows = defaultdict(lambda: defaultdict(dict))
    for r in read_jsonl(LC / "ablation_rows.jsonl"):
        v = r["variant"]
        for H, lc in r["lc"].items():
            rows[v][("lc", H)][r["world"], r["w"]] = lc
        for H, m in r.get("metrics", {}).items():
            rows[v][("srmse", H)][r["world"], r["w"]] = \
                m.get("s_rmse")
            rows[v][("delay", H)][r["world"], r["w"]] = \
                m.get("delay_err")
    ev_by_var = defaultdict(list)
    for e in read_jsonl(LC / "ablation_events.jsonl"):
        ev_by_var[e["variant"]].append(e)

    def cb_rate(evs):
        by = defaultdict(list)
        for e in evs:
            by[(e["world"], e["w"], e["vid"])].append(e)
        n_cb = 0
        for key, ee in by.items():
            ee.sort(key=lambda e: e["step"])
            hit = False
            for i, e in enumerate(ee):
                for e2 in ee[i + 1:]:
                    if e2["step"] - e["step"] <= 20 and \
                            e["key"].startswith("disc") and \
                            e2["key"].startswith("disc") and \
                            e["key"] != e2["key"]:
                        hit = True
                        break
                if hit:
                    break
            n_cb += int(hit)
        return n_cb, len(by)

    abl = {}
    for v in rows:
        lc120 = rows[v][("lc", "H120")]
        disc = sum(x["disc_left"] + x["disc_right"] for x in
                   lc120.values()) if lc120 else 0
        merg = sum(x["merge"] for x in lc120.values()) if lc120 else 0
        n = len(lc120) if lc120 else 0
        evs = ev_by_var.get(v, [])
        n_cb, n_veh = cb_rate(evs)
        abl[v] = dict(
            windows=n,
            disc_total_H120=disc, disc_per_window=round(disc / max(n, 1),
                                                        3),
            merge_total_H120=merg,
            merge_per_window=round(merg / max(n, 1), 3),
            vehicles_with_disc=round(n_veh / max(n, 1), 2),
            disc_changeback_rate=round(n_cb / max(n_veh, 1), 3))
        for H in ("H10", "H120"):
            sr = [x for x in rows[v][("srmse", H)].values()
                  if x == x]
            de = [x for x in rows[v][("delay", H)].values()
                  if x == x]
            abl[v][f"s_rmse_{H}_mean"] = round(sum(sr) / max(len(sr), 1),
                                               2) if sr else None
            abl[v][f"delay_err_{H}_mean"] = round(sum(de) /
                                                  max(len(de), 1), 2) \
                if de else None
    order = json.loads((LC / "ablation_meta.json").read_text(
        encoding="utf-8"))["variants"] if \
        (LC / "ablation_meta.json").exists() else sorted(abl)
    T["ablation_descriptive_NOT_ADOPTED"] = {k: abl[k] for k in order
                                             if k in abl}

    (LC / "tables.json").write_text(json.dumps(T, indent=1),
                                    encoding="utf-8")
    write_md(T)
    print("wrote", LC / "tables.json")
    return T


def write_md(T: dict) -> None:
    lines = ["# WD-01-PA2G-LC1 decomposition tables (machine-generated)",
             ""]
    h = T["headline_post_lc1"]
    lines += ["## Headline v2g LC counts (post-LC1-fixes eval)", "",
              "| model|H | n | merge p/t | disc p/t |", "|---|---|---|---|"]
    for k in sorted(h):
        v = h[k]
        lines.append(f"| {k} | {v['n']} | {v['lc_merge_pred']}/"
                     f"{v['lc_merge_true']} | {v['lc_disc_pred']}/"
                     f"{v['lc_disc_true']} |")
    lp = T["lane_pair_H120"]
    lines += ["", "## Lane pair, events in (w, w+120], tracked cohort",
              "", "| class | truth | pred | pred/truth |", "|---|---|---|---|"]
    for c in ("merge", "disc_left", "disc_right"):
        tv, pv = lp["truth"].get(c, 0), lp["pred"].get(c, 0)
        lines.append(f"| {c} | {tv} | {pv} | "
                     f"{pv / tv:.2f}x |" if tv else
                     f"| {c} | {tv} | {pv} | - |")
    tb = T["merge_s10mbins"]
    lines += ["", "## Merge position, 10 m bins on the merge edge "
              "[378.66, 454.10) m", "",
              "| s bin (m) | truth | pred |", "|---|---|---|"]
    for k in sorted(set(tb["truth"]) | set(tb["pred"]),
                    key=lambda x: float(x.split("-")[0])):
        lines.append(f"| {k} | {tb['truth'].get(k, 0)} | "
                     f"{tb['pred'].get(k, 0)} |")
    db = T["disc_s100mbins"]
    lines += ["", "## Discretionary events, 100 m s-bins (from-lane s)",
              "", "| s bin (m) | truth | pred |", "|---|---|---|"]
    for k in sorted(set(db["truth"]) | set(db["pred"]),
                    key=lambda x: float(x.split("-")[0])):
        lines.append(f"| {k} | {db['truth'].get(k, 0)} | "
                     f"{db['pred'].get(k, 0)} |")
    rg = T["regime_H120"]
    keys = sorted(set(rg["truth"]) | set(rg["pred"]))
    lines += ["", "## Local regime at the event (same-lane neighbours "
              "within +-50 m | their mean speed)", "",
              "| regime | truth | pred |", "|---|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {rg['truth'].get(k, 0)} | "
                     f"{rg['pred'].get(k, 0)} |")
    cv = T["cav_vs_hdv_H120"]
    lines += ["", "## CAV vs HDV", "",
              "| side | truth events | pred events | truth /1000 v-s | "
              "pred /1000 v-s |", "|---|---|---|---|---|"]
    for k in ("cav", "hdv"):
        lines.append(f"| {k} | {cv['truth'].get(k, 0)} | "
                     f"{cv['pred'].get(k, 0)} | "
                     f"{cv['truth_per_1000_vs'].get(k, 0)} | "
                     f"{cv['pred_per_1000_vs'].get(k, 0)} |")
    ts = T["time_since_start_H120"]
    lines += ["", "## Time since window start", "",
              "| bin (s) | truth | pred |", "|---|---|---|"]
    for k in ("0-10", "10-30", "30-60", "60-120"):
        lines.append(f"| {k} | {ts['truth'].get(k, 0)} | "
                     f"{ts['pred'].get(k, 0)} |")
    rp = T["repeats_per_vehicle_H120"]
    lines += ["", "## Repeats per vehicle (120 s rollout)", "",
              "| metric | truth | pred |", "|---|---|---|"]
    for k in sorted(set(rp["truth"]["dist"]) | set(rp["pred"]["dist"])):
        lines.append(f"| vehicles with {k} events | "
                     f"{rp['truth']['dist'].get(k, 0)} | "
                     f"{rp['pred']['dist'].get(k, 0)} |")
    lines.append(f"| vehicles with a change | "
                 f"{rp['truth']['vehicles_with_change']} | "
                 f"{rp['pred']['vehicles_with_change']} |")
    lines.append(f"| changed back (opp. disc) within 10 s | "
                 f"{rp['truth']['changed_back_within_10s']} "
                 f"({rp['truth']['changeback_rate']}) | "
                 f"{rp['pred']['changed_back_within_10s']} "
                 f"({rp['pred']['changeback_rate']}) |")
    mo_t, mo_p = T["merge_outcomes_truth_at_w"], \
        T["merge_outcomes_pred_at_w"]
    lines += ["", "## Merge outcomes, accel-lane vehicles at w", "",
              "| metric | truth | pred |", "|---|---|---|"]
    for k in ("n", "merged", "stopped", "never_H120",
              "dwell_from_w_mean", "dwell_from_w_median",
              "dwell_from_edge_entry_mean", "dwell_from_edge_entry_median"):
        lines.append(f"| {k} | {mo_t.get(k)} | {mo_p.get(k)} |")
    mp_ = T["merge_position"]
    lines += ["", "## Merge position stats", "",
              f"| side | n | mean s (m) | share below 415 m |",
              "|---|---|---|---|"]
    for k in ("truth", "pred"):
        v = mp_[k]
        lines.append(f"| {k} | {v['n']} | {v['mean_s']} | "
                     f"{v['below_415']} |")
    ww = T["worst_windows_H120"]
    lines += ["", "## Worst windows by disc surplus (H120 slice)", "",
              "| world | w | disc pred | disc true | surplus | merge p/t |",
              "|---|---|---|---|---|---|"]
    for x in ww[:12]:
        lines.append(f"| {x['world']} | {x['w']} | {x['disc_pred']} | "
                     f"{x['disc_true']} | {x['disc_surplus']} | "
                     f"{x['merge_pred']}/{x['merge_true']} |")
    rt = T["rejection_tally"]
    lines += ["", "## Rejected decision tallies (pred, all windows)",
              "", "| kind|reason | total |", "|---|---|"]
    for k, v in sorted(rt["totals"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    ab = T.get("ablation_descriptive_NOT_ADOPTED", {})
    if ab:
        lines += ["", "## Descriptive ablations (LABELLED: NOT ADOPTED, "
                  "NOT SELECTED; params_nominal.json untouched)", "",
                  "| variant | disc/window (H120) | merge/window (H120) | "
                  "disc change-back rate | s_RMSE H10 | s_RMSE H120 | "
                  "delay_err H10 |", "|---|---|---|---|---|---|---|"]
        for k, v in ab.items():
            lines.append(
                f"| {k} | {v['disc_per_window']} | "
                f"{v['merge_per_window']} | {v['disc_changeback_rate']} | "
                f"{v.get('s_rmse_H10_mean')} | "
                f"{v.get('s_rmse_H120_mean')} | "
                f"{v.get('delay_err_H10_mean')} |")
    lines += ["", "Delay-scope note (owner decision (c)): Part A is "
              "tracked-scope; Part B carries untracked arrivals.", ""]
    (LC / "tables.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
