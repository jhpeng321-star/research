"""Markdown tables for HORIZON_CHECK.md from agg_results.json (part A) and B_analysis.json (B).

    python scripts/horizon_check/make_tables.py --a <agg_results.json> --b <B_analysis.json>
"""
from __future__ import annotations

import argparse
import json


def f(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


def table_a(res: dict, set_name: str) -> list:
    sets = res["sets"][set_name]
    lines = [f"### A: {set_name} (n = {next(iter(sets.values()))['n_states']} states; roster-only, entries off)",
             "",
             "| model | h (s) | HDV med abs ds (m) | zone count rel.err / r | zone mean speed MAE (m/s) / r "
             "| queued cells MAE / r | cell density MAE (veh/km) / r | queue-flag recall / precision "
             "| cum. zone delay rel.err / r | cum. zone reward rel.err / r | cum. hard-brake rel.err / r |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, blk in sets.items():
        for h, x in blk["h"].items():
            z, c = x["zone"], x["cell"]
            rw = x["team_reward_cumulative"]["zone"]
            q = c["queue_flag"]
            lines.append(
                f"| {tag} | {h} | {f(x['hdv_abs_ds_median_of_state_medians_m'], 1)} "
                f"| {f(z['vehicle_count'].get('rel_err'))} / {f(z['vehicle_count'].get('pearson'))} "
                f"| {f(z['mean_speed_mps'].get('mae'))} / {f(z['mean_speed_mps'].get('pearson'))} "
                f"| {f(z['queued_cells'].get('mae'))} / {f(z['queued_cells'].get('pearson'))} "
                f"| {f(c['density_veh_per_km'].get('mae'), 1)} / {f(c['density_veh_per_km'].get('pearson'))} "
                f"| {f(q.get('recall'))} / {f(q.get('precision'))} "
                f"| {f(rw['delay'].get('rel_err'))} / {f(rw['delay'].get('pearson'))} "
                f"| {f(rw['reward'].get('rel_err'))} / {f(rw['reward'].get('pearson'))} "
                f"| {f(rw['hard_brake'].get('rel_err'))} / {f(rw['hard_brake'].get('pearson'))} |")
    lines.append("")
    return lines


def table_b(res: dict, tags: list, metrics=("delay_all", "delay_cohort", "reward_all",
                                            "reward_cohort", "team_delay_all")) -> list:
    lines = [f"### B: imagined vs SUMO arm differences D_X (all controllers pooled; "
             f"{res['n_points']} decision points)", "",
             "| WM | metric | h | n | sign agree | sign agree, larger-half abs D_SUMO | Pearson | Spearman "
             "| med abs D SUMO | med abs D WM | slope WM~SUMO |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        for m in metrics:
            for h in ("30", "60", "120"):
                a = res["pairs"][tag]["ALL_CONTROLLERS"][h][m]["all_arms"]
                lines.append(f"| {tag} | {m} | {h} | {a['n']} | {f(a.get('sign_agreement'))} "
                             f"| {f(a.get('sign_agreement_top_half'))} | {f(a.get('pearson'))} "
                             f"| {f(a.get('spearman'))} | {f(a.get('sumo_median_abs'), 1)} "
                             f"| {f(a.get('imag_median_abs'), 1)} | {f(a.get('slope_img_on_sumo'))} |")
    lines.append("")
    return lines


def table_b_ctrl(res: dict, tags: list, metric="delay_all") -> list:
    lines = [f"### B: per controller and arm kind ({metric}; sign agreement / Pearson)", "",
             "| WM | controller | h | all arms | sign agree, larger-half abs D_SUMO | ALL | GROUP | ONLY_i "
             "| decision top-1 | regret | regret / spread |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        for c in res["controllers"]:
            for h in ("30", "60", "120"):
                p = res["pairs"][tag][c][h][metric]
                d = res["decisions"][tag][c][h][metric]

                def sp(x):
                    return f"{f(x.get('sign_agreement'))} / {f(x.get('pearson'))}"
                lines.append(f"| {tag} | {c} | {h} | {sp(p['all_arms'])} "
                             f"| {f(p['all_arms'].get('sign_agreement_top_half'))} | {sp(p.get('ALL', {}))} "
                             f"| {sp(p.get('GROUP', {}))} | {sp(p.get('ONLY', {}))} "
                             f"| {f(d.get('top1_agreement'))} | {f(d.get('mean_regret'), 1)} "
                             f"| {f(d.get('mean_regret_rel_to_spread'))} |")
    lines.append("")
    return lines


def table_b_agg(res: dict, tags: list) -> list:
    lines = ["### B: imagined all-native B vs SUMO B (full traffic incl. arrivals)", "",
             "| WM | h | HDV med abs ds (m) | zone count: WM vs all traffic rel.err / r | WM vs t0 cohort "
             "| persistence(t0) vs all | zone speed MAE WM vs all / r | persistence | cum. road delay "
             "WM vs all rel.err (bias) | WM vs cohort rel.err (bias) | cohort share of zone vehicles "
             "| source-wait share of team delay |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        for h, x in res["aggregate_B"][tag].items():
            zc, zs, d = x["zone_count"], x["zone_mean_speed"], x["cum_road_delay_B"]
            lines.append(
                f"| {tag} | {h} | {f(x['B_hdv_abs_ds_median_m'], 1)} "
                f"| {f(zc['img_vs_all_traffic'].get('rel_err'))} / {f(zc['img_vs_all_traffic'].get('pearson'))} "
                f"| {f(zc['img_vs_t0_cohort'].get('rel_err'))} / {f(zc['img_vs_t0_cohort'].get('pearson'))} "
                f"| {f(zc['persistence_t0_vs_all_traffic'].get('rel_err'))} / "
                f"{f(zc['persistence_t0_vs_all_traffic'].get('pearson'))} "
                f"| {f(zs['img_vs_all_traffic'].get('mae'))} / {f(zs['img_vs_all_traffic'].get('pearson'))} "
                f"| {f(zs['persistence_t0_vs_all_traffic'].get('mae'))} / "
                f"{f(zs['persistence_t0_vs_all_traffic'].get('pearson'))} "
                f"| {f(d['img_vs_all'].get('rel_err'))} ({f(d['img_vs_all'].get('bias'), 0)}) "
                f"| {f(d['img_vs_cohort'].get('rel_err'))} ({f(d['img_vs_cohort'].get('bias'), 0)}) "
                f"| {f(x['cohort_share_of_zone_vehicles'])} | {f(x['source_wait_share_of_team_delay'])} |")
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--tags", nargs="+", default=None)
    args = ap.parse_args()
    out = []
    if args.a:
        a = json.load(open(args.a, encoding="utf-8"))
        for s in a["sets"]:
            out += table_a(a, s)
    if args.b:
        b = json.load(open(args.b, encoding="utf-8"))
        tags = args.tags or b["wms"]
        out += table_b(b, tags) + table_b_ctrl(b, tags) + table_b_ctrl(b, tags, "reward_all") \
            + table_b_agg(b, tags)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
