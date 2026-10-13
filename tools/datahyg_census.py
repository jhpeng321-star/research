"""DATAHYG census: designed-controller content of the C4 WM TRAIN/VAL windows (read-only).

    PYTHONPATH=src python tools/datahyg_census.py [--data E:/research-dense/data_cache]
        [--lists E:/research-dense/runs/active/c4_wm_lists] [--horizon 240] [--out docs/datahyg]

Reads only TRAIN packs/sidecars named by the C4 lists (pack_v3 / pack_dspeed / the switching
direct-mix arm) and never a guard file, CAL, holdout or sealed path.  Reports, per pack and
split: windows kept/dropped by reason, CAV-hours by controller before/after, the direct-mix
"clean prefix" distribution (mask option) and the exogenous random-direct CAV share after.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wms.data import controller_hygiene as H

FORBID = ("holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "sealed")
STAGES_TICKS = (20, 60, 120, 240)


def guard(path: Path) -> Path:
    if any(f in str(path).lower() for f in FORBID):
        raise SystemExit(f"refused {path}")
    return path


def load_json(path: Path):
    return json.loads(guard(path).read_text(encoding="utf-8"))


class Pack:
    def __init__(self, root: Path, names=("veh_step", "veh_mask", "target_mask")):
        self.root = guard(root)
        self.a = {n: np.load(root / f"{n}.npy", mmap_mode="r") for n in names}
        self.items = load_json(root / "manifest.json")["items"]
        self.runs = load_json(root / "INDEX.json")["runs"]

    def run_of(self, tick: int) -> dict:
        for r in self.runs:
            if r["tick_start"] <= tick < r["tick_end"]:
                return r
        raise KeyError(tick)

    def rows(self, lo: int, hi: int):
        vs = np.asarray(self.a["veh_step"][lo:hi, :, 7], np.float32) > 0.5
        vm = np.asarray(self.a["veh_mask"][lo:hi]).astype(bool)
        tm = np.asarray(self.a["target_mask"][lo:hi]).astype(bool)
        return vm, tm, vs & vm


def identity_book(root: Path):
    ids = np.load(guard(root) / "identities.npy", mmap_mode="r")
    pos = np.load(root / "global_positions.npy", mmap_mode="r")
    vmap = load_json(root / "vehicle_map.json")
    row = {int(p): i for i, p in enumerate(pos.tolist())}

    def slots(position: int) -> dict[int, str]:
        r = np.asarray(ids[row[int(position)]])
        return {int(k): str(vmap[int(v)][2]) for k, v in enumerate(r) if v >= 0}
    return slots


def directmix(data: Path, lists: Path, horizon: int) -> dict:
    sw = data / "wmn1_switching_train"
    pack = Pack(sw / "pack_switching_train")
    ctrl_tl = np.load(sw / "pack_switching_train" / "controller_timeline.npy", mmap_mode="r")
    slots_of = identity_book(sw / "identities_switching_train")
    by_run = defaultdict(list)
    for pos, it in enumerate(pack.items):
        by_run[(it["world"], it["arm"])].append((int(it["tick"]), pos))
    out = {}
    for split, fname in (("train", "windows_train_directmix_trainpart.json"),
                         ("val", "windows_val_directmix.json")):
        runs = load_json(lists / fname)["windows"]
        rows_out = []
        for w in runs:
            items = sorted(by_run[(w["world"], w["arm"])])
            snaps = [(t, slots_of(p)) for t, p in items]
            for (tick0, pos), (_t, cohort) in zip(items, snaps):
                run = pack.run_of(tick0)
                hi = min(run["tick_end"], tick0 + horizon + 121)
                vm, tm, cav = pack.rows(tick0, hi)
                rec = np.asarray(ctrl_tl[pos]).astype(np.int64)
                cav0 = {k: v for k, v in cohort.items() if cav[0, k]}
                res = H.directmix_damping_ticks(
                    w["world"], tick0, horizon, veh_mask=vm, target_mask=tm, is_cav=cav,
                    cohort_vid=cav0, recorded_ctrl=rec, snapshots=snaps,
                    run_start_tick=run["tick_start"])
                bad = res["damping"] | res["unknown"]
                prefix = int(np.argmax(bad)) if bad.any() else len(bad)
                # CAV-hours by recorded controller of the t0 cohort over the horizon
                keep = min(len(bad), horizon + 1)
                ctl = rec[np.minimum(np.arange(keep) // 2, rec.shape[0] - 1)]
                alive = H.same_vehicle_chain(tm[:keep], vm[:keep],
                                             np.isin(np.arange(vm.shape[1]), list(cav0))) \
                    & cav[:keep]
                cnt = np.bincount(ctl[alive], minlength=4) * 0.5 / 3600.0
                rows_out.append({"world": w["world"], "tick": tick0, "position": pos,
                                 "prefix_ticks": prefix, "stats": res["stats"],
                                 "cohort_h": cnt.tolist(),
                                 "other_h": float(((cav[:keep] & ~alive).sum()) * 0.5 / 3600)})
        n = len(rows_out)
        pref = np.asarray([r["prefix_ticks"] for r in rows_out])
        st = Counter()
        for r in rows_out:
            st.update(r["stats"])
        coh = np.sum([r["cohort_h"] for r in rows_out], axis=0)
        clean = pref >= horizon + 1
        coh_kept = (np.sum([r["cohort_h"] for r, c in zip(rows_out, clean) if c], axis=0)
                    if clean.any() else np.zeros(4))
        out[split] = {
            "n_items": n,
            "drop_option": {"kept": int(clean.sum()), "dropped_damping": int((~clean).sum()),
                            "drop_fraction": round(float((~clean).mean()), 4)},
            "clean_for_stage": {f"{t // 2}s": int((pref >= t + 1).sum()) for t in STAGES_TICKS},
            "prefix_ticks_quantiles": {q: float(np.quantile(pref, q))
                                       for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "mask_option": {
                "kept_prefix_ge_10s": int((pref >= 21).sum()),
                "supervised_tick_fraction": round(float(np.minimum(pref, horizon + 1).sum()
                                                        / (n * (horizon + 1))), 4)},
            "cohort_cav_hours_by_recorded_controller_before":
                dict(zip(H_CONTROLLERS, np.round(coh, 3).tolist())),
            "cohort_cav_hours_by_recorded_controller_after_drop":
                dict(zip(H_CONTROLLERS, np.round(coh_kept, 3).tolist())),
            "non_cohort_cav_hours_before": round(sum(r["other_h"] for r in rows_out), 3),
            "identity_and_schedule_check": dict(st),
        }
        out[split]["_items"] = rows_out
    return out


H_CONTROLLERS = ("native", "damping", "pi-external-u", "direct")


def dspeed(data: Path, lists: Path, horizon: int) -> dict:
    pack = Pack(data / "pack_dspeed")
    slots_of = identity_book(data / "identities_dspeed")
    key_pos = {}
    for pos, it in enumerate(pack.items):
        key_pos[(it["chunk"], it["world"], it["arm"])] = pos
    out = {}
    for split, fname in (("train", "windows_train_native_c4.json"),
                         ("val", "windows_val_native_c4.json")):
        wins = [w for w in load_json(lists / fname)["windows"] if w["pack"] == "pack_dspeed"]
        res = []
        for w in wins:
            pos = key_pos[(w["chunk"], w["world"], w["arm"])]
            it = pack.items[pos]
            run = pack.run_of(int(it["tick"]))
            lo, hi = int(it["tick"]), run["tick_end"]
            vm, tm, cav = pack.rows(lo, hi)
            plan = H.DspeedRun(w["world"], w["arm"], slots_of(pos), vm, tm, cav[0])
            allowed = plan.allowed_starts(horizon)
            # valid starts: unshifted t0 = row 0, shifted rows 1..room
            room = len(vm) - 1 - horizon
            starts = np.arange(0, max(1, room + 1))
            ok = allowed[starts]
            directed_h = float((plan.directed & cav).sum() * 0.5 / 3600)
            cav_h = float(cav.sum() * 0.5 / 3600)
            # expected window composition, one window per item draw, start uniform over the
            # valid starts (before) / the allowed starts (strict): CAV rows, directed rows
            # (old label: every CAV row was "direct"), directed rows inside designed ticks
            per_row = np.stack([cav.sum(1), (plan.directed & cav).sum(1),
                                ((plan.directed & cav).sum(1) * plan.designed)], 1)
            csum = np.concatenate((np.zeros((1, 3)), np.cumsum(per_row, 0)))
            hi_rows = np.minimum(len(vm), starts + horizon + 1)
            win = csum[hi_rows] - csum[starts]
            comp_before = win.mean(0)
            comp_after = win[ok].mean(0) if ok.any() else np.zeros(3)
            res.append({"key": w["key"], "n_selected": len(plan.selected_slots),
                        "n_designed_segments": plan.n_designed_segments,
                        "designed_tick_fraction": float(plan.designed.mean()),
                        "unshifted_allowed": bool(ok[0]), "allowed_start_fraction":
                        float(ok.mean()), "any_allowed": bool(ok.any()),
                        "directed_h": directed_h, "cav_h": cav_h,
                        "directed_h_clean": float((plan.directed & cav
                                                   & ~plan.designed[:, None]).sum() * 0.5 / 3600),
                        "win_before": comp_before.tolist(), "win_after": comp_after.tolist()})
        n = len(res)
        out[split] = {
            "n_items": n,
            "items_dropped_no_allowed_start": int(sum(not r["any_allowed"] for r in res)),
            "unshifted_window_refused": int(sum(not r["unshifted_allowed"] for r in res)),
            "mean_allowed_start_fraction": round(float(np.mean(
                [r["allowed_start_fraction"] for r in res])), 4),
            "items_with_designed_segments": int(sum(r["n_designed_segments"] > 0 for r in res)),
            "cav_hours": round(sum(r["cav_h"] for r in res), 3),
            "old_label_direct_cav_hours": round(sum(r["cav_h"] for r in res), 3),
            "directed_cav_hours_before": round(sum(r["directed_h"] for r in res), 3),
            "directed_cav_hours_outside_designed_ticks": round(
                sum(r["directed_h_clean"] for r in res), 3),
            "per_window_draw_cav_rows_before_after": {
                lab: [round(float(np.mean([r[f"win_{when}"][j] for r in res])), 1)
                      for when in ("before", "after")]
                for j, lab in enumerate(("cav", "directed", "directed_in_designed"))},
            "_items": res,
        }
        w = out[split]["per_window_draw_cav_rows_before_after"]
        out[split]["directed_share_of_cav_rows_before_after"] = [
            round(w["directed"][i] / max(1e-9, w["cav"][i]), 4) for i in (0, 1)]
        out[split]["random_directed_share_before_after"] = [
            round((w["directed"][i] - w["directed_in_designed"][i]) / max(1e-9, w["cav"][i]), 4)
            for i in (0, 1)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/research-dense/data_cache")
    ap.add_argument("--lists", default="E:/research-dense/runs/active/c4_wm_lists")
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--out", default="docs/datahyg")
    ap.add_argument("--only", choices=("directmix", "dspeed"), default=None)
    a = ap.parse_args(argv)
    data, lists = Path(a.data), Path(a.lists)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"horizon_ticks": a.horizon}
    if a.only in (None, "directmix"):
        doc["directmix"] = directmix(data, lists, a.horizon)
    if a.only in (None, "dspeed"):
        doc["dspeed"] = dspeed(data, lists, a.horizon)
    items = {k: {s: v.pop("_items") for s, v in doc[k].items()} for k in ("directmix", "dspeed")
             if k in doc}
    name = f"census_h{a.horizon}" + (f"_{a.only}" if a.only else "")
    (out / f"{name}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    (out / f"{name}_items.json").write_text(json.dumps(items), encoding="utf-8")
    json.dump(doc, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
