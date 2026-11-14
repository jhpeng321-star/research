"""D1: minimal independent reference rollout (pure numpy) vs engine.Rollout
for persistence and plain-IDM kinds on validation windows.  Any mismatch
beyond tolerance localises an engine defect (bookkeeping, births, exits,
neighbour search, integration)."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402

DT = 0.5
KMAX = 240
LANE_LIMIT = (33.79, 33.79, 24.0)
VT_N = dict(tau=1.1, a_max=2.6, b=4.5, s0=2.5, sf=1.0)


def ref_rollout(ra, k0, kind):
    """Independent numpy rollout.  Returns {tick: {vid: (lane, s, v)}}."""
    idx = ra.tick_rows(k0)
    idx = idx[ra.lane_m[idx] >= 0]
    st = {}
    for i in idx:
        st[int(ra.vid[i])] = [int(ra.lane_m[i]), float(ra.s[i]),
                              float(ra.v[i])]
    lengths = P.lane_lengths()
    births = defaultdict(list)
    bm = ra.b_dep & (ra.b_tick > k0) & (ra.b_tick <= k0 + KMAX)
    bm &= ra.b_lane >= 0
    for i in np.where(bm)[0]:
        births[int(ra.b_tick[i]) - k0].append(
            (int(ra.b_vid[i]), int(ra.b_lane[i]), float(ra.b_s[i]),
             float(ra.b_v[i])))
    snaps = {}
    for j in range(KMAX + 1):
        if j > 0:
            for (tvid, l, s0, v0) in births.get(j, ()):
                st[int(tvid)] = [int(l), float(s0),
                                 min(float(v0), 1.2 * LANE_LIMIT[l])]
            if kind == "idm":
                # mirror of p3lib.idm_acc spec: gap=250 when no leader
                # within 250 (interaction ~ tiny but NOT zero); same
                # clamps.  Leader = next larger s in the lane.
                by_lane = defaultdict(list)
                for vid, (l, s, v) in st.items():
                    by_lane[l].append((s, v, vid))
                new_a = {}
                for l, lst in by_lane.items():
                    lst.sort()                      # ascending s = rear first
                    for x in range(len(lst)):
                        s_f, v_f, vid_f = lst[x]
                        if x + 1 < len(lst):
                            s_l, v_l, _ = lst[x + 1]
                            gap = s_l - s_f
                            dv = v_f - v_l
                            if gap <= 0 or gap > 250:
                                gap, dv = 250.0, 0.0
                        else:
                            gap, dv = 250.0, 0.0
                        lim = 31.29 if (l < 2 or s_f >= 378.66) else 22.22
                        v0 = VT_N["sf"] * lim
                        s_star = VT_N["s0"] + max(
                            0.0, v_f * VT_N["tau"]
                            + v_f * dv / (2 * np.sqrt(VT_N["a_max"]
                                                      * VT_N["b"])))
                        acc = VT_N["a_max"] * (
                            1.0 - min(v_f / v0, 4.0) ** 4
                            - (s_star / max(gap, 0.3)) ** 2)
                        new_a[vid_f] = max(-9.0, min(3.0, acc))
                for vid, (l, s, v) in st.items():
                    st[vid][2] = max(0.0, min(1.2 * LANE_LIMIT[l],
                                              v + new_a[vid] * DT))
            gone = []
            for vid, (l, s, v) in st.items():
                st[vid][1] = s + v * DT
                if st[vid][1] > lengths[l]:
                    gone.append(vid)
            for vid in gone:
                del st[vid]
        snaps[j] = {vid: (l, s, v) for vid, (l, s, v) in st.items()}
    return snaps


def engine_snaps(root, t0, kind, dev):
    r = E.Rollout(root, t0, kind=kind, dev=dev).run()
    out = {}
    for j, sn in r.snapshots.items():
        out[j] = {int(v): (int(l), float(s), float(vv))
                  for v, l, s, vv in zip(sn["vid"], sn["lane"], sn["s"],
                                         sn["v"])}
    return out


def compare(a, b):
    """Per-tick s/v max abs diff over common vids; reports missing/extra."""
    s_diff = v_diff = 0.0
    miss = extra = 0
    for j in a:
        A, B = a[j], b.get(j, {})
        for vid, (l, s, v) in A.items():
            if vid not in B:
                miss += 1
                continue
            s_diff = max(s_diff, abs(s - B[vid][1]))
            v_diff = max(v_diff, abs(v - B[vid][2]))
        for vid in B:
            if vid not in A:
                extra += 1
    return s_diff, v_diff, miss, extra


def main():
    dev = P.device()
    splits = P.roots_by_split()
    out = {}
    for root in splits["val"][:2]:
        for t0 in (240.0, 360.0):
            ra = P.root_arrays(root)
            for kind in ("persistence", "idm"):
                ref = ref_rollout(ra, int(round((t0 - 60.0) / 0.5)), kind)
                eng = engine_snaps(root, t0,
                                   "persistence" if kind == "persistence"
                                   else "idm_normal", dev)
                s_d, v_d, miss, extra = compare(ref, eng)
                key = f"{root}@t{int(t0)}:{kind}"
                out[key] = dict(s_max_diff=round(s_d, 6),
                                v_max_diff=round(v_d, 6),
                                ref_only=miss, eng_only=extra)
                print(key, out[key])
    ok = all(v["s_max_diff"] < 1e-6 and v["v_max_diff"] < 1e-6
             and v["ref_only"] == 0 and v["eng_only"] == 0
             for v in out.values())
    print("D1 MATCH:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
