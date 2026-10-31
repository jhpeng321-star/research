"""Sign/geometry detectors. Events are candidates until member-release support exists."""
from collections import defaultdict
import numpy as np

from .io import Tape

MERGE_START = 0.
MERGE_END = 75.44
RAMP = 2
RECEIVING = 1
BEHAVIOURS = ("yield_to_merge", "gap_creation", "cross_lane_hold", "wave_damping")


def greater(a, b, eps):
    return a - b > eps * max(abs(a), abs(b), np.finfo(float).tiny)


def leader(row, identity):
    v = row[identity]
    candidates = [(w.x, j) for j, w in row.items() if w.lane == v.lane and w.x > v.x]
    return min(candidates)[1] if candidates else None


def follower(row, identity):
    v = row[identity]
    candidates = [(w.x, j) for j, w in row.items() if w.lane == v.lane and w.x < v.x]
    return max(candidates)[1] if candidates else None


def gap(row, rear, front):
    if rear not in row or front not in row:
        return None
    a, b = row[rear], row[front]
    return b.x - b.length - a.x if a.lane == b.lane and b.x > a.x else None


def validate_release(actual: Tape, reference: Tape, members, reference_kind="native_member_release"):
    """Fail closed: clock/state identity plus explicit observable member command release.

    Caller must also bind provenance (kind/member list) in the pair index. A same-seed
    whole-controller arm comparison is never sufficient for this function's caller.
    """
    if not members or not np.array_equal(actual.time, reference.time):
        raise ValueError("Release pair needs members and identical clocks")
    if actual.meta.get("seed") != reference.meta.get("seed"):
        raise ValueError("Release seed mismatch")
    a, b = actual.rows[0], reference.rows[0]
    if a.keys() != b.keys():
        raise ValueError("Initial identity mismatch")
    eps = max(actual.eps, reference.eps)
    for j in a:
        x, y = a[j], b[j]
        if x.lane != y.lane or x.cav != y.cav:
            raise ValueError("Initial lane/CAV mismatch")
        for p, q in ((x.x, y.x), (x.v, y.v), (x.length, y.length)):
            if greater(abs(p - q), eps * max(abs(p), abs(q)), eps):
                raise ValueError("Initial state mismatch")
    if not all(j in a and a[j].cav for j in members):
        raise ValueError("Release member absent or not CAV at branch start")
    released = 0
    for row in reference.rows:
        for j in members:
            if j in row:
                v=row[j]
                if reference_kind=="native_member_release":
                    if v.controller not in (0,"native") or v.issued is not None:
                        raise ValueError("Reference member has an external command or unknown native status")
                elif reference_kind in ("policy_mean_intervention","policy_mean_reference"):
                    if v.controller=="policy_mean":
                        if v.policy_mean is None or v.issued is None or not np.isfinite(v.policy_mean):
                            raise ValueError("Policy-mean baseline needs its recorded unperturbed command")
                        if greater(abs(v.issued-v.policy_mean),eps*max(abs(v.issued),abs(v.policy_mean)),eps):
                            raise ValueError("Reference issued command differs from policy mean")
                    elif v.controller!="native" or v.issued is not None:
                        raise ValueError("Reference contains sampled or unknown control")
                else: raise ValueError("Unknown reference intervention")
                released += 1
    if not released:
        raise ValueError("No observed release exposure")


def _headway(row, rear, front):
    g = gap(row, rear, front)
    return g / row[rear].v if g is not None and row[rear].v > 0 else None


def _runs(mask):
    start = None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            yield start, i - 1
            start = None


def _crossing(tape, identity, begin):
    for i in range(max(1, begin), len(tape.time)):
        a, b = tape.rows[i - 1].get(identity), tape.rows[i].get(identity)
        if a and b and a.lane == b.lane and a.x < MERGE_START <= b.x:
            return float(tape.time[i - 1] + (MERGE_START - a.x) / (b.x - a.x) *
                         (tape.time[i] - tape.time[i - 1]))
    return None


def detect(tape, reference=None, released_members=(), reference_kind="native_member_release"):
    if reference is not None:
        validate_release(tape, reference, released_members,reference_kind)
    events = []
    eps = tape.eps if reference is None else max(tape.eps, reference.eps)
    active = {}
    complete_neighbours=tape.coverage.get("trace_scope","full_observable_fleet") in (
        "full_observable_fleet","members_and_complete_neighbours")

    def close(key, end, accepted=None, reason="gap_closed"):
        state = active.pop(key)
        rear, front = key
        first, peak = state["first"], state["peak"]
        startgap, peakgap = state["startgap"], state["peakgap"]
        h0 = _headway(tape.rows[first], rear, front)
        h1 = _headway(tape.rows[peak], rear, front)
        ev = dict(behaviour="yield_to_merge" if accepted else "gap_creation", members=[rear],
                  leader=front, start_s=float(tape.time[first]), end_s=float(tape.time[end]),
                  start_index=first, end_index=end, opening_m=peakgap-startgap,
                  headway_change_s=h1-h0 if h0 is not None and h1 is not None else None,
                  ramp_fit_observed=state["fits"], accepted_vehicle=accepted,
                  end_reason=reason, evidence="observational_candidate", group=[rear])
        a0,a1=tape.rows[first][rear],tape.rows[peak][rear]
        ev["ballistic_braking_contribution_m"]=(a0.x+a0.v*(tape.time[peak]-tape.time[first]))-a1.x
        if accepted:
            ev["merge_time_interval_s"] = [float(tape.time[end-1]), float(tape.time[end])]
            ev["accepted_slot_gap_m"]=gap(tape.rows[end],rear,front)
            ev["accepted_slot_change_m"]=ev["accepted_slot_gap_m"]-startgap
        if reference is not None and rear in released_members:
            comparison=end if accepted else peak
            actual_gap=gap(tape.rows[comparison],rear,front)
            actual_headway=_headway(tape.rows[comparison],rear,front)
            g0, g1 = (gap(reference.rows[i], rear, front) for i in (first, comparison))
            if g0 is not None and g1 is not None:
                dg = (actual_gap-startgap)-(g1-g0)
                ev["extra_opening_vs_release_m"] = dg
                a0,a1=tape.rows[first][rear],tape.rows[comparison][rear]
                b0,b1=reference.rows[first][rear],reference.rows[comparison][rear]
                deficit=(b1.x-b0.x)-(a1.x-a0.x)
                braking=(a0.v-a1.v)-(b0.v-b1.v)
                ev["cav_displacement_deficit_vs_release_m"]=deficit
                ev["extra_speed_loss_vs_release_mps"]=braking
                hr0, hr1 = (_headway(reference.rows[i], rear, front) for i in (first, comparison))
                ev["extra_headway_change_vs_release_s"] = (actual_headway-h0)-(hr1-hr0) if all(
                    x is not None for x in (h0,actual_headway,hr0,hr1)) else None
                if (greater(actual_gap-startgap,g1-g0,eps)
                        and greater(b1.x-b0.x,a1.x-a0.x,eps)
                        and greater(a0.v-a1.v,b0.v-b1.v,eps)):
                    if not accepted:
                        ev["evidence"] = "paired_release_gap_support"
                    else:
                        # Earlier acceptance means the reference vehicle is still ramp-side
                        # at the actual acceptance clock, with continuously observed history.
                        history = [r.get(accepted) for r in reference.rows[first:end+1]]
                        if all(v is not None and v.lane == RAMP for v in history):
                            later = next((i for i in range(end+1,len(reference.time))
                                          if accepted in reference.rows[i-1] and accepted in reference.rows[i]
                                          and reference.rows[i-1][accepted].lane == RAMP
                                          and reference.rows[i][accepted].lane == RECEIVING), None)
                            ev["reference_merge_time_interval_s"] = ([float(reference.time[later-1]),
                                float(reference.time[later])] if later is not None else None)
                            ev["reference_acceptance_right_censored"] = later is None
                            ev["evidence"] = "paired_release_yield_support"
        events.append(ev)

    for i in (range(1, len(tape.time)) if complete_neighbours else ()):
        prev, row = tape.rows[i-1], tape.rows[i]
        continuing_keys = set(active)
        for key in list(active):
            rear, front = key
            g = gap(row, rear, front)
            state = active[key]
            if g is None or row[rear].lane != RECEIVING:
                close(key, i-1, reason="identity_or_lane_censored")
                continue
            initial=tape.rows[state["first"]][rear]
            ballistic=initial.x+initial.v*(tape.time[i]-tape.time[state["first"]])
            if not greater(ballistic,row[rear].x,eps):
                close(key,i-1,reason="braking_deficit_closed")
                continue
            inserted = [j for j,w in row.items() if j in prev and prev[j].lane == RAMP
                        and w.lane == RECEIVING and row[rear].x <= w.x-w.length
                        and w.x <= row[front].x-row[front].length]
            if greater(g, state["peakgap"], eps):
                state.update(peak=i, peakgap=g)
            if inserted:
                close(key, i, accepted=sorted(inserted)[0], reason="accepted")
            elif leader(row, rear) != front:
                close(key, i-1, reason="slot_changed_censored")
            elif not greater(g, state["startgap"], eps):
                close(key, i, reason="gap_closed")
        for rear, v in prev.items():
            if not (v.cav and v.lane == RECEIVING and rear in row and row[rear].lane == RECEIVING):
                continue
            front = leader(prev, rear)
            if front is None or (rear,front) in continuing_keys:
                continue
            g0, g1 = gap(prev,rear,front), gap(row,rear,front)
            if g1 is None or not greater(v.v,row[rear].v,eps) or not greater(g1,g0,eps):
                continue
            ballistic=v.x+v.v*(tape.time[i]-tape.time[i-1])
            if not greater(ballistic,row[rear].x,eps):
                continue
            key = rear, front
            fits = any(w.lane == RAMP and 0 <= w.x <= MERGE_END
                       and v.x <= w.x-w.length and w.x <= prev[front].x-prev[front].length
                       for w in prev.values())
            active[key] = dict(first=i-1, peak=i, startgap=g0, peakgap=g1, fits=fits)
            inserted = [j for j,w in row.items() if j in prev and prev[j].lane == RAMP
                        and w.lane == RECEIVING and row[rear].x <= w.x-w.length
                        and w.x <= row[front].x-row[front].length]
            if inserted:
                close(key,i,accepted=sorted(inserted)[0],reason="accepted")
        for key,state in active.items():
            rear,front=key
            state["fits"] |= any(w.lane == RAMP and 0 <= w.x <= MERGE_END
                                 and row[rear].x <= w.x-w.length
                                 and w.x <= row[front].x-row[front].length for w in row.values())
    for key in list(active):
        close(key,len(tape.time)-1,reason="observation_end_censored")

    # Cross-lane body overlap distinguishes a wall from unrelated upstream braking.
    wall_intervals = defaultdict(list)
    for i in range(1,len(tape.time)):
        a,b=tape.rows[i-1],tape.rows[i]
        ids=sorted(j for j,v in b.items() if j in a and v.cav and v.lane in (0,1)
                   and a[j].lane == v.lane and v.x < MERGE_START
                   and greater(a[j].v,v.v,eps)
                   and greater(a[j].x+a[j].v*(tape.time[i]-tape.time[i-1]),v.x,eps))
        edges={j:set() for j in ids}
        for n,j in enumerate(ids):
            for k in ids[n+1:]:
                x,y=b[j],b[k]
                if x.lane != y.lane and min(x.x,y.x) > max(x.x-x.length,y.x-y.length):
                    edges[j].add(k); edges[k].add(j)
        unseen=set(ids)
        while unseen:
            stack=[min(unseen)]; group=set()
            while stack:
                j=stack.pop()
                if j in group: continue
                group.add(j); unseen.discard(j); stack.extend(edges[j]-group)
            if len(group)>1: wall_intervals[tuple(sorted(group))].append(i)
    for members,indices in wall_intervals.items():
        mask=np.zeros(len(tape.time),bool); mask[indices]=True
        for lo,hi in _runs(mask):
            first=lo-1
            ev=dict(behaviour="cross_lane_hold",members=list(members),group=list(members),
                    start_s=float(tape.time[first]),end_s=float(tape.time[hi]),
                    start_index=first,end_index=hi,evidence="observational_candidate",
                    follower_crossing_delays_s={})
            if reference is not None and set(members)<=set(released_members):
                for j in members:
                    f=follower(tape.rows[first],j)
                    if f:
                        ta,tb=_crossing(tape,f,first),_crossing(reference,f,first)
                        if ta is not None and tb is not None:
                            ev["follower_crossing_delays_s"][f]=ta-tb
                if ev["follower_crossing_delays_s"] and all(greater(d,0,eps) for d in
                        ev["follower_crossing_delays_s"].values()):
                    ev["evidence"]="paired_release_metering_support"
            events.append(ev)

    # Fixed follower-chain, same-clock temporal variance, no changing-population variance.
    cav_ids=sorted({j for row in tape.rows for j,v in row.items() if v.cav and v.lane in (0,1)}) if complete_neighbours else []
    for j in cav_ids:
        slowing=[False]+[j in a and j in b and a[j].lane==b[j].lane and b[j].lane in (0,1)
                        and greater(a[j].v,b[j].v,eps)
                        and greater(a[j].x+a[j].v*(tape.time[i]-tape.time[i-1]),b[j].x,eps)
                        for i,(a,b) in enumerate(zip(tape.rows[:-1],tape.rows[1:]),1)]
        for lo,hi in _runs(slowing):
            first=lo-1
            chain=[]; f=follower(tape.rows[first],j)
            while f is not None:
                chain.append(f); f=follower(tape.rows[first],f)
            supported=[]; metrics={}
            vc=float(np.var([r[j].v for r in tape.rows[first:hi+1]]))
            preceding=j
            for f in chain:
                if not all(f in r and preceding in r and leader(r,f)==preceding for r in tape.rows[first:hi+1]):
                    break
                vf=float(np.var([r[f].v for r in tape.rows[first:hi+1]]))
                comp=vc
                if reference is not None and j in released_members:
                    if not all(f in r and r[f].lane==tape.rows[first][f].lane
                               for r in reference.rows[first:hi+1]):
                        break
                    comp=float(np.var([r[f].v for r in reference.rows[first:hi+1]]))
                metrics[f]=dict(variance_m2_s2=vf,comparison_variance_m2_s2=comp,reduction_m2_s2=comp-vf)
                if greater(comp,vf,eps): supported.append(f)
                preceding=f
            if supported:
                events.append(dict(behaviour="wave_damping",members=[j],group=[j],followers=supported,
                    follower_metrics=metrics,start_s=float(tape.time[first]),end_s=float(tape.time[hi]),
                    start_index=first,end_index=hi,evidence="paired_release_variance_support" if reference is not None
                    and j in released_members else "observational_attenuation"))
    events.sort(key=lambda e:(e["start_s"],e["behaviour"],e["members"]))
    for i,e in enumerate(events): e["event_id"]=i
    if reference is not None:
        for e in events:
            e["reference_intervention"]=reference_kind
            e["intervention_members"]=list(released_members)
            if reference_kind in ("policy_mean_intervention","policy_mean_reference") and e["evidence"].startswith("paired_release_"):
                e["evidence"]=e["evidence"].replace("paired_release_","paired_policy_mean_",1)
    exposure=sum((tape.time[i]-tape.time[i-1])*sum(v.cav for j,v in tape.rows[i-1].items()
                 if j in tape.rows[i]) for i in range(1,len(tape.time)))
    coverage=dict(tape.coverage,disabled_detectors=[] if complete_neighbours else
                  ["yield_to_merge","gap_creation","wave_damping"])
    return dict(events=events,exposure_cav_s=float(exposure),
                exposure_episode_s=float(tape.time[-1]-tape.time[0]),coverage=coverage)
