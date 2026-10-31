"""Run with python -m tools.coopbehav.report --views ROOT --out OUTPUT.

Optional --pairs index.json contains {pairs:[{id, factual, reference, members,
kind:'native_member_release', seed, arm, branch:{t0_s,horizon_s}, C_G, D_G,
sum_D_i}]}. Paths point to E3DATA directories (or normalized JSONL traces).
The three credit fields must reconcile before entering exploratory discovery.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

from .detectors import BEHAVIOURS, detect, follower, gap, leader
from .discovery import discover
from .io import Tape, Vehicle, check_seed, load_tape, manifest_entries, read_json, safe


def write_json(path, value):
    safe(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_trace(path, meta):
    """Normalized JSONL: {time_s, vehicles:[{vid,s_m,v_mps,length_m,lane,
    is_cav,controller,issued_acceleration_mps2}]}. No inferred length/CAV flags.
    All-frame export is required; partial neighbourhood exports must declare scope.
    """
    path=safe(path)
    if path.is_dir(): return load_tape(path,meta)
    check_seed(meta["seed"])
    opener=gzip.open if path.suffix==".gz" else open
    times=[]; rows=[]
    with opener(path,"rt",encoding="utf-8") as stream:
        for line in stream:
            frame=json.loads(line)
            if frame.get("phase")=="prefix": continue
            row={}
            action_rows={str(a["vid"]):a for a in frame.get("actions",[])}
            for v in frame["vehicles"]:
                j=str(v["vid"])
                if j in row: raise ValueError("Duplicate trace identity")
                issued=v.get("issued_acceleration_mps2")
                if issued is not None and not np.isfinite(issued): issued=None
                row[j]=Vehicle(float(v["s_m"]),float(v["v_mps"]),float(v["length_m"]),
                               int(v["lane"]),bool(v["is_cav"]),issued,v.get("controller"),
                               action_rows.get(j,{}).get("policy_mean"))
            times.append(float(frame["time_s"])); rows.append(row)
    branch=meta.get("branch",{})
    start=float(branch.get("t0_s",times[0])); end=start+float(branch.get("horizon_s",times[-1]-start))
    keep=[i for i,t in enumerate(times) if start<=t<=end]
    return Tape(np.array(times)[keep],[rows[i] for i in keep],meta,
                float(meta.get("position_dtype_epsilon",np.finfo(np.float64).eps)),
                {path.name:hashlib.sha256(path.read_bytes()).hexdigest()},
                dict(trace_scope=meta.get("trace_scope","unspecified"),
                     limitations=meta.get("trace_limitations",[]),
                     requested_branch_interval_s=[start,end],
                     recorded_interval_s=[times[keep[0]],times[keep[-1]]]))


def signature(tape,members,result):
    """Fixed identities and initial immediate neighbours; dimensions have named units."""
    m=set(map(str,members)); values=defaultdict(list)
    initial=tape.rows[0]
    neighbours=set()
    for j in m & initial.keys():
        neighbours.update(x for x in (leader(initial,j),follower(initial,j)) if x is not None)
    neighbours-=m
    for j in sorted(m):
        observed=[(i,r[j]) for i,r in enumerate(tape.rows) if j in r]
        if not observed: continue
        v=np.array([w.v for _,w in observed]); x=np.array([w.x for _,w in observed])
        values["member_speed_mps"].extend(v.tolist())
        values["member_temporal_speed_variance_m2_s2"].append(float(np.var(v)))
        values["member_speed_change_mps"].append(float(v[-1]-v[0]))
        values["member_displacement_m"].append(float(x[-1]-x[0]))
        values["member_upstream_fraction"].extend((x<0).astype(float).tolist())
        values["member_lane_changes"].append(sum(a.lane!=b.lane and a.lane>=0 and b.lane>=0
            for (ia,a),(ib,b) in zip(observed[:-1],observed[1:]) if ib==ia+1))
        values["member_issued_acceleration_mps2"].extend(w.issued for _,w in observed if w.issued is not None)
        for (ia,a),(ib,b) in zip(observed[:-1],observed[1:]):
            if ib==ia+1: values["member_realized_acceleration_mps2"].append(
                (b.v-a.v)/float(tape.time[ib]-tape.time[ia]))
        front=leader(initial,j) if j in initial else None
        if front:
            g0,g1=gap(initial,j,front),gap(tape.rows[-1],j,front)
            if g0 is not None and g1 is not None: values["member_gap_change_m"].append(g1-g0)
        for i,w in observed:
            ramps=[abs(v.x-w.x) for v in tape.rows[i].values() if v.lane==2]
            if ramps: values["nearest_ramp_longitudinal_distance_m"].append(min(ramps))
    for j in sorted(neighbours):
        observed=[r[j] for r in tape.rows if j in r]
        if observed:
            values["neighbour_speed_mps"].extend(v.v for v in observed)
            values["neighbour_temporal_speed_variance_m2_s2"].append(float(np.var([v.v for v in observed])))
            values["neighbour_speed_change_mps"].append(observed[-1].v-observed[0].v)
    features={k:float(np.mean(v)) for k,v in values.items() if v}
    features["observed_member_fraction"]=sum(j in r for r in tape.rows for j in m)/(len(tape.rows)*len(m)) if m else 0.
    observed_commands=[r[j].issued is not None for r in tape.rows for j in m if j in r]
    features["member_command_observed_fraction"]=float(np.mean(observed_commands)) if observed_commands else 0.
    explained=sorted({e["behaviour"] for e in result["events"] if m & set(e["members"])})
    return dict(features=features,explained_by=explained,initial_neighbours=sorted(neighbours))


def example_plot(tape,event,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharex=True,gridspec_kw={"width_ratios":[1.5,1]})
    if event is None:
        for ax in axes: ax.axis("off")
        axes[0].text(.1,.5,"No observed event in this source/arm",transform=axes[0].transAxes)
    else:
        lo,hi=event["start_index"],event["end_index"]
        # Plot the whole event plus one neighbouring frame for boundary context.
        indices=range(max(0,lo-1),min(len(tape.time),hi+2))
        members=set(event["members"])
        neighbors=set(event.get("followers",[]))|{event.get("leader"),event.get("accepted_vehicle")}
        for i in (lo,hi):
            for j in members & tape.rows[i].keys():
                neighbors.update(x for x in (leader(tape.rows[i],j),follower(tape.rows[i],j)) if x)
        ids=members|(neighbors-{None})
        for j in sorted(ids):
            tt=[]; xx=[]; vv=[]; lanes=[]
            for i in indices:
                v=tape.rows[i].get(j)
                tt.append(tape.time[i]); xx.append(v.x if v else np.nan)
                vv.append(v.v if v else np.nan); lanes.append(v.lane if v else -1)
            label=f"{j} {'CAV' if j in members else 'neighbour'}"
            line=axes[0].plot(tt,xx,lw=2 if j in members else 1,alpha=.9,label=label)[0]
            axes[0].scatter(tt,xx,c=lanes,cmap="viridis",vmin=0,vmax=2,s=10)
            axes[1].plot(tt,vv,color=line.get_color(),lw=2 if j in members else 1)
        for ax in axes:
            ax.axvspan(tape.time[lo],tape.time[hi],color="#cbd5e1",alpha=.35)
            ax.set_xlabel("Time (s)"); ax.grid(alpha=.2)
        axes[0].axhspan(0,75.44,color="#fde68a",alpha=.25)
        axes[0].set_ylabel("Longitudinal position s (m)")
        axes[1].set_ylabel("Speed (m/s)")
        axes[0].legend(fontsize=7,loc="best")
        fig.suptitle(event["behaviour"].replace("_"," ")+" | "+event["evidence"],fontsize=11)
    fig.tight_layout(); fig.savefig(safe(path),dpi=140); plt.close(fig)


def run_report(views,out,pairs=None,arms=("mappo","ctrl0"),label="source"):
    out=safe(out); out.mkdir(parents=True,exist_ok=True)
    source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ("report.py","io.py","detectors.py","discovery.py")}
    episodes=[]; all_events=[]; member_rows=[]; samples={}; discovery_rows=[]; input_rows=[]
    aggregate=defaultdict(lambda:dict(episodes=0,exposure_cav_s=0.,exposure_episode_s=0.,
                                    counts=Counter(),member_event_counts=Counter(),supported_counts=Counter(),
                                    eligible_episodes=Counter(),behaviour_cav_s=Counter(),behaviour_episode_s=Counter(),
                                    ramp_fit_gap_count=0))

    def consume(tape,identity,arm,reference=None,members=(),pair=None):
        result=detect(tape,reference,members,pair["kind"] if pair else "native_member_release")
        record=dict(source=label,episode=identity,arm=arm,seed=tape.meta.get("seed"),
                    exposure_cav_s=result["exposure_cav_s"],exposure_episode_s=result["exposure_episode_s"],
                    coverage=result["coverage"],counts=dict(Counter(e["behaviour"] for e in result["events"])))
        episodes.append(record)
        agg=aggregate[arm]; agg["episodes"]+=1
        for k in ("exposure_cav_s","exposure_episode_s"): agg[k]+=result[k]
        disabled=result["coverage"]["disabled_detectors"]
        for b in BEHAVIOURS:
            if b not in disabled:
                agg["eligible_episodes"][b]+=1
                agg["behaviour_cav_s"][b]+=result["exposure_cav_s"]
                agg["behaviour_episode_s"][b]+=result["exposure_episode_s"]
            else: record["counts"][b]=None
        per_member=defaultdict(Counter)
        for e in result["events"]:
            full=dict(e,source=label,episode=identity,arm=arm,seed=tape.meta.get("seed"))
            if pair is not None: full.update(intervention_members=list(members),intervention_id=identity)
            all_events.append(full)
            agg["counts"][e["behaviour"]]+=1
            agg["member_event_counts"][e["behaviour"]]+=len(e["members"])
            if e["behaviour"]=="gap_creation" and e["ramp_fit_observed"]: agg["ramp_fit_gap_count"]+=1
            if e["evidence"].startswith("paired_"): agg["supported_counts"][e["behaviour"]]+=1
            for j in e["members"]: per_member[j][e["behaviour"]]+=1
            key=(arm,e["behaviour"])
            score=e.get("opening_m",e["end_s"]-e["start_s"])
            if key not in samples or score>samples[key][0]: samples[key]=(score,tape,e)
        cav_ids={j for row in tape.rows for j,v in row.items() if v.cav}
        for j in sorted(cav_ids):
            exposure=sum((tape.time[i]-tape.time[i-1]) for i in range(1,len(tape.time))
                         if j in tape.rows[i-1] and tape.rows[i-1][j].cav and j in tape.rows[i])
            member_rows.append(dict(source=label,episode=identity,arm=arm,vid=j,exposure_cav_s=float(exposure),
                                    **{b:per_member[j][b] if b not in disabled else None for b in BEHAVIOURS}))
        input_rows.append(dict(episode=identity,meta=tape.meta,hashes=tape.hashes,coverage=tape.coverage,
                               reference_hashes=reference.hashes if reference else None))
        if pair is not None:
            fields=[pair.get(k) for k in ("C_G","D_G","sum_D_i")]
            if all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in fields):
                cg,dg,solo=fields
                if len(members)<2 and cg!=0:
                    raise ValueError("Singleton groups cannot have nonzero interaction credit")
                tolerance=np.finfo(float).eps*max(abs(cg),abs(dg),abs(solo),np.finfo(float).tiny)*len(members)
                if abs(cg-(dg-solo))>tolerance: raise ValueError("Unreconciled C_G in pair index")
                window=float(pair["intervention_window_s"])
                if window<=0: raise ValueError("Positive intervention window required for signatures")
                stop=tape.time[0]+window
                keep=[i for i,t in enumerate(tape.time) if t<=stop]
                cropped=Tape(tape.time[keep],[tape.rows[i] for i in keep],tape.meta,tape.eps)
                relevant=dict(events=[e for e in result["events"] if e["start_s"]<=stop])
                sig=signature(cropped,members,relevant)
                discovery_rows.append(dict(id=identity,C_G=cg,signature_window_s=[float(cropped.time[0]),
                    float(cropped.time[-1])],credit_outcome=pair.get("credit_outcome","unspecified"),**sig))
        print(json.dumps(dict(episode=identity,arm=arm,events=len(result["events"]))),flush=True)

    for root in views:
        for path,meta in manifest_entries(root,arms):
            consume(load_tape(path,meta),str(meta.get("episode",path.name)),meta["arm"])
    if pairs is not None:
        index=read_json(pairs)
        for pair in index["pairs"]:
            if pair.get("kind") not in ("native_member_release","policy_mean_intervention","policy_mean_reference"):
                raise ValueError("Explicit native-release or policy-mean intervention required")
            check_seed(pair["seed"])
            if len(set(pair["members"]))!=len(pair["members"]) or len(pair["members"])<1:
                raise ValueError("Pair index needs unique member identities")
            def resolve(name):
                p=Path(pair[name]); return safe(p if p.is_absolute() else Path(pairs).parent/p)
            meta={k:pair[k] for k in ("seed","arm","branch","trace_scope","trace_limitations",
                 "position_dtype_epsilon","source_design","source_initial_state_sha256") if k in pair}
            for key in ("trace_scope","trace_limitations","position_dtype_epsilon"):
                if key not in meta and key in index: meta[key]=index[key]
            if meta.get("trace_scope","").startswith("full_observable_slot_fleet"):
                meta["trace_scope"]="full_observable_fleet"
            actual,reference=load_trace(resolve("factual"),meta),load_trace(resolve("reference"),meta)
            for name,tape in (("factual",actual),("reference",reference)):
                expected=pair.get(name+"_sha256")
                if expected is not None and (len(tape.hashes)!=1 or next(iter(tape.hashes.values()))!=expected):
                    raise ValueError("Paired-source trace hash mismatch")
            consume(actual,str(pair["id"]),pair["arm"],reference,list(map(str,pair["members"])),pair)
    summaries={}
    for arm,agg in sorted(aggregate.items()):
        summaries[arm]={k:v for k,v in agg.items() if not isinstance(v,Counter)}
        summaries[arm]["behaviours"]={}
        for b in BEHAVIOURS:
            n=agg["counts"][b]; m=agg["member_event_counts"][b]
            eligible=agg["eligible_episodes"][b]; exposure=agg["behaviour_cav_s"][b]
            episode_exposure=agg["behaviour_episode_s"][b]
            summaries[arm]["behaviours"][b]=dict(count=n if eligible else None,member_event_count=m if eligible else None,
                eligible_episodes=eligible,eligible_exposure_cav_s=exposure,
                paired_supported_count=agg["supported_counts"][b],
                events_per_cav_hour=n*3600/exposure if exposure else None,
                member_events_per_cav_hour=m*3600/exposure if exposure else None,
                events_per_episode_hour=n*3600/episode_exposure if episode_exposure else None)
            sample=samples.get((arm,b))
            example_plot(sample[1] if sample else None,sample[2] if sample else None,out/f"{arm}_{b}.png")
    summary=dict(source=label,interpretation="Descriptive; event signs are not positive C_G evidence",
                 controllers=summaries,episodes=len(episodes),events=len(all_events),
                 rates="events/CAV-hour; member-events/CAV-hour; groups additionally events/episode-hour",
                 examples="largest gap opening, otherwise longest event, first source order breaks ties",
                 missing_behaviours="Zero observed events is conditional on recorded coverage, not proof of absence")
    summary["analysis_source_sha256"]=source_hashes
    summary["source_manifest_sha256"]={str(root):hashlib.sha256(safe(Path(root)/
        "C4_EPISODES_MANIFEST.json").read_bytes()).hexdigest() for root in views}
    if pairs: summary["pair_index_sha256"]=hashlib.sha256(safe(pairs).read_bytes()).hexdigest()
    write_json(out/"summary.json",summary); write_json(out/"episodes.json",episodes)
    write_json(out/"inputs.json",input_rows); write_json(out/"signatures.json",discovery_rows)
    write_json(out/"discovery.json",discover(discovery_rows))
    with gzip.open(out/"events.jsonl.gz","wt",encoding="utf-8") as stream:
        for e in all_events: stream.write(json.dumps(e,allow_nan=False)+"\n")
    with (out/"members.csv").open("w",newline="",encoding="utf-8") as stream:
        fields=["source","episode","arm","vid","exposure_cav_s",*BEHAVIOURS]
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(member_rows)
    write_json(out/"COMPLETE.json",dict(summary_sha256=hashlib.sha256((out/"summary.json").read_bytes()).hexdigest(),
                                       episodes=len(episodes),events=len(all_events)))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views",action="append",default=[])
    p.add_argument("--out",required=True); p.add_argument("--pairs")
    p.add_argument("--arms",nargs="+",default=["mappo","ctrl0"]); p.add_argument("--label",default="source")
    args=p.parse_args()
    if not args.views and not args.pairs: p.error("At least one --views or --pairs source required")
    run_report(args.views,args.out,args.pairs,args.arms,args.label)


if __name__=="__main__": main()
