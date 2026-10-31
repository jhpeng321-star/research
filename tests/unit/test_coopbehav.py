"""Observable geometry tests; no simulator, model, driver truth or network files."""
from dataclasses import replace
import json

import numpy as np
import pytest

from tools.coopbehav.detectors import detect, validate_release
from tools.coopbehav.io import Tape, Vehicle, load_tape, safe


def opening(merge=True, native=False):
    rows=[]
    for t in range(5):
        rows.append({
            "c":Vehicle(10*t-(0 if native else t*t),10-(0 if native else 2*t),5,1,True,
                        None if native else -2,0 if native else 3),
            "l":Vehicle(40+10*t,10,5,1,False),
            "r":Vehicle(20+10*t,10,5,1 if merge and t>=2 else 2,False),
            "f":Vehicle(-30+8*t-(.5*t*t if native else 0),8-(t if native else 0),5,1,False)})
    return Tape(np.arange(5),rows,{"seed":969550})


def events(tape,kind,**kw):
    return [e for e in detect(tape,**kw)["events"] if e["behaviour"]==kind]


def test_yield_requires_actual_insertion_into_opened_slot():
    ev=events(opening(),"yield_to_merge")
    assert len(ev)==1
    assert ev[0]["accepted_vehicle"]=="r"
    assert ev[0]["merge_time_interval_s"]==[1.,2.]
    assert ev[0]["opening_m"]==4
    assert ev[0]["headway_change_s"]>0
    assert ev[0]["evidence"]=="observational_candidate"


def test_member_release_support_and_acceptance_censoring():
    a,b=opening(),opening(merge=False,native=True)
    ev=events(a,"yield_to_merge",reference=b,released_members=["c"])[0]
    assert ev["evidence"]=="paired_release_yield_support"
    assert ev["extra_opening_vs_release_m"]==4
    assert ev["reference_acceptance_right_censored"]


def test_no_earlier_acceptance_is_not_supported_yield():
    ev=events(opening(),"yield_to_merge",reference=opening(native=True),released_members=["c"])[0]
    assert ev["evidence"]=="observational_candidate"


def test_gap_creation_without_acceptance_and_censoring():
    ev=events(opening(merge=False),"gap_creation")[0]
    assert ev["opening_m"]==16
    assert ev["ramp_fit_observed"]
    assert ev["accepted_vehicle"] is None
    assert ev["end_reason"]=="observation_end_censored"
    paired=events(opening(merge=False),"gap_creation",reference=opening(False,True),released_members=["c"])[0]
    assert paired["evidence"]=="paired_release_gap_support"


def test_gap_without_ramp_is_only_geometric_opening():
    tape=opening(False)
    for r in tape.rows: del r["r"]
    ev=events(tape,"gap_creation")[0]
    assert not ev["ramp_fit_observed"]


def test_leader_acceleration_without_cav_braking_is_not_yield():
    tape=opening()
    for i,r in enumerate(tape.rows):
        r["c"]=replace(r["c"],x=10*i,v=10)
        r["l"]=replace(r["l"],x=40+10*i+i*i,v=10+2*i)
    assert not events(tape,"yield_to_merge")
    assert not events(tape,"gap_creation")


def test_identity_loss_terminates_slot_and_never_counts_merge():
    tape=opening()
    for r in tape.rows[2:]: r["replacement"]=r.pop("c")
    assert not events(tape,"yield_to_merge")
    ev=events(tape,"gap_creation")[0]
    assert ev["end_reason"]=="identity_or_lane_censored"


def wall(offset=0,native=False):
    rows=[]
    for t in range(5):
        x=-25+10*t-(0 if native else t*t)
        v=10-(0 if native else 2*t)
        rows.append({"a":Vehicle(x,v,5,0,True,None if native else -2,0 if native else 3),
                     "b":Vehicle(x+offset,v,5,1,True,None if native else -2,0 if native else 3),
                     "fa":Vehicle(-30+10*t-(0 if native else t*t),v,5,0,False),
                     "fb":Vehicle(-30+10*t-(0 if native else t*t),v,5,1,False)})
    return Tape(np.arange(5),rows,{"seed":969551})


def test_wall_body_overlap_distinct_lanes_and_contiguous_timing():
    ev=events(wall(),"cross_lane_hold")
    assert len(ev)==1
    assert ev[0]["members"]==["a","b"]
    assert ev[0]["start_s"]==0 and ev[0]["end_s"]==4
    assert not events(wall(offset=6),"cross_lane_hold")
    tape=wall()
    for row in tape.rows: row["b"]=replace(row["b"],lane=0)
    assert not events(tape,"cross_lane_hold")


def test_wall_metering_requires_observed_later_followers_crossing():
    # Extend a slow wall and its followers until their actual crossings are observed.
    a,b=wall(),wall(native=True)
    for t in range(5,10):
        a.time=np.append(a.time,t); b.time=np.append(b.time,t)
        a.rows.append({j:replace(v,x=v.x+2) for j,v in a.rows[-1].items()})
        b.rows.append({j:replace(v,x=v.x+10) for j,v in b.rows[-1].items()})
    ev=events(a,"cross_lane_hold",reference=b,released_members=["a","b"])[0]
    assert ev["evidence"]=="paired_release_metering_support"
    assert all(d>0 for d in ev["follower_crossing_delays_s"].values())


def test_temporal_follower_damping_and_paired_same_identity_comparison():
    ev=events(opening(False),"wave_damping")[0]
    assert ev["followers"]==["f"]
    assert ev["follower_metrics"]["f"]["variance_m2_s2"]==0
    assert ev["evidence"]=="observational_attenuation"
    ev=events(opening(False),"wave_damping",reference=opening(False,True),released_members=["c"])[0]
    assert ev["evidence"]=="paired_release_variance_support"
    assert ev["follower_metrics"]["f"]["reduction_m2_s2"]==2


def test_no_fabricated_damping_when_follower_identity_changes():
    tape=opening(False)
    for row in tape.rows[1:]: del row["f"]
    assert not events(tape,"wave_damping")


def test_pair_guards_reject_initial_mismatch_and_external_reference():
    with pytest.raises(ValueError,match="external command"):
        validate_release(opening(),opening(False),["c"])
    bad=opening(False,True); bad.rows[0]["c"]=replace(bad.rows[0]["c"],x=1)
    with pytest.raises(ValueError,match="state mismatch"):
        validate_release(opening(),bad,["c"])


@pytest.mark.parametrize("path",["sealed/a","x/heldout/a","x/cal_split/a","chunk5/a","j4b/a",
    "inc1/a","hetdata_truth/x","vTypes.xml","s964100/x","s962123/x","s977456/x"])
def test_restricted_paths_fail_before_read(path):
    with pytest.raises(ValueError,match="Restricted"):
        safe(path)


def test_loader_ignores_history_and_never_carries_action_across_slot_recycling(tmp_path):
    t=np.arange(5,dtype=float)
    kin=np.zeros((5,1,7),np.float32); kin[:,:,0]=np.arange(5)[:,None]
    kin[:,:,2]=1; kin[:,:,4]=1; kin[:,:,6]=5
    arrays=dict(time_s=t,veh_kin=kin,veh_mask=np.ones((5,1),bool),is_cav=np.ones((5,1),bool),
                vid=np.array([[1],[1],[2],[2],[2]]),control_time_s=np.array([0.,3.]),
                controller=np.full((2,1),3),issued_acceleration_mps2=np.full((2,1),-.2))
    for name,a in arrays.items(): np.save(tmp_path/(name+".npy"),a)
    (tmp_path/"episode.json").write_text(json.dumps(dict(seed=969550,branch=dict(t0_s=1,horizon_s=3))))
    tape=load_tape(tmp_path)
    assert tape.time.tolist()==[1.,2.,3.,4.]
    assert tape.rows[0]["1"].issued==-.2
    assert tape.rows[1]["2"].issued is None
    assert tape.rows[2]["2"].issued==-.2
    assert tape.eps==np.finfo(np.float32).eps


def test_constant_speeds_do_not_generate_physical_events():
    tape=opening(False,True)
    for t,row in enumerate(tape.rows): row["f"]=replace(row["f"],v=8,x=-30+8*t)
    assert detect(tape)["events"]==[]


def test_nonfinite_clock_and_vehicle_geometry_rejected():
    with pytest.raises(ValueError): Tape([0,np.nan],[{},{}])
    with pytest.raises(ValueError): Tape([0,1],[{"c":Vehicle(0,1,0,1,True)},{}])
