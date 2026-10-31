import gzip
import json
from dataclasses import replace

import pytest

from tools.coopbehav.report import load_trace, run_report, signature
from tools.coopbehav.detectors import detect
from tests.unit.test_coopbehav import opening


def export(tape,path):
    with gzip.open(path,"wt",encoding="utf-8") as f:
        for t,row in zip(tape.time,tape.rows):
            f.write(json.dumps(dict(time_s=float(t),vehicles=[dict(vid=j,s_m=v.x,v_mps=v.v,
                length_m=v.length,lane=v.lane,is_cav=v.cav,controller=v.controller,
                issued_acceleration_mps2=v.issued) for j,v in row.items()]))+"\n")


def test_complete_paired_report_has_counts_member_rates_plots_and_credit_clustering(tmp_path):
    a,b=opening(),opening(False,True)
    for r in a.rows: r["l"]=replace(r["l"],cav=True,controller=3,issued=0.)
    for r in b.rows: r["l"]=replace(r["l"],cav=True,controller=0)
    export(a,tmp_path/"group.jsonl.gz")
    export(b,tmp_path/"baseline.jsonl.gz")
    index=dict(pairs=[dict(id="synthetic",factual="group.jsonl.gz",reference="baseline.jsonl.gz",
        members=["c","l"],kind="native_member_release",seed=969550,arm="mappo",C_G=1.,D_G=2.,sum_D_i=1.,
        branch=dict(t0_s=0,horizon_s=4),intervention_window_s=4,trace_scope="full_observable_fleet")])
    (tmp_path/"index.json").write_text(json.dumps(index))
    out=tmp_path/"report"
    result=run_report([],out,tmp_path/"index.json")
    assert result["controllers"]["mappo"]["behaviours"]["yield_to_merge"]["paired_supported_count"]==1
    assert result["controllers"]["mappo"]["exposure_cav_s"]==8
    for kind in ("yield_to_merge","gap_creation","cross_lane_hold","wave_damping"):
        assert (out/f"mappo_{kind}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads((out/"discovery.json").read_text())["counts"]["ranked"]==1
    assert (out/"members.csv").read_text().count("synthetic")==2
    assert json.loads((out/"COMPLETE.json").read_text())["episodes"]==1


def test_signatures_keep_state_action_and_fixed_neighbour_evidence():
    tape=opening(False)
    sig=signature(tape,["c"],detect(tape))
    assert sig["features"]["member_issued_acceleration_mps2"]==-2
    assert sig["features"]["member_speed_change_mps"]==-8
    assert sig["initial_neighbours"]==["f","l"]
    assert "gap_creation" in sig["explained_by"]


def test_jsonl_reader_never_infers_missing_vehicle_length(tmp_path):
    path=tmp_path/"trace.jsonl"
    path.write_text(json.dumps(dict(time_s=0,vehicles=[dict(vid="x",s_m=0,v_mps=1,lane=1,is_cav=True)]))+"\n")
    with pytest.raises(KeyError,match="length_m"):
        load_trace(path,dict(seed=969550))
