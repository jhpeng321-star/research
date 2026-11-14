"""Queue-only centre fairness: no withdrawal or rescoring of active requests."""
import json
import os
from .ls_broker import request_order


def put(root,name,value,stamp):
    d=root/'requests';d.mkdir(exist_ok=True);p=d/(name+'.json')
    p.write_text(json.dumps(value));os.utime(p,(stamp,stamp));return p


def test_centre_precedes_unassigned_long_search_without_changing_record(tmp_path):
    b=put(tmp_path,'B',{'group':['a','b']},10)
    r=put(tmp_path,'R',{'mode':'centre','group':['c','d']},20)
    before={p:p.read_bytes() for p in (b,r)}
    assert request_order(tmp_path)==[r,b]
    assert {p:p.read_bytes() for p in (b,r)}==before


def test_fifo_retained_within_both_classes(tmp_path):
    b1=put(tmp_path,'B1',{},10);b2=put(tmp_path,'B2',{},11)
    r1=put(tmp_path,'R1',{'mode':'centre'},20);r2=put(tmp_path,'R2',{'mode':'centre'},21)
    assert request_order(tmp_path)==[r1,r2,b1,b2]


def test_malformed_request_does_not_break_priority_index(tmp_path):
    b=put(tmp_path,'bad',{},10);b.write_text('{broken')
    r=put(tmp_path,'R',{'mode':'centre'},20)
    assert request_order(tmp_path)==[r,b]
