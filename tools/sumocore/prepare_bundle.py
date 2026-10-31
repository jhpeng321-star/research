"""Runtime direct-insertion adapter, not a saved-state/static-export shortcut.

Uses authoritative existing geometry. The shared helper's static XML exporter
is consumed for cross-checking when available; this adapter prepares synthetic
future demand and a TraCI insertion bundle rather than exporting original types.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
from demand import sample,stream
from inference import infer,sumo_attributes

BASE=Path('/home/ubuntu/wmdata20261031/jobs')
NET=Path('/home/ubuntu/e3final/runtime/scenarios/idm_v5_7_extended_zone/osm.v5_7.extended.net.xml')
CHECKPOINT=Path('/home/ubuntu/e3final/inputs/wm_s_best.pt')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def network():
    root=ET.parse(NET).getroot()
    lanes={r.attrib['id']:dict(length=float(r.attrib['length']),edge=e.attrib['id'],
                              index=int(r.attrib['index']))
           for e in root.findall('edge') if not e.attrib['id'].startswith(':') for r in e.findall('lane')}
    successors={}
    for c in root.findall('connection'):
        a,b=c.attrib['from'],c.attrib['to']
        if not a.startswith(':') and not b.startswith(':'):successors.setdefault(a,set()).add(b)
    def route(edge):
        edges=[edge]
        while edges[-1] in successors:
            nxt=successors[edges[-1]];assert len(nxt)==1,('ambiguous visible destination',edges[-1],nxt)
            value=next(iter(nxt));assert value not in edges
            edges.append(value)
        return edges
    return lanes,route


def prepare(job,out):
    from traffic_shepherd.c4.geometry import LANE_TABLE,HERE_OFFSETS
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    p=BASE/f"s{job['world']}_native"
    start=torch.load(p/'start.pt',map_location='cpu',weights_only=False)
    require_development_case(job['world'],start['demand'],'exposed');check_case(job['world'],start['demand'])
    with np.load(p/'start_history.npz',allow_pickle=False) as z:raw=dict(z)
    state=start['state'];active=state['present'][0].numpy()
    assert not bool((state['junction'][0]&state['present'][0]).any())
    lanes,route=network()
    requests,estimate=sample(start,job['world'],job['rng'],job['horizon'])
    params={};inference_receipt=None
    if job['variant']=='inferred':params,inference_receipt=infer(start,raw,CHECKPOINT)
    vehicles=[];cavs=[];hdv=[]
    for k in np.flatnonzero(active):
        vid=start['vehicle_ids'][int(raw['vid_index'][-1,k])]
        lane=int(state['lane'][0,k]);s=float(state['s_m'][0,k])
        candidates=[]
        for (edge,index),(model_lane,offset) in LANE_TABLE.items():
            if model_lane!=lane:continue
            lane_id=f'{edge}_{index}';pos=s+HERE_OFFSETS[lane]-offset
            if lane_id in lanes and 0<=pos<lanes[lane_id]['length']:
                candidates.append((offset,lane_id,pos,edge,index))
        assert len(candidates)==1,('observable inverse geometry ambiguous',vid,s,lane,candidates)
        _,lane_id,pos,edge,index=candidates[0]
        cav=bool(state['is_cav'][0,k])
        if cav:cavs.append(vid)
        else:hdv.append(vid)
        vehicles.append(dict(id=vid,lane_id=lane_id,lane_index=index,position=pos,
                             speed=float(state['v_mps'][0,k]),route=route(edge),is_cav=cav))
    donors=[params[v] for v in hdv] if params else []
    type_rng=stream(job['world'],job['rng'],'entrant.settings')
    xml=ET.Element('routes')
    type_cache={}
    def typename(attrs):
        key=json.dumps(attrs,sort_keys=True)
        if key not in type_cache:
            name='sc.'+str(len(type_cache));type_cache[key]=name
            ET.SubElement(xml,'vType',dict(id=name,carFollowModel='IDM',laneChangeModel='LC2013',sigma='0',
                                          **{k:str(v) for k,v in attrs.items()}))
        return type_cache[key]
    # All types precede vehicles in XML; initial population inserted via TraCI.
    for v in vehicles:
        attrs=sumo_attributes() if v['is_cav'] or not params else params[v['id']]
        v['type']=typename(attrs)
    for r in requests:
        attrs=sumo_attributes() if r['is_cav'] or not donors else donors[int(type_rng.integers(len(donors)))]
        r['type']=typename(attrs)
        if r['is_cav']:cavs.append(r['id'])
    entrances={name:next(edge for (edge,index),(lane,offset) in LANE_TABLE.items()
                          if offset==0 and (lane==2)==(name=='ramp')) for name in ('mainline','ramp')}
    for name,edge in entrances.items():ET.SubElement(xml,'route',dict(id=name,edges=' '.join(route(edge))))
    for r in requests:
        ET.SubElement(xml,'vehicle',dict(id=r['id'],type=r['type'],route=r['entrance'],
                                      depart=str(r['depart']),departLane='best',departSpeed='max'))
    routes=out/'generated.rou.xml';ET.ElementTree(xml).write(routes,encoding='utf-8',xml_declaration=True)
    config=ET.Element('configuration');inputs=ET.SubElement(config,'input')
    ET.SubElement(inputs,'net-file',value=str(NET));ET.SubElement(inputs,'route-files',value=str(routes))
    processing=ET.SubElement(config,'processing')
    ET.SubElement(processing,'time-to-teleport',value='-1')
    ET.SubElement(processing,'collision.action',value='warn')
    conf=out/'generated.sumocfg';ET.ElementTree(config).write(conf,encoding='utf-8',xml_declaration=True)
    receipt=dict(world=job['world'],t0=300,inputs={str(x):sha(x) for x in (p/'start.pt',p/'start_history.npz',NET)},
        demand=estimate,settings=inference_receipt,initial_count=len(vehicles),
        future_request_count=len(requests),initial_cav_count=sum(v['is_cav'] for v in vehicles),
        future_request_sha256=hashlib.sha256(json.dumps([{k:v for k,v in r.items() if k!='type'} for r in requests],sort_keys=True).encode()).hexdigest(),
        generated_routes_sha256=sha(routes),initial_vehicle_sha256=hashlib.sha256(json.dumps(vehicles,sort_keys=True).encode()).hexdigest(),
        hidden_state_loaded=False,truth_access=False)
    with (out/'input_receipt.json').open('x') as f:json.dump(receipt,f,indent=1)
    return dict(vehicles=vehicles,cav_ids=cavs,requests=requests,receipt=receipt),conf
