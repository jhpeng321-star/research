"""Frozen recorded-state diagnostic; no simulation or fitting entry point."""
from pathlib import Path
import copy
from dataclasses import asdict, replace
import gzip
import hashlib
import inspect
import json
import os
import sys
import time
import traceback

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  CUDA_VISIBLE_DEVICES='')
ROOT = Path('/root/autodl-tmp/c4/c1pidiag20261031')
UNITS = ROOT/'units_v2'
EVIDENCE = ROOT/'evidence_v2'
REF = Path('/root/autodl-tmp/c4/c1refresh')
sys.path.insert(0, '/root/autodl-tmp/c4/speedhead20261030/tools/speedhead')
import wm_worker as frozen
sys.path.insert(0, str(REF/'code/tools/c1refresh'))
import c1r_run as legacy
import numpy as np
import torch
from wms.config.train import TrainConfig
from wms.eval.backend import WmsBackend, default_tools, feasibility_projection
from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.model.checkpoint import load_checkpoint
from wms.eval.normalize import recorded_for
from wms.eval.metrics import frame_at, present_by_id

BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')
HASHES = {k:v[2] for k,v in legacy.MODELS.items()}
HASHES['wm2'] = 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
CACHE = {}


def audit(event, args):
    if event == 'subprocess.Popen':
        raise PermissionError('inference cannot launch subprocesses')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).lower()
        if any(b in p for b in BAD) or p.endswith(('.xml', '.xml.gz')):
            raise PermissionError('excluded input')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, allow_nan=False, indent=1)
        f.flush()
        os.fsync(f.fileno())


def init():
    torch.set_num_threads(1)
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    sys.addaudithook(audit)
    legacy.install_pivot_diag_stub()


def record(e):
    assert not any(b in str(e).lower() for b in BAD)
    p = REF/'states/states'/(e['sid']+'.json.gz')
    assert digest(p) == e['file_sha256']
    with gzip.open(p, 'rt') as f:
        r = json.load(f)
    seed = int(r.get('seed', 0))
    assert not (964000 <= seed <= 964199 or 962000 <= seed <= 962999 or 977000 <= seed <= 977999)
    return r


def history(r, mode):
    r = copy.deepcopy(r)
    h = r.get('history')
    if mode == 'snapshot':
        from wms.eval.rollout_inputs import t0_vehicles
        _, ids, kin, _ = t0_vehicles(r)
        r['history'] = dict(ids=ids, kin=kin[None].tolist(), mask=[[True]*len(ids)],
            target_mask=[], history_s=60., history_stride_s=1., dt_s=.5,
            available_span_s=0., run_start_limited=True)
    else:
        seconds = float(mode[1:])
        assert h and float(h.get('available_span_s', 0)) >= seconds, 'genuine history unavailable'
        n = int(seconds/.5)+1
        for key in ('kin', 'mask'):
            h[key] = h[key][-n:]
        h['target_mask'] = h.get('target_mask', [])[-max(0, n-1):]
        h['available_span_s'] = seconds
        h['run_start_limited'] = seconds < 60
    return r


def predictor(tag, population, residual, lanes):
    key = (tag, population, residual, lanes)
    if key in CACHE:
        return CACHE[key]
    if tag == 'wm2':
        path = frozen.WM_PATH
        models, receipt = frozen.load_members([str(path)], [str(frozen.WM_CONFIG)], 'cpu')
        m = models[0]
        config = torch.load(path, map_location='cpu', weights_only=False)['config']
        projection = None
    else:
        path = REF/'ckpt'/tag/'wm_s.pt'
        kind = legacy.MODELS[tag][1]
        loaded = load_checkpoint(path, device='cpu', strict=True, model_kind=kind,
                                 config_factory=TrainConfig.from_legacy_flat)
        m, config = loaded.model.eval(), dict(loaded.legacy_config)
        projection = loaded.projection or feasibility_projection(config)
        receipt = dict(physics=asdict(m.physics))
    assert digest(path) == HASHES[tag]
    m.requires_grad_(False)
    m.cav_exec_cap = 'off'
    m.diag_encoder_population = population
    if lanes == 'mobil':
        # Entire learned lane pipeline -> deterministic physical MOBIL prior.
        for attr in ('lane_v5', 'lane_hazard', 'lane_change_head'):
            if hasattr(m, attr): setattr(m, attr, None)
        # Courtesy's pivot-intent features still require the frozen pivot net.
        # Keep that feature producer; bypass only its lane-decision application.
        if getattr(m, 'merge_pivot', None) is not None:
            m._apply_merge_pivot = lambda state, out, s, v, lane, *a, **kw: lane
        for attr in ('_apply_merge_threshold','_apply_merge_frailty','_apply_merge_core'):
            if hasattr(m, attr):
                setattr(m, attr, lambda state, kin, present, out, lane, *a, **kw: lane)
        m.lane_model, m.lane_head_mode, m.lane_decision = 'residual', 'off', 'argmax'
        changes = dict(lane_rule='mobil')
        if hasattr(m.physics, 'lane_learnable'): changes['lane_learnable'] = False
        if hasattr(m.physics, 'lane_change_secure_gap'): changes['lane_change_secure_gap'] = False
        m.physics = replace(m.physics, **changes)
    original = m.forward
    def forward(state, **kw):
        out = original(state, **kw)
        if not residual:
            out['accel_mps2'] = out['physics_accel_mps2'].clamp(m.physics.accel_min_mps2, m.physics.accel_max_mps2)
            out['accel_delta_mps2'] = torch.zeros_like(out['accel_mps2'])
            out['raw_residual_mps2'] = torch.zeros_like(out['accel_mps2'])
        if lanes == 'mobil':
            from wms.model.driver_encoder import PARAMETER_NAMES
            params = kw.get('driver_params')
            if params is None and 'driver_parameters' in out:
                params = {name:out['driver_parameters'][...,i] for i,name in enumerate(PARAMETER_NAMES)}
            out['lane_logits'] = m.mobil_merge_logits(state, params)
        return out
    m.forward = forward
    backend = WmsBackend(model=m, config=config, tools=default_tools(), projection=projection)
    pred = WmsPredictor(backend, RolloutFlags())
    info = dict(checkpoint=str(path), sha256=HASHES[tag], physics=asdict(m.physics),
                model_spec=asdict(m.spec), runtime=str(frozen.FIX), receipt=receipt)
    CACHE[key] = (pred, info)
    return pred, info


def endpoints(r, frames, arm):
    out = {}
    rec = recorded_for(r, arm)
    roster = {str(v['id']) for v in r['initial_state']['vehicles'] if not v.get('is_cav', False)}
    for h in (60, 120):
        p = present_by_id(frame_at(frames, r['t0_s']+h))
        t = present_by_id(frame_at(rec, r['t0_s']+h))
        ids = sorted(roster & set(p) & set(t))
        errors = {i:dict(ds=float(p[i]['s'])-float(t[i]['s']),
                         dv=float(p[i]['v'])-float(t[i]['v']),
                         lane_match=p[i]['lane']==t[i]['lane']) for i in ids}
        assert all(np.isfinite(v[k]) for v in errors.values() for k in ('ds','dv'))
        out[str(h)] = dict(ids=errors, n=len(ids), n_truth=len(roster & set(t)),
                           median=float(np.median([abs(v['ds']) for v in errors.values()])) if ids else None)
    return out


def run(u):
    e, tag, hist, pop, residual, lanes, mode = u
    name = '__'.join((e['sid'], tag, hist, 'pop' if pop else 'inf', 'r1' if residual else 'r0', lanes, mode))
    target = UNITS/(name+'.json')
    if target.exists():
        old = json.loads(target.read_text())
        assert old['unit'] == list(u)
        return old['error']
    start = time.monotonic()
    value = dict(unit=u, name=name, sid=e['sid'], world=e['world'], cohort=e['cohort'],
        tag=tag, history=hist, population=pop, residual=residual, lanes=lanes, mode=mode,
        pid=os.getpid(), error=None)
    try:
        r = history(record(e), hist)
        pred, info = predictor(tag, pop, residual, lanes)
        seed = legacy.unit_seed(r, e['arm'], 0)
        legacy.seed_all(seed)
        if mode == 'tf':
            frames = legacy.predict_teacher(pred, r, e['arm'], 120.)
        else:
            frames, _ = legacy.predict_closed(pred, r, e['arm'], 120., plant='head')
        value.update(endpoints=endpoints(r, frames, e['arm']), model=info, seed=seed)
    except Exception:
        value['error'] = traceback.format_exc()
    value['seconds'] = time.monotonic()-start
    write(target, value)
    return value['error']


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    init()
    panel = json.loads((ROOT/'PANEL.json').read_text())['states']
    if args.smoke:
        for tag in ('c1struct', 'wm13e', 'wm2', 'bbtraj'):
            e = panel[0]
            print(tag, run((e,tag,'snapshot',False,True,'original','tf')), flush=True)
