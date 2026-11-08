"""CLEAN5C4 privileged teacher gate (MAIN 2026-11-08): the offer gate may read the full plant table at train time.
(1) offline/runtime parity of onset_bin('privileged') on a recorded tape slice (wmv_p20) via the host reducer
read path and via the LCEXEC protocol path into offer_ok; (2) the policy observation / sample is byte-identical
with the teacher gate on and off; (3) the registered privileged admission result."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

from . import clean5b_restart as c5b, fullpol_bank as fb, fullpol_situational as fs, lc_families as lf
from .test_lc_families_cpu import (Capture, Prefer, canon, controller, e2e_rows, lc_cfg, make, step,  # noqa: F401
                                   wz)
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401

C4SIT = Path(__file__).resolve().parents[2]/'clean5c'/'c4sit'


def _reducer():
    spec = importlib.util.spec_from_file_location('sit_reduce_under_test', C4SIT.parents[1]/'ops'/'sit_reduce.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slice():
    tapes = {}
    for line in gzip.open(C4SIT/'fixture'/'c4sit_tape_slice.jsonl.gz', 'rt'):
        d = json.loads(line)
        tapes[(d['job'], int(round(d['time_s'])))] = d['vehicles']
    return tapes, json.loads((C4SIT/'fixture'/'c4sit_acts.json').read_text(encoding='utf-8'))


def test_privileged_parity_offline_runtime_on_recorded_tape(tmp_path, cfg, wz):
    tapes, acts = _slice()
    red = _reducer()
    assert {a['ponset_bin']['cong'] for a in acts} == {'free', 'congested'}
    for a in acts:
        # offline: the host reducer's tape read + onset_bin('privileged') == the recorded reducer output ==
        # the original S-MULTICRIT-ACT bin of the same activation (full-tape features, verbatim)
        path = tmp_path/f"{a['job']}.jsonl.gz"
        with gzip.open(path, 'wt') as fh:
            fh.write(json.dumps(dict(time_s=a['t'], vehicles=tapes[(a['job'], a['t'])]))+'\n')
        rows = red.tape_rows(path, [a['t']])[a['t']]
        own = next(r for r in rows if r['vid'] == a['holder'])
        offline = fs.onset_bin(own, rows, 'privileged')
        assert offline == a['ponset_bin'] == fs.situation_bin(a['sit'])
        # runtime: the same tape rows as the backend table -> LCEXEC protocol (privileged_rows) -> state ->
        # generator view -> offer_ok(teacher_rows) -> the same function
        table = {r['vid']: r for r in tapes[(a['job'], a['t'])]}
        cavs = {v: r for v, r in table.items() if r['is_cav']}
        proto = lf.Protocol(lf.settings(lc_cfg(cfg, cap=3)))
        sensing = {v: [r for r in table.values() if abs(r['s_m']-cavs[v]['s_m']) <= fs.SENSOR_RANGE_M] for v in cavs}
        proto.receive(a['t'], cavs, sensing, {}, privileged_rows=list(table.values()))
        state = proto.state(a['holder'], a['t'])
        teacher = state['teacher_rows']
        assert [dict(r) for r in teacher] == [lf._row(r) for r in table.values()]
        sensed = [r for r in state['sensed'] if r['vid'] != a['holder']]
        cond = lambda b: dict(situation=dict(rule=fs.RULE, mode='privileged', bins=[b]))  # noqa: E731
        own_rt = next(r for r in state['sensed'] if r['vid'] == a['holder'])
        assert fs.situation_ok(cond(offline), own_rt, sensed, teacher) is True
        other = dict(offline, cong='congested' if offline['cong'] == 'free' else 'free')
        assert fs.situation_ok(cond(other), own_rt, sensed, teacher) is False
        assert fs.situation_ok(cond(offline), own_rt, sensed, None) is False          # no table: closed
        # without the table the protocol state carries no teacher rows (default path unchanged)
        proto.receive(a['t'], cavs, sensing, {})
        assert 'teacher_rows' not in proto.state(a['holder'], a['t'])


def test_offer_ok_threads_teacher_rows_only_when_present(monkeypatch):
    from . import fullpol_templates as ft
    seen = []
    real = fb.offer_ok
    monkeypatch.setattr(fb, 'offer_ok', lambda c, o, rows, **kw: seen.append(kw) or real(c, o, rows, **kw))
    fp_cls = type('FP', (), {'from_dict': staticmethod(lambda d: d)})
    monkeypatch.setattr(ft, '_lc', lambda: type('LC', (), {'FreeProgram': fp_cls}))
    monkeypatch.setattr(ft, '_view', lambda frame: None)
    cond = dict(trigger='lanes', own_wm_lanes=[0], zone_s=[-1800., -400.], spread_m=300.,
                roles=[dict(wm_lane=0), dict(wm_lane=1)], gate=None, start_gate=None, slots=1, cooldown_s=120,
                situation=dict(rule=fs.RULE, mode='privileged', bins=[dict(cong='free', lane_diff='L2notfaster')]))
    tpl = {'x': 1, 'source': dict(variant='wm_x')}
    slot = dict(classes=['verified'], templates=[dict(template=tpl, offer_condition=cond)])
    r = lambda v, s, ln: dict(vid=v, s_m=float(s), lane=ln, v_mps=25., is_cav=True)  # noqa: E731
    own, pair = r('e', -1000, 0), [r('e', -1000, 0), r('p', -1100, 1)]
    state = dict(frame=None, rows=pair, fullpol=slot, now=500, ledger={})
    rng = np.random.default_rng(0)
    assert ft.propose(state, own, rng) == [] and seen[-1] == {}                         # privileged, no table
    full = pair+[r('q', -200, 1)]
    assert ft.propose(dict(state, teacher_rows=full), own, rng) == [tpl] and seen[-1] == dict(teacher_rows=full)


def test_policy_observation_byte_identical_gate_on_off(cfg, wz):
    """Same episode with the teacher gate on (privileged templates -> the backend hands the full table to the
    protocol) vs off: identical samples (incl. every observation tensor) and requests; the table never enters
    a sample."""
    outs = {}
    for mode in ('privileged', 'own_sensing'):
        raw = lc_cfg(cfg, cap=3)
        raw['lc_families']['generator_args'] = dict(templates=dict(B=[dict(template={}, offer_condition=dict(
            situation=dict(rule=fs.RULE, mode=mode, bins=[])))], R=[]))
        p = make(raw)
        assert p.teacher_gate is (mode == 'privileged')
        p.trigger.trigger_head = Prefer(p.trigger)
        c = controller(p)
        events = []
        for t in range(300, 306):
            env = Capture()
            s, req = step(p, c, e2e_rows(t), t, env)
            events.append([s, req, env.ticks])
            assert (p.lc_families.teacher_rows is not None) is (mode == 'privileged')
        outs[mode] = canon(events)
    assert outs['privileged'] == outs['own_sensing'] and 'teacher' not in outs['privileged']


def test_needs_teacher():
    t = lambda mode: dict(offer_condition=dict(situation=dict(mode=mode)))  # noqa: E731
    assert fs.needs_teacher(dict(templates=dict(B=[t('privileged')], R=[t('privileged')])))
    assert not fs.needs_teacher(dict(templates=[t('own_sensing')])) and not fs.needs_teacher({})
    assert not fs.needs_teacher(None)


def test_registered_privileged_admission_doc(cfg, wz):
    from .lc_families import CONFIG_KEY
    doc = json.loads((C4SIT/'GUIDANCE_C4SIT_priv.json').read_text(encoding='utf-8'))
    adm = [p for p in doc['situational']['pairs'] if p['admitted']]
    assert [(p['family'], p['bin'], p['effect']['n'], p['effect']['clusters']) for p in adm] == [
        ('mut_006c290e03ad3d3c', dict(cong='free', lane_diff='L2notfaster'), 15, 9)]
    assert adm[0]['effect']['hi'] < 0 and doc['situational']['mode'] == 'privileged'
    assert [(r['regime'], r['family'], r['arm']) for r in doc['rows']] == [('D2800_p20', 'mut_006c290e03ad3d3c', 'B')]
    assert c5b.check_p20_guidance(doc, c5b.P20_CELLS, wm=True) == ('D2400_p20', 'D3200_p20')
    out = dict(cfg)
    out.update(c5b.keys_c(desc=wz, guidance_doc=doc, provenance='c4sit test'))
    tpl = out[CONFIG_KEY]['generator_args']['templates']
    assert tpl['B'] == tpl['R'] and len(tpl['B']) == 1
    assert tpl['B'][0]['offer_condition']['situation']['mode'] == 'privileged'
    assert fs.needs_teacher(out[CONFIG_KEY]['generator_args'])
