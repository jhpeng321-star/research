"""CLEAN5C4 privileged teacher bin (MAIN 2026-11-08; REV-SPD B1 fix): the privileged bin is used ONLY where B's
EV-BC target is chosen (offline, recorded tape, no RNG). Slate candidacy uses only the non-privileged conditions.
(1) offline reducer / training-time teacher parity on a recorded tape slice; (2) with the gate OPEN, the slate,
samples, observations and action RNG draws are identical with privileged mode on and off; (3) only B's EV-BC
target labels differ; (4) the registered doc and its LF file sha."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from . import clean5b_restart as c5b, fullpol_learn as fl, fullpol_situational as fs, lc_families as lf
from .test_lc_families_cpu import (Capture, Prefer, canon, controller, e2e_rows, lc_cfg, make, step,  # noqa: F401
                                   wz)
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401

C4SIT = Path(__file__).resolve().parents[2]/'clean5c'/'c4sit'
DOC = C4SIT/'GUIDANCE_C4SIT_priv.json'
FAM = 'mut_006c290e03ad3d3c'
OPEN = dict(cong='free', lane_diff='L2notfaster')


def _doc():
    return json.loads(DOC.read_bytes())


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


def _tape(path, t, vehicles):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as fh:
        fh.write(json.dumps(dict(time_s=float(t), vehicles=vehicles))+'\n')
    return path


def test_privileged_parity_reducer_vs_training_teacher_on_recorded_tape(tmp_path):
    tapes, acts = _slice()
    red = _reducer()
    assert {a['ponset_bin']['cong'] for a in acts} == {'free', 'congested'}
    for a in acts:
        path = _tape(tmp_path/a['job']/'observable.jsonl.gz', a['t'], tapes[(a['job'], a['t'])])
        rows = red.tape_rows(path, [a['t']])[a['t']]
        own = next(r for r in rows if r['vid'] == a['holder'])
        offline = fs.onset_bin(own, rows, 'privileged')                        # host reducer path
        assert offline == a['ponset_bin'] == fs.situation_bin(a['sit'])       # == the S-MULTICRIT-ACT bin
        teacher = fs.TapeTeacher(path)                                         # training-time path (same tape)
        assert teacher.bin(a['holder'], a['t']) == offline
        row = dict(family=FAM, situational=dict(mode='privileged', bins=[offline]))
        assert fs.teacher_admit(row, teacher, a['holder'], a['t']) is True
        other = dict(offline, cong='congested' if offline['cong'] == 'free' else 'free')
        assert fs.teacher_admit(dict(row, situational=dict(mode='privileged', bins=[other])), teacher,
                                a['holder'], a['t']) is False
        assert fs.teacher_admit(row, teacher, a['holder'], a['t']+1) is False         # second not recorded
        assert fs.teacher_admit(row, fs.TapeTeacher(tmp_path/'missing.gz'), a['holder'], a['t']) is False
        assert fs.teacher_admit(dict(family=FAM), None, a['holder'], a['t']) is True  # no block: unchanged


def _wall_rows():
    """Acting CAV on WM lane 1 with one CAV on every WM lane 0/1/2 inside 200 m, all in the trigger zone."""
    def r(vid, lane, s, cav=True, v=25.):
        return dict(vid=vid, s_m=float(s), v_mps=float(v), lane=lane, is_cav=cav, length_m=5.)
    return [r('e', 1, -1000.), r('p0', 0, -1050.), r('p2', 2, -950.), r('h', 1, -1080., cav=False)]


def _settings(cfg, cond):
    tpl = _doc()['clean5c']['templates'][0]
    entry = dict(template=deepcopy(tpl['template']), offer_condition=cond, family=FAM, cells=['D2800_p20'])
    raw = lc_cfg(cfg, cap=3, generator=None)
    from .fullpol_ev import GENERATOR
    raw['lc_families'].update(generator=GENERATOR,
                              generator_args=dict(classes=['verified'], structured=[], templates=dict(B=[entry], R=[entry])))
    return lf.settings(raw)


@pytest.mark.parametrize('bins', [[OPEN], [dict(cong='congested', lane_diff='L2faster')]])
def test_slate_and_rng_identical_privileged_on_off(cfg, wz, bins):
    """The real admitted template on the real LCEXEC slate path: the ProgramOption slate and the generator RNG
    stream are identical whether the template carries the privileged situation (open or closed) or none."""
    cond = deepcopy(_doc()['clean5c']['templates'][0]['offer_condition'])
    plain = {k: v for k, v in cond.items() if k != 'situation'}
    priv = dict(plain, situation=dict(cond['situation'], bins=bins))
    out = {}
    for name, c in (('plain', plain), ('priv', priv)):
        proto = lf.Protocol(dict(_settings(cfg, c), arm='B'))
        rows = _wall_rows()
        table = {r['vid']: r for r in rows}
        cavs = {v: r for v, r in table.items() if r['is_cav']}
        proto.receive(500, cavs, {v: [r for r in rows if abs(r['s_m']-cavs[v]['s_m']) <= 100.] for v in cavs}, {})
        rng = np.random.default_rng(3)
        options = lf.pool('free_program_initiate', proto.state('e', 500), table['e'], rng)
        out[name] = (len(options), canon([repr(o) for o in options]), json.dumps(rng.bit_generator.state))
    assert out['plain'] == out['priv']
    assert out['plain'][0] >= 1                                           # the template IS on the slate (open)


def test_episode_samples_identical_with_privileged_templates(cfg, wz):
    """A controlled episode (samples incl. every observation tensor, requests, env ticks) is identical with the
    privileged-situation templates (gate open) and with the same templates without a situation block."""
    cond = deepcopy(_doc()['clean5c']['templates'][0]['offer_condition'])
    outs = {}
    for name, c in (('plain', {k: v for k, v in cond.items() if k != 'situation'}), ('priv', cond)):
        raw = lc_cfg(cfg, cap=3)
        entry = dict(template=deepcopy(_doc()['clean5c']['templates'][0]['template']), offer_condition=c)
        raw['lc_families']['generator_args'] = dict(templates=dict(B=[entry], R=[entry]))
        p = make(raw)
        p.trigger.trigger_head = Prefer(p.trigger)
        ctl = controller(p)
        events = []
        for t in range(300, 306):
            env = Capture()
            s, req = step(p, ctl, e2e_rows(t), t, env)
            events.append([s, req, env.ticks])
        outs[name] = canon(events)
    assert outs['plain'] == outs['priv'] and 'privileged' not in outs['priv']


def _trigger(vid):
    tpl = _doc()['clean5c']['templates'][0]['template']
    entry = dict(family='free_program_initiate', template=tpl)
    return dict(context=dict(members=[vid]), tables=[[None, entry]], mask=[[True, True]])


def test_only_b_ev_bc_target_labels_differ(tmp_path, cfg):
    """B's EV-BC target on the same recorded slate: the family where the teacher bin is admitted, none where it is
    not; without the privileged block the family is targeted regardless. No RNG is used."""
    tapes, acts = _slice()
    rows_doc = {r['family']: r for r in _doc()['rows']}
    assert rows_doc[FAM]['situational']['mode'] == 'privileged' and rows_doc[FAM]['situational']['bins'] == [OPEN]
    plain_rows = {FAM: {k: v for k, v in rows_doc[FAM].items() if k != 'situational'}}
    labels = {}
    run = dict(cfg, run_root=str(tmp_path))
    for a in acts:
        ep = dict(id=f"ep_{a['job']}", world=dict(cell_key='D2800_p20'))
        _tape(tmp_path/'jobs'/ep['id']/'observable.jsonl.gz', a['t'], tapes[(a['job'], a['t'])])
        counts = Counter()
        admit = fl._teacher(run, ep, rows_doc, counts)
        before = np.random.get_state()[1].copy()
        priv = fl.ev_targets(run, _trigger(a['holder']), rows_doc, admit(_trigger(a['holder']), a['t']))
        plain = fl.ev_targets(run, _trigger(a['holder']), plain_rows)
        assert fl._teacher(run, ep, plain_rows, Counter()) is None                       # no block: no teacher
        assert (np.random.get_state()[1] == before).all()
        labels[a['ponset_bin']['cong']] = (priv[0][0], plain[0][0], dict(counts))
    assert labels['free'] == (1, 1, {'teacher_admitted': 1})                    # admitted bin: same target
    assert labels['congested'] == (0, 1, {'teacher_blocked': 1})                # other bin: no target (B only)


def test_r_gets_no_targets_and_b_r_share_the_slate(cfg, wz):
    from .lc_families import CONFIG_KEY
    doc = _doc()
    assert all(r['arm'] == 'B' for r in doc['rows'])
    out = dict(cfg)
    out.update(c5b.keys_c(desc=wz, guidance_doc=doc, provenance='c4sit test'))
    tpl = out[CONFIG_KEY]['generator_args']['templates']
    assert tpl['B'] == tpl['R'] and len(tpl['B']) == 1
    table_b, _ = fl.ev_table(out, 'B')
    table_r, receipt_r = fl.ev_table(out, 'R')
    assert list(table_b) == ['D2800_p20'] and list(table_b['D2800_p20']) == [FAM]
    assert table_r == {} and receipt_r['r_without_matched_sweep']


def test_registered_doc_and_file_sha():
    data = DOC.read_bytes()
    assert b'\r' not in data
    sha = hashlib.sha256(data).hexdigest()
    assert (C4SIT/'GUIDANCE_C4SIT_priv.json.sha256').read_text(encoding='utf-8').split()[0] == sha
    doc = json.loads(data)
    adm = [p for p in doc['situational']['pairs'] if p['admitted']]
    assert [(p['family'], p['bin'], p['effect']['n'], p['effect']['clusters']) for p in adm] == [(FAM, OPEN, 15, 9)]
    assert adm[0]['effect']['hi'] < 0 and doc['situational']['mode'] == 'privileged'
    assert doc['source']['rule'].startswith(fs.RULE)
    assert c5b.check_p20_guidance(doc, c5b.P20_CELLS, wm=True) == ('D2400_p20', 'D3200_p20')
